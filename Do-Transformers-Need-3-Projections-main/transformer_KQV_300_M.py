"""
Complete GPT Training Pipeline with QKV Attention
Modified to use locally downloaded SlimPajama data
Single file, ready to run on 4× A100 80GB

Usage:
    python train_qkv_local_data.py

Requirements:
    pip install torch transformers datasets accelerate clearml flash-attn tqdm pyyaml
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import math
import time
import yaml
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer
from datasets import load_from_disk  # CHANGED: Use load_from_disk instead of load_dataset
from tqdm import tqdm
import numpy as np

try:
    from flash_attn import flash_attn_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False
    print("⚠️  Flash Attention not available, using standard attention (will be slower)")

try:
    from clearml import Task
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    print("⚠️  ClearML not available, logging will be limited")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    vocab_size: int = 50304
    n_positions: int = 2048
    n_layer: int = 20
    n_embd: int = 1024
    n_head: int = 16
    n_inner: int = 4096
    
    embd_pdrop: float = 0.0
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.0
    
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_bias: bool = True
    tie_word_embeddings: bool = True
    
    use_flash_attention: bool = True


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Data paths - CHANGED: Add paths to local datasets
    train_data_path: str = "./slimpajama_train"
    val_data_path: str = "./slimpajama_validation"
    
    # Training
    total_tokens: int = 10_000_000_000  # CHANGED: 10B tokens (matches your download)
    micro_batch_size: int = 6          # Per GPU
    gradient_accumulation: int = 24
    
    num_workers: int = 16  
    prefetch_factor: int = 6 

    # Optimizer
    learning_rate: float = 2e-4
    min_lr: float = 2e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Schedule
    warmup_steps: int = 500
    lr_scheduler: str = "cosine"
    
    # Checkpointing
    save_interval: int = 1000
    eval_interval: int = 200
    log_interval: int = 10
    
    # Evaluation
    eval_tokens: int = 10_000_000  # 10M tokens
    
    # System
    seed: int = 42
    output_dir: str = "./outputs_kqv"
    
    # ClearML
    clearml_project: str = "GPT-QKV-Comparison"
    clearml_task: str = "GPT-300M-QKV-10B"  # CHANGED: Updated task name


# ============================================================================
# MODEL: QKV ATTENTION (unchanged)
# ============================================================================

class QKVAttention(nn.Module):
    """Multi-head QKV attention (standard transformer)"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.attn_pdrop
        self.use_flash = config.use_flash_attention and FLASH_AVAILABLE
        
        # QKV projection: 3× parameters (Q, K, V)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.use_bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.use_bias)
        
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_positions, config.n_positions))
            .view(1, 1, config.n_positions, config.n_positions)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        # Calculate Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        
        if self.use_flash:
            # Flash Attention
            out = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=True
            )
        else:
            # Standard attention
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2)
        
        out = out.contiguous().view(B, T, C)
        out = self.c_proj(out)
        out = self.resid_dropout(out)
        
        return out


class MLP(nn.Module):
    """Feed-forward network"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner, bias=config.use_bias)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd, bias=config.use_bias)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.resid_pdrop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = QKVAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT Language Model with QKV attention"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        
        self.h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        if config.tie_word_embeddings:
            self.lm_head = lambda x: F.linear(x, self.wte.weight)
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = input_ids.device
        b, t = input_ids.size()
        
        tok_emb = self.wte(input_ids)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.wpe(pos)
        
        x = self.drop(tok_emb + pos_emb)
        
        for block in self.h:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.n_positions else idx[:, -self.config.n_positions:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


# ============================================================================
# DATA LOADING - CHANGED: Use local datasets
# ============================================================================

class LocalSlimPajamaDataset(Dataset):
    """Dataset from locally downloaded SlimPajama data"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        seq_length: int = 2048,
        max_examples: Optional[int] = None
    ):
        print(f"Loading dataset from {data_path}...")
        self.dataset = load_from_disk(data_path)
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        if max_examples is not None:
            self.dataset = self.dataset.select(range(min(max_examples, len(self.dataset))))
        
        print(f"Loaded {len(self.dataset):,} examples")
        
        # Pre-compute total tokens for progress tracking
        print("Computing dataset statistics...")
        self.total_tokens = 0
        sample_size = min(1000, len(self.dataset))
        for i in range(sample_size):
            text = self.dataset[i]['text']
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            self.total_tokens += len(tokens)
        
        # Extrapolate to full dataset
        self.total_tokens = int(self.total_tokens * len(self.dataset) / sample_size)
        print(f"Estimated total tokens: {self.total_tokens / 1e9:.2f}B")
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        
        # Tokenize
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.seq_length + 1
        )
        
        # Pad if necessary
        if len(tokens) < self.seq_length + 1:
            tokens = tokens + [self.tokenizer.eos_token_id] * (self.seq_length + 1 - len(tokens))
        
        # Create input and labels
        input_ids = torch.tensor(tokens[:self.seq_length], dtype=torch.long)
        labels    = input_ids.clone() 
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }


def get_dataloader(
    data_path: str,
    tokenizer,
    batch_size: int,
    seq_length: int,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    shuffle: bool = True,
    max_examples: Optional[int] = None
) -> DataLoader:
    """Create dataloader from local dataset"""
    dataset = LocalSlimPajamaDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        seq_length=seq_length,
        max_examples=max_examples
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True  # Drop last incomplete batch for consistent gradient accumulation
    )


# ============================================================================
# TRAINING UTILITIES (unchanged)
# ============================================================================

def get_lr(step: int, config: TrainingConfig, total_steps: int) -> float:
    """Learning rate schedule with warmup and cosine decay"""
    if step < config.warmup_steps:
        # Linear warmup
        return config.learning_rate * (step + 1) / config.warmup_steps
    
    # Cosine decay
    progress = (step - config.warmup_steps) / (total_steps - config.warmup_steps)
    progress = max(0.0, min(1.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr + (config.learning_rate - config.min_lr) * cosine


def count_parameters(model: nn.Module) -> Dict[str, float]:
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'total_M': total / 1e6,
        'trainable': trainable,
        'trainable_M': trainable / 1e6,
    }


class AverageMeter:
    """Track running average"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ============================================================================
# EVALUATION (unchanged)
# ============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    max_batches: int,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model on validation set"""
    # Save original training state
    was_training = model.training
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    
    try:
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Use autocast for evaluation too
            with autocast(dtype=torch.bfloat16):
                _, loss = model(input_ids, labels=labels)
            
            batch_tokens = input_ids.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
        
        # Calculate metrics INSIDE try block
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        result = {
            'loss': avg_loss,
            'perplexity': perplexity,
            'bpc': avg_loss / math.log(2),
        }
        
    except Exception as e:
        print(f"⚠️  Evaluation failed: {e}")
        # Return dummy values to continue training
        result = {
            'loss': float('inf'),
            'perplexity': float('inf'),
            'bpc': float('inf'),
        }
    
    finally:
        # Always restore training state and clear cache
        if was_training:
            model.train()
        torch.cuda.empty_cache()
    
    return result


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    tokenizer,
    prompts: list,
    max_length: int = 100,
    device: torch.device = None
) -> Dict[str, str]:
    """Generate text samples"""
    model.eval()
    samples = {}
    
    # Unwrap DDP if needed
    unwrapped_model = model.module if hasattr(model, 'module') else model
    
    for i, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        # Use autocast for generation
        with autocast(dtype=torch.bfloat16):
            output = unwrapped_model.generate(  # ✅ Use unwrapped model
                input_ids,
                max_new_tokens=max_length,
                temperature=0.8,
                top_k=50
            )        
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        samples[f'prompt_{i+1}'] = generated
    
    model.train()
    return samples


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train():
    """Main training function"""
    
    # ========================================================================
    # SETUP
    # ========================================================================
    
    # Initialize distributed
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    
    is_main_process = local_rank == 0
    
    # Configs
    model_config = ModelConfig()
    train_config = TrainingConfig()
    
    # Set seed
    torch.manual_seed(train_config.seed + local_rank)
    np.random.seed(train_config.seed + local_rank)
    
    # Create output directory
    if is_main_process:
        os.makedirs(train_config.output_dir, exist_ok=True)
        
        # Save configs
        with open(f"{train_config.output_dir}/model_config.yaml", 'w') as f:
            yaml.dump(asdict(model_config), f)
        with open(f"{train_config.output_dir}/train_config.yaml", 'w') as f:
            yaml.dump(asdict(train_config), f)
    
    # Initialize ClearML
    task = None
    if is_main_process and CLEARML_AVAILABLE:
        task = Task.init(
            project_name=train_config.clearml_project,
            task_name=train_config.clearml_task
        )
        task.connect(asdict(model_config))
        task.connect(asdict(train_config))
        
    # ========================================================================
    # MODEL
    # ========================================================================
    
    if is_main_process:
        print("=" * 80)
        print("INITIALIZING MODEL")
        print("=" * 80)
    
    model = GPT(model_config).to(device)
    
    if is_main_process:
        params = count_parameters(model)
        print(f"Total parameters: {params['total_M']:.2f}M")
        print(f"Trainable parameters: {params['trainable_M']:.2f}M")
        
        if task:
            task.get_logger().report_single_value("model/total_params_M", params['total_M'])
    
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        find_unused_parameters=False
    )

    scaler = GradScaler()

    # ========================================================================
    # DATA - CHANGED: Load from local paths
    # ========================================================================
    
    if is_main_process:
        print("\n" + "=" * 80)
        print("LOADING DATA")
        print("=" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Check if data paths exist
    if is_main_process:
        if not os.path.exists(train_config.train_data_path):
            raise FileNotFoundError(f"Training data not found at {train_config.train_data_path}")
        if not os.path.exists(train_config.val_data_path):
            raise FileNotFoundError(f"Validation data not found at {train_config.val_data_path}")
    
    train_loader = get_dataloader(
        data_path=train_config.train_data_path,
        tokenizer=tokenizer,
        batch_size=train_config.micro_batch_size,
        seq_length=model_config.n_positions,
        num_workers=train_config.num_workers,
        prefetch_factor=train_config.prefetch_factor,
        shuffle=True
    )
    
    val_loader = get_dataloader(
        data_path=train_config.val_data_path,
        tokenizer=tokenizer,
        batch_size=train_config.micro_batch_size,
        seq_length=model_config.n_positions,
        num_workers=train_config.num_workers,
        prefetch_factor=train_config.prefetch_factor,
        shuffle=False
    )
    
    # ========================================================================
    # OPTIMIZER & SCHEDULER
    # ========================================================================
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay
    )
    
    # Calculate total steps
    tokens_per_batch = (
        train_config.micro_batch_size * 
        model_config.n_positions * 
        train_config.gradient_accumulation * 
        world_size
    )
    total_steps = train_config.total_tokens // tokens_per_batch
    
    # Calculate evaluation batches
    eval_batches = train_config.eval_tokens // (train_config.micro_batch_size * model_config.n_positions)
    
    if is_main_process:
        print(f"\nTokens per batch: {tokens_per_batch:,}")
        print(f"Total steps: {total_steps:,}")
        print(f"Warmup steps: {train_config.warmup_steps}")
        print(f"Eval batches: {eval_batches}")
        print(f"Dataset size: {len(train_loader.dataset):,} examples")
        print(f"Batches per epoch: {len(train_loader):,}")
        print(f"Epochs: {total_steps / len(train_loader):.2f}")
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    
    if is_main_process:
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
    
    model.train()
    
    # Metrics
    loss_meter = AverageMeter()
    step = 0
    tokens_seen = 0
    start_time = time.time()
    
    # Training state
    train_iter = iter(train_loader)
    epoch = 0
    
    while step < total_steps:
        
        # ====================================================================
        # TRAINING STEP
        # ====================================================================
        
        optimizer.zero_grad()
        
        # Gradient accumulation
        for micro_step in range(train_config.gradient_accumulation):
            try:
                batch = next(train_iter)
            except StopIteration:
                # New epoch
                epoch += 1
                if is_main_process:
                    print(f"\n[EPOCH {epoch}] Starting new epoch...")
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass with autocast for mixed precision
            with autocast(dtype=torch.bfloat16):
                _, loss = model(input_ids, labels=labels)
            loss = loss / train_config.gradient_accumulation
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            loss_meter.update(loss.item() * train_config.gradient_accumulation)
        
        # Gradient clipping
        # Unscale gradients before clipping
        scaler.unscale_(optimizer)
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            train_config.grad_clip
        )
        
        # Safety: Skip update if gradients exploded
        if grad_norm > 100.0:
            print(f"⚠️  WARNING: Gradient norm {grad_norm:.2f} is too high! Skipping update.")
            optimizer.zero_grad()
            scaler.update()  # Still update scaler
            continue
        
        # Optimizer step with learning rate schedule
        lr = get_lr(step, train_config, total_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Optimizer step with gradient scaling
        scaler.step(optimizer)
        scaler.update()
        
        # Update counters
        step += 1
        tokens_seen += tokens_per_batch
        
        # ====================================================================
        # LOGGING
        # ====================================================================
        
        if is_main_process and step % train_config.log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_seen / elapsed
            
            perplexity = math.exp(loss_meter.avg)
            
            print(
                f"Step {step}/{total_steps} | "
                f"Epoch {epoch} | "
                f"Loss: {loss_meter.avg:.4f} | "
                f"PPL: {perplexity:.2f} | "
                f"LR: {lr:.2e} | "
                f"Tokens/sec: {tokens_per_sec:.0f} | "
                f"Grad: {grad_norm:.2f}"
            )
            
            if task:
                task.get_logger().report_scalar(
                    "train", "loss", iteration=step, value=loss_meter.avg
                )
                task.get_logger().report_scalar(
                    "train", "perplexity", iteration=step, value=perplexity
                )
                task.get_logger().report_scalar(
                    "train", "learning_rate", iteration=step, value=lr
                )
                task.get_logger().report_scalar(
                    "train", "grad_norm", iteration=step, value=grad_norm
                )
                task.get_logger().report_scalar(
                    "train", "tokens_per_second", iteration=step, value=tokens_per_sec
                )
                task.get_logger().report_scalar(
                    "train", "epoch", iteration=step, value=epoch
                )
            
            loss_meter.reset()
        
        # ====================================================================
        # EVALUATION
        # ====================================================================
        
        if is_main_process and step % train_config.eval_interval == 0:
            print(f"\n[EVAL] Running validation...")
            eval_start = time.time()
            
            val_metrics = evaluate(
                model,
                val_loader,
                max_batches=eval_batches,
                device=device
            )
            
            eval_time = time.time() - eval_start
            
            print(
                f"[EVAL] Val Loss: {val_metrics['loss']:.4f} | "
                f"Val PPL: {val_metrics['perplexity']:.2f} | "
                f"Time: {eval_time:.0f}s"
            )
            
            if task:
                task.get_logger().report_scalar(
                    "val", "loss", iteration=step, value=val_metrics['loss']
                )
                task.get_logger().report_scalar(
                    "val", "perplexity", iteration=step, value=val_metrics['perplexity']
                )
        
        # ====================================================================
        # CHECKPOINTING
        # ====================================================================
        
        if is_main_process and step % train_config.save_interval == 0:
            checkpoint_path = f"{train_config.output_dir}/checkpoint_step_{step}.pt"
            print(f"\n[CHECKPOINT] Saving to {checkpoint_path}")
            
            # Save model
            torch.save({
                'step': step,
                'epoch': epoch,
                'tokens_seen': tokens_seen,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_config': asdict(model_config),
                'train_config': asdict(train_config),
            }, checkpoint_path)
            
            # Generate samples
            prompts = [
                "Once upon a time",
                "The meaning of life is",
                "In a shocking finding, scientist discovered"
            ]
            
            samples = generate_samples(model, tokenizer, prompts, device=device)
            
            # Save samples
            samples_path = f"{train_config.output_dir}/samples_step_{step}.txt"
            with open(samples_path, 'w') as f:
                for key, text in samples.items():
                    f.write(f"{key}:\n{text}\n\n")
            
            print(f"[SAMPLES] Saved to {samples_path}")
    
    # ========================================================================
    # FINAL EVALUATION
    # ========================================================================
    
    if is_main_process:
        print("\n" + "=" * 80)
        print("FINAL EVALUATION")
        print("=" * 80)
        
        val_metrics = evaluate(
            model,
            val_loader,
            max_batches=eval_batches,
            device=device
        )
        
        print(f"Final Val Loss: {val_metrics['loss']:.4f}")
        print(f"Final Val Perplexity: {val_metrics['perplexity']:.2f}")
        
        # Save final checkpoint
        final_path = f"{train_config.output_dir}/final_model.pt"
        torch.save({
            'step': step,
            'epoch': epoch,
            'tokens_seen': tokens_seen,
            'model_state_dict': model.state_dict(),
            'val_metrics': val_metrics,
            'model_config': asdict(model_config),
            'train_config': asdict(train_config),
        }, final_path)
        
        print(f"\n✅ Training complete! Final model saved to {final_path}")
        print(f"Total time: {(time.time() - start_time) / 3600:.2f} hours")
        print(f"Total epochs: {epoch}")
    
    dist.destroy_process_group()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    train()