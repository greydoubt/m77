"""
Complete GPT Training Pipeline with Grouped-Query Attention (GQA) - 1.2B Model
Modified to use locally downloaded SlimPajama data
Distributed training on 8× A100 40GB - OPTIMIZED

GQA uses:
- Full Q projection with n_head heads (32 in this case)
- Reduced K,V projections with n_kv_groups heads (8 in this case)
- Each KV group is shared by n_head/n_kv_groups query heads

This is the approach used in Llama 2, Mistral, and other modern LLMs.

Usage:
    torchrun --nproc_per_node=8 transformer_GQA_1.2B.py

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
from torch.cuda.amp import GradScaler
from torch.amp import autocast

from transformers import AutoTokenizer
from datasets import load_from_disk
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
    """Model architecture configuration - 1.2B parameters"""
    vocab_size: int = 50304
    n_positions: int = 2048
    n_layer: int = 22  
    n_embd: int = 2048        
    n_head: int = 32         
    n_kv_groups: int = 8      
    n_inner: int = 8192        
    
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
    """Training configuration - Optimized for 8× A100 40GB"""
    # Data paths
    train_data_path: str = "./slimpajama_data/train"
    val_data_path: str = "./slimpajama_validation"
    
    # Training - OPTIMIZED
    total_tokens: int = 10_000_000_000  # 10B tokens
    micro_batch_size: int = 2           # ← OPTIMIZED: For 40GB GPUs
    gradient_accumulation: int = 36     # ← OPTIMIZED: Maintain effective batch
    
    num_workers: int = 12               # ← OPTIMIZED: 8 → 12
    prefetch_factor: int = 6            # ← OPTIMIZED: 4 → 6

    # Optimizer
    learning_rate: float = 6e-5
    min_lr: float = 6e-6
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Schedule
    warmup_steps: int = 1000
    lr_scheduler: str = "cosine"
    
    # Checkpointing
    save_interval: int = 1000
    eval_interval: int = 500            # ← OPTIMIZED: 200 → 500
    log_interval: int = 10
    
    # Evaluation
    eval_tokens: int = 10_000_000  # 10M tokens
    
    # System
    seed: int = 42
    output_dir: str = "./outputs_1.2B_gqa"
    
    # ClearML
    clearml_project: str = "GPT-QKV-Comparison"
    clearml_task: str = "GPT-1.2B-GQA-8-10B"


# ============================================================================
# MODEL: GROUPED-QUERY ATTENTION (GQA)
# ============================================================================

class GQAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) - Used in Llama 2, Mistral, etc.
    
    Key characteristics:
    - Full Q projection with n_head heads (e.g., 32 heads)
    - Reduced K,V projections with n_kv_groups heads (e.g., 8 groups)
    - Each KV group is shared by n_head/n_kv_groups query heads
    
    Benefits:
    - Maintains most of QKV's expressiveness (separate Q, K, V)
    - Reduces KV cache by n_head/n_kv_groups factor (4× in this case)
    - Production-proven in modern LLMs
    
    Example with n_head=32, n_kv_groups=8:
    - 32 separate query heads
    - 8 key heads (each shared by 4 query heads)
    - 8 value heads (each shared by 4 query heads)
    - Cache stores 8 KV heads instead of 32 (75% reduction!)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_groups == 0, \
            f"n_head ({config.n_head}) must be divisible by n_kv_groups ({config.n_kv_groups})"
        
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.n_kv_groups = config.n_kv_groups
        self.n_head_per_group = config.n_head // config.n_kv_groups
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.attn_pdrop
        self.use_flash = config.use_flash_attention and FLASH_AVAILABLE
        
        # Full query projection (all n_head heads)
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=config.use_bias)
        
        # REDUCED KV projections (only n_kv_groups heads)
        # This is the KEY difference from standard QKV attention!
        kv_dim = self.n_kv_groups * self.head_dim
        self.c_kv = nn.Linear(config.n_embd, 2 * kv_dim, bias=config.use_bias)
        
        # Output projection (same as all variants)
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
        """
        Args:
            x: [batch, seq_len, n_embd]
            
        Returns:
            out: [batch, seq_len, n_embd]
        """
        B, T, C = x.size()
        
        # Query projection: Full n_head heads
        q = self.c_q(x)  # [B, T, n_embd]
        q = q.view(B, T, self.n_head, self.head_dim)  # [B, T, n_head, head_dim]
        
        # Key, Value projections: Only n_kv_groups heads
        kv = self.c_kv(x)  # [B, T, 2 * n_kv_groups * head_dim]
        k, v = kv.split(self.n_kv_groups * self.head_dim, dim=2)
        k = k.view(B, T, self.n_kv_groups, self.head_dim)  # [B, T, n_kv_groups, head_dim]
        v = v.view(B, T, self.n_kv_groups, self.head_dim)
        
        # Repeat KV heads to match number of query heads
        # Each KV group is used by n_head_per_group query heads
        # Example: [B, T, 8, head_dim] -> [B, T, 32, head_dim] with each group repeated 4 times
        k = k.repeat_interleave(self.n_head_per_group, dim=2)  # [B, T, n_head, head_dim]
        v = v.repeat_interleave(self.n_head_per_group, dim=2)
        
        # Now q, k, v all have shape [B, T, n_head, head_dim]
        # But k and v came from fewer parameters!
        
        # Choose attention implementation
        if self.use_flash:
            # Flash Attention: [B, T, n_head, head_dim]
            out = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=True
            )
        else:
            # Standard attention
            # Transpose to [B, n_head, T, head_dim]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # Attention scores: [B, n_head, T, T]
            # This is standard attention - nothing special here
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            # Apply causal mask
            attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            
            # Softmax
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            
            # Apply attention to values
            out = torch.matmul(attn, v)  # [B, n_head, T, head_dim]
            out = out.transpose(1, 2)    # [B, T, n_head, head_dim]
        
        # Merge heads
        out = out.contiguous().view(B, T, C)
        
        # Output projection
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
    """Single transformer block with GQA"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GQAttention(config)  # GQA instead of standard QKV!
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT_GQA(nn.Module):
    """GPT Language Model with Grouped-Query Attention (GQA)"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # Transformer blocks with GQA
        self.h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # LM head (tied with token embeddings if specified)
        if config.tie_word_embeddings:
            self.lm_head = lambda x: F.linear(x, self.wte.weight)
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights (GPT-2 style)"""
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
        """
        Args:
            input_ids: [batch, seq_len] token indices
            labels: [batch, seq_len] target token indices (for training)
            
        Returns:
            logits: [batch, seq_len, vocab_size]
            loss: scalar (if labels provided)
        """
        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.config.n_positions, f"Sequence length {t} exceeds maximum {self.config.n_positions}"
        
        # Token embeddings
        tok_emb = self.wte(input_ids)  # [b, t, n_embd]
        
        # Position embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # [1, t]
        pos_emb = self.wpe(pos)  # [1, t, n_embd]
        
        # Combined embeddings
        x = self.drop(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.h:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # LM head
        logits = self.lm_head(x)  # [b, t, vocab_size]
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Cross entropy loss
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
        """Generate tokens autoregressively"""
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.config.n_positions else idx[:, -self.config.n_positions:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # [batch, vocab_size]
            
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


# ============================================================================
# DATA LOADING
# ============================================================================

class LocalSlimPajamaDataset(Dataset):
    """Memory-mapped dataset using pre-chunked Arrow files"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        seq_length: int = 2048,
        max_examples: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        if rank == 0:
            print(f"Loading dataset from {data_path}...")
        
        chunks_dir = Path(data_path) / "chunks"
        
        if chunks_dir.exists():
            chunk_paths = sorted(chunks_dir.glob("chunk_*"))
            
            if rank == 0:
                print(f"📦 Found {len(chunk_paths)} chunks")
            
            # Each rank gets its own chunks
            self.chunk_paths = [
                chunk_paths[i] for i in range(len(chunk_paths)) 
                if i % world_size == rank
            ]
            
            print(f"[Rank {rank}] Loading {len(self.chunk_paths)} chunks (memory-mapped)")
            
            # Load but DON'T concatenate - keeps memory-mapped
            self.datasets = [load_from_disk(str(p)) for p in self.chunk_paths]
            
            # Build cumulative index
            self.cumulative_lengths = [0]
            for ds in self.datasets:
                self.cumulative_lengths.append(
                    self.cumulative_lengths[-1] + len(ds)
                )
            
            self.total_length = self.cumulative_lengths[-1]
            print(f"[Rank {rank}] ✅ Mapped {self.total_length:,} examples")
            
        else:
            self.datasets = [load_from_disk(data_path)]
            self.cumulative_lengths = [0, len(self.datasets[0])]
            self.total_length = len(self.datasets[0])
        
        if max_examples is not None:
            self.total_length = min(max_examples, self.total_length)
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        # Find which chunk
        chunk_idx = 0
        for i in range(len(self.cumulative_lengths) - 1):
            if idx < self.cumulative_lengths[i + 1]:
                chunk_idx = i
                local_idx = idx - self.cumulative_lengths[i]
                break
        
        text = self.datasets[chunk_idx][local_idx]['text']
        
        # Tokenize
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.seq_length + 1
        )
        
        if len(tokens) < self.seq_length + 1:
            tokens = tokens + [self.tokenizer.eos_token_id] * (self.seq_length + 1 - len(tokens))
        
        input_ids = torch.tensor(tokens[:self.seq_length], dtype=torch.long)
        labels    = input_ids.clone() 
        
        return {'input_ids': input_ids, 'labels': labels}

def get_dataloader(
    data_path: str,
    tokenizer,
    batch_size: int,
    seq_length: int,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    shuffle: bool = True,
    max_examples: Optional[int] = None,
    rank: int = 0,
    world_size: int = 1
) -> DataLoader:
    """Create dataloader from local dataset"""
    dataset = LocalSlimPajamaDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        seq_length=seq_length,
        max_examples=max_examples,
        rank=rank,
        world_size=world_size
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True
    )


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def get_lr(step: int, config: TrainingConfig, total_steps: int) -> float:
    """Learning rate schedule with warmup and cosine decay"""
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    
    progress = (step - config.warmup_steps) / (total_steps - config.warmup_steps)
    progress = max(0.0, min(1.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr + (config.learning_rate - config.min_lr) * cosine


def count_parameters(model: nn.Module) -> Dict[str, float]:
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count attention parameters specifically
    attn_params = sum(
        p.numel() for name, p in model.named_parameters()
        if 'attn' in name and 'ln' not in name
    )
    
    return {
        'total': total,
        'total_M': total / 1e6,
        'trainable': trainable,
        'trainable_M': trainable / 1e6,
        'attention': attn_params,
        'attention_M': attn_params / 1e6,
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
# EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    max_batches: int,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model on validation set"""
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
            
            with autocast('cuda', dtype=torch.bfloat16):
                _, loss = model(input_ids, labels=labels)
            
            batch_tokens = input_ids.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        result = {
            'loss': avg_loss,
            'perplexity': perplexity,
            'bpc': avg_loss / math.log(2),
        }
        
    except Exception as e:
        print(f"⚠️  Evaluation failed: {e}")
        result = {
            'loss': float('inf'),
            'perplexity': float('inf'),
            'bpc': float('inf'),
        }
    
    finally:
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
    
    unwrapped_model = model.module if hasattr(model, 'module') else model
    
    for i, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        with autocast('cuda', dtype=torch.bfloat16):
            output = unwrapped_model.generate(
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
    
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    
    is_main_process = local_rank == 0
    
    model_config = ModelConfig()
    train_config = TrainingConfig()
    
    torch.manual_seed(train_config.seed + local_rank)
    np.random.seed(train_config.seed + local_rank)
    
    if is_main_process:
        os.makedirs(train_config.output_dir, exist_ok=True)
        
        with open(f"{train_config.output_dir}/model_config.yaml", 'w') as f:
            yaml.dump(asdict(model_config), f)
        with open(f"{train_config.output_dir}/train_config.yaml", 'w') as f:
            yaml.dump(asdict(train_config), f)
    
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
        print("INITIALIZING 1.2B GQA MODEL (Grouped-Query Attention)")
        print("=" * 80)
        print(f"⭐ GQA-{model_config.n_kv_groups}: Industry-proven approach (Llama 2, Mistral)")
        print(f"   • {model_config.n_head} query heads (full expressiveness)")
        print(f"   • {model_config.n_kv_groups} KV groups (cache efficiency)")
        print(f"   • Each KV group shared by {model_config.n_head // model_config.n_kv_groups} Q heads")
    
    model = GPT_GQA(model_config).to(device)
    
    if is_main_process:
        params = count_parameters(model)
        print(f"\nTotal parameters: {params['total_M']:.2f}M")
        print(f"Trainable parameters: {params['trainable_M']:.2f}M")
        print(f"Attention parameters: {params['attention_M']:.2f}M")
        
        # Calculate KV cache size
        kv_cache_per_token = 2 * model_config.n_kv_groups * model_config.n_layer * (model_config.n_embd // model_config.n_head) * 2  # 2 bytes for FP16
        kv_cache_at_2k = kv_cache_per_token * 2048 / (1024**2)
        kv_cache_at_32k = kv_cache_per_token * 32768 / (1024**2)
        
        print(f"\n📊 KV CACHE COMPARISON (1.2B models @ 32k context):")
        print(f"   QKV/KV (32 heads): ~5.9 GB")
        print(f"   GQA-8 (this model): {kv_cache_at_32k:.2f} GB")
        print(f"   Cache reduction: {(1 - model_config.n_kv_groups/model_config.n_head)*100:.0f}%")
        print(f"\n💡 Expected: GQA maintains quality similar to QKV with better cache efficiency")
        
        if task:
            task.get_logger().report_single_value("model/total_params_M", params['total_M'])
            task.get_logger().report_single_value("model/attention_params_M", params['attention_M'])
            task.get_logger().report_single_value("model/kv_cache_gb_at_32k", kv_cache_at_32k)
    
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        find_unused_parameters=False
    )
    
    scaler = GradScaler()
    
    # ========================================================================
    # DATA
    # ========================================================================
    
    if is_main_process:
        print("\n" + "=" * 80)
        print("LOADING DATA")
        print("=" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
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
        shuffle=True,
        rank=local_rank,
        world_size=world_size
    )
    
    val_loader = get_dataloader(
        data_path=train_config.val_data_path,
        tokenizer=tokenizer,
        batch_size=train_config.micro_batch_size,
        seq_length=model_config.n_positions,
        num_workers=train_config.num_workers,
        prefetch_factor=train_config.prefetch_factor,
        shuffle=False,
        rank=local_rank,
        world_size=world_size
    )
    
    # ========================================================================
    # OPTIMIZER & SCHEDULER
    # ========================================================================
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay,
        fused=True  # ← OPTIMIZATION: Fused AdamW
    )
    
    tokens_per_batch = (
        train_config.micro_batch_size * 
        model_config.n_positions * 
        train_config.gradient_accumulation * 
        world_size
    )
    total_steps = train_config.total_tokens // tokens_per_batch
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
        print("🔬 Testing GQA at 1.2B scale")
        print("   Expected: Similar quality to QKV with 75% cache reduction")
    
    model.train()
    
    loss_meter = AverageMeter()
    step = 0
    tokens_seen = 0
    start_time = time.time()
    
    train_iter = iter(train_loader)
    epoch = 0
    
    while step < total_steps:
        
        # ====================================================================
        # TRAINING STEP
        # ====================================================================
        
        optimizer.zero_grad()
        
        for micro_step in range(train_config.gradient_accumulation):
            try:
                batch = next(train_iter)
            except StopIteration:
                epoch += 1
                if is_main_process:
                    print(f"\n[EPOCH {epoch}] Starting new epoch...")
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast('cuda', dtype=torch.bfloat16):
                _, loss = model(input_ids, labels=labels)
            loss = loss / train_config.gradient_accumulation
            
            scaler.scale(loss).backward()
            loss_meter.update(loss.item() * train_config.gradient_accumulation)
        
        scaler.unscale_(optimizer)
        
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            train_config.grad_clip
        )
        
        if grad_norm > 100.0:
            print(f"⚠️  WARNING: Gradient norm {grad_norm:.2f} is too high! Skipping update.")
            optimizer.zero_grad()
            scaler.update()
            continue
        
        lr = get_lr(step, train_config, total_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        scaler.step(optimizer)
        scaler.update()
        
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
                task.get_logger().report_scalar("train", "loss", iteration=step, value=loss_meter.avg)
                task.get_logger().report_scalar("train", "perplexity", iteration=step, value=perplexity)
                task.get_logger().report_scalar("train", "learning_rate", iteration=step, value=lr)
                task.get_logger().report_scalar("train", "grad_norm", iteration=step, value=grad_norm)
                task.get_logger().report_scalar("train", "tokens_per_second", iteration=step, value=tokens_per_sec)
                task.get_logger().report_scalar("train", "epoch", iteration=step, value=epoch)
            
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
                task.get_logger().report_scalar("val", "loss", iteration=step, value=val_metrics['loss'])
                task.get_logger().report_scalar("val", "perplexity", iteration=step, value=val_metrics['perplexity'])
        
        # ====================================================================
        # CHECKPOINTING
        # ====================================================================
        
        if is_main_process and step % train_config.save_interval == 0:
            checkpoint_path = f"{train_config.output_dir}/checkpoint_step_{step}.pt"
            print(f"\n[CHECKPOINT] Saving to {checkpoint_path}")
            
            try:
                torch.save({
                    'step': step,
                    'epoch': epoch,
                    'tokens_seen': tokens_seen,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_config': asdict(model_config),
                    'train_config': asdict(train_config),
                }, checkpoint_path)
                
                prompts = [
                    "Once upon a time",
                    "The meaning of life is",
                    "In a shocking finding, scientist discovered"
                ]
                
                samples = generate_samples(model, tokenizer, prompts, device=device)
                
                samples_path = f"{train_config.output_dir}/samples_step_{step}.txt"
                with open(samples_path, 'w') as f:
                    for key, text in samples.items():
                        f.write(f"{key}:\n{text}\n\n")
                
                print(f"[SAMPLES] Saved to {samples_path}")
            except Exception as e:
                print(f"⚠️  Checkpoint/sampling failed: {e}")
                print("Continuing training...")
    
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
        print(f"\n🎯 GQA combines quality and efficiency!")
        
        final_path = f"{train_config.output_dir}/final_model.pt"
        try:
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
        except Exception as e:
            print(f"⚠️  Final checkpoint save failed: {e}")
        
        print(f"Total time: {(time.time() - start_time) / 3600:.2f} hours")
        print(f"Total epochs: {epoch}")
    
    dist.destroy_process_group()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    train()