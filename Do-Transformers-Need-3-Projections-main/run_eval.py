"""
LM Evaluation Harness Wrapper for Custom 1.2B Transformer Models

Usage:
    python lm_eval.py --model kqv  --checkpoint ./outputs_1.2B_kqv/checkpoint_step_7000.pt
    python lm_eval.py --model kv   --checkpoint ./outputs_1.2B_qkv_k_equals_v/checkpoint_step_7000.pt
    python lm_eval.py --model gqa  --checkpoint ./outputs_1.2B_gqa/checkpoint_step_7000.pt
    python lm_eval.py --model mqa  --checkpoint ./outputs_1.2B_mqa/checkpoint_step_8000.pt
    python lm_eval.py --model qgqa --checkpoint ./outputs_1.2B_q_gqa/checkpoint_step_8000.pt
    python lm_eval.py --model qmqa --checkpoint ./outputs_1.2B_qmqa/checkpoint_step_8000.pt
"""

import os
import sys
import json
import argparse
import importlib.util
import torch
import torch.nn.functional as F
from dataclasses import dataclass, fields
from typing import Optional, List, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True,
                    choices=['kqv', 'kv', 'gqa', 'mqa', 'qgqa', 'qmqa'])
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--tasks', type=str,
                    default='hellaswag,piqa,arc_easy,arc_challenge,winogrande')
parser.add_argument('--num_fewshot', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--output_path', type=str, default=None)
args = parser.parse_args()

# ============================================================================
# LOAD MODEL CLASS
# Files are in the same directory as this script
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_class(filename, class_name):
    filepath = os.path.join(SCRIPT_DIR, filename)
    spec = importlib.util.spec_from_file_location("model_module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

# Exact filenames as they appear on disk
MODEL_MAP = {
    'kqv':  ('transformer_KQV_1_2B.py',   'GPT'),
    'kv':   ('transformer_KV_1_1_2B.py',  'GPT_QKV_KEqualsV'),
    'gqa':  ('transformer_GQA_1_2B.py',   'GPT_GQA'),
    'mqa':  ('transformer_MQA_1_2B.py',   'GPT_MQA'),
    'qgqa': ('transformer_QGQA_1_2B.py',  'GPT_Q_GQA'),
    'qmqa': ('transformer_QMQA_1_2B.py',  'GPT_QMQA'),
}

print(f"\nLoading model variant: {args.model}")
filename, class_name = MODEL_MAP[args.model]
ModelClass = load_class(filename, class_name)
print(f"  Loaded {class_name} from {filename}")

# ============================================================================
# LOAD CHECKPOINT
# ============================================================================

print(f"\nLoading checkpoint: {args.checkpoint}")
ckpt = torch.load(args.checkpoint, map_location='cpu')

model_cfg_dict = ckpt['model_config']
print(f"  Tokens seen: {ckpt.get('tokens_seen', 'unknown'):,}")

@dataclass
class ModelConfig:
    vocab_size: int = 50304
    n_positions: int = 2048
    n_layer: int = 22
    n_embd: int = 2048
    n_head: int = 32
    n_kv_groups: int = 1
    n_inner: int = 8192
    embd_pdrop: float = 0.0
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.0
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_bias: bool = True
    tie_word_embeddings: bool = True
    use_flash_attention: bool = False

valid_keys = {f.name for f in fields(ModelConfig)}
config = ModelConfig(**{k: v for k, v in model_cfg_dict.items() if k in valid_keys})
config.use_flash_attention = False  # always off for eval

model = ModelClass(config)
# model.load_state_dict(ckpt['model_state_dict'], strict=True)
state_dict = ckpt['model_state_dict']
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=True)
model.eval()
model.to(args.device)

n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params/1e9:.2f}B")
print(f"  Device: {args.device}")

# ============================================================================
# TOKENIZER
# ============================================================================

print("\nLoading tokenizer (GPT-2)...")
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# ============================================================================
# LM-EVAL WRAPPER
# ============================================================================

from lm_eval.api.model import LM
from lm_eval.api.instance import Instance


class CustomLM(LM):

    def __init__(self, model, tokenizer, device, batch_size=8, max_length=2048):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens) -> str:
        return self.tokenizer.decode(tokens)

    def _encode_pair(self, context: str, continuation: str):
        ctx_tokens = self.tok_encode(context)
        cont_tokens = self.tok_encode(continuation)
        max_ctx = self._max_length - len(cont_tokens) - 1
        if max_ctx < 1:
            max_ctx = 1
        ctx_tokens = ctx_tokens[-max_ctx:]
        return ctx_tokens, cont_tokens

    @torch.no_grad()
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        results = []
        batch_reqs = [req.args for req in requests]

        for i in tqdm(range(0, len(batch_reqs), self._batch_size),
                      desc="loglikelihood", leave=False):
            batch = batch_reqs[i: i + self._batch_size]
            encoded = [self._encode_pair(ctx, cont) for ctx, cont in batch]

            for ctx_tokens, cont_tokens in encoded:
                full_tokens = ctx_tokens + cont_tokens
                if len(full_tokens) > self._max_length:
                    full_tokens = full_tokens[-self._max_length:]

                input_ids = torch.tensor([full_tokens], dtype=torch.long).to(self._device)

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits, _ = self.model(input_ids)

                cont_len = len(cont_tokens)
                ctx_len = len(full_tokens) - cont_len

                cont_logits = logits[0, ctx_len - 1: ctx_len - 1 + cont_len]
                cont_ids = torch.tensor(cont_tokens, dtype=torch.long).to(self._device)

                log_probs = F.log_softmax(cont_logits, dim=-1)
                token_log_probs = log_probs[range(cont_len), cont_ids]
                total_log_prob = token_log_probs.sum().item()

                greedy_tokens = cont_logits.argmax(dim=-1)
                is_greedy = (greedy_tokens == cont_ids).all().item()

                results.append((total_log_prob, bool(is_greedy)))

        return results

    @torch.no_grad()
    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        results = []
        for req in tqdm(requests, desc="loglikelihood_rolling", leave=False):
            text = req.args[0]
            tokens = self.tok_encode(text)
            total_log_prob = 0.0

            for start in range(0, len(tokens), self._max_length - 1):
                chunk = tokens[start: start + self._max_length]
                if len(chunk) < 2:
                    continue
                input_ids = torch.tensor([chunk], dtype=torch.long).to(self._device)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits, _ = self.model(input_ids)
                shift_logits = logits[0, :-1]
                shift_labels = input_ids[0, 1:]
                log_probs = F.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs[range(len(shift_labels)), shift_labels]
                total_log_prob += token_log_probs.sum().item()

            results.append(total_log_prob)
        return results

    def generate_until(self, requests: List[Instance]) -> List[str]:
        results = []
        for req in requests:
            context, gen_kwargs = req.args
            ctx_tokens = self.tok_encode(context)
            if len(ctx_tokens) > self._max_length - 10:
                ctx_tokens = ctx_tokens[-(self._max_length - 10):]
            input_ids = torch.tensor([ctx_tokens], dtype=torch.long).to(self._device)
            max_new = gen_kwargs.get('max_gen_toks', 50)
            output = self.model.generate(input_ids, max_new_tokens=max_new,
                                         temperature=1.0, top_k=50)
            generated = output[0, len(ctx_tokens):]
            results.append(self.tok_decode(generated.tolist()))
        return results


# ============================================================================
# RUN EVALUATION
# ============================================================================

print(f"\nInitializing lm-eval wrapper...")
lm = CustomLM(
    model=model,
    tokenizer=tokenizer,
    device=args.device,
    batch_size=args.batch_size,
    max_length=config.n_positions
)

from lm_eval import simple_evaluate

tasks = [t.strip() for t in args.tasks.split(',')]
print(f"Tasks: {tasks}")
print(f"Few-shot: {args.num_fewshot}\n")

results = simple_evaluate(
    model=lm,
    tasks=tasks,
    num_fewshot=args.num_fewshot,
    batch_size=args.batch_size,
    device=args.device,
    log_samples=False,
)

# ============================================================================
# PRINT + SAVE RESULTS
# ============================================================================

print("\n" + "="*60)
print(f"RESULTS — {args.model.upper()} | {args.num_fewshot}-shot")
print("="*60)

task_results = {}
for task_name, task_result in results['results'].items():
    acc = task_result.get('acc,none',
          task_result.get('acc_norm,none',
          task_result.get('acc_norm', None)))
    if acc is not None:
        acc_pct = acc * 100
        print(f"  {task_name:<25} {acc_pct:.2f}%")
        task_results[task_name] = acc_pct

if task_results:
    avg = sum(task_results.values()) / len(task_results)
    print(f"\n  {'Average':<25} {avg:.2f}%")
    task_results['average'] = avg

print("="*60)

output_path = args.output_path or f"lm_eval_results_{args.model}.json"
save_data = {
    'model': args.model,
    'checkpoint': args.checkpoint,
    'num_fewshot': args.num_fewshot,
    'tasks': task_results,
}

with open(output_path, 'w') as f:
    json.dump(save_data, f, indent=2)

print(f"\nSaved to: {output_path}")