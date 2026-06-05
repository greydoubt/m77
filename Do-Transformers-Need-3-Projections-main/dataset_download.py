"""
SlimPajama Dataset Downloader
Downloads SlimPajama data in chunks with checkpointing and resume capability
"""

from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer
import os
import json
from pathlib import Path
from datetime import datetime

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def download_split_robust(
    split_name, 
    target_tokens,
    chunk_size=1_000_000_000,
    output_dir="./slimpajama_data",
    resume=True
):
    """Download dataset with chunking, checkpointing, and resume capability."""
    
    # Setup directories
    split_dir = Path(output_dir) / split_name
    chunks_dir = split_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_file = split_dir / "checkpoint.json"
    
    # Load checkpoint if resuming
    start_tokens = 0
    start_chunk = 0
    if resume and checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            start_tokens = checkpoint['total_tokens']
            start_chunk = checkpoint['chunk_num'] + 1
            print(f"\n Resuming from checkpoint:")
            print(f"   Already downloaded: {start_tokens/1e9:.2f}B tokens")
            print(f"   Starting from chunk {start_chunk}")
            
            if start_tokens >= target_tokens:
                print(f"✅ Already complete! ({start_tokens/1e9:.2f}B tokens)")
                return start_tokens
    
    print(f"\ Downloading {split_name} split")
    print(f"   Target: {target_tokens/1e9:.1f}B tokens (~{target_tokens/1e9*3:.0f}GB)")
    print(f"   Chunk size: {chunk_size/1e9:.1f}B tokens (~{chunk_size/1e9*3:.0f}GB)")
    print(f"   Output: {split_dir}")
    print()
    
    # Stream dataset
    ds = load_dataset(
        "cerebras/SlimPajama-627B", 
        split=split_name,
        streaming=True
    )
    
    # Skip examples if resuming
    if start_tokens > 0:
        print(f" Skipping to resume point...")
        skip_count = 0
        for example in ds:
            text = example['text']
            tokens = tokenizer.encode(text, add_special_tokens=False)
            skip_count += len(tokens)
            if skip_count >= start_tokens:
                break
        print(f"✅ Skipped {skip_count/1e9:.2f}B tokens\n")
    
    total_tokens = start_tokens
    chunk_tokens = 0
    chunk_examples = []
    chunk_num = start_chunk
    
    start_time = datetime.now()
    
    try:
        for example in ds:
            text = example['text']
            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_count = len(tokens)
            
            total_tokens += token_count
            chunk_tokens += token_count
            chunk_examples.append(example)
            
            # Progress update every 1000 examples
            if len(chunk_examples) % 1000 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                tokens_per_sec = (total_tokens - start_tokens) / elapsed if elapsed > 0 else 0
                eta_seconds = (target_tokens - total_tokens) / tokens_per_sec if tokens_per_sec > 0 else 0
                
                print(f"Progress: {total_tokens/1e9:.2f}B / {target_tokens/1e9:.1f}B tokens "
                      f"({100*total_tokens/target_tokens:.1f}%) | "
                      f"Chunk {chunk_num}: {len(chunk_examples)} examples | "
                      f"ETA: {eta_seconds/60:.0f}min")
            
            # Save chunk when reaching chunk_size
            if chunk_tokens >= chunk_size:
                _save_chunk(chunks_dir, chunk_num, chunk_examples, chunk_tokens)
                _save_checkpoint(checkpoint_file, total_tokens, chunk_num)
                
                # Reset for next chunk
                chunk_examples = []
                chunk_tokens = 0
                chunk_num += 1
            
            # Stop when target reached
            if total_tokens >= target_tokens:
                print(f"\n Target reached!")
                break
        
        # Save final partial chunk if any
        if chunk_examples:
            _save_chunk(chunks_dir, chunk_num, chunk_examples, chunk_tokens)
            _save_checkpoint(checkpoint_file, total_tokens, chunk_num)
        
        # Create metadata file
        _save_metadata(split_dir, split_name, total_tokens, chunk_num + 1, target_tokens)
        
        print(f"\n{'='*70}")
        print(f"✅ DOWNLOAD COMPLETE!")
        print(f"{'='*70}")
        print(f"Split: {split_name}")
        print(f"Total tokens: {total_tokens/1e9:.2f}B")
        print(f"Total chunks: {chunk_num + 1}")
        print(f"Storage: ~{(chunk_num + 1) * 3}GB")
        print(f"Location: {split_dir}")
        print(f"Time: {(datetime.now() - start_time).total_seconds()/60:.1f} minutes")
        print(f"{'='*70}\n")
        
        return total_tokens
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Download interrupted by user")
        if chunk_examples:
            print(f"💾 Saving current chunk before exit...")
            _save_chunk(chunks_dir, chunk_num, chunk_examples, chunk_tokens)
            _save_checkpoint(checkpoint_file, total_tokens, chunk_num)
        print(f"✅ Checkpoint saved. Run again with resume=True to continue.")
        return total_tokens
    
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        if chunk_examples:
            print(f" Saving current chunk before exit...")
            _save_chunk(chunks_dir, chunk_num, chunk_examples, chunk_tokens)
            _save_checkpoint(checkpoint_file, total_tokens, chunk_num)
        raise


def _save_chunk(chunks_dir, chunk_num, examples, tokens):
    """Save a chunk to disk"""
    chunk_path = chunks_dir / f"chunk_{chunk_num:03d}"
    print(f"\n Saving chunk {chunk_num}")
    print(f"   Examples: {len(examples):,}")
    print(f"   Tokens: {tokens/1e9:.2f}B (~{tokens/1e9*3:.1f}GB)")
    
    chunk_ds = Dataset.from_list(examples)
    chunk_ds.save_to_disk(str(chunk_path))
    print(f"   ✅ Saved to {chunk_path}\n")


def _save_checkpoint(checkpoint_file, total_tokens, chunk_num):
    """Save checkpoint for resume"""
    checkpoint = {
        'total_tokens': total_tokens,
        'chunk_num': chunk_num,
        'timestamp': datetime.now().isoformat()
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def _save_metadata(split_dir, split_name, total_tokens, num_chunks, target_tokens):
    """Save metadata about the download"""
    metadata = {
        'split': split_name,
        'total_tokens': total_tokens,
        'target_tokens': target_tokens,
        'num_chunks': num_chunks,
        'tokenizer': 'gpt2',
        'dataset': 'cerebras/SlimPajama-627B',
        'download_date': datetime.now().isoformat()
    }
    metadata_file = split_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_file}")


def load_split(split_name, output_dir="./slimpajama_data", combine=False):
    """Load a downloaded split."""
    split_dir = Path(output_dir) / split_name
    chunks_dir = split_dir / "chunks"
    
    if not chunks_dir.exists():
        raise ValueError(f"No data found at {split_dir}")
    
    chunk_paths = sorted(chunks_dir.glob("chunk_*"))
    
    print(f"\nFound {len(chunk_paths)} chunks in {chunks_dir}")
    
    if combine:
        print(f"⚠️  Loading all chunks into memory...")
        datasets = [Dataset.load_from_disk(str(p)) for p in chunk_paths]
        combined = concatenate_datasets(datasets)
        print(f"✅ Combined dataset: {len(combined):,} examples")
        return combined
    else:
        print(f"💡 Tip: Use these paths to load chunks individually for distributed training")
        return [str(p) for p in chunk_paths]


if __name__ == "__main__":
    print("="*70)
    print("SLIMPAJAMA DATASET DOWNLOADER")
    print("="*70)
    print()
    
    # Download train split (24B tokens)
    train_tokens = download_split_robust(
        split_name="train",
        target_tokens=10_000_000_000,
        chunk_size=1_000_000_000,
        output_dir="./slimpajama_data",
        resume=True
    )
    
    # Download validation split (10M tokens)
    val_tokens = download_split_robust(
        split_name="validation",
        target_tokens=10_000_000,
        chunk_size=10_000_000,
        output_dir="./slimpajama_data",
        resume=True
    )
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Train: {train_tokens/1e9:.2f}B tokens (~{train_tokens/1e9*3:.0f}GB)")
    print(f"Validation: {val_tokens/1e6:.1f}M tokens (~{val_tokens/1e6*3:.0f}MB)")
    print(f"Location: ./slimpajama_data/")
    print("="*70)