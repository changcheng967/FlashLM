#!/usr/bin/env python3
"""Quick eval + generation from v9.1 checkpoint."""
import os, sys, math, json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CKPT_PATH = SCRIPT_DIR / 'out_v91' / 'checkpoint.pt'
DATA_DIR = SCRIPT_DIR / 'data'
SEQ_LEN = 256
BATCH_SIZE = 8
N_THREADS = 4

os.environ['OMP_NUM_THREADS'] = str(N_THREADS)
os.environ['MKL_NUM_THREADS'] = str(N_THREADS)
torch.set_num_threads(N_THREADS)
torch.set_num_interop_threads(1)

# Import model definition
sys.path.insert(0, str(SCRIPT_DIR))
from train_v91 import ReckoningV2, RMSNorm, prepare_data

def main():
    if not CKPT_PATH.exists():
        print(f"No checkpoint at {CKPT_PATH}")
        return

    ckpt = torch.load(str(CKPT_PATH), map_location='cpu', weights_only=False)
    print(f"Checkpoint: step {ckpt.get('step','?')}, "
          f"tokens {ckpt.get('tokens_seen','?'):,}, "
          f"time {ckpt.get('elapsed_total',0)/60:.1f}m, "
          f"best_val {ckpt.get('best_val','?')}")

    # Load data for tokenizer + validation
    from tokenizers import Tokenizer
    tok_path = DATA_DIR / 'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tok_path))

    # Build model and load weights
    model = ReckoningV2(
        vocab=4096, d_model=384, n_layers=6, d_mem=64,
        conv_k=7, d_ff=1536, seq_len=256, dropout=0.0)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # Eval PPL
    val_bin = DATA_DIR / 'val.bin'
    if val_bin.exists():
        val_data = np.fromfile(str(val_bin), dtype=np.uint16).astype(np.int32)
        n = (len(val_data) - 1) // SEQ_LEN
        losses = []
        n_batches = min(100, n // BATCH_SIZE)
        with torch.no_grad():
            for _ in range(n_batches):
                bx, by = [], []
                for _ in range(BATCH_SIZE):
                    i = np.random.randint(0, n) * SEQ_LEN
                    chunk = val_data[i:i + SEQ_LEN + 1]
                    bx.append(chunk[:-1])
                    by.append(chunk[1:])
                x = torch.tensor(np.stack(bx), dtype=torch.long)
                y = torch.tensor(np.stack(by), dtype=torch.long)
                loss = model(x, targets=y)
                if not torch.isnan(loss):
                    losses.append(loss.item())
        if losses:
            avg = sum(losses) / len(losses)
            ppl = math.exp(min(avg, 10))
            print(f"\nVal PPL: {ppl:.2f} (loss {avg:.4f}, {len(losses)} batches)")

    # Generate samples
    print("\n" + "=" * 60)
    print("GENERATION")
    print("=" * 60)
    seeds = ["Once upon a time", "The little girl", "A cat sat",
             "There was a", "One day"]
    for temp in [0.1, 0.5, 0.8, 1.0]:
        print(f"\n--- Temperature {temp} ---")
        for seed_text in seeds[:3]:
            try:
                seed_ids = tokenizer.encode(seed_text).ids
                seed = torch.tensor([seed_ids], dtype=torch.long)
                gen = model.generate(seed, 100, temperature=temp,
                                     top_p=0.9, freq_penalty=1.0)
                text = tokenizer.decode(gen[0].tolist())
                print(f"  [{seed_text}] {text[:200]}")
            except Exception as e:
                print(f"  [{seed_text}] ERROR: {e}")

if __name__ == '__main__':
    main()
