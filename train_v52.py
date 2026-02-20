#!/usr/bin/env python3
import os
import sys
import time
import math
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# THREADING FIX — Must be set before any Torch operations
# ============================================================================
# Set this immediately to avoid "parallel work has started" error
# Wrap in try-except to handle cases where parallel work has already started
try:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
except RuntimeError:
    # If threads already set, try to at least set the environment variables
    pass

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

# ============================================================================
# CONFIGURATION — Max Quality for 5GB RAM
# ============================================================================
CONFIG = {
    # Model - Scaled up for quality
    'vocab': 4096,         # Will be updated dynamically based on tokenizer
    'd_model': 256,        # Increased from 128
    'n_layers': 6,         # Increased from 4
    'n_heads': 4,          # Increased from 2
    'd_head': 64,
    'd_ffn': 512,          # Increased from 256
    'dropout': 0.1,        # Added for regularization

    # Training
    'seq_len': 128,        # Increased from 64 for better context
    'batch_size': 4,       # Reduced to fit larger model
    'grad_accum': 8,       # Increased to maintain effective batch size
    'lr': 5e-4,            # Slightly lower for stability with larger model
    'min_lr': 1e-5,
    'warmup_steps': 100,
    'weight_decay': 0.01,
    'grad_clip': 1.0,
    'betas': (0.9, 0.95),

    # Schedule
    'total_hours': 2.0,    # Longer training for quality
    'save_every': 600,
    'eval_every': 200,
    'log_every': 20,
    'gen_every': 200,

    # Data
    'data_dir': 'data_v52_maxq',
    'out_dir': 'out_v52_maxq',
    'max_train_tokens': None,  # None = use all data
}

# ============================================================================
# ROTARY POSITIONAL EMBEDDINGS (RoPE)
# ============================================================================
class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding for better length generalization."""
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        
    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: (B, H, T, D)
    # cos, sin: (T, D)
    cos = cos.unsqueeze(0).unsqueeze(0) # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# ============================================================================
# FAST LINEAR
# ============================================================================
class FastLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        nn.init.kaiming_normal_(self.weight, mode='fan_out')

# ============================================================================
# ATTENTION WITH RoPE
# ============================================================================
class FastCausalAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_head, dropout=0.1, max_seq_len=2048):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = d_head ** -0.5
        
        total_dim = n_heads * d_head
        self.qkv = FastLinear(d_model, 3 * total_dim)
        self.out = FastLinear(total_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # RoPE
        self.rotary = RotaryEmbedding(d_head, max_seq_len)

    def forward(self, x):
        B, T, D = x.shape
        
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary(x, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.out(out)

# ============================================================================
# FEED FORWARD
# ============================================================================
class FastFFN(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1):
        super().__init__()
        self.up = FastLinear(d_model, d_ffn * 2)
        self.down = FastLinear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.up(x)
        h1, h2 = h.chunk(2, dim=-1)
        return self.dropout(self.down(F.gelu(h1) * h2)) # GELU often smoother than SiGLU

# ============================================================================
# TRANSFORMER BLOCK
# ============================================================================
class NovaBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_head, d_ffn, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = FastCausalAttention(d_model, n_heads, d_head, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FastFFN(d_model, d_ffn, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# ============================================================================
# NOVA-IGNITION LM
# ============================================================================
class NovaIgnitionLM(nn.Module):
    def __init__(self, vocab, d_model, n_layers, n_heads, d_head, d_ffn, dropout=0.1, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.seq_len = CONFIG['seq_len']
        
        self.embed = nn.Embedding(vocab, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            NovaBlock(d_model, n_heads, d_head, d_ffn, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_out = nn.LayerNorm(d_model)
        self.head = FastLinear(d_model, vocab)
        
        # Weight tying
        self.head.weight = self.embed.weight
        
        # Init
        self._init_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'═'*60}")
        print(f"🚀 FlashLM v5.2 'Nova-Ignition' (Max Quality Edition)")
        print(f"{'═'*60}")
        print(f"   Parameters:      {total_params:,}")
        print(f"   Model Dimension: {d_model}")
        print(f"   Layers:          {n_layers}")
        print(f"   Vocab Size:      {vocab}")
        print(f"   Est. RAM:        ~{total_params * 4 * 3 / 1024**3:.2f} GB (Training)")
        print(f"{'═'*60}\n")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.shape
        h = self.dropout(self.embed(x))
        
        for block in self.blocks:
            h = block(h)
        
        logits = self.head(self.ln_out(h))
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return loss
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.seq_len:]
            logits = self(ctx)[:, -1, :] / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        
        self.train()
        return idx
# ============================================================================
# FAST DATASET — Fixed uint16 conversion
# ============================================================================
class FastDataset(Dataset):
    def __init__(self, bin_path, seq_len):
        self.seq_len = seq_len
        # Load as uint16 (storage format)
        self.data = np.memmap(str(bin_path), dtype=np.uint16, mode='r')
        self.n = (len(self.data) - 1) // seq_len
        
    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        i = idx * self.seq_len
        chunk = self.data[i : i + self.seq_len + 1]
        
        # FIX: Cast to int32 BEFORE torch.from_numpy to avoid uint16 error
        x = torch.from_numpy(chunk[:-1].astype(np.int32))
        y = torch.from_numpy(chunk[1:].astype(np.int32))
        
        return x.long(), y.long()

def prepare_data(config):
    data_dir = Path(config['data_dir'])
    data_dir.mkdir(exist_ok=True)
    
    train_bin = data_dir / "train.bin"
    val_bin = data_dir / "val.bin"
    tok_path = data_dir / "tokenizer.json"
    meta_path = data_dir / "meta.json"
    
    # Check if we need to regenerate data
    needs_regen = False
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        # If vocab requirement changed or files missing, regen
        if meta.get('vocab') != config['vocab'] or not train_bin.exists():
            needs_regen = True
    else:
        needs_regen = True

    if not needs_regen:
        print(f"✅ Data ready in {data_dir}")
        return str(tok_path)
    
    print(f"\n{'═'*60}")
    print(f"📦 PREPARING DATA (This may take a few minutes)")
    print(f"{'═'*60}")
    
    train_txt = data_dir / "stories.txt"
    
    if not train_txt.exists():
        print("📥 Downloading TinyStories...")
        import urllib.request
        # Using the training split for better quality, or valid split for speed
        url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
        urllib.request.urlretrieve(url, train_txt)
        print(f"   Downloaded: {train_txt.stat().st_size / 1e6:.1f} MB")
    
    # Tokenizer
    print(f"🔤 Training tokenizer (Vocab: {config['vocab']})...")
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel()
    trainer = BpeTrainer(
        vocab_size=config['vocab'], 
        min_frequency=2,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
    )
    tokenizer.train(files=[str(train_txt)], trainer=trainer)
    tokenizer.save(str(tok_path))
    
    # Check actual vocab size
    actual_vocab = tokenizer.get_vocab_size()
    print(f"   Actual Vocab Size: {actual_vocab}")
    
    # Tokenize
    print(f"🔢 Tokenizing data...")
    with open(train_txt, 'r', encoding='utf-8') as f:
        text = f.read()
    
    stories = [s.strip() for s in text.split('\n\n') if len(s.strip()) > 50]
    
    tokens = []
    eos_id = tokenizer.token_to_id("<eos>") or 0
    
    # Limit stories to fit in RAM/Disk for this demo
    # TinyStories Valid is ~30MB, Train is ~2GB. 
    # We'll process up to ~20M tokens to keep preparation fast but data rich.
    max_tokens = 20_000_000 
    
    for i, story in enumerate(stories):
        tokens.extend(tokenizer.encode(story).ids)
        tokens.append(eos_id)
        if len(tokens) >= max_tokens: 
            print(f"   Reached token limit: {len(tokens):,}")
            break
        if i % 5000 == 0 and i > 0:
            print(f"   Processed {i} stories...", end='\r')
    
    tokens = tokens[:max_tokens]
    
    split = int(len(tokens) * 0.95)
    
    np.array(tokens[:split], dtype=np.uint16).tofile(str(train_bin))
    np.array(tokens[split:], dtype=np.uint16).tofile(str(val_bin))
    
    print(f"   Train: {split:,} tokens")
    print(f"   Val:   {len(tokens) - split:,} tokens")
    
    # Save meta to avoid regen loops
    with open(meta_path, 'w') as f:
        json.dump({'vocab': config['vocab'], 'actual_vocab': actual_vocab}, f)
        
    print(f"{'═'*60}\n")
    
    return str(tok_path)

# ============================================================================
# LR SCHEDULE
# ============================================================================
def get_lr(step, config, total_steps):
    if step < config['warmup_steps']:
        return config['lr'] * (step + 1) / config['warmup_steps']
    
    progress = (step - config['warmup_steps']) / max(1, total_steps - config['warmup_steps'])
    return config['min_lr'] + 0.5 * (config['lr'] - config['min_lr']) * (1 + math.cos(math.pi * progress))

# ============================================================================
# EVALUATION
# ============================================================================
@torch.no_grad()
def evaluate(model, val_data, seq_len, max_batches=20):
    model.eval()
    losses = []
    n = (len(val_data) - 1) // seq_len
    
    if n == 0: return {'loss': 99.0, 'ppl': 99.0}

    for _ in range(min(max_batches, n // 4)):
        batch_x, batch_y = [], []
        for _ in range(4):
            i = np.random.randint(0, n) * seq_len
            chunk = val_data[i:i + seq_len + 1]
            batch_x.append(chunk[:-1])
            batch_y.append(chunk[1:])
        
        x = torch.tensor(np.stack(batch_x), dtype=torch.long)
        y = torch.tensor(np.stack(batch_y), dtype=torch.long)
        
        loss = model(x, targets=y)
        losses.append(loss.item())
    
    model.train()
    avg = sum(losses) / len(losses)
    return {'loss': avg, 'ppl': math.exp(min(avg, 20))}

# ============================================================================
# MAIN
# ============================================================================
def train():
    config = CONFIG
    out_dir = Path(config['out_dir'])
    out_dir.mkdir(exist_ok=True)
    
    # REMOVED: Threading lines moved to top of file to fix RuntimeError
    
    # Prepare data
    tok_path = prepare_data(config)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tok_path)
    
    # DYNAMIC VOCAB FIX:
    actual_vocab = tokenizer.get_vocab_size()
    config['vocab'] = actual_vocab # Update config with real size
    
    val_raw = np.fromfile(str(Path(config['data_dir']) / 'val.bin'), dtype=np.uint16)
    val_data = val_raw.astype(np.int32)
    print(f"📊 Val: {len(val_data):,} tokens\n")
    
    train_ds = FastDataset(
        str(Path(config['data_dir']) / 'train.bin'),
        config['seq_len']
    )
    
    train_dl = DataLoader(
        train_ds, 
        batch_size=config['batch_size'],
        shuffle=True, 
        num_workers=0,
        drop_last=True,
        pin_memory=False
    )
    
    print("🏗️  Building model...")
    model = NovaIgnitionLM(
        vocab=config['vocab'],
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        d_head=config['d_head'],
        d_ffn=config['d_ffn'],
        dropout=config['dropout']
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        betas=config['betas'],
        weight_decay=config['weight_decay'],
        fused=False
    )
    
    # Steps calc
    steps_per_epoch = len(train_dl) // config['grad_accum']
    toks_per_step = config['batch_size'] * config['grad_accum'] * config['seq_len']
    # Estimate speed at ~400 tok/s for larger model
    estimated_steps = int(config['total_hours'] * 3600 * 400 / toks_per_step)
    
    # Resume
    step, tokens_seen, best_val = 0, 0, float('inf')
    ckpt_path = out_dir / 'latest.pt'
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            step = ckpt['step']
            tokens_seen = ckpt['tokens']
            best_val = ckpt.get('best_val', float('inf'))
            print(f"📂 Resumed: step {step}, {tokens_seen/1e6:.1f}M tokens\n")
        except Exception as e:
            print(f"⚠️ Checkpoint load failed: {e}. Starting fresh.")
    
    json.dump(config, open(out_dir / 'config.json', 'w'), indent=2)
    
    prompts = ["Once upon a time", "The little dog", "A magical forest"]
    
    print(f"{'═'*60}")
    print(f"🚀 TRAINING (Max Quality Mode)")
    print(f"{'═'*60}")
    print(f"   Target Duration: {config['total_hours']}h")
    print(f"   Est. Steps:      ~{estimated_steps:,}")
    print(f"{'═'*60}\n")
    
    t_start = time.time()
    log_loss = 0.0
    model.train()
    train_iter = iter(train_dl)
    
    while True:
        elapsed = time.time() - t_start
        if elapsed / 3600 >= config['total_hours']:
            print(f"\n⏰ Time limit reached")
            break
        
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        
        for _ in range(config['grad_accum']):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                x, y = next(train_iter)
            
            loss = model(x, targets=y)
            (loss / config['grad_accum']).backward()
            accum_loss += loss.item()
            tokens_seen += x.numel()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        
        lr = get_lr(step, config, estimated_steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        
        optimizer.step()
        step += 1
        log_loss += accum_loss / config['grad_accum']
        
        if step % 50 == 0:
            gc.collect()
        
        if step % config['log_every'] == 0:
            tps = tokens_seen / elapsed if elapsed > 0 else 0
            avg_loss = log_loss / config['log_every']
            ppl = math.exp(min(avg_loss, 20))
            print(f"Step {step:5d} │ Loss {avg_loss:.4f} │ PPL {ppl:6.1f} │ "
                  f"LR {lr:.1e} │ {tps:,.0f} tok/s")
            log_loss = 0.0
        
        if step % config['eval_every'] == 0:
            metrics = evaluate(model, val_data, config['seq_len'])
            if metrics['loss'] < best_val:
                best_val = metrics['loss']
                torch.save(model.state_dict(), out_dir / 'best.pt')
                print(f"  ✦ VAL │ Loss {metrics['loss']:.4f} │ PPL {metrics['ppl']:.1f} ★ BEST")
            else:
                print(f"  ✦ VAL │ Loss {metrics['loss']:.4f} │ PPL {metrics['ppl']:.1f}")
        
        if step % config['gen_every'] == 0 and step > 0:
            print(f"\n{'─'*50}")
            model.eval()
            for p in prompts:
                ids = tokenizer.encode(p).ids
                x = torch.tensor([ids], dtype=torch.long)
                out = model.generate(x, max_new_tokens=40, temperature=0.7, top_k=40)
                text = tokenizer.decode(out[0].tolist())
                print(f"  > {text[:150]}")
            model.train()
            print(f"{'─'*50}\n")
        
        if step % config['save_every'] == 0:
            torch.save({
                'step': step,
                'tokens': tokens_seen,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val': best_val,
            }, out_dir / 'latest.pt')
            
    # Final
    print(f"\n{'═'*60}")
    print(f"✅ TRAINING COMPLETE")
    print(f"{'═'*60}")
    print(f"   Final PPL: {math.exp(min(evaluate(model, val_data, config['seq_len'])['loss'], 20)):.2f}")
    print(f"   Output Dir: {out_dir}")

if __name__ == '__main__':
    train()
