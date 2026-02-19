"""Download TinyStories, train BPE-8K tokenizer, save as binary."""
import os
import numpy as np
from pathlib import Path


def main():
    DATA = Path("data")
    DATA.mkdir(exist_ok=True)

    # ── Download ──
    train_file = DATA / "train.txt"
    val_file = DATA / "val.txt"

    if not train_file.exists():
        print("📥 Downloading TinyStories train...")
        os.system(f"wget -q --show-progress -O {train_file} "
                  "'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt'")
    if not val_file.exists():
        print("📥 Downloading TinyStories val...")
        os.system(f"wget -q --show-progress -O {val_file} "
                  "'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt'")

    print(f"✅ Train: {train_file.stat().st_size / 1e9:.2f} GB")
    print(f"✅ Val:   {val_file.stat().st_size / 1e6:.1f} MB")

    # ── Train tokenizer ──
    print("\n🔤 Training BPE-8K tokenizer...")
    from tokenizers import ByteLevelBPETokenizer

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(train_file)],
        vocab_size=8192,
        min_frequency=2,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
    )
    tokenizer.save_model(str(DATA), "bpe8k")
    
    # Save as single file for easy loading
    from tokenizers import Tokenizer
    tok = Tokenizer(tokenizer._tokenizer)
    tok.save(str(DATA / "tokenizer.json"))
    print("✅ Tokenizer saved")

    # ── Tokenize to binary ──
    print("\n🔢 Tokenizing...")
    tok = Tokenizer.from_file(str(DATA / "tokenizer.json"))

    def tokenize_to_bin(txt_path, bin_path):
        tokens = []
        batch = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                batch.append(line)
                if len(batch) >= 10000:
                    for enc in tok.encode_batch(batch):
                        tokens.extend(enc.ids)
                    batch = []
                    if (i + 1) % 500000 == 0:
                        print(f"    {i+1:,} lines → {len(tokens):,} tokens")
            if batch:
                for enc in tok.encode_batch(batch):
                    tokens.extend(enc.ids)

        arr = np.array(tokens, dtype=np.uint16)
        arr.tofile(bin_path)
        print(f"  ✅ {len(arr):,} tokens → {bin_path} ({arr.nbytes/1e6:.1f} MB)")
        return len(arr)

    n_train = tokenize_to_bin(train_file, DATA / "train.bin")
    n_val = tokenize_to_bin(val_file, DATA / "val.bin")

    print(f"\n{'═'*50}")
    print(f"📊 READY")
    print(f"   Train: {n_train:,} tokens")
    print(f"   Val:   {n_val:,} tokens")
    print(f"   At 14K tok/s: {n_train/14000/3600:.1f}h per epoch")
    print(f"   24h = ~{24*14000*3600/n_train:.1f} epochs")
    print(f"{'═'*50}")


if __name__ == "__main__":
    main()
