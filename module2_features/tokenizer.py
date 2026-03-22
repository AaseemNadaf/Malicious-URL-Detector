"""
Module 2 — Feature Engineering & Tokenization
===============================================
Project : Malicious URL Detector (CNN + XAI)
File    : module2_features/tokenizer.py
Run     : python tokenizer.py
Outputs :
    data/processed/X_train.npy   ← training sequences
    data/processed/X_val.npy     ← validation sequences
    data/processed/X_test.npy    ← test sequences
    data/processed/y_train.npy   ← training labels
    data/processed/y_val.npy     ← validation labels
    data/processed/y_test.npy    ← test labels
    data/processed/char_vocab.json  ← character → integer mapping
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────

INPUT_PATH   = "data/processed/dataset_clean.csv"
OUTPUT_DIR   = "data/processed"

# Every URL will be padded or truncated to this length.
# From Module 1 we know: mean=39, 75th percentile=44, max=500.
# 200 captures ~98% of URLs without cutting them off.
MAX_LEN      = 200

# Train / Validation / Test split ratios
TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
TEST_RATIO   = 0.10   # must sum to 1.0

RANDOM_SEED  = 42


# ─────────────────────────────────────────────
# 1.  HELPERS
# ─────────────────────────────────────────────

def log(msg: str):
    print(f"  [INFO]  {msg}")


# ─────────────────────────────────────────────
# 2.  BUILD CHARACTER VOCABULARY
# ─────────────────────────────────────────────

def build_vocab(urls: pd.Series) -> dict:
    """
    Scan every character that appears in the URL corpus and assign
    it a unique integer starting from 1.

    We reserve index 0 as the PADDING token — when a URL is shorter
    than MAX_LEN, we fill the remaining positions with 0.

    Returns a dict like: {'h': 1, 't': 2, 'p': 3, ':': 4, ...}
    """
    log("Building character vocabulary from all URLs...")

    # Collect every unique character across all URLs
    all_chars = set()
    for url in urls:
        all_chars.update(str(url))

    # Sort for reproducibility (same order every run)
    sorted_chars = sorted(all_chars)

    # Assign integer index starting at 1 (0 is reserved for padding)
    vocab = {char: idx + 1 for idx, char in enumerate(sorted_chars)}

    log(f"  Vocabulary size: {len(vocab)} unique characters (+ 1 padding token = {len(vocab)+1} total)")
    log(f"  Sample mappings: { {k: vocab[k] for k in list(vocab)[:10]} }")

    return vocab


# ─────────────────────────────────────────────
# 3.  ENCODE URLS → INTEGER SEQUENCES
# ─────────────────────────────────────────────

def encode_urls(urls: pd.Series, vocab: dict, max_len: int) -> np.ndarray:
    """
    Convert each URL string into a fixed-length integer array.

    Steps for each URL:
      1. Convert each character to its vocab integer (unknown chars → 0)
      2. Truncate if longer than max_len
      3. Pad with 0s at the END if shorter than max_len

    Result shape: (num_urls, max_len)

    Example with max_len=10:
      "http://ab"  → [9,21,21,17,3,2,2,6,24,0]   (padded with 1 zero)
      "http://abcdefghijklmnop" → [9,21,21,17,3,2,2,6,24,25]  (truncated)
    """
    log(f"Encoding {len(urls):,} URLs to integer sequences (max_len={max_len})...")

    sequences = np.zeros((len(urls), max_len), dtype=np.int32)

    for i, url in enumerate(urls):
        url_str = str(url)
        for j, char in enumerate(url_str[:max_len]):   # truncate at max_len
            sequences[i, j] = vocab.get(char, 0)       # 0 for unknown chars

        # Log progress every 100k rows
        if (i + 1) % 100_000 == 0:
            log(f"  Encoded {i+1:,} / {len(urls):,} URLs...")

    log(f"  Encoding complete. Output shape: {sequences.shape}")
    return sequences


# ─────────────────────────────────────────────
# 4.  TRAIN / VALIDATION / TEST SPLIT
# ─────────────────────────────────────────────

def split_data(X: np.ndarray, y: np.ndarray):
    """
    Split into train / val / test sets.

    Why three splits?
    - Train  (80%): the model learns from this
    - Val    (10%): we monitor this during training to catch overfitting
    - Test   (10%): held out completely — used ONCE at the end to report
                    final accuracy. Never touched during training.

    We use stratify=y to ensure each split has the same 50/50
    class balance as the full dataset.
    """
    log(f"Splitting data: {TRAIN_RATIO*100:.0f}% train / "
        f"{VAL_RATIO*100:.0f}% val / {TEST_RATIO*100:.0f}% test ...")

    # First split: carve out the test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=y
    )

    # Second split: split the remaining data into train and val
    # val_ratio relative to the trainval portion
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio_adjusted,
        random_state=RANDOM_SEED,
        stratify=y_trainval
    )

    log(f"  Train : {X_train.shape[0]:,} samples")
    log(f"  Val   : {X_val.shape[0]:,} samples")
    log(f"  Test  : {X_test.shape[0]:,} samples")

    # Verify class balance in each split
    for name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        mal_pct = y_split.mean() * 100
        log(f"  {name} malicious ratio: {mal_pct:.1f}%")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────
# 5.  SAVE OUTPUTS
# ─────────────────────────────────────────────

def save_outputs(vocab, X_train, X_val, X_test, y_train, y_val, y_test):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save vocab as JSON so Flask API and XAI module can load it
    vocab_path = os.path.join(OUTPUT_DIR, "char_vocab.json")
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)
    log(f"Saved vocabulary → {vocab_path}")

    # Save numpy arrays — .npy is fast to load (much faster than CSV)
    splits = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }
    for name, arr in splits.items():
        path = os.path.join(OUTPUT_DIR, f"{name}.npy")
        np.save(path, arr)
        log(f"Saved {name} → {path}  shape={arr.shape}  dtype={arr.dtype}")


# ─────────────────────────────────────────────
# 6.  SANITY CHECK
# ─────────────────────────────────────────────

def sanity_check(vocab: dict):
    """
    Decode a few sequences back to URLs to visually verify
    the encoding round-trip is correct.
    """
    log("\n── Sanity check — decode a few sequences back to URLs ────")

    # Build reverse vocab: integer → character
    reverse_vocab = {v: k for k, v in vocab.items()}

    # Load a small slice of training data
    X = np.load(os.path.join(OUTPUT_DIR, "X_train.npy"))
    y = np.load(os.path.join(OUTPUT_DIR, "y_train.npy"))
    df_orig = pd.read_csv(INPUT_PATH)

    print()
    for i in range(3):
        # Decode: convert ints back to chars, stop at padding (0)
        decoded = ''.join(
            reverse_vocab.get(idx, '?')
            for idx in X[i]
            if idx != 0          # skip padding tokens
        )
        label_str = "MALICIOUS" if y[i] == 1 else "SAFE"
        print(f"  [{label_str}] Decoded : {decoded}")

    print()
    log("If decoded URLs look like real URLs above, encoding is correct!")


# ─────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  MODULE 2 — Feature Engineering & Tokenization")
    print("="*55 + "\n")

    # --- Load cleaned dataset from Module 1 ---
    log(f"Loading dataset from {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)
    log(f"  Loaded {len(df):,} rows")

    urls   = df['url']
    labels = df['label'].values.astype(np.int32)

    # --- Build vocabulary ---
    vocab = build_vocab(urls)

    # --- Encode URLs to integer sequences ---
    X = encode_urls(urls, vocab, MAX_LEN)

    # --- Split ---
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, labels)

    # --- Save ---
    save_outputs(vocab, X_train, X_val, X_test, y_train, y_val, y_test)

    # --- Sanity check ---
    sanity_check(vocab)

    print("\n" + "="*55)
    print("  Module 2 complete! Ready for Module 3 (Colab).")
    print("="*55 + "\n")