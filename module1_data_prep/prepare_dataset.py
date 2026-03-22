"""
Module 1 — Dataset Preparation & Cleaning
==========================================
Project : Malicious URL Detector (CNN + XAI)
File    : module1_data_prep/prepare_dataset.py
Run     : python prepare_dataset.py
Output  : data/processed/dataset_clean.csv
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.utils import resample

# ─────────────────────────────────────────────
# 0.  CONFIGURATION — edit paths if needed
# ─────────────────────────────────────────────

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"

PHIUSIIL_PATH = os.path.join(RAW_DIR, "phiusiil.csv")
URLHAUS_PATH  = os.path.join(RAW_DIR, "urlhaus.txt")
TRANCO_PATH   = os.path.join(RAW_DIR, "tranco.csv")

OUTPUT_PATH   = os.path.join(PROCESSED_DIR, "dataset_clean.csv")

# How many URLs to keep total (balanced: half safe, half malicious)
# Increase if your machine can handle it. 100k is a good starting point.
TARGET_TOTAL  = 800_000

# Maximum character length we'll accept for a URL
MAX_URL_LEN   = 500
MIN_URL_LEN   = 5


# ─────────────────────────────────────────────
# 1.  HELPERS
# ─────────────────────────────────────────────

def log(msg):
    """Simple logger so we can see what's happening."""
    print(f"  [INFO]  {msg}")


def is_valid_url(url: str) -> bool:
    """
    Basic sanity check.
    We don't need a perfect URL parser — we just want to drop obvious junk
    like empty strings, file paths, or placeholder values.
    """
    if not isinstance(url, str):
        return False
    url = url.strip()
    if len(url) < MIN_URL_LEN or len(url) > MAX_URL_LEN:
        return False
    # Must start with http/https or at least contain a dot (bare domains)
    if not re.search(r'https?://|[a-zA-Z0-9]\.[a-zA-Z]{2,}', url):
        return False
    return True


def clean_url(url: str) -> str:
    """
    Light normalisation — we do NOT strip too much because the CNN will learn
    from raw character patterns (e.g. long subdomains, suspicious TLDs).
    """
    url = url.strip()
    # Remove surrounding quotes that sometimes appear in CSVs
    url = url.strip('"').strip("'")
    # Lowercase the scheme and domain only (path stays as-is)
    # Simple approach: just lowercase everything — URLs are case-insensitive
    # for scheme/host anyway, and path case rarely matters for our task.
    url = url.lower()
    return url


# ─────────────────────────────────────────────
# 2.  LOAD EACH DATASET
# ─────────────────────────────────────────────

def load_phiusiil(path: str) -> pd.DataFrame:
    """
    PhiUSIIL CSV has many feature columns.
    We only need the raw URL and the label.
    Label column is typically 'label': 1 = phishing, 0 = benign.
    """
    log(f"Loading PhiUSIIL from {path}")
    df = pd.read_csv(path, low_memory=False)

    # Print columns so you can verify on your machine
    log(f"  Columns found: {list(df.columns)}")

    # Common column names in PhiUSIIL — adjust if yours differ
    url_col   = None
    label_col = None

    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ('url', 'urls', 'url_raw', 'phishing_url'):
            url_col = col
        if col_lower in ('label', 'class', 'type', 'result', 'status', 'tag'):
            label_col = col

    if url_col is None or label_col is None:
        raise ValueError(
            "Could not auto-detect URL or label column in PhiUSIIL.\n"
            f"Columns are: {list(df.columns)}\n"
            "Set url_col and label_col manually in load_phiusiil()."
        )

    df = df[[url_col, label_col]].rename(columns={url_col: 'url', label_col: 'label'})

    # ── PhiUSIIL label convention (IMPORTANT — this dataset is flipped) ──────
    # status = 1 → legitimate = SAFE   → our label 0
    # status = 0 → phishing   = MALICIOUS → our label 1
    # We flip with (1 - status) to match project convention: 1=malicious, 0=safe
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])
    df['label'] = (1 - df['label'].astype(int))  # flip: 0→1 (malicious), 1→0 (safe)
    df = df[df['label'].isin([0, 1])]

    # ── Normalise URLs ───────────────────────────────────────────────────────
    # PhiUSIIL mixes bare domains (e.g. google.com) with full URLs (http://...).
    # Prepend http:// to bare domains so all entries are consistent.
    def ensure_scheme(url: str) -> str:
        url = str(url).strip()
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        return url

    df['url'] = df['url'].apply(ensure_scheme)

    log(f"  Loaded {len(df):,} rows | malicious: {(df.label==1).sum():,} | safe: {(df.label==0).sum():,}")
    return df


def load_urlhaus(path: str) -> pd.DataFrame:
    """
    URLhaus downloads as a .txt file but is formatted like a CSV inside.
    The file has a large comment block at the top (lines starting with '#'),
    followed by a header row, then data rows.

    Example structure:
        # URLhaus Database Dump
        # ...more comments...
        #
        "id","dateadded","url","url_status","last_online","threat","tags","urlhaus_link","reporter"
        "123","2024-01-01","http://evil.com/malware","online",...

    We skip all comment lines and parse the rest as a normal CSV.
    All URLs in URLhaus are malicious — label = 1.
    """
    log(f"Loading URLhaus from {path}")

    # Read all lines — the header is on a line starting with "# id,"
    # so we can't blindly skip all # lines. We need to keep that one.
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        raw_lines = f.readlines()

    cleaned = []
    for line in raw_lines:
        stripped = line.strip()
        if stripped.startswith('# id,'):
            # This IS the header — strip the leading "# " and keep it
            cleaned.append(line[2:])  # remove "# " prefix
        elif stripped.startswith('#'):
            # Pure comment line — skip
            continue
        else:
            cleaned.append(line)

    if not cleaned:
        raise ValueError("URLhaus file appears to be empty or all comments.")

    # Re-parse the cleaned lines as a CSV (header is the first non-comment line)
    from io import StringIO
    df = pd.read_csv(StringIO(''.join(cleaned)), low_memory=False)

    # Strip whitespace from column names (sometimes present)
    df.columns = df.columns.str.strip().str.strip('"')
    log(f"  Columns found: {list(df.columns)}")

    # Find URL column (case-insensitive)
    url_col = None
    for col in df.columns:
        if col.lower() == 'url':
            url_col = col
            break

    if url_col is None:
        raise ValueError(
            f"Could not find 'url' column in URLhaus file.\n"
            f"Columns found: {list(df.columns)}\n"
            f"First few lines of file:\n{''.join(cleaned[:5])}"
        )

    df = df[[url_col]].rename(columns={url_col: 'url'})

    # Strip quotes that sometimes wrap values in URLhaus exports
    df['url'] = df['url'].astype(str).str.strip('"').str.strip()
    df['label'] = 1  # All URLhaus entries are malicious

    # Drop any rows where URL ended up empty after stripping
    df = df[df['url'].str.len() > 0]

    log(f"  Loaded {len(df):,} malicious URLs from URLhaus")
    return df


def load_tranco(path: str) -> pd.DataFrame:
    """
    Tranco list is just: rank,domain (e.g. '1,google.com').
    We prepend 'http://' to make them proper URLs.
    All entries are benign — label = 0.
    """
    log(f"Loading Tranco from {path}")

    # Tranco has no header
    df = pd.read_csv(path, header=None, names=['rank', 'domain'], low_memory=False)
    log(f"  Loaded {len(df):,} domains")

    # Convert domain → URL
    df['url'] = 'http://' + df['domain'].str.strip()
    df['label'] = 0  # All Tranco entries are benign

    df = df[['url', 'label']]
    log(f"  Converted {len(df):,} domains to URLs (all safe)")
    return df


# ─────────────────────────────────────────────
# 3.  MERGE & CLEAN
# ─────────────────────────────────────────────

def merge_datasets(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Concatenate all source dataframes into one."""
    combined = pd.concat(dfs, ignore_index=True)
    log(f"Combined total rows (before cleaning): {len(combined):,}")
    return combined


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
    1. Drop rows where URL is null or empty
    2. Apply is_valid_url filter
    3. Normalise URL text
    4. Remove duplicates (same URL appearing in multiple sources)
    5. Report final class distribution
    """
    log("Cleaning dataset...")

    # Step 1: Drop nulls
    before = len(df)
    df = df.dropna(subset=['url', 'label'])
    log(f"  Dropped {before - len(df):,} rows with null url/label")

    # Step 2: Validate URLs
    before = len(df)
    df['url'] = df['url'].astype(str)
    df = df[df['url'].apply(is_valid_url)]
    log(f"  Dropped {before - len(df):,} invalid URLs (too short/long, no domain pattern)")

    # Step 3: Normalise
    df['url'] = df['url'].apply(clean_url)

    # Step 4: Remove duplicates — keep first occurrence
    before = len(df)
    df = df.drop_duplicates(subset='url', keep='first')
    log(f"  Dropped {before - len(df):,} duplicate URLs")

    # Report
    mal_count  = (df.label == 1).sum()
    safe_count = (df.label == 0).sum()
    log(f"  After cleaning — malicious: {mal_count:,} | safe: {safe_count:,}")

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# 4.  BALANCE CLASSES
# ─────────────────────────────────────────────

def balance_and_sample(df: pd.DataFrame, total: int) -> pd.DataFrame:
    """
    CNNs work best with balanced classes (equal 0s and 1s).
    We sample equal numbers from each class up to total/2 each.

    If one class has fewer samples than needed, we use all of them
    and adjust the other class to match (undersample the majority).
    """
    log("Balancing classes...")

    per_class = total // 2

    malicious = df[df.label == 1]
    safe      = df[df.label == 0]

    # How many can we actually take?
    n_mal  = min(per_class, len(malicious))
    n_safe = min(per_class, len(safe))

    # Make both classes the same size (take the smaller of the two targets)
    n_each = min(n_mal, n_safe)

    malicious_sampled = malicious.sample(n=n_each, random_state=42)
    safe_sampled      = safe.sample(n=n_each, random_state=42)

    balanced = pd.concat([malicious_sampled, safe_sampled], ignore_index=True)

    # Shuffle the final dataset
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    log(f"  Final dataset: {len(balanced):,} URLs | {n_each:,} malicious + {n_each:,} safe")
    return balanced


# ─────────────────────────────────────────────
# 5.  SAVE
# ─────────────────────────────────────────────

def save_dataset(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    log(f"Saved clean dataset → {path}")
    log(f"  Shape: {df.shape}")
    log(f"  Columns: {list(df.columns)}")


# ─────────────────────────────────────────────
# 6.  QUICK SANITY CHECK
# ─────────────────────────────────────────────

def sanity_check(path: str):
    """
    Reload the saved file and print a few rows so you can visually
    confirm everything looks right before moving to Module 2.
    """
    log("\n── Sanity check ──────────────────────────────────")
    df = pd.read_csv(path)

    print(f"\n  Total rows : {len(df):,}")
    print(f"  Label dist :\n{df['label'].value_counts().to_string()}\n")
    print("  Sample malicious URLs:")
    print(df[df.label == 1]['url'].head(5).to_string(index=False))
    print("\n  Sample safe URLs:")
    print(df[df.label == 0]['url'].head(5).to_string(index=False))

    # Check for any remaining nulls
    nulls = df.isnull().sum()
    if nulls.any():
        print(f"\n  WARNING — nulls found:\n{nulls[nulls > 0]}")
    else:
        print("\n  No nulls found. Dataset looks clean!")

    # Check URL length stats
    df['url_len'] = df['url'].str.len()
    print(f"\n  URL length stats:")
    print(df['url_len'].describe().round(1).to_string())


# ─────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  MODULE 1 — Dataset Preparation & Cleaning")
    print("="*55 + "\n")

    # --- Load ---
    df_phiusiil = load_phiusiil(PHIUSIIL_PATH)
    df_urlhaus  = load_urlhaus(URLHAUS_PATH)
    df_tranco   = load_tranco(TRANCO_PATH)

    # --- Merge ---
    df_combined = merge_datasets(df_phiusiil, df_urlhaus, df_tranco)

    # --- Clean ---
    df_clean = clean_dataset(df_combined)

    # --- Balance ---
    df_final = balance_and_sample(df_clean, total=TARGET_TOTAL)

    # --- Save ---
    save_dataset(df_final, OUTPUT_PATH)

    # --- Sanity check ---
    sanity_check(OUTPUT_PATH)

    print("\n" + "="*55)
    print("  Module 1 complete! Ready for Module 2.")
    print("="*55 + "\n")