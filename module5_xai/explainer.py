"""
Module 5 - LIME XAI Explainer
==============================
Project : Malicious URL Detector (CNN + XAI)
File    : module5_xai/explainer.py
Imported by module4_api/app.py
"""

import re
import numpy as np
from typing import Callable


# ==============================================================
# 1. URL TOKENIZER
# ==============================================================

def tokenize_url(url: str) -> list:
    """
    Split URL into meaningful tokens that LIME can turn on/off.

    'http://paypa1-secure-login.tk/verify/account?id=123'
    -> ['http', '://', 'paypa1', '-', 'secure', '-', 'login',
        '.', 'tk', '/', 'verify', '/', 'account', '?', 'id', '=', '123']
    """
    url = str(url).strip().lower()
    tokens = []

    # Extract scheme
    scheme_match = re.match(r'^(https?)(://)', url)
    if scheme_match:
        tokens.append(scheme_match.group(1))
        tokens.append(scheme_match.group(2))
        url = url[scheme_match.end():]

    # Split on special chars, keeping delimiters
    parts = re.split(r'([./?&=\-@:#])', url)
    for part in parts:
        if part:
            tokens.append(part)

    return tokens


def get_content_tokens(tokens: list) -> list:
    """Return only non-delimiter tokens (the ones LIME will perturb)."""
    DELIMITERS = {'.', '/', '?', '&', '=', '-', '@', ':', '#', '://'}
    return [t for t in tokens if t not in DELIMITERS]


def reconstruct_url(tokens: list, mask: np.ndarray) -> str:
    """
    Reconstruct URL from tokens, replacing masked-out content
    tokens with 'a'. Delimiters are always preserved.
    """
    DELIMITERS = {'.', '/', '?', '&', '=', '-', '@', ':', '#', '://'}
    result   = []
    mask_pos = 0

    for token in tokens:
        if token in DELIMITERS:
            result.append(token)
        else:
            if mask_pos < len(mask) and mask[mask_pos] == 0:
                result.append('a')
            else:
                result.append(token)
            mask_pos += 1

    return ''.join(result)


# ==============================================================
# 2. LIME EXPLAINER
# ==============================================================

class URLLimeExplainer:
    """
    LIME explainer for URL malicious prediction.

    Usage:
        explainer = URLLimeExplainer(predict_fn, encode_fn, feature_fn)
        results = explainer.explain(url)
    """

    def __init__(self,
                 predict_fn: Callable,
                 encode_fn: Callable,
                 feature_fn: Callable,
                 num_samples: int = 300):
        self.predict_fn  = predict_fn
        self.encode_fn   = encode_fn
        self.feature_fn  = feature_fn
        self.num_samples = num_samples


    def explain(self, url: str) -> list:
        """
        Run LIME on a URL. Returns list of token dicts sorted
        by absolute importance descending.

        Even when model confidence is 1.0, relative importance
        differences between tokens are meaningful.
        """
        url_lower    = url.strip().lower()
        all_tokens   = tokenize_url(url_lower)
        content_toks = get_content_tokens(all_tokens)
        n_tokens     = len(content_toks)

        if n_tokens == 0:
            return []

        # Generate random perturbations
        np.random.seed(42)
        masks    = np.random.randint(0, 2, size=(self.num_samples, n_tokens))
        masks[0] = np.ones(n_tokens, dtype=int)   # first = full URL

        # Run model on each perturbed URL
        predictions = np.zeros(self.num_samples)
        feats       = self.feature_fn(url_lower)   # features stay fixed

        for i, mask in enumerate(masks):
            perturbed      = reconstruct_url(all_tokens, mask)
            seq            = self.encode_fn(perturbed)
            predictions[i] = float(
                self.predict_fn([seq, feats], verbose=0)[0][0]
            )

        # Compute kernel weights - samples closer to original get higher weight
        distances = np.sqrt(np.sum((masks - 1) ** 2, axis=1))
        kernel_w  = np.sqrt(np.exp(-(distances ** 2) / (n_tokens ** 2)))

        # Weighted least squares to get token importances
        try:
            W          = np.diag(kernel_w)
            Xw         = W @ masks.astype(float)
            yw         = W @ predictions
            Xw_inter   = np.column_stack([Xw, kernel_w])
            coeffs, _, _, _ = np.linalg.lstsq(Xw_inter, yw, rcond=None)
            importances = coeffs[:n_tokens]
        except Exception:
            # Fallback: simple correlation
            importances = np.array([
                float(np.corrcoef(masks[:, i], predictions)[0, 1])
                for i in range(n_tokens)
            ])

        # Normalize importances to [-1, 1] range so thresholds are consistent
        # regardless of model confidence level
        max_abs = np.max(np.abs(importances))
        if max_abs > 0:
            importances = importances / max_abs

        # Build result list
        results = []
        for i, token in enumerate(content_toks):
            results.append({
                "token":      token,
                "importance": round(float(importances[i]), 4),
                "position":   i
            })

        # Sort by absolute importance descending
        results.sort(key=lambda x: abs(x["importance"]), reverse=True)
        return results


# ==============================================================
# 3. PATTERN MATCHING FOR HUMAN-READABLE EXPLANATIONS
# ==============================================================

# Free / abused TLDs
FREE_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top",
    "pw", "cc", "su", "click", "loan", "work",
    "date", "racing", "win", "download"
}

# Phishing keywords
PHISHING_KEYWORDS = {
    "secure", "security", "verify", "verification",
    "login", "signin", "account", "update", "confirm",
    "banking", "paypal", "amazon", "apple", "google",
    "microsoft", "password", "credential", "suspend",
    "alert", "warning", "urgent", "limited", "locked",
    "recover", "restore", "validate", "authenticate"
}

# Known typosquats
TYPOSQUATS = {
    "paypa1", "g00gle", "micros0ft", "arnazon",
    "linkedln", "faceb00k", "twitterr", "paypai",
    "amaz0n", "gooogle", "microsofft"
}

# Exploit paths
EXPLOIT_PATHS = {
    "wp-admin", "wp-login", "shell", "c99", "r57",
    "phpmyadmin", "config", "setup", "install", "backup"
}


def classify_token(token: str) -> str:
    """Map a token to a plain-English reason. Returns '' if not suspicious."""
    t = token.lower().strip()

    if not t or t in {'http', 'https', 'www', 'com', 'org', 'net', 'io'}:
        return ""

    if t in FREE_TLDS:
        return "uses a free TLD commonly abused in malicious URLs"

    if re.match(r'^\d{1,3}$', t):
        return "contains a raw IP address instead of a domain name"

    if t in TYPOSQUATS:
        return "resembles a brand name misspelling (typosquatting)"

    if t in EXPLOIT_PATHS:
        return "contains a path segment associated with web exploits"

    if t in PHISHING_KEYWORDS:
        return "contains a keyword commonly used in phishing pages"

    if re.search(r'%[0-9a-f]{2}', t):
        return "contains URL-encoded characters used to obfuscate malicious URLs"

    if len(t) > 20 and re.match(r'^[a-z0-9]+$', t):
        return "contains a long random-looking string typical of generated malicious domains"

    # Partial keyword match for things like "paypa1" or "secure-login"
    for kw in PHISHING_KEYWORDS:
        if kw in t and kw != t:
            return "contains a keyword commonly used in phishing pages"

    return ""


def build_lime_explanation(url: str, label: str,
                            lime_results: list) -> dict:
    """
    Convert LIME token weights into structured human-readable explanation.

    Returns:
        {
            "summary"   : "Plain-English one-sentence explanation",
            "reasons"   : ["reason 1", "reason 2"],
            "top_tokens": [{"token", "importance", "reason"}, ...]
        }
    """
    if label == "safe":
        return {
            "summary":    "This URL does not exhibit common malicious patterns.",
            "reasons":    [],
            "top_tokens": []
        }

    if not lime_results:
        return {
            "summary":    "This URL was flagged based on its overall character pattern.",
            "reasons":    [],
            "top_tokens": []
        }

    # Take top tokens by absolute importance (normalized so threshold is 0.05)
    # Use top 6 regardless of threshold - let classify_token filter noise
    top_candidates = [r for r in lime_results
                      if r["importance"] > 0][:6]

    reasons    = []
    top_tokens = []

    for result in top_candidates:
        token  = result["token"]
        imp    = result["importance"]
        reason = classify_token(token)

        entry = {
            "token":      token,
            "importance": imp,
            "reason":     reason if reason else "suspicious character pattern"
        }
        top_tokens.append(entry)

        if reason and reason not in reasons:
            reasons.append(reason)

    # Build summary
    if reasons:
        if len(reasons) == 1:
            summary = f"Flagged because this URL {reasons[0]}."
        else:
            summary = f"Flagged because this URL {reasons[0]}; and {reasons[1]}."
    elif top_tokens:
        token_list = ", ".join(f"'{t['token']}'" for t in top_tokens[:3])
        summary = f"Flagged due to suspicious tokens: {token_list}."
    else:
        summary = "This URL matches patterns commonly seen in malicious URLs."

    return {
        "summary":    summary,
        "reasons":    reasons,
        "top_tokens": top_tokens
    }


# ==============================================================
# 4. QUICK TEST
# ==============================================================

if __name__ == "__main__":
    test_urls = [
        "http://paypa1-secure-login.tk/verify/account",
        "http://192.168.1.1/admin/shell.php",
        "https://google.com.fake-login.xyz/signin",
    ]
    print("Module 5 - URL Tokenizer Test\n")
    for url in test_urls:
        tokens  = tokenize_url(url)
        content = get_content_tokens(tokens)
        print(f"URL     : {url}")
        print(f"All     : {tokens}")
        print(f"Content : {content}")
        for t in content:
            r = classify_token(t)
            if r:
                print(f"  '{t}' -> {r}")
        print()