import re
import numpy as np
from typing import Callable
from functools import lru_cache


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

    scheme_match = re.match(r'^(https?)(://)', url)
    if scheme_match:
        tokens.append(scheme_match.group(1))
        tokens.append(scheme_match.group(2))
        url = url[scheme_match.end():]

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
    Reconstruct URL replacing masked content tokens with 'a'.
    Delimiters are always preserved.
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
# 2. OPTIMIZED LIME EXPLAINER
# ==============================================================

class URLLimeExplainer:
    """
    Optimized LIME explainer using batched model predictions.

    Key optimization: instead of calling model.predict() N times
    (once per perturbed URL), we encode all N URLs into a single
    numpy array and call model.predict() ONCE with batch_size=N.
    This uses GPU parallelism and reduces Python overhead by ~10x.
    """

    def __init__(self,
                 predict_fn: Callable,
                 encode_fn: Callable,
                 feature_fn: Callable,
                 num_samples: int = 150):
        """
        Args:
            predict_fn  : model.predict(inputs, verbose=0)
            encode_fn   : function(url_str) -> np.ndarray (1, MAX_LEN)
            feature_fn  : function(url_str) -> np.ndarray (1, N_FEATURES)
            num_samples : number of perturbations (150 is sweet spot)
        """
        self.predict_fn  = predict_fn
        self.encode_fn   = encode_fn
        self.feature_fn  = feature_fn
        self.num_samples = num_samples
        self._cache      = {}   # url -> lime_results


    def explain(self, url: str) -> list:
        """
        Run LIME on a URL. Returns token importance list.
        Results are cached so repeated calls for the same URL are instant.
        """
        url_lower = url.strip().lower()

        # Cache hit - return immediately
        if url_lower in self._cache:
            return self._cache[url_lower]

        result = self._run_lime(url_lower)
        self._cache[url_lower] = result
        return result


    def _run_lime(self, url: str) -> list:
        all_tokens   = tokenize_url(url)
        content_toks = get_content_tokens(all_tokens)
        n_tokens     = len(content_toks)

        if n_tokens == 0:
            return []

        # Generate random perturbation masks
        np.random.seed(42)
        masks    = np.random.randint(0, 2, size=(self.num_samples, n_tokens))
        masks[0] = np.ones(n_tokens, dtype=int)   # first = full URL

        # ── Batch encode all perturbed URLs ────────────────────
        # Build (num_samples, MAX_LEN) sequence array in one pass
        # instead of calling encode_fn num_samples times separately
        MAX_LEN    = self.encode_fn(url).shape[1]
        N_FEATURES = self.feature_fn(url).shape[1]

        seq_batch  = np.zeros((self.num_samples, MAX_LEN),    dtype=np.int32)
        feat_batch = np.zeros((self.num_samples, N_FEATURES), dtype=np.float32)

        # Features stay the same for all perturbations (only sequence changes)
        base_feats = self.feature_fn(url)[0]   # shape (N_FEATURES,)

        for i, mask in enumerate(masks):
            perturbed       = reconstruct_url(all_tokens, mask)
            seq_batch[i]    = self.encode_fn(perturbed)[0]
            feat_batch[i]   = base_feats   # reuse same features

        # ── Single batched model call ──────────────────────────
        # This is the key optimization - one predict() call instead
        # of num_samples individual calls. GPU processes all in parallel.
        predictions = self.predict_fn(
            [seq_batch, feat_batch],
            verbose=0,
            batch_size=self.num_samples
        ).flatten()

        # ── Weighted linear regression ─────────────────────────
        distances  = np.sqrt(np.sum((masks - 1) ** 2, axis=1))
        kernel_w   = np.sqrt(np.exp(-(distances ** 2) / (n_tokens ** 2)))

        try:
            W          = np.diag(kernel_w)
            Xw         = W @ masks.astype(float)
            yw         = W @ predictions
            Xw_inter   = np.column_stack([Xw, kernel_w])
            coeffs, _, _, _ = np.linalg.lstsq(Xw_inter, yw, rcond=None)
            importances = coeffs[:n_tokens]
        except Exception:
            importances = np.array([
                float(np.corrcoef(masks[:, i], predictions)[0, 1])
                for i in range(n_tokens)
            ])

        # Normalize to [-1, 1]
        max_abs = np.max(np.abs(importances))
        if max_abs > 0:
            importances = importances / max_abs

        results = []
        for i, token in enumerate(content_toks):
            results.append({
                "token":      token,
                "importance": round(float(importances[i]), 4),
                "position":   i
            })

        results.sort(key=lambda x: abs(x["importance"]), reverse=True)
        return results


    def clear_cache(self):
        """Clear the URL result cache."""
        self._cache.clear()


# ==============================================================
# 3. PATTERN MATCHING FOR HUMAN-READABLE EXPLANATIONS
# ==============================================================

FREE_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top",
    "pw", "cc", "su", "click", "loan", "work",
    "date", "racing", "win", "download"
}

PHISHING_KEYWORDS = {
    "secure", "security", "verify", "verification",
    "login", "signin", "account", "update", "confirm",
    "banking", "paypal", "amazon", "apple", "google",
    "microsoft", "password", "credential", "suspend",
    "alert", "warning", "urgent", "limited", "locked",
    "recover", "restore", "validate", "authenticate"
}

TYPOSQUATS = {
    "paypa1", "g00gle", "micros0ft", "arnazon",
    "linkedln", "faceb00k", "twitterr", "paypai",
    "amaz0n", "gooogle", "microsofft"
}

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
            "summary"   : "Plain-English explanation sentence",
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

    top_candidates = [r for r in lime_results if r["importance"] > 0][:6]

    reasons    = []
    top_tokens = []

    for result in top_candidates:
        token  = result["token"]
        imp    = result["importance"]
        reason = classify_token(token)

        top_tokens.append({
            "token":      token,
            "importance": imp,
            "reason":     reason if reason else "suspicious character pattern"
        })

        if reason and reason not in reasons:
            reasons.append(reason)

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
        print(f"Content : {content}")
        for t in content:
            r = classify_token(t)
            if r:
                print(f"  '{t}' -> {r}")
        print()