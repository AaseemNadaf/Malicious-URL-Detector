"""
Module 4 - Flask REST API (with Module 5 LIME XAI)
====================================================
Project : Malicious URL Detector (CNN + XAI)
File    : module4_api/app.py
Run     : python app.py
"""

import os
import sys
import json
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, MaxPooling1D,
    GlobalMaxPooling1D, Dense, Dropout,
    Concatenate, SpatialDropout1D
)
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import LIME explainer from Module 5
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'module5_xai'))
from explainer import URLLimeExplainer, build_lime_explanation


# ==============================================================
# 0. CONFIG
# ==============================================================

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR      = os.path.join(BASE_DIR, "models")
WEIGHTS_PATH    = os.path.join(MODELS_DIR, "url_cnn_weights.weights.h5")
VOCAB_PATH      = os.path.join(MODELS_DIR, "char_vocab.json")
THRESHOLD_PATH  = os.path.join(MODELS_DIR, "threshold.json")

MAX_LEN         = 200
EMBEDDING_DIM   = 128
N_FEATURES      = 10


# ==============================================================
# 1. TRUSTED DOMAIN WHITELIST
# ==============================================================

TRUSTED_DOMAINS = {
    # Google
    "google.com", "www.google.com", "mail.google.com",
    "drive.google.com", "docs.google.com", "accounts.google.com",
    "play.google.com", "maps.google.com", "news.google.com",
    "calendar.google.com", "meet.google.com", "chat.google.com",
    "photos.google.com", "translate.google.com",
    # GitHub
    "github.com", "www.github.com", "gist.github.com",
    "raw.githubusercontent.com", "githubusercontent.com",
    # YouTube
    "youtube.com", "www.youtube.com", "youtu.be",
    # Amazon
    "amazon.com", "www.amazon.com", "amazon.in", "amazon.co.uk",
    "amazon.de", "amazon.fr", "amazon.co.jp",
    # Microsoft
    "microsoft.com", "www.microsoft.com", "outlook.com", "outlook.live.com",
    "office.com", "www.office.com", "live.com", "hotmail.com",
    "azure.com", "visualstudio.com", "onedrive.live.com",
    "teams.microsoft.com", "sharepoint.com",
    # Apple
    "apple.com", "www.apple.com", "icloud.com", "www.icloud.com",
    "support.apple.com", "developer.apple.com",
    # Meta
    "facebook.com", "www.facebook.com", "m.facebook.com",
    "instagram.com", "www.instagram.com",
    "whatsapp.com", "www.whatsapp.com", "web.whatsapp.com",
    "messenger.com", "www.messenger.com",
    # Twitter / X
    "twitter.com", "www.twitter.com", "x.com", "www.x.com",
    # Professional
    "linkedin.com", "www.linkedin.com",
    # Entertainment
    "netflix.com", "www.netflix.com",
    "spotify.com", "www.spotify.com",
    "twitch.tv", "www.twitch.tv",
    "primevideo.com", "www.primevideo.com",
    "disneyplus.com", "www.disneyplus.com",
    "hulu.com", "www.hulu.com",
    # Sports & News
    "formula1.com", "www.formula1.com",
    "espn.com", "www.espn.com",
    "bbc.com", "www.bbc.com", "bbc.co.uk", "www.bbc.co.uk",
    "cnn.com", "www.cnn.com",
    "nytimes.com", "www.nytimes.com",
    "reuters.com", "www.reuters.com",
    "theguardian.com", "www.theguardian.com",
    "forbes.com", "www.forbes.com",
    "bloomberg.com", "www.bloomberg.com",
    "techcrunch.com", "www.techcrunch.com",
    "nba.com", "www.nba.com",
    "fifa.com", "www.fifa.com",
    "uefa.com", "www.uefa.com",
    # Knowledge
    "wikipedia.org", "www.wikipedia.org", "en.wikipedia.org",
    "reddit.com", "www.reddit.com", "old.reddit.com",
    "stackoverflow.com", "www.stackoverflow.com",
    "stackexchange.com", "www.stackexchange.com",
    "quora.com", "www.quora.com",
    "medium.com", "www.medium.com",
    # Search
    "yahoo.com", "www.yahoo.com",
    "bing.com", "www.bing.com",
    "duckduckgo.com", "www.duckduckgo.com",
    "baidu.com", "www.baidu.com",
    # Dev tools
    "npmjs.com", "www.npmjs.com",
    "pypi.org", "www.pypi.org",
    "docker.com", "www.docker.com", "hub.docker.com",
    "gitlab.com", "www.gitlab.com",
    "bitbucket.org", "www.bitbucket.org",
    "heroku.com", "www.heroku.com",
    "vercel.com", "www.vercel.com",
    "netlify.com", "www.netlify.com",
    "digitalocean.com", "www.digitalocean.com",
    "aws.amazon.com", "console.aws.amazon.com",
    "cloud.google.com", "portal.azure.com",
    "colab.research.google.com", "kaggle.com", "www.kaggle.com",
    # Productivity & tools
    "dropbox.com", "www.dropbox.com",
    "paypal.com", "www.paypal.com",
    "wordpress.com", "www.wordpress.com",
    "discord.com", "www.discord.com", "discordapp.com",
    "zoom.us", "www.zoom.us",
    "slack.com", "www.slack.com", "app.slack.com",
    "notion.so", "www.notion.so",
    "figma.com", "www.figma.com",
    "canva.com", "www.canva.com",
    "trello.com", "www.trello.com",
    "atlassian.com", "www.atlassian.com",
    "asana.com", "www.asana.com",
    "airtable.com", "www.airtable.com",
    # E-commerce
    "ebay.com", "www.ebay.com",
    "etsy.com", "www.etsy.com",
    "shopify.com", "www.shopify.com",
    "aliexpress.com", "www.aliexpress.com",
    "flipkart.com", "www.flipkart.com",
    # Finance
    "stripe.com", "www.stripe.com", "dashboard.stripe.com",
    "coinbase.com", "www.coinbase.com",
    # Infrastructure
    "cloudflare.com", "www.cloudflare.com",
    "amazonaws.com", "s3.amazonaws.com",
    "googleapis.com", "gstatic.com",
    "akamai.com", "www.akamai.com",
    # Education
    "coursera.org", "www.coursera.org",
    "udemy.com", "www.udemy.com",
    "khanacademy.org", "www.khanacademy.org",
    "edx.org", "www.edx.org",
    "mit.edu", "www.mit.edu",
    "stanford.edu", "www.stanford.edu",
}


def extract_hostname(url: str) -> str:
    url = str(url).strip().lower()
    url = re.sub(r'^https?://', '', url)
    host = url.split('/')[0]
    host = host.split(':')[0]
    host = host.split('?')[0]
    return host


def is_trusted(url: str) -> bool:
    return extract_hostname(url) in TRUSTED_DOMAINS


# ==============================================================
# 2. BUILD MODEL ARCHITECTURE
# ==============================================================

def build_model(vocab_size: int, max_len: int,
                embedding_dim: int, n_features: int) -> Model:

    seq_input = Input(shape=(max_len,), name="url_input")
    x = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="char_embedding"
    )(seq_input)
    x = SpatialDropout1D(0.2, name="spatial_dropout")(x)

    b1 = Conv1D(128, kernel_size=2, activation='relu',
                padding='same', name="conv1_k2")(x)
    b1 = MaxPooling1D(pool_size=2, name="pool1_k2")(b1)
    b1 = Conv1D(64, kernel_size=2, activation='relu',
                padding='same', name="conv2_k2")(b1)
    b1 = GlobalMaxPooling1D(name="gpool_k2")(b1)

    b2 = Conv1D(128, kernel_size=3, activation='relu',
                padding='same', name="conv1_k3")(x)
    b2 = MaxPooling1D(pool_size=2, name="pool1_k3")(b2)
    b2 = Conv1D(64, kernel_size=3, activation='relu',
                padding='same', name="conv2_k3")(b2)
    b2 = GlobalMaxPooling1D(name="gpool_k3")(b2)

    b3 = Conv1D(128, kernel_size=5, activation='relu',
                padding='same', name="conv1_k5")(x)
    b3 = MaxPooling1D(pool_size=2, name="pool1_k5")(b3)
    b3 = Conv1D(64, kernel_size=5, activation='relu',
                padding='same', name="conv2_k5")(b3)
    b3 = GlobalMaxPooling1D(name="gpool_k5")(b3)

    cnn_out = Concatenate(name="cnn_merge")([b1, b2, b3])

    feat_input = Input(shape=(n_features,), name="feature_input")
    feat_out = Dense(32, activation='relu', name="feat_dense")(feat_input)

    merged = Concatenate(name="final_merge")([cnn_out, feat_out])
    out = Dense(256, activation='relu',    name="dense1")(merged)
    out = Dropout(0.5,                     name="dropout1")(out)
    out = Dense(128, activation='relu',    name="dense2")(out)
    out = Dropout(0.4,                     name="dropout2")(out)
    out = Dense(64,  activation='relu',    name="dense3")(out)
    out = Dropout(0.3,                     name="dropout3")(out)
    output = Dense(1, activation='sigmoid', name="output")(out)

    return Model(inputs=[seq_input, feat_input], outputs=output,
                 name="HybridURLCNN")


# ==============================================================
# 3. LOAD RESOURCES AT STARTUP
# ==============================================================

print("\n" + "=" * 55)
print("  Loading resources...")
print("=" * 55)

print("  Loading vocabulary...")
with open(VOCAB_PATH) as f:
    vocab = json.load(f)
VOCAB_SIZE = len(vocab) + 1
print(f"  Vocab size       : {VOCAB_SIZE}")

print("  Loading threshold...")
with open(THRESHOLD_PATH) as f:
    THRESHOLD = json.load(f)["threshold"]
print(f"  Threshold        : {THRESHOLD:.4f}")

print("  Building model...")
model = build_model(VOCAB_SIZE, MAX_LEN, EMBEDDING_DIM, N_FEATURES)

print("  Loading weights...")
model.load_weights(WEIGHTS_PATH)
print("  Weights loaded.")

# Warm-up
_dummy_seq   = np.zeros((1, MAX_LEN),    dtype=np.int32)
_dummy_feats = np.zeros((1, N_FEATURES), dtype=np.float32)
model.predict([_dummy_seq, _dummy_feats], verbose=0)
print("  Model warmed up.")
print(f"  Trusted domains  : {len(TRUSTED_DOMAINS)}")
print("=" * 55 + "\n")


# ==============================================================
# 4. PREPROCESSING
# ==============================================================

def encode_url(url: str) -> np.ndarray:
    url = str(url).strip().lower()
    sequence = np.zeros((1, MAX_LEN), dtype=np.int32)
    for i, char in enumerate(url[:MAX_LEN]):
        sequence[0, i] = vocab.get(char, 0)
    return sequence


def extract_features(url: str) -> np.ndarray:
    SUSPICIOUS_TLDS = ['.tk', '.ml', '.ga', '.cf', '.gq',
                       '.xyz', '.top', '.pw', '.cc', '.su']
    IP_PATTERN = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
    f = [
        len(url) / 200,
        url.count('.') / 10,
        url.count('-') / 10,
        url.count('/') / 10,
        url.count('@') / 5,
        1.0 if IP_PATTERN.search(url) else 0.0,
        1.0 if any(t in url for t in SUSPICIOUS_TLDS) else 0.0,
        1.0 if len(url) > 75 else 0.0,
        url.count('=') / 5,
        url.count('?') / 3,
    ]
    return np.array([f], dtype=np.float32)


# ==============================================================
# 5. INITIALISE LIME EXPLAINER
# ==============================================================

lime_explainer = URLLimeExplainer(
    predict_fn  = model.predict,
    encode_fn   = encode_url,
    feature_fn  = extract_features,
    num_samples = 300    # 300 perturbations - good balance of speed vs accuracy
)

print("  LIME explainer initialised.")


# ==============================================================
# 6. FLASK APP
# ==============================================================

app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":          "ok",
        "threshold":       THRESHOLD,
        "vocab_size":      VOCAB_SIZE,
        "trusted_domains": len(TRUSTED_DOMAINS)
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "url" not in data:
        return jsonify({"error": "Request body must include a 'url' field."}), 400

    url = str(data["url"]).strip()
    if not url:
        return jsonify({"error": "URL cannot be empty."}), 400
    if len(url) > 2048:
        return jsonify({"error": "URL too long (max 2048 characters)."}), 400

    # Whitelist check
    if is_trusted(url):
        return jsonify({
            "url":          url,
            "label":        "safe",
            "confidence":   1.0,
            "risk_score":   0.0,
            "explanation":  "This domain is on the trusted whitelist.",
            "top_tokens":   [],
            "top_segments": [],
            "whitelisted":  True
        })

    # Preprocess
    seq   = encode_url(url)
    feats = extract_features(url)

    # Predict
    risk_score   = float(model.predict([seq, feats], verbose=0)[0][0])
    is_malicious = risk_score >= THRESHOLD
    label        = "malicious" if is_malicious else "safe"
    confidence   = risk_score if is_malicious else (1.0 - risk_score)

    # LIME explanation (only run for malicious to save time on safe URLs)
    if is_malicious:
        lime_results  = lime_explainer.explain(url)
        xai_output    = build_lime_explanation(url, label, lime_results)
    else:
        lime_results  = []
        xai_output    = build_lime_explanation(url, label, [])

    return jsonify({
        "url":          url,
        "label":        label,
        "confidence":   round(confidence, 4),
        "risk_score":   round(risk_score, 4),
        "explanation":  xai_output["summary"],
        "reasons":      xai_output["reasons"],
        "top_tokens":   xai_output["top_tokens"],
        "top_segments": lime_results[:5],
        "whitelisted":  False
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error."}), 500


# ==============================================================
# 7. RUN
# ==============================================================

if __name__ == "__main__":
    print("  Listening on http://localhost:5000")
    print("  GET  /health")
    print("  POST /predict  { url: '...' }\n")
    app.run(debug=True, host="0.0.0.0", port=5000)