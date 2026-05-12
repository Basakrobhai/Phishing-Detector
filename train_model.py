"""
train_model.py
──────────────
Train a Random Forest classifier on the PhiUSIIL / UCI phishing dataset.

Usage
-----
  python train_model.py
  python train_model.py --dataset path/to/dataset.csv

The script can run in two modes:
  1. With a real dataset (CSV with a 'label' column, 1=phishing, 0=legit)
  2. Synthetic demo mode — generates a small balanced synthetic dataset so
     the pipeline can be tested without downloading real data.

The trained model bundle is saved to:
  model_data/model.pkl
"""

import os
import argparse
import pickle
import numpy as np

try:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "model_data")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

# Feature keys must match what feature_extractor.extract_features() returns
FEATURE_KEYS = [
    "url_length",
    "dot_count",
    "hyphen_count",
    "at_symbol",
    "double_slash_redirect",
    "hex_encoding_count",
    "subdomain_count",
    "uses_ip_address",
    "uses_https",
    "non_standard_port",
    "query_param_count",
    "path_depth",
    "has_redirect_param",
    "keyword_count",
    "trusted_tld",
]

# ── Synthetic dataset ──────────────────────────────────────────────────────────

def _synthetic_dataset(n=4000):
    """
    Generate a balanced synthetic dataset for demo / CI purposes.

    Legit URLs vary widely in the real world — youtube.com has a 'www'
    subdomain, google.com/search has query params, github.com/user/repo
    has deep paths.  We generate from MULTIPLE legit archetypes so the
    model learns that subdomains / query params / longer paths are normal.
    """
    rng = np.random.default_rng(42)
    half = n // 2

    # ── Multiple legit archetypes (mixed together) ─────────────────────────
    # Each represents a real class of legitimate URL:
    #   simple homepage, CDN/subdomain, long-path, heavy-query
    legit_archetypes = np.array([
        # len  dots hyph  @    //   hex  sub  ip   https port qpar pdep redir kw   tld
        [ 28,   2,   0,   0,   0,   0,   0,   0,   1,   0,   0,   1,   0,   0,   1 ],  # simple: google.com
        [ 35,   3,   0,   0,   0,   0,   1,   0,   1,   0,   0,   1,   0,   0,   1 ],  # subdomain: www.youtube.com
        [ 60,   3,   0,   0,   0,   0,   1,   0,   1,   0,   2,   3,   0,   0,   1 ],  # path: github.com/user/repo
        [ 80,   4,   0,   0,   0,   0,   0,   0,   1,   0,   4,   2,   0,   0,   1 ],  # query-heavy: search engine
        [ 45,   3,   1,   0,   0,   0,   0,   0,   1,   0,   1,   2,   0,   0,   1 ],  # hyphen domain: my-site.com
    ], dtype=float)

    noise_legit = np.array([
        12,  1,   1,  0.05, 0.05, 0.5, 0.4, 0.05, 0.1, 0.05, 1.5, 1,  0.05, 0.2, 0.1
    ])

    # Sample equally from all legit archetypes
    per_arch = half // len(legit_archetypes)
    legit_parts = []
    for arch in legit_archetypes:
        chunk = rng.normal(arch, noise_legit, (per_arch, len(FEATURE_KEYS)))
        legit_parts.append(chunk)
    legit = np.vstack(legit_parts)[:half]

    # ── Phishing archetype ─────────────────────────────────────────────────
    phish_archetypes = np.array([
        # len   dots hyph  @    //   hex  sub  ip   https port qpar pdep redir kw   tld
        [ 110,   7,   3,   0,   1,   4,   3,   0,   0,   0,   6,   5,   1,   3,   0 ],  # classic phish
        [  90,   5,   2,   0,   0,   2,   2,   0,   0,   0,   3,   3,   1,   2,   0 ],  # moderate phish
        [ 140,   8,   4,   1,   1,   6,   4,   0,   0,   1,   8,   6,   1,   4,   0 ],  # heavy phish
        [  55,   4,   3,   0,   0,   1,   2,   1,   0,   0,   2,   3,   0,   1,   1 ],  # IP-based phish
    ], dtype=float)

    noise_phish = np.array([
        25,  2,   1.5, 0.3, 0.4, 2,  1,  0.3, 0.35, 0.4, 3,  2,  0.4, 1,  0.35
    ])

    per_arch_p = half // len(phish_archetypes)
    phish_parts = []
    for arch in phish_archetypes:
        chunk = rng.normal(arch, noise_phish, (per_arch_p, len(FEATURE_KEYS)))
        phish_parts.append(chunk)
    phish = np.vstack(phish_parts)[:half]

    # Clip to sane ranges and binarise boolean columns
    for data in (legit, phish):
        data[:, 0]  = np.clip(data[:, 0], 10, 500)    # url_length
        for col in [3, 4, 7, 8, 9, 12]:               # binary features
            data[:, col] = (data[:, col] > 0.5).astype(float)
        data[:, 14] = (data[:, 14] > 0.5).astype(float)  # trusted_tld
        np.clip(data, 0, None, out=data)

    X = np.vstack([legit, phish])
    y = np.array([0] * len(legit) + [1] * len(phish))
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# ── CSV dataset loader ─────────────────────────────────────────────────────────

def _load_csv(path: str):
    df = pd.read_csv(path)
    # Flexible label column names
    label_col = next(
        (c for c in df.columns if c.lower() in ("label", "class", "phishing", "result")),
        None,
    )
    if label_col is None:
        raise ValueError("CSV must contain a 'label' column (0=legit, 1=phishing).")
    y = df[label_col].values
    # Intersect with available feature columns
    available = [k for k in FEATURE_KEYS if k in df.columns]
    missing   = set(FEATURE_KEYS) - set(available)
    if missing:
        print(f"  Warning: {len(missing)} feature(s) missing from CSV, filling with 0: {missing}")
    for m in missing:
        df[m] = 0
    X = df[FEATURE_KEYS].values.astype(float)
    return X, y


# ── Training ───────────────────────────────────────────────────────────────────

def train(dataset_path: str | None = None):
    if not ML_AVAILABLE:
        print("scikit-learn / pandas not installed — skipping model training.")
        print("Install with:  pip install scikit-learn pandas")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)

    if dataset_path:
        print(f"Loading dataset: {dataset_path}")
        X, y = _load_csv(dataset_path)
    else:
        print("No dataset provided — generating synthetic demo dataset (n=2000)")
        X, y = _synthetic_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"Training RandomForest on {len(X_train)} samples …")
    clf = RandomForestClassifier(
        n_estimators    = 200,
        max_depth       = 12,
        min_samples_leaf= 2,
        class_weight    = "balanced",
        random_state    = 42,
        n_jobs          = -1,
    )
    clf.fit(X_train, y_train)

    print("\n── Test-set performance ──")
    print(classification_report(y_test, clf.predict(X_test),
                                target_names=["Legit", "Phishing"]))

    bundle = {
        "model":        clf,
        "scaler":       scaler,
        "feature_keys": FEATURE_KEYS,
        "synthetic":    dataset_path is None,   # flag so detector knows
    }
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(bundle, fh)
    print(f"Model saved → {MODEL_PATH}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train phishing detection model")
    parser.add_argument("--dataset", default=None,
                        help="Path to CSV dataset (optional; uses synthetic data if omitted)")
    args = parser.parse_args()
    train(args.dataset)
