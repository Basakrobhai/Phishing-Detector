# PhishGuard — Phishing Detection Tool

A modular Python tool that analyses URLs and classifies them as **SAFE**, **SUSPICIOUS**, or **PHISHING** using rule-based heuristics, optional machine learning, and optional Google Safe Browsing API integration.

---

## Project Structure

```
phishing_detector/
├── app.py                  ← Entry point (Flask web UI + CLI)
├── feature_extractor.py    ← URL feature extraction (all heuristics, WHOIS, SSL)
├── detector.py             ← Scoring engine + ML blend + GSB integration
├── train_model.py          ← Model training script (Random Forest)
├── requirements.txt
├── model_data/
│   └── model.pkl           ← Generated after running train_model.py
└── templates/
    └── index.html          ← Web UI
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run — Web Interface (default)

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

### 3. Run — Single URL (CLI)

```bash
python app.py --url https://paypal-secure-login.net/verify
```

### 4. Run — Interactive CLI

```bash
python app.py --cli
```

---

## Train the ML Model

**Without a dataset** (uses synthetic demo data):

```bash
python train_model.py
```

**With a real dataset** (CSV with a `label` column: 0=legit, 1=phishing):

```bash
python train_model.py --dataset /path/to/dataset.csv
```

> Recommended datasets: [UCI Phishing Dataset](https://archive.ics.uci.edu/dataset/327/phishing+websites), [PhiUSIIL](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)

---

## Optional: Google Safe Browsing API

Set your API key as an environment variable before running:

```bash
export GSB_API_KEY="your_key_here"
python app.py
```

Get a key at: https://developers.google.com/safe-browsing/v4/get-started

---

## How Scoring Works

Each suspicious feature contributes points to a **risk score**:

| Score Range | Classification |
|-------------|---------------|
| < 3         | ✅ SAFE        |
| 3 – 6       | ⚠️ SUSPICIOUS  |
| > 6         | 🚨 PHISHING    |

### Rules (selected)

| Rule | Points |
|------|--------|
| URL length > 75 chars | +1.0 |
| URL length > 150 chars | +1.5 |
| Contains `@` symbol | +2.0 |
| IP address as hostname | +2.5 |
| No HTTPS | +1.0 |
| Brand name in subdomain (typosquat) | +3.0 |
| SSL cert mismatch | +3.0 |
| Domain < 30 days old | +2.5 |
| Domain < 180 days old | +1.5 |
| Suspicious keyword in URL | +1.0 |
| Multiple suspicious keywords | +1.5 |
| Double-slash redirect trick | +1.5 |
| Non-standard port | +1.5 |
| Google Safe Browsing hit | +5.0 |

### ML Integration

If `model_data/model.pkl` exists, a Random Forest prediction is blended:
- High-confidence phishing prediction (≥ 70%) → score raised to phishing threshold
- High-confidence safe prediction (≥ 85%) → score reduced by 1.0

---

## API

The Flask app exposes a JSON API:

```
POST /analyse
Content-Type: application/json

{ "url": "https://example.com" }
```

Response:
```json
{
  "url": "https://example.com",
  "classification": "SAFE",
  "risk_score": 0.0,
  "triggered_rules": [],
  "ml_label": "SAFE",
  "ml_confidence": 0.94,
  "gsb_flagged": null,
  "features": { ... },
  "errors": []
}
```

---

## Environment Variables

| Variable     | Default | Description                          |
|-------------|---------|--------------------------------------|
| `GSB_API_KEY` | —      | Google Safe Browsing API key         |
| `FLASK_PORT`  | 5000   | Web server port                      |
| `FLASK_DEBUG` | 0      | Set to `1` for Flask debug mode      |
