"""
app.py
──────
Dual-mode entry point:
  • Web interface  →  python app.py
  • CLI            →  python app.py --url https://example.com
                      python app.py --cli  (interactive prompt)

Environment variables:
  GSB_API_KEY   — Google Safe Browsing API key (optional)
  FLASK_PORT    — Web server port (default 5000)
  FLASK_DEBUG   — Set to '1' for debug mode
"""

import os
import sys
import json
import argparse
from flask import Flask, request, jsonify, render_template # type: ignore

from feature_extractor import extract_features
from detector import analyse, LABEL_SAFE, LABEL_SUSPICIOUS, LABEL_PHISHING

app = Flask(__name__)

GSB_API_KEY = os.environ.get("GSB_API_KEY")


# Helpers

def _analyse_url(url: str) -> dict:
    features = extract_features(url)
    result   = analyse(features, gsb_api_key=GSB_API_KEY)
    return {
        "url":             result.url,
        "classification":  result.classification,
        "risk_score":      result.risk_score,
        "triggered_rules": [{"rule": r, "points": p} for r, p in result.triggered_rules],
        "ml_label":        result.ml_label,
        "ml_confidence":   result.ml_confidence,
        "gsb_flagged":     result.gsb_flagged,
        "features": {
            "url_length":            features.get("url_length"),
            "uses_https":            features.get("uses_https"),
            "uses_ip_address":       features.get("uses_ip_address"),
            "subdomain_count":       features.get("subdomain_count"),
            "domain_age_days":       features.get("domain_age_days"),
            "keyword_count":         features.get("keyword_count"),
            "suspicious_keywords":   features.get("suspicious_keywords", []),
            "typosquat_targets":     features.get("typosquat_targets", []),
            "ssl_match":             features.get("ssl_match"),
            "hostname":              features.get("hostname"),
            "resolved_ip":           features.get("resolved_ip"),
            "trusted_tld":           features.get("trusted_tld"),
        },
        "errors": features.get("errors", []),
    }


# Web routes

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyse", methods=["POST"])
def analyse_route():
    data = request.get_json(silent=True) or {}
    url  = (data.get("url") or request.form.get("url") or "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    try:
        result = _analyse_url(url)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# CLI mode

def cli_single(url: str):
    from feature_extractor import extract_features
    from detector import analyse, DetectionResult
    print(f"\nAnalysing: {url}\n{'─'*60}")
    features = extract_features(url)
    result: DetectionResult = analyse(features, gsb_api_key=GSB_API_KEY)
    print(result.summary())
    print()


def cli_interactive():
    print("Phishing Detector — Interactive CLI")
    print("Type a URL to analyse, or 'quit' to exit.\n")
    while True:
        try:
            url = input("URL> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if url.lower() in ("quit", "exit", "q"):
            break
        if url:
            cli_single(url)


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phishing Detection Tool")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--url",  metavar="URL", help="Analyse a single URL and exit")
    mode.add_argument("--cli",  action="store_true", help="Interactive CLI mode")
    parser.add_argument("--port", type=int, default=int(os.environ.get("FLASK_PORT", 5000)))
    parser.add_argument("--debug", action="store_true",
                        default=os.environ.get("FLASK_DEBUG") == "1")
    args = parser.parse_args()

    if args.url:
        cli_single(args.url)
    elif args.cli:
        cli_interactive()
    else:
        print(f"Starting web interface on http://localhost:{args.port}")
        app.run(host="0.0.0.0", port=args.port, debug=args.debug)
