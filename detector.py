"""
detector.py
───────────
Rule-based scoring engine.  Optionally blends in an ML prediction when a
trained model is available.

Risk score thresholds:
  < 3   → SAFE
  3–6   → SUSPICIOUS
  > 6   → PHISHING
"""

from __future__ import annotations
import os
import pickle
from dataclasses import dataclass, field
from typing import Optional

# ── Thresholds ─────────────────────────────────────────────────────────────────

SAFE_THRESHOLD       = 3
PHISHING_THRESHOLD   = 6

LABEL_SAFE           = "SAFE"
LABEL_SUSPICIOUS     = "SUSPICIOUS"
LABEL_PHISHING       = "PHISHING"

# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    url:              str
    classification:   str
    risk_score:       float
    triggered_rules:  list[tuple[str, float]] = field(default_factory=list)
    ml_label:         Optional[str]           = None
    ml_confidence:    Optional[float]         = None
    gsb_flagged:      Optional[bool]          = None
    feature_snapshot: dict                    = field(default_factory=dict)
    analysis_errors:  list[str]               = field(default_factory=list)

    @property
    def emoji(self) -> str:
        return {"SAFE": "✅", "SUSPICIOUS": "⚠️", "PHISHING": "🚨"}.get(
            self.classification, "❓"
        )

    def summary(self) -> str:
        lines = [
            f"{self.emoji}  Classification : {self.classification}",
            f"   Risk score     : {self.risk_score:.1f}  "
            f"(safe < {SAFE_THRESHOLD}, phishing > {PHISHING_THRESHOLD})",
        ]
        if self.triggered_rules:
            lines.append("\n   Triggered rules:")
            for rule, pts in self.triggered_rules:
                lines.append(f"     +{pts:.1f}  {rule}")
        if self.ml_label:
            conf = f"{self.ml_confidence*100:.0f}%" if self.ml_confidence else "?"
            lines.append(f"\n   ML prediction  : {self.ml_label} ({conf} confidence)")
        if self.gsb_flagged is True:
            lines.append("   Google Safe Browsing: URL is in threat database")
        if self.analysis_errors:
            lines.append("\n   Non-fatal errors during analysis:")
            for e in self.analysis_errors:
                lines.append(f"     • {e}")
        return "\n".join(lines)


# ── Rule engine ────────────────────────────────────────────────────────────────

def _add(rules: list, label: str, points: float, condition: bool):
    if condition:
        rules.append((label, points))


def rule_score(features: dict) -> tuple[float, list[tuple[str, float]]]:
    """
    Evaluate every heuristic rule against *features*.
    Returns (total_score, triggered_rules).
    """
    triggered: list[tuple[str, float]] = []
    a = lambda label, pts, cond: _add(triggered, label, pts, cond)

    # ── URL structure rules ────────────────────────────────────────────────────
    a("URL is very long (>75 chars)",       1.0, features.get("url_length", 0) > 75)
    a("URL is extremely long (>150 chars)", 1.5, features.get("url_length", 0) > 150)
    a("Contains '@' symbol",               2.0, features.get("at_symbol", False))
    a("Double slash redirect trick",        1.5, features.get("double_slash_redirect", False))
    a("Uses IP address instead of domain",  3.0, features.get("uses_ip_address", False))
    a("No HTTPS",                           1.0, not features.get("uses_https", True))
    a("Non-standard port",                  1.5, features.get("non_standard_port", False))
    a("Excessive hyphens in hostname (>2)", 1.0, features.get("hyphen_count", 0) > 2)
    a("Many subdomains (>2)",               1.5, features.get("subdomain_count", 0) > 2)
    a("High hex-encoding in URL (>3)",      1.0, features.get("hex_encoding_count", 0) > 3)
    a("Redirect parameter in query string", 1.0, features.get("has_redirect_param", False))
    a("Many query parameters (>5)",         0.5, features.get("query_param_count", 0) > 5)
    a("Deep path (>5 segments)",            0.5, features.get("path_depth", 0) > 5)
    a("Many dots in URL (>5)",              0.5, features.get("dot_count", 0) > 5)

    # ── Keyword / brand rules ──────────────────────────────────────────────────
    kw_count = features.get("keyword_count", 0)
    a("Suspicious keyword in URL (×1)",     1.0, kw_count >= 1)
    a("Multiple suspicious keywords (×3+)", 1.5, kw_count >= 3)

    typos = features.get("typosquat_targets", [])
    a("Brand name embedded in hostname (typosquat)", 3.0, len(typos) > 0)

    # ── Domain / certificate rules ─────────────────────────────────────────────
    age = features.get("domain_age_days")
    a("Domain very new (<30 days)",         2.5, age is not None and age < 30)
    a("Domain new (<180 days)",             1.5, age is not None and 30 <= age < 180)

    ssl_match = features.get("ssl_match")
    a("SSL certificate mismatch",           3.0, ssl_match is False)

    a("Untrusted TLD",                      0.5, not features.get("trusted_tld", True))

    total = sum(pts for _, pts in triggered)
    return round(total, 2), triggered


# ── Classification ─────────────────────────────────────────────────────────────

def classify(score: float) -> str:
    if score < SAFE_THRESHOLD:
        return LABEL_SAFE
    if score <= PHISHING_THRESHOLD:
        return LABEL_SUSPICIOUS
    return LABEL_PHISHING


# ── ML integration (optional) ──────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_data", "model.pkl")


def _load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        with open(MODEL_PATH, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def _model_is_synthetic() -> bool:
    bundle = _load_model()
    if bundle is None:
        return False
    return bundle.get("synthetic", False)


ML_DISPLAY_THRESHOLD = 0.75   # don't show ML prediction below this confidence

def _ml_predict(features: dict) -> tuple[Optional[str], Optional[float]]:
    """
    Returns (label, confidence) or (None, None) if model unavailable or
    confidence is below ML_DISPLAY_THRESHOLD (unreliable prediction).
    """
    bundle = _load_model()
    if bundle is None:
        return None, None
    try:
        model    = bundle["model"]
        vec_keys = bundle["feature_keys"]
        X = [[features.get(k, 0) for k in vec_keys]]
        label_id = model.predict(X)[0]
        proba    = model.predict_proba(X)[0]
        conf     = float(max(proba))
        # Suppress low-confidence predictions — not reliable enough to show
        if conf < ML_DISPLAY_THRESHOLD:
            return None, None
        label = LABEL_PHISHING if label_id == 1 else LABEL_SAFE
        return label, conf
    except Exception:
        return None, None


# ── Google Safe Browsing (optional) ────────────────────────────────────────────

def _gsb_check(url: str, api_key: str | None) -> Optional[bool]:
    """Returns True if URL is flagged, False if clean, None if check skipped."""
    if not api_key:
        return None
    import urllib.request, json
    endpoint = (
        "https://safebrowsing.googleapis.com/v4/threatMatches:find"
        f"?key={api_key}"
    )
    payload = {
        "client": {"clientId": "phishing_detector", "clientVersion": "1.0"},
        "threatInfo": {
            "threatTypes": [
                "MALWARE", "SOCIAL_ENGINEERING",
                "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION",
            ],
            "platformTypes":    ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries":    [{"url": url}],
        },
    }
    try:
        req  = urllib.request.Request(
            endpoint,
            data    = json.dumps(payload).encode(),
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        return bool(data.get("matches"))
    except Exception:
        return None


# ── Main entry point ───────────────────────────────────────────────────────────

def analyse(features: dict, gsb_api_key: str | None = None) -> DetectionResult:
    """
    Run the full detection pipeline on pre-extracted *features*.
    Returns a populated DetectionResult.
    """
    score, rules = rule_score(features)

    # Optional ML — informational only, never overrides the rule score.
    # Suppressed entirely if the model was trained on synthetic data,
    # since synthetic models don't generalise to real URLs reliably.
    ml_label, ml_conf = (None, None)
    if not _model_is_synthetic():
        ml_label, ml_conf = _ml_predict(features)

    # Optional GSB
    gsb_flagged = _gsb_check(features.get("raw_url", ""), gsb_api_key)
    if gsb_flagged:
        rules.append(("Google Safe Browsing: URL is in threat database", 5.0))
        score += 5.0

    classification = classify(score)

    return DetectionResult(
        url              = features.get("raw_url", ""),
        classification   = classification,
        risk_score       = round(score, 2),
        triggered_rules  = rules,
        ml_label         = ml_label,
        ml_confidence    = ml_conf,
        gsb_flagged      = gsb_flagged,
        feature_snapshot = features,
        analysis_errors  = features.get("errors", []),
    )
