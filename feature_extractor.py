"""
feature_extractor.py
────────────────────
Extracts heuristic and structural features from a URL for phishing detection.
All network calls (WHOIS, DNS, SSL) are individually try/caught so a single
failure never kills the whole analysis.
"""

import re
import socket
import ssl
import ipaddress
from urllib.parse import urlparse
from datetime import datetime, timezone

try:
    import whois as _whois          # python-whois
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False

# ── Constants ──────────────────────────────────────────────────────────────────

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "account", "update", "banking",
    "confirm", "password", "signin", "webscr", "ebayisapi", "paypal",
    "authenticate", "validation", "wallet", "credential", "free", "lucky",
    "bonus", "winner", "click", "submit", "access", "support", "service",
    "invoice", "alert", "suspended", "unusual", "locked",
]

TRUSTED_TLDS = {".com", ".org", ".net", ".edu", ".gov", ".co", ".io", ".uk"}

_BRANDS = [
    "paypal", "apple", "google", "amazon", "microsoft", "facebook",
    "netflix", "instagram", "twitter", "linkedin", "chase", "wellsfargo",
    "bankofamerica", "citibank", "ebay", "dropbox", "office365", "outlook",
]

# ── Internal helpers ───────────────────────────────────────────────────────────

def _is_ip(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def _domain_age_days(hostname: str):
    """Return domain age in days, or None on failure."""
    if not WHOIS_AVAILABLE:
        return None
    try:
        w = _whois.whois(hostname)
        creation = w.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if not creation:
            return None
        if creation.tzinfo is None:
            creation = creation.replace(tzinfo=timezone.utc)
        return max((datetime.now(timezone.utc) - creation).days, 0)
    except Exception:
        return None


def _ssl_domain_match(hostname: str):
    """
    True  → cert SAN/CN matches hostname
    False → mismatch detected
    None  → could not verify
    """
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(
            socket.create_connection((hostname, 443), timeout=5),
            server_hostname=hostname,
        ) as ssock:
            cert = ssock.getpeercert()
        san = cert.get("subjectAltName", ())
        names = [v for k, v in san if k == "DNS"]
        if not names:
            cn = dict(x[0] for x in cert.get("subject", ())).get("commonName", "")
            names = [cn] if cn else []
        for name in names:
            pattern = re.escape(name).replace(r"\*", r"[^.]+")
            if re.fullmatch(pattern, hostname, re.IGNORECASE):
                return True
        return False
    except Exception:
        return None


def _detect_typosquats(hostname: str) -> list:
    hostname_lower = hostname.lower()
    hits = []
    for brand in _BRANDS:
        if brand in hostname_lower:
            parts = hostname_lower.split(".")
            root = parts[-2] if len(parts) >= 2 else parts[0]
            if root != brand:            # brand appears in subdomain/path → suspicious
                hits.append(brand)
    return hits


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_features(raw_url: str) -> dict:
    """
    Analyse *raw_url* and return a flat dict of features plus an 'errors' list.
    Never raises — all failures are captured in 'errors'.
    """
    errors = []

    # Normalise scheme
    if not re.match(r"^https?://", raw_url, re.IGNORECASE):
        raw_url = "http://" + raw_url

    try:
        parsed = urlparse(raw_url)
    except Exception as exc:
        return {"errors": [f"URL parse failure: {exc}"], "raw_url": raw_url}

    hostname  = (parsed.hostname or "").lower()
    path      = parsed.path  or ""
    query     = parsed.query or ""
    full      = raw_url
    lower_full = full.lower()

    # ── URL structure ──────────────────────────────────────────────────────────
    url_length          = len(full)
    dot_count           = full.count(".")
    hyphen_count        = hostname.count("-")
    at_symbol           = "@" in full
    double_slash        = "//" in path
    hex_encoding_count  = len(re.findall(r"%[0-9a-fA-F]{2}", full))
    subdomain_count     = max(len(hostname.split(".")) - 2, 0)
    uses_ip             = _is_ip(hostname)
    uses_https          = parsed.scheme.lower() == "https"
    non_standard_port   = parsed.port not in (None, 80, 443)
    query_param_count   = len(query.split("&")) if query else 0
    path_depth          = len([p for p in path.split("/") if p])
    has_redirect_param  = bool(
        re.search(r"(redirect|return|goto|url)=", query, re.IGNORECASE)
    )

    # ── Keyword / brand signals ────────────────────────────────────────────────
    suspicious_keywords_found = [kw for kw in SUSPICIOUS_KEYWORDS if kw in lower_full]
    typosquat_targets         = _detect_typosquats(hostname)

    # ── Domain info ────────────────────────────────────────────────────────────
    tld         = ("." + hostname.split(".")[-1]) if "." in hostname else ""
    trusted_tld = tld.lower() in TRUSTED_TLDS

    domain_age_days = None
    try:
        domain_age_days = _domain_age_days(hostname)
    except Exception as exc:
        errors.append(f"WHOIS: {exc}")

    ssl_match = None
    if uses_https:
        try:
            ssl_match = _ssl_domain_match(hostname)
        except Exception as exc:
            errors.append(f"SSL: {exc}")

    resolved_ip = None
    try:
        resolved_ip = socket.gethostbyname(hostname)
    except Exception as exc:
        errors.append(f"DNS: {exc}")

    return {
        # URL structure
        "url_length":           url_length,
        "dot_count":            dot_count,
        "hyphen_count":         hyphen_count,
        "at_symbol":            at_symbol,
        "double_slash_redirect":double_slash,
        "hex_encoding_count":   hex_encoding_count,
        "subdomain_count":      subdomain_count,
        "uses_ip_address":      uses_ip,
        "uses_https":           uses_https,
        "non_standard_port":    non_standard_port,
        "query_param_count":    query_param_count,
        "path_depth":           path_depth,
        "has_redirect_param":   has_redirect_param,
        # Content signals
        "suspicious_keywords":  suspicious_keywords_found,
        "keyword_count":        len(suspicious_keywords_found),
        "typosquat_targets":    typosquat_targets,
        # Domain
        "hostname":             hostname,
        "tld":                  tld,
        "trusted_tld":          trusted_tld,
        "domain_age_days":      domain_age_days,
        "resolved_ip":          resolved_ip,
        "ssl_match":            ssl_match,
        # Meta
        "raw_url":              raw_url,
        "parsed_scheme":        parsed.scheme,
        "errors":               errors,
    }
