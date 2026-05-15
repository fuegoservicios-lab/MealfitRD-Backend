"""[P2-AUDIT-4 · 2026-05-15] Test parser-based: el CSP de `vercel.json` ya
NO whitelistea `fonts.googleapis.com` ni `fonts.gstatic.com` en `connect-src`
ni `font-src`.

Por qué este test:
    Convención del repo `P3-SELF-HOST-FONTS`: el E2E test
    (`frontend/e2e/golden_path.spec.js`) rechaza requests a `fonts.gstatic.com`.
    Pero el CSP de vercel.json los seguía whitelisteando — incoherencia
    documental detectada en audit 2026-05-15. Si el CSP los permite, una
    regresión accidental (e.g., `<link rel="stylesheet" href="https://fonts.
    googleapis.com/...">`) pasaría el CSP pero rompería el E2E test sin pista
    obvia.

Fix esperado:
    - Remover `https://fonts.googleapis.com` de `connect-src`.
    - Remover `https://fonts.gstatic.com` de `connect-src` y `font-src`.
    - Remover `https://fonts.googleapis.com` de `style-src`.

Drift detection:
    - `vercel.json` no contiene `fonts.googleapis.com` ni `fonts.gstatic.com`
      en el `value` del CSP header (Content-Security-Policy o
      Content-Security-Policy-Report-Only).
    - Mantenemos los otros 5 headers obligatorios (HSTS, nosniff, DENY,
      Referrer-Policy, Permissions-Policy) intactos.

Cross-link convention (P2-HIST-AUDIT-14): slug `p2_audit_4`.

Tooltip-anchor: P2-AUDIT-4-START | gap audit 2026-05-15
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_VERCEL_JSON = _REPO_ROOT / "frontend" / "vercel.json"


@pytest.fixture(scope="module")
def vercel_src() -> str:
    return _VERCEL_JSON.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def vercel_config(vercel_src: str) -> dict:
    return json.loads(vercel_src)


def _csp_value(config: dict) -> str:
    """Localiza el header CSP (enforced o Report-Only) y devuelve su valor."""
    for entry in config.get("headers", []):
        for h in entry.get("headers", []):
            key = h.get("key", "")
            if key in ("Content-Security-Policy", "Content-Security-Policy-Report-Only"):
                return h.get("value", "")
    return ""


# ---------------------------------------------------------------------------
# 1. fonts.googleapis.com ya NO aparece en el CSP value
# ---------------------------------------------------------------------------
def test_csp_no_fonts_googleapis(vercel_config: dict):
    csp = _csp_value(vercel_config)
    assert csp, "P2-AUDIT-4 regresión: header CSP ausente. Esperado al menos uno."
    assert "fonts.googleapis.com" not in csp, (
        "P2-AUDIT-4 regresión: `fonts.googleapis.com` reaparece en el CSP. "
        "Incoherencia con P3-SELF-HOST-FONTS (E2E test rechaza requests a "
        "ese host). Remover del CSP value."
    )


def test_csp_no_fonts_gstatic(vercel_config: dict):
    csp = _csp_value(vercel_config)
    assert "fonts.gstatic.com" not in csp, (
        "P2-AUDIT-4 regresión: `fonts.gstatic.com` reaparece en el CSP. "
        "Incoherencia con P3-SELF-HOST-FONTS (E2E test rechaza requests a "
        "ese host). Remover del CSP value."
    )


# ---------------------------------------------------------------------------
# 2. Los 6 headers obligatorios (P1-VERCEL-SECURITY-HEADERS) siguen presentes
# ---------------------------------------------------------------------------
_REQUIRED_HEADERS = [
    "Strict-Transport-Security",
    "X-Content-Type-Options",
    "X-Frame-Options",
    "Referrer-Policy",
    "Permissions-Policy",
]


@pytest.mark.parametrize("header_name", _REQUIRED_HEADERS)
def test_required_security_headers_still_present(vercel_config: dict, header_name: str):
    """P1-VERCEL-SECURITY-HEADERS exige 6 headers obligatorios. La limpieza
    de fonts NO debe haber removido ningún otro header."""
    keys = set()
    for entry in vercel_config.get("headers", []):
        for h in entry.get("headers", []):
            keys.add(h.get("key", ""))
    assert header_name in keys, (
        f"P2-AUDIT-4 regresión colateral: header `{header_name}` desapareció "
        f"de vercel.json. El cleanup de fonts NO debe haber afectado los 5 "
        f"headers obligatorios de P1-VERCEL-SECURITY-HEADERS."
    )


def test_csp_header_present(vercel_config: dict):
    """El CSP header (Report-Only o enforced) sigue presente — la limpieza
    NO debe haberlo removido entero, solo el whitelist de fonts."""
    csp = _csp_value(vercel_config)
    assert csp, (
        "P2-AUDIT-4 regresión: el header CSP completo desapareció. Esperado "
        "al menos `Content-Security-Policy-Report-Only` con las directivas "
        "default-src, script-src, connect-src, etc."
    )


# ---------------------------------------------------------------------------
# 3. CSP sigue conteniendo directivas críticas (la limpieza fue mínima)
# ---------------------------------------------------------------------------
_REQUIRED_DIRECTIVES = [
    "default-src",
    "script-src",
    "connect-src",
    "frame-src",
    "object-src",
    "frame-ancestors",
]


@pytest.mark.parametrize("directive", _REQUIRED_DIRECTIVES)
def test_csp_directive_present(vercel_config: dict, directive: str):
    csp = _csp_value(vercel_config)
    assert directive in csp, (
        f"P2-AUDIT-4 regresión colateral: directiva `{directive}` "
        f"desapareció del CSP. La limpieza solo debía remover hosts de "
        f"fuentes externas, no estructura del CSP."
    )
