"""[P3-NEW-PAYPAL-FALLBACK · 2026-05-15] Anchor + regression guard.

`frontend/src/components/dashboard/PaymentModal.jsx:60` tenía:

    "client-id": import.meta.env.VITE_PAYPAL_CLIENT_ID || "ARVcVpVZ-...",

El fallback era un client_id de PayPal real (no placeholder). Si Vercel
perdía la env var (typo en deploy parcial, rotación incompleta), el SDK
seguía cargando el ID hardcoded del commit — posiblemente apuntando al
merchant equivocado o un client deshabilitado. Anti-pattern porque
oculta misconfig en lugar de fail-loud.

Post-fix: si `VITE_PAYPAL_CLIENT_ID` está unset, el componente lanza
Error visible que dispara el ErrorBoundary global (mejor "modal roto +
alert SRE" que "pago procesado contra merchant incorrecto"). PayPal
client_id es público en bundle (no es secreto), pero el riesgo es
operacional.

Defensas que el test enforza:
  1. Anchor `P3-NEW-PAYPAL-FALLBACK` presente en PaymentModal.jsx.
  2. Cero fallback `|| "ARV..."` (client_id real hardcoded) en el callsite.
  3. El gate `if (!_paypalClientId) throw new Error(...)` está presente
     ANTES de `initialOptions`.
  4. Anchor presente en este archivo (cross-link guard P2-HIST-AUDIT-14).
"""

from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PAYMENT = _REPO_ROOT / "frontend" / "src" / "components" / "dashboard" / "PaymentModal.jsx"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_anchor_present():
    src = _read(_PAYMENT)
    assert "P3-NEW-PAYPAL-FALLBACK" in src, (
        "Falta anchor `P3-NEW-PAYPAL-FALLBACK` en PaymentModal.jsx."
    )


def test_no_hardcoded_client_id_fallback():
    """Cero `VITE_PAYPAL_CLIENT_ID || "ARV..."` en el callsite. Acepta
    fallbacks placeholder de plan IDs (P-ANNUAL_BASIC_PLACEHOLDER, etc.)
    porque son intencionales para tiers no lanzados."""
    src = _read(_PAYMENT)
    # Pattern: VITE_PAYPAL_CLIENT_ID || "AR..." (client IDs reales empiezan
    # con A en sandbox, A o L en live; placeholder usa otro prefijo).
    bad = re.search(
        r'VITE_PAYPAL_CLIENT_ID\s*\|\|\s*["\']A[A-Za-z0-9_-]{20,}["\']',
        src,
    )
    assert bad is None, (
        f"PaymentModal.jsx tiene fallback hardcoded de `VITE_PAYPAL_CLIENT_ID`: "
        f"{bad.group(0)!r}. Reemplazar por gate fail-loud "
        f"`if (!_paypalClientId) throw new Error(...)`."
    )


def test_fail_loud_gate_present():
    """`if (!_paypalClientId) throw new Error(...)` debe aparecer en el archivo
    antes de cualquier uso del client-id."""
    src = _read(_PAYMENT)
    pat = re.compile(
        r"if\s*\(\s*!_paypalClientId\s*\)\s*\{?\s*throw\s+new\s+Error",
        re.DOTALL,
    )
    assert pat.search(src), (
        "PaymentModal.jsx debe tener `if (!_paypalClientId) throw new Error(...)` "
        "antes del `initialOptions` con `client-id`. Sin esto, el componente no "
        "fail-loud cuando la env var falta."
    )


def test_initial_options_uses_local_var():
    """`initialOptions["client-id"]` debe usar la variable `_paypalClientId`,
    NO `import.meta.env.VITE_PAYPAL_CLIENT_ID` directo (eso reabre el
    bypass del gate fail-loud)."""
    src = _read(_PAYMENT)
    # Buscar bloque initialOptions y verificar que client-id usa variable
    m = re.search(r"initialOptions\s*=\s*\{[^}]*?[\"']client-id[\"']\s*:\s*([^,\n]+)", src, re.DOTALL)
    assert m is not None, "No se encontró `initialOptions[\"client-id\"]` en PaymentModal.jsx."
    rhs = m.group(1).strip()
    assert "_paypalClientId" in rhs, (
        f"`initialOptions[\"client-id\"]` debe usar la variable local "
        f"`_paypalClientId`, no `import.meta.env.VITE_PAYPAL_CLIENT_ID` directo. "
        f"Encontrado: {rhs!r}"
    )


def test_anchor_present_in_test_file():
    src = _read(Path(__file__))
    assert "P3-NEW-PAYPAL-FALLBACK" in src
