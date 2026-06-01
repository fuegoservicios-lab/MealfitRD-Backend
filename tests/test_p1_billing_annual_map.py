"""[P1-BILLING-ANNUAL-MAP · 2026-05-31] `_build_paypal_plan_tier_map` debe
mapear TANTO la variante mensual (`PAYPAL_PLAN_{TIER}_ID`) como la anual
(`PAYPAL_PLAN_{TIER}_ANNUAL_ID`) al mismo tier interno.

Vector pre-fix:
    El frontend (PaymentModal.jsx) usa 6 plan IDs distintos — 3 mensuales y 3
    anuales (`VITE_PAYPAL_PLAN_{TIER}[_ANNUAL]`). El backend solo leía los 3
    mensuales. Cuando un usuario pagaba el plan ANUAL, PayPal devolvía el plan_id
    anual; `_build_paypal_plan_tier_map().get(plan_id_anual)` retornaba None y el
    handler `/verify` lanzaba HTTPException(400, "Plan no reconocido") DESPUÉS de
    que PayPal ya había cobrado → cobro-sin-upgrade (limbo, chargeback/soporte).
    Contraparte directa de los P0-BILLING ya cerrados.

Lo que enforza este test:
    A) El mapping contempla el sufijo `_ANNUAL` (parser-based, siempre corre).
    B) El anchor `P1-BILLING-ANNUAL-MAP` permanece en source.
    C-E) Funcional (skip si deps ausentes): ambas variantes resuelven al tier
         correcto; una env var anual sola basta; vars ausentes se omiten.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_BILLING_PY = _BACKEND_ROOT / "routers" / "billing.py"

_PAYPAL_VARS = [
    "PAYPAL_PLAN_BASIC_ID",
    "PAYPAL_PLAN_PLUS_ID",
    "PAYPAL_PLAN_ULTRA_ID",
    "PAYPAL_PLAN_BASIC_ANNUAL_ID",
    "PAYPAL_PLAN_PLUS_ANNUAL_ID",
    "PAYPAL_PLAN_ULTRA_ANNUAL_ID",
]


@pytest.fixture(scope="module")
def billing_src() -> str:
    return _BILLING_PY.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Parser-based (sin deps — corre siempre)
# ---------------------------------------------------------------------------

def test_a_mapping_contemplates_annual_suffix(billing_src: str):
    """El helper debe iterar el sufijo `_ANNUAL` además del mensual."""
    assert "def _build_paypal_plan_tier_map" in billing_src, (
        "P1-BILLING-ANNUAL-MAP: helper `_build_paypal_plan_tier_map` ausente."
    )
    assert "_ANNUAL" in billing_src, (
        "P1-BILLING-ANNUAL-MAP: el mapping no contempla `_ANNUAL`. Un pago del "
        "plan anual quedaría sin mapear → 400 cobro-sin-upgrade."
    )
    # Loop canónico sobre ambos sufijos.
    assert re.search(
        r'for\s+suffix\s+in\s+\(\s*""\s*,\s*"_ANNUAL"\s*\)', billing_src
    ), (
        "P1-BILLING-ANNUAL-MAP: se espera el loop "
        "`for suffix in (\"\", \"_ANNUAL\")` que cubre mensual + anual."
    )


def test_b_anchor_present(billing_src: str):
    assert "P1-BILLING-ANNUAL-MAP" in billing_src, (
        "P1-BILLING-ANNUAL-MAP: anchor ausente en billing.py."
    )


# ---------------------------------------------------------------------------
# Funcional (skip si routers.billing no es importable en este entorno)
# ---------------------------------------------------------------------------

def _load_func():
    if str(_BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(_BACKEND_ROOT))
    try:
        from routers.billing import _build_paypal_plan_tier_map
    except Exception as exc:  # pragma: no cover - entorno sin deps
        pytest.skip(f"routers.billing no importable en este entorno: {exc}")
    return _build_paypal_plan_tier_map


def test_c_both_variants_map_to_tier(monkeypatch):
    func = _load_func()
    for v in _PAYPAL_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("PAYPAL_PLAN_BASIC_ID", "P-MONTH-BASIC")
    monkeypatch.setenv("PAYPAL_PLAN_PLUS_ID", "P-MONTH-PLUS")
    monkeypatch.setenv("PAYPAL_PLAN_ULTRA_ID", "P-MONTH-ULTRA")
    monkeypatch.setenv("PAYPAL_PLAN_BASIC_ANNUAL_ID", "P-YEAR-BASIC")
    monkeypatch.setenv("PAYPAL_PLAN_PLUS_ANNUAL_ID", "P-YEAR-PLUS")
    monkeypatch.setenv("PAYPAL_PLAN_ULTRA_ANNUAL_ID", "P-YEAR-ULTRA")

    assert func() == {
        "P-MONTH-BASIC": "basic",
        "P-MONTH-PLUS": "plus",
        "P-MONTH-ULTRA": "ultra",
        "P-YEAR-BASIC": "basic",
        "P-YEAR-PLUS": "plus",
        "P-YEAR-ULTRA": "ultra",
    }


def test_d_annual_only_resolves(monkeypatch):
    """Si solo está seteado el ID anual de un tier, igual debe mapear."""
    func = _load_func()
    for v in _PAYPAL_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("PAYPAL_PLAN_PLUS_ANNUAL_ID", "P-YEAR-PLUS-ONLY")
    assert func().get("P-YEAR-PLUS-ONLY") == "plus"


def test_e_missing_vars_omitted(monkeypatch):
    """Sin env vars, el mapping queda vacío (degradación segura: rechaza)."""
    func = _load_func()
    for v in _PAYPAL_VARS:
        monkeypatch.delenv(v, raising=False)
    assert func() == {}
