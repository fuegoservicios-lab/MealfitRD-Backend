"""[P0-BILLING-1 · 2026-05-12] `/api/subscription/verify` debe derivar `tier`
server-side desde el `plan_id` que PayPal devuelve, NO desde `data.get("tier")`.

Vector pre-fix:
    El handler hacía `UPDATE user_profiles SET plan_tier = body.tier` tras
    verificar `sub_data.get("status") == "ACTIVE"`. Pero NO verificaba que el
    `plan_id` real de la subscription matchease el `tier` solicitado.

    Atacante con sub real de "basic" ($X/mes) puede:
      1. POST /api/subscription/verify {user_id, subscriptionID, tier="ultra"}
      2. PayPal confirma status=ACTIVE (sub real es válida).
      3. Backend UPDATE plan_tier="ultra".
      4. Acceso ilimitado por el precio de "basic".

Fix:
    Mapping `PayPal plan_id` → tier interno construido desde env vars
    `PAYPAL_PLAN_BASIC_ID` / `PAYPAL_PLAN_PLUS_ID` / `PAYPAL_PLAN_ULTRA_ID`.
    El handler:
      1. Verifica sub status=ACTIVE en PayPal.
      2. Lee `verified_plan_id = sub_data.get("plan_id")`.
      3. `server_tier = _build_paypal_plan_tier_map()[verified_plan_id]`.
      4. UPDATE usa `server_tier`, NO `data.get("tier")`.
    El `data.get("tier")` queda como `client_hint_tier` solo para logear
    divergencia (audit/observabilidad).

Lo que este test enforza:
    A) `_build_paypal_plan_tier_map` existe en `routers/billing.py`.
    B) El UPDATE final usa una variable `tier_to_assign` (server-derived),
       NO `tier` (client-trusted).
    C) `sub_data.get("plan_id")` aparece en el flujo entre `status==ACTIVE`
       check y el UPDATE (sin ese SELECT no hay forma de derivar tier).
    D) Anchor `P0-BILLING-1-TIER-SERVER-SIDE` o `P0-BILLING-1` permanece
       en source — protege contra refactor que borre la convención.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_BILLING_PY = _BACKEND_ROOT / "routers" / "billing.py"


@pytest.fixture(scope="module")
def billing_src() -> str:
    return _BILLING_PY.read_text(encoding="utf-8")


def test_a_build_paypal_plan_tier_map_helper_exists(billing_src: str):
    """`_build_paypal_plan_tier_map` declarado y lee las 3 env vars."""
    assert "def _build_paypal_plan_tier_map" in billing_src, (
        "P0-BILLING-1: helper `_build_paypal_plan_tier_map` ausente. "
        "Restaurar — es el mapping plan_id→tier server-side."
    )
    for tier in ("BASIC", "PLUS", "ULTRA"):
        var = f"PAYPAL_PLAN_{tier}_ID"
        assert var in billing_src, (
            f"P0-BILLING-1: env var {var!r} no referenciada en billing.py. "
            f"Las 3 vars son requeridas para mapear plan_id→tier."
        )


def test_b_update_uses_server_derived_tier_not_client_body(billing_src: str):
    """El UPDATE `plan_tier=` en /verify DEBE usar `tier_to_assign`
    (server-derived), NUNCA `data.get("tier")` o variable `tier` cruda
    asignada desde el body.

    Patrón canónico esperado (líneas reales del archivo):
        res = supabase.table("user_profiles").update({
            "plan_tier": tier_to_assign,
            ...
        }).eq("id", user_id).execute()
    """
    # Aislar el handler /verify (entre el decorator @router.post("/verify") y
    # el siguiente @router.post o EOF). Esto evita falsos positivos del /cancel.
    verify_match = re.search(
        r'@router\.post\("/verify"\)(.*?)(?=@router\.post|@webhooks_router\.post|\Z)',
        billing_src,
        re.DOTALL,
    )
    assert verify_match is not None, "Handler /verify no encontrado en billing.py."
    verify_body = verify_match.group(1)

    # El UPDATE debe contener `"plan_tier": tier_to_assign`. Si aparece
    # `"plan_tier": tier,` (variable cruda del body) o `"plan_tier": data.get("tier")`
    # → regresión P0.
    assert re.search(r'"plan_tier"\s*:\s*tier_to_assign', verify_body), (
        "P0-BILLING-1 regresión: el UPDATE de plan_tier no usa "
        "`tier_to_assign` (server-derived). Verificar que el cliente NO "
        "puede inyectar tier."
    )
    assert not re.search(r'"plan_tier"\s*:\s*data\.get\(\s*[\'"]tier[\'"]', verify_body), (
        "P0-BILLING-1 regresión: el UPDATE pasa `data.get('tier')` "
        "directo — cliente puede inyectar tier arbitrario."
    )
    # Defensa adicional: el patrón legacy `tier = data.get("tier")` que
    # luego se usa en UPDATE está prohibido. Lo aceptamos solo como
    # `client_hint_tier = (data.get("tier") or "")` (variable distinta).
    legacy_pattern = re.search(
        r'^\s*tier\s*=\s*data\.get\(\s*[\'"]tier[\'"]\s*\)',
        verify_body,
        re.MULTILINE,
    )
    assert legacy_pattern is None, (
        "P0-BILLING-1 regresión: `tier = data.get('tier')` reaparece. "
        "El patrón canónico es "
        "`client_hint_tier = (data.get('tier') or '').strip().lower()` — "
        "la variable separada deja claro que NO se usa para asignar."
    )


def test_c_plan_id_extracted_between_status_check_and_update(billing_src: str):
    """`sub_data.get("plan_id")` debe aparecer ANTES del UPDATE y DESPUÉS
    del check `status != "ACTIVE"`. Sin ese extract no hay forma de
    derivar tier server-side.
    """
    verify_match = re.search(
        r'@router\.post\("/verify"\)(.*?)(?=@router\.post|@webhooks_router\.post|\Z)',
        billing_src,
        re.DOTALL,
    )
    assert verify_match is not None
    verify_body = verify_match.group(1)

    status_idx = verify_body.find('status != "ACTIVE"')
    plan_id_idx = verify_body.find('sub_data.get("plan_id")')
    update_idx = verify_body.find('"plan_tier": tier_to_assign')

    assert plan_id_idx > 0, (
        "P0-BILLING-1: `sub_data.get('plan_id')` no encontrado. "
        "El handler no extrae el plan_id real de PayPal."
    )
    assert status_idx > 0 and plan_id_idx > status_idx, (
        "P0-BILLING-1: `plan_id` debe extraerse DESPUÉS del status check. "
        "Caso contrario podría aceptar subs no-ACTIVE."
    )
    assert update_idx > plan_id_idx, (
        "P0-BILLING-1: el UPDATE debe ocurrir DESPUÉS de derivar el tier. "
        "Order: status check → plan_id → server_tier → UPDATE."
    )


def test_d_anchor_present_in_source(billing_src: str):
    """Anchor `P0-BILLING-1-TIER-SERVER-SIDE` (o `P0-BILLING-1`) presente
    en `billing.py`. Sin él, un refactor cosmético podría borrar la
    convención sin pasar por este test.
    """
    anchors = ("P0-BILLING-1-TIER-SERVER-SIDE", "P0-BILLING-1")
    assert any(a in billing_src for a in anchors), (
        f"P0-BILLING-1: ningún anchor {anchors} en billing.py. "
        f"Restaurar el marker en el bloque comentario del handler."
    )
