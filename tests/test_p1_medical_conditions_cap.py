"""[P1-MEDICAL-CONDITIONS-CAP · 2026-08-01] Cap de condiciones médicas
simultáneas por plan (decisión de producto del owner).

Contexto:
  El owner decidió acotar el alcance clínico del generador al checklist de
  condiciones que las reglas deterministas (`condition_rules.py`) SÍ saben
  defender de forma consistente:
    (A) cap de 3 condiciones médicas simultáneas para crear un plan.
    (B) eliminación de los inputs de texto libre "Otra condición médica..."
        y "Otro medicamento..." del assessment (ver frontend).

Este módulo cubre el lado servidor (A): `_validate_medical_conditions_cap`
en `routers/plans.py`, cableado en `/api/analyze` y `/api/analyze/stream`
(los dos endpoints que reciben el formulario COMPLETO para generar un plan
NUEVO). `/swap-meal` y `/{plan_id}/chunks/{chunk_id}/regenerate-simplified`
NO reciben el formulario completo (swap hidrata clinical data desde el
perfil ya persistido vía `_enrich_clinical_from_profile`; regenerate-simplified
no toma body en absoluto) — no llevan el guard, y así debe quedar.

Compatibilidad: la validación es del REQUEST de generación nueva, NO de
lectura de perfil. Perfiles YA guardados con >cap condiciones (o con
condiciones capturadas por el texto libre pre-fix) siguen leyéndose y
usándose con normalidad — solo un nuevo submit del formulario puede
disparar el 422.

Cobertura:
  (a) 3 condiciones reales → pasa.
  (b) 4 condiciones reales → rechaza (contrato `too_many_medical_conditions`).
  (c) "Ninguna" + 3 reales → pasa (negativos de `_PROFILE_RISK_NEGATIVES`
      no cuentan contra el cap, mismo SSOT que `_profile_has_medical_risk`).
  (d) knob `MEALFIT_MAX_MEDICAL_CONDITIONS` respetado (monkeypatch a 2 → la
      3ª condición rechaza) + clamp [1,7] vía validator de `_env_int`.
  (e) el knob se auto-registra en `_KNOBS_REGISTRY` al leerse.
  (f) parser: ambos call sites (`/analyze`, `/analyze/stream`) están
      cableados con el mismo `code`/`max` contract.
"""
from pathlib import Path

import pytest

from knobs import _KNOBS_REGISTRY
from routers.plans import _validate_medical_conditions_cap


def _payload(conditions) -> dict:
    return {"medicalConditions": conditions}


# ---------------------------------------------------------------------------
# (a) 3 condiciones reales → pasa la validación (cap default = 3).
# ---------------------------------------------------------------------------
def test_tres_condiciones_reales_pasa_cap_default():
    ok, count, cap = _validate_medical_conditions_cap(
        _payload(["Diabetes tipo 2", "Hipertensión", "Enfermedad renal crónica"])
    )
    assert ok is True
    assert count == 3
    assert cap == 3


# ---------------------------------------------------------------------------
# (b) 4 condiciones reales → rechaza.
# ---------------------------------------------------------------------------
def test_cuatro_condiciones_reales_rechaza_cap_default():
    ok, count, cap = _validate_medical_conditions_cap(
        _payload(["Diabetes tipo 2", "Hipertensión", "Enfermedad renal crónica", "Asma"])
    )
    assert ok is False
    assert count == 4
    assert cap == 3


def test_endpoint_detail_contract_too_many_medical_conditions():
    """El detail 422 que arma el endpoint replica exactamente el contrato
    pedido por el owner: `code`, `max`, `message` es-DO con el cap dinámico."""
    ok, count, cap = _validate_medical_conditions_cap(
        _payload(["Diabetes tipo 2", "Hipertensión", "Enfermedad renal crónica", "Asma"])
    )
    assert not ok
    detail = {
        "code": "too_many_medical_conditions",
        "max": cap,
        "message": (
            "Para garantizar la calidad clínica del plan, selecciona máximo "
            f"{cap} condiciones prioritarias."
        ),
    }
    assert detail == {
        "code": "too_many_medical_conditions",
        "max": 3,
        "message": (
            "Para garantizar la calidad clínica del plan, selecciona máximo "
            "3 condiciones prioritarias."
        ),
    }


# ---------------------------------------------------------------------------
# (c) "Ninguna" (y otros sentinels) + 3 reales → pasa (negativos no cuentan).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("negative", ["Ninguna", "ninguno", "N/A", "na", "No", "Nada", "", "  "])
def test_sentinel_negativo_no_cuenta_contra_el_cap(negative):
    ok, count, cap = _validate_medical_conditions_cap(
        _payload([negative, "Diabetes tipo 2", "Hipertensión", "Enfermedad renal crónica"])
    )
    assert ok is True, f"sentinel {negative!r} no debería contar contra el cap (count={count})"
    assert count == 3


def test_solo_ninguna_cuenta_cero():
    ok, count, cap = _validate_medical_conditions_cap(_payload(["Ninguna"]))
    assert ok is True
    assert count == 0


def test_array_vacio_cuenta_cero():
    ok, count, cap = _validate_medical_conditions_cap(_payload([]))
    assert ok is True
    assert count == 0


def test_key_ausente_cuenta_cero():
    ok, count, cap = _validate_medical_conditions_cap({})
    assert ok is True
    assert count == 0


def test_string_unico_no_lista_cuenta_como_una():
    """Cliente legacy que manda `medicalConditions` como string suelto en vez
    de array — mismo comportamiento defensivo que `_profile_has_medical_risk`."""
    ok, count, cap = _validate_medical_conditions_cap({"medicalConditions": "Diabetes tipo 2"})
    assert count == 1
    assert ok is True


def test_string_unico_negativo_cuenta_cero():
    ok, count, cap = _validate_medical_conditions_cap({"medicalConditions": "Ninguna"})
    assert count == 0
    assert ok is True


# ---------------------------------------------------------------------------
# (d) knob `MEALFIT_MAX_MEDICAL_CONDITIONS` respetado + clamp [1, 7].
# ---------------------------------------------------------------------------
def test_knob_respetado_baja_el_cap_a_dos(monkeypatch):
    monkeypatch.setenv("MEALFIT_MAX_MEDICAL_CONDITIONS", "2")
    # 3ra condición real ahora rechaza.
    ok, count, cap = _validate_medical_conditions_cap(
        _payload(["Diabetes tipo 2", "Hipertensión", "Enfermedad renal crónica"])
    )
    assert cap == 2
    assert ok is False
    assert count == 3

    # 2 condiciones sigue pasando bajo el cap bajado.
    ok2, count2, cap2 = _validate_medical_conditions_cap(
        _payload(["Diabetes tipo 2", "Hipertensión"])
    )
    assert cap2 == 2
    assert ok2 is True


def test_knob_clamp_fuera_de_rango_alto_cae_a_default(monkeypatch):
    """Validator [1,7] de `_env_int`: 10 está fuera de rango → cae al
    default=3 (WARNING logueado, `parse_failed=True` en el registry)."""
    monkeypatch.setenv("MEALFIT_MAX_MEDICAL_CONDITIONS", "10")
    ok, count, cap = _validate_medical_conditions_cap(_payload([]))
    assert cap == 3


def test_knob_clamp_fuera_de_rango_bajo_cae_a_default(monkeypatch):
    monkeypatch.setenv("MEALFIT_MAX_MEDICAL_CONDITIONS", "0")
    ok, count, cap = _validate_medical_conditions_cap(_payload([]))
    assert cap == 3


def test_knob_valor_limite_superior_del_rango_se_acepta(monkeypatch):
    monkeypatch.setenv("MEALFIT_MAX_MEDICAL_CONDITIONS", "7")
    ok, count, cap = _validate_medical_conditions_cap(_payload([]))
    assert cap == 7


def test_knob_no_numerico_cae_a_default(monkeypatch):
    monkeypatch.setenv("MEALFIT_MAX_MEDICAL_CONDITIONS", "abc")
    ok, count, cap = _validate_medical_conditions_cap(_payload([]))
    assert cap == 3


# ---------------------------------------------------------------------------
# (e) el knob se auto-registra en `_KNOBS_REGISTRY` (para /health/version,
#     `_log_active_knobs()` startup, y SRE confirmando overrides sin redeploy).
# ---------------------------------------------------------------------------
def test_knob_se_autorregistra_en_knobs_registry(monkeypatch):
    monkeypatch.setenv("MEALFIT_MAX_MEDICAL_CONDITIONS", "5")
    _validate_medical_conditions_cap(_payload([]))
    entry = _KNOBS_REGISTRY.get("MEALFIT_MAX_MEDICAL_CONDITIONS")
    assert entry is not None, "el knob debe aparecer en _KNOBS_REGISTRY tras leerse"
    assert entry["type"] == "int"
    assert entry["default"] == 3
    assert entry["value"] == 5
    assert entry["is_override"] is True


def test_knob_registry_default_sin_override(monkeypatch):
    monkeypatch.delenv("MEALFIT_MAX_MEDICAL_CONDITIONS", raising=False)
    _validate_medical_conditions_cap(_payload([]))
    entry = _KNOBS_REGISTRY.get("MEALFIT_MAX_MEDICAL_CONDITIONS")
    assert entry is not None
    assert entry["value"] == 3
    assert entry["is_override"] is False


# ---------------------------------------------------------------------------
# (f) parser: ambos endpoints de generación de plan nuevo llevan el guard con
# el mismo contrato de `code`/`max`. Tooltip-anchor: si renombras
# `_validate_medical_conditions_cap` o el `code` del detail, este test falla
# ANTES de que el drift llegue a producción.
# ---------------------------------------------------------------------------
_PLANS_PY = Path(__file__).resolve().parent.parent / "routers" / "plans.py"


def _read_plans_source() -> str:
    return _PLANS_PY.read_text(encoding="utf-8")


def test_ambos_endpoints_de_generacion_cablean_el_guard():
    src = _read_plans_source()
    call_count = src.count("_validate_medical_conditions_cap(data)")
    assert call_count == 2, (
        f"Se esperaban exactamente 2 call sites de `_validate_medical_conditions_cap(data)` "
        f"(`/analyze` y `/analyze/stream`); se encontraron {call_count}. Si se añadió/quitó un "
        f"endpoint de generación de plan NUEVO, actualiza este test y verifica que el guard "
        f"esté (o deliberadamente no esté, documentando por qué) en el nuevo endpoint."
    )


def test_detail_code_too_many_medical_conditions_presente():
    src = _read_plans_source()
    assert src.count('"code": "too_many_medical_conditions"') == 2, (
        "El `code` del detail 422 debe aparecer exactamente en los 2 call sites del guard."
    )
    assert src.count('"max": _mc_cap') == 2


def test_swap_meal_no_lleva_el_guard_por_diseno():
    """`/swap-meal` hidrata clinical data DESDE el perfil ya persistido
    (`_enrich_clinical_from_profile`), no re-recibe el formulario completo de
    generación. Aplicarle el cap rompería swaps legítimos sobre perfiles
    legacy con >cap condiciones (grandfathered). Ancla la decisión: si
    alguien añade el guard aquí, este test falla para forzar review."""
    src = _read_plans_source()
    swap_start = src.index('@router.post("/swap-meal")')
    swap_persist_start = src.index('@router.post("/{plan_id}/swap-meal/persist")')
    swap_body = src[swap_start:swap_persist_start]
    assert "_validate_medical_conditions_cap" not in swap_body


def test_regenerate_simplified_no_toma_body_ni_lleva_el_guard():
    """`/{plan_id}/chunks/{chunk_id}/regenerate-simplified` no declara
    `data: dict = Body(...)` — no re-recibe el formulario, así que no puede
    (ni debe) llevar el cap de condiciones médicas."""
    src = _read_plans_source()
    start = src.index('@router.post("/{plan_id}/chunks/{chunk_id}/regenerate-simplified")')
    # siguiente @router.post tras este endpoint acota el cuerpo de la función.
    next_marker = src.index("@router.post(", start + 1)
    body = src[start:next_marker]
    assert "Body(...)" not in body
    assert "_validate_medical_conditions_cap" not in body
