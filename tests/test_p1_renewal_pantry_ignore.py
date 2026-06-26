"""[P1-RENEWAL-PANTRY-IGNORE · 2026-06-26] Variety-first: el endpoint de plan completo ignora la nevera.

INCIDENTE RAÍZ (user d4bc3af5, corr=9040fc1d, 2026-06-26 07:00): tras restockear su nevera (39 ítems),
una renovación por variedad fue capturada por el PANTRY GUARD estricto (`unauthorized=15`), agotó sus 2
retries y entregó un plan matemático de emergencia (`_is_fallback=True`, band 0.0, "Pollo y Arroz"
idéntico los 3 días). El skip previo (P1-PANTRY-GUARD-REGEN-SKIP) dependía de que el request llevara
`update_reason` en el payload — señal LEAKY: el entry-point usado no lo enviaba → el guard se aplicó.

FIX: `_run_pantry_validation_for_initial_chunk` salta el guard por default (knob
MEALFIT_INITIAL_CHUNK_PANTRY_GUARD=False) — la generación de plan COMPLETO es variety-first y su lista
de compras DEFINE qué comprar, no se amarra a la nevera previa. Los flujos pantry-aware reales
(Cambiar Plato /swap-meal, día completo /regenerate-day) son endpoints SEPARADOS. Anchor:
P1-RENEWAL-PANTRY-IGNORE.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import constants

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _import_plans_or_skip():
    """routers.plans arrastra llm_provider, que en algunos venvs de test rompe al instanciar el
    cliente DeepSeek (TypeError: object.__init__). Saltar limpio en ese entorno — la cobertura
    estructural (parser-based abajo) ancla el contrato igual."""
    try:
        from routers import plans as plans_mod
        return plans_mod
    except Exception as e:  # pragma: no cover - depende del venv
        pytest.skip(f"routers.plans no importable en este venv: {type(e).__name__}: {e}")


def test_knob_default_off_significa_skip():
    """Default OFF = el guard estricto NO corre en generación inicial (ignora nevera)."""
    assert constants.INITIAL_CHUNK_PANTRY_GUARD_ENABLED is False


def test_knob_registrado():
    from knobs import get_knobs_registry_snapshot
    # Forzar registro evaluando el helper (idempotente).
    from knobs import _env_bool
    _env_bool("MEALFIT_INITIAL_CHUNK_PANTRY_GUARD", False)
    snap = get_knobs_registry_snapshot()
    assert "MEALFIT_INITIAL_CHUNK_PANTRY_GUARD" in snap


def test_skip_devuelve_result_intacto_con_nevera_llena_y_sin_update_reason(monkeypatch):
    """El caso EXACTO del incidente: nevera con 39 ítems + sin update_reason → el guard NO corre,
    el plan se devuelve intacto (mismo objeto), sin retries pantry ni emergency fallback."""
    from fastapi import BackgroundTasks
    plans_mod = _import_plans_or_skip()

    # Asegurar el default (skip).
    monkeypatch.setattr(constants, "INITIAL_CHUNK_PANTRY_GUARD_ENABLED", False)

    sentinel = {"days": [{"meals": [{"ingredients": ["pollo guisado", "arroz blanco", "aguacate"]}]}],
                "calories": 2100}
    nevera = [f"item_{i}" for i in range(39)]  # 39 > PANTRY_GUARD_MIN_ITEMS

    out = plans_mod._run_pantry_validation_for_initial_chunk(
        result=sentinel,
        pipeline_data={},
        history=[],
        taste_profile="",
        memory_ctx="",
        background_tasks=BackgroundTasks(),
        actual_user_id="d4bc3af5",
        pantry_ingredients=nevera,
        transport_label="P0-2 SSE",
        update_reason=None,  # <-- el campo ausente que causaba el bug
    )
    # Devuelto intacto, sin flags de degradación pantry.
    assert out is sentinel
    assert "_initial_chunk_pantry_degraded" not in out
    assert "_pantry_supplement_required" not in out


def test_skip_no_corre_validador_pesado(monkeypatch):
    """El skip ocurre ANTES de importar/llamar al validador con retries (que requiere DB/LLM).
    Si el validador se invocara, este test fallaría con AssertionError explícito."""
    from fastapi import BackgroundTasks
    plans_mod = _import_plans_or_skip()
    monkeypatch.setattr(constants, "INITIAL_CHUNK_PANTRY_GUARD_ENABLED", False)

    import cron_tasks
    def _boom(*a, **k):
        raise AssertionError("el validador pantry con retries NO debe llamarse en variety-first")
    monkeypatch.setattr(cron_tasks, "_validate_and_retry_initial_chunk_against_pantry", _boom)

    res = {"days": [{"meals": [{"ingredients": ["pescado"]}]}]}
    out = plans_mod._run_pantry_validation_for_initial_chunk(
        result=res, pipeline_data={}, history=[], taste_profile="", memory_ctx="",
        background_tasks=BackgroundTasks(), actual_user_id="u",
        pantry_ingredients=[f"x{i}" for i in range(20)], update_reason=None,
    )
    assert out is res


def test_wiring_marker_presente():
    src = open(os.path.join(_ROOT, "routers", "plans.py"), encoding="utf-8").read()
    assert "P1-RENEWAL-PANTRY-IGNORE" in src
    assert "INITIAL_CHUNK_PANTRY_GUARD_ENABLED" in src


def test_skip_estructura_antes_del_validador():
    """Parser-based (robusto a venv roto): dentro de `_run_pantry_validation_for_initial_chunk`, el
    early-return por variety-first (`if not _IPG_ENABLED: ... return result`) DEBE aparecer ANTES del
    skip por update_reason Y ANTES de la llamada al validador con retries. Así el caso del incidente
    (nevera llena + sin update_reason) se corta antes de tocar el guard."""
    src = open(os.path.join(_ROOT, "routers", "plans.py"), encoding="utf-8").read()
    start = src.index("def _run_pantry_validation_for_initial_chunk")
    end = src.index("\ndef ", start + 1)
    body = src[start:end]
    i_knob = body.index("if not _IPG_ENABLED:")
    i_return = body.index("return result", i_knob)
    i_update_reason = body.index("if update_reason:")
    i_validator = body.index("_validate_and_retry_initial_chunk_against_pantry")
    assert i_knob < i_return < i_update_reason, "el skip variety-first debe ir antes del skip por update_reason"
    assert i_return < i_validator, "el skip variety-first debe cortar ANTES de invocar el validador con retries"
