"""[P1-NONCHUNKED-PERSIST-SYNC · 2026-06-15] Persistencia no-chunked síncrona (gap-audit G2).

Bug original (gap-audit 2026-06-15, G2):
    El branch CHUNKED persiste síncrono y propaga `_persist_failed` (→ 503 / error-event / KV-failed en los
    3 consumidores). El branch NO-CHUNKED hacía el INSERT como `background_tasks.add_task(_save_plan_and_
    track_background, ...)` fire-and-forget: si fallaba (pool exhaustion, statement_timeout, CHECK I8,
    serialization), el usuario YA había recibido 200/complete con un plan que NO existe en meal_plans —
    historial/dashboard vacíos al recargar, falsa sensación de éxito, solo un alert SRE.

Cierre:
    `_save_plan_and_track_background(..., return_id=True)` corre el INSERT SÍNCRONO y PROPAGA la excepción
    (en vez de tragarla) + retorna el UUID. El branch no-chunked del router lo llama inline, setea
    `result["id"]` en éxito y `result["_persist_failed"]=True` en fallo → los MISMOS 3 consumidores lo
    propagan. El tracking de frecuencias queda best-effort (su fallo no afecta la persistencia ya exitosa).

Tests: funcional (mocks, sin DB/LLM) del helper + contrato parser-based del wiring del router.
"""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import services

_PLANS_SRC = (Path(__file__).resolve().parent.parent / "routers" / "plans.py").read_text(encoding="utf-8")

# Plan mínimo SIN ingredientes → raw_ingredients=[] → el bloque de freq-tracking y resolution_coverage
# se saltan (cero DB), aislando el path de persistencia que G2 endurece.
_PLAN = {"days": [], "calories": 1500, "macros": {"protein": "100g", "carbs": "150g", "fats": "50g"}}


def _patch_common(monkeypatch, save_mock):
    import nutrition_db
    monkeypatch.setattr(services, "generate_plan_title", lambda pd: "Plan de prueba", raising=True)
    monkeypatch.setattr(services, "check_recent_meal_plan_exists", lambda *a, **k: False, raising=True)
    monkeypatch.setattr(services, "_persist_plan_persist_failed_alert", lambda *a, **k: None, raising=True)
    monkeypatch.setattr(services, "save_new_meal_plan_atomic", save_mock, raising=True)
    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", lambda *a, **k: object(), raising=False)


def test_sync_propagates_insert_failure(monkeypatch):
    """return_id=True: un INSERT fallido DEBE propagar la excepción al caller (para que marque _persist_failed)."""
    save = MagicMock(side_effect=RuntimeError("pool exhausted"))
    _patch_common(monkeypatch, save)
    with pytest.raises(RuntimeError):
        services._save_plan_and_track_background("u1", dict(_PLAN), None, return_id=True)


def test_sync_returns_plan_id_on_success(monkeypatch):
    """return_id=True: retorna el UUID y propaga return_id=True al INSERT atómico."""
    save = MagicMock(return_value="plan-uuid-123")
    _patch_common(monkeypatch, save)
    pid = services._save_plan_and_track_background("u1", dict(_PLAN), None, return_id=True)
    assert pid == "plan-uuid-123"
    assert save.call_args.kwargs.get("return_id") is True, "el INSERT debe recibir return_id=True"


def test_background_mode_swallows_failure(monkeypatch):
    """return_id=False (default, modo BackgroundTask): NO propaga el fallo (el usuario ya tiene respuesta)."""
    save = MagicMock(side_effect=RuntimeError("boom"))
    _patch_common(monkeypatch, save)
    assert services._save_plan_and_track_background("u1", dict(_PLAN), None) is None  # no raise


def test_dedup_skip_returns_none_without_insert(monkeypatch):
    save = MagicMock(return_value="x")
    _patch_common(monkeypatch, save)
    monkeypatch.setattr(services, "check_recent_meal_plan_exists", lambda *a, **k: True, raising=True)
    assert services._save_plan_and_track_background("u1", dict(_PLAN), None, return_id=True) is None
    save.assert_not_called()


# ── Contrato del wiring del router (parser-based) ──
def _postprocess_body() -> str:
    start = _PLANS_SRC.find("def _postprocess_pipeline_result")
    end = _PLANS_SRC.find("\ndef _attach_pantry_degraded_response_meta")
    assert start != -1 and end != -1
    return _PLANS_SRC[start:end]


def test_router_nonchunked_persists_sync_not_fire_and_forget():
    body = _postprocess_body()
    assert "P1-NONCHUNKED-PERSIST-SYNC" in body, "Falta el tooltip-anchor del branch no-chunked."
    assert "return_id=True" in body, "El branch no-chunked debe persistir con return_id=True (síncrono)."
    assert 'result["id"] = _pid' in body, "Debe asignar result['id'] tras persistir."
    assert 'result["_persist_failed"] = True' in body, "Debe marcar _persist_failed si el INSERT falla."
    # El INSERT no-chunked NO debe seguir siendo fire-and-forget vía BackgroundTask.
    assert not re.search(r"add_task\(\s*\n?\s*_save_plan_and_track_background", body), (
        "El INSERT no-chunked NO debe correr como background_tasks.add_task(_save_plan_and_track_background) "
        "(G2): un fallo quedaría silencioso (200/complete con plan perdido)."
    )
