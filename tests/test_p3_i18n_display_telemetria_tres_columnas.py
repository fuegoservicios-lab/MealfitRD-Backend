"""[P3-I18N-DISPLAY-TELEMETRIA-CON-TRES-COLUMNAS-MUERTAS · 2026-08-23] La fila de
``pipeline_metrics`` que deja cada ciclo de enriquecimiento salía con ``duration_ms``,
``retries`` y ``tokens_estimated`` SIEMPRE en 0: `duration_ms` nadie lo ponía en el
resumen, `retries` leía una clave (`batches_failed`) que nadie escribía, y
`tokens_estimated` era un ``0`` literal. La fila existía (P2-DISPLAY-SIN-TELEMETRIA-
RESULTADO) y sólo el jsonb decía algo: un panel que agrupe por las columnas numéricas ve
ceros.

Se mide ejecutando el ciclo real con los dobles del test hermano (`_FakeLLM` declara
`usage_metadata`) y capturando el INSERT.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plan_display_i18n as pdi  # noqa: E402
from test_p1_plan_display_i18n import (  # noqa: E402
    _FakeLLM, _FakeResponse, _base_meal, _set_plan, _valid_response_for_base_meal, engine,  # noqa: F401
)

_MARKER = "P3-I18N-DISPLAY-TELEMETRIA-CON-TRES-COLUMNAS-MUERTAS"


@pytest.fixture(autouse=True)
def _reset():
    _FakeLLM.NEXT_RESPONSE = None
    _FakeLLM.NEXT_EXCEPTION = None
    _FakeLLM.captured_prompts = []
    _FakeLLM.invoke_count = 0
    yield


@pytest.fixture
def filas(monkeypatch):
    """Captura el INSERT de `_emit_result_telemetry` y silencia la alerta."""
    out = []

    def _fake_write(sql, params=None, *a, **k):
        if "pipeline_metrics" in str(sql):
            out.append(params)
        return None

    monkeypatch.setattr(pdi, "execute_sql_write", _fake_write)
    monkeypatch.setattr(pdi, "_emit_degraded_alert", lambda *a, **k: None)
    return out


def _columnas(params):
    # (user_id, plan_id, node, duration_ms, retries, tokens_estimated, confidence, metadata)
    return {"duration_ms": params[3], "retries": params[4], "tokens_estimated": params[5]}


def test_un_ciclo_limpio_trae_duracion_y_tokens_y_cero_reintentos(engine, filas):
    _set_plan(engine, [_base_meal()])
    _FakeLLM.NEXT_RESPONSE = _valid_response_for_base_meal()  # 120 + 80 tokens
    pdi.enrich_plan_display("plan-1", "user-1", "en-US")
    assert filas, "el ciclo no dejó fila en pipeline_metrics"
    c = _columnas(filas[-1])
    assert c["tokens_estimated"] == 200, f"los tokens del provider no llegan a la columna: {c} [{_MARKER}]"
    assert c["retries"] == 0, c
    assert isinstance(c["duration_ms"], int) and c["duration_ms"] >= 0
    # El reloj es el del ciclo: un entero pequeño, no 0 por construcción. Se comprueba que
    # el jsonb lo trae también (la fuente es la misma).
    import json
    meta = json.loads(filas[-1][7])
    assert "duration_ms" in meta and meta["tokens_estimated"] == 200


def test_una_excepcion_del_provider_cuenta_como_reintento(engine, filas, monkeypatch):
    """El ciclo reencola el lote cuando el provider falla; esa invocación EXTRA es un
    reintento y tiene que verse en la columna."""
    _set_plan(engine, [_base_meal()])
    respuestas = [RuntimeError("provider 503"), _valid_response_for_base_meal()]

    class _LLMQueFallaUnaVez(_FakeLLM):
        def invoke(self, messages):
            type(self).invoke_count += 1
            r = respuestas.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

    monkeypatch.setattr(pdi, "build_chat_llm", lambda model, **kwargs: _LLMQueFallaUnaVez(**kwargs))
    pdi.enrich_plan_display("plan-2", "user-1", "en-US")
    assert filas
    c = _columnas(filas[-1])
    assert c["retries"] >= 1, f"la invocación extra no cuenta como reintento: {c} [{_MARKER}]"
    assert c["tokens_estimated"] == 200, "los tokens de la invocación buena se pierden"


def test_tokens_de_es_robusto():
    assert pdi._tokens_de(_FakeResponse("x", {"input_tokens": 3, "output_tokens": 4})) == 7
    assert pdi._tokens_de(_FakeResponse("x", None)) == 0
    assert pdi._tokens_de(_FakeResponse("x", {"input_tokens": None})) == 0
    assert pdi._tokens_de(object()) == 0
