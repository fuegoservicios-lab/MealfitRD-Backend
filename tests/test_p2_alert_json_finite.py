"""[P2-ALERT-JSON-FINITE · 2026-07-05]

Bug vivo (corr=237cf108): `[P2-2/POST-SWAP-ALERT] INSERT system_alerts falló:
InvalidTextRepresentation: invalid input syntax for type json`. Causa: `delta_pct` de las
divergencias puede ser float('inf') (lado esperado 0 → división; misma clase del
INF-RESPONSE-500 de 2026-07-04) y `json.dumps` serializa inf/nan como Infinity/NaN — que es
JSON INVÁLIDO para Postgres `::jsonb`. La alerta de coherencia post-swap se perdía en silencio
best-effort. Fix: sanitizador `_json_finite` (no-finitos → None) + `allow_nan=False` como
cinturón (si un no-finito se cuela, ValueError explícito en vez de Infinity silencioso).
"""
import json
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


@pytest.fixture()
def go():
    import graph_orchestrator as g
    return g


def _emit_with(go, monkeypatch, delta):
    captured = {}

    def _fake_write(sql, params):
        captured["sql"] = sql
        captured["params"] = params
        return 1

    monkeypatch.setattr(go, "execute_sql_write", _fake_write)
    monkeypatch.setattr(go, "execute_sql_query", lambda *a, **k: None)  # sin cooldown abierto
    ok = go._emit_post_swap_coherence_alert(
        user_id="u-test", plan_id="p-test",
        divergences=[{"food": "Miel", "side": "expected", "hypothesis": "unknown",
                      "magnitude": True, "delta_pct": delta}],
        hyp_counter={"unknown": 1}, critical_total=5, household=1.0)
    return ok, captured


def test_inf_delta_pct_produces_valid_json(go, monkeypatch):
    ok, captured = _emit_with(go, monkeypatch, float("inf"))
    assert ok is True, "el INSERT ya no explota con delta_pct=inf"
    meta_json = captured["params"][3]
    assert "Infinity" not in meta_json and "NaN" not in meta_json, \
        f"Postgres ::jsonb rechaza Infinity/NaN: {meta_json[:200]}"
    parsed = json.loads(meta_json)  # JSON válido
    assert parsed["divergences_sample"][0]["delta_pct"] is None, \
        "el no-finito se sanea a None (la fila sigue siendo diagnóstica)"


def test_nan_delta_pct_produces_valid_json(go, monkeypatch):
    ok, captured = _emit_with(go, monkeypatch, float("nan"))
    assert ok is True
    parsed = json.loads(captured["params"][3])
    assert parsed["divergences_sample"][0]["delta_pct"] is None


def test_finite_delta_pct_preserved(go, monkeypatch):
    ok, captured = _emit_with(go, monkeypatch, 0.42)
    assert ok is True
    parsed = json.loads(captured["params"][3])
    assert parsed["divergences_sample"][0]["delta_pct"] == 0.42, \
        "los finitos NO se tocan (el sanitizador solo caza inf/nan)"


def test_allow_nan_false_belt_present():
    i = _GO.index("[P2-ALERT-JSON-FINITE · 2026-07-05]")
    win = _GO[i:i + 3000]
    assert "allow_nan=False" in win, \
        "cinturón: si un no-finito se cuela al dumps, ValueError explícito (no Infinity silencioso)"
    assert "_json_finite" in win
