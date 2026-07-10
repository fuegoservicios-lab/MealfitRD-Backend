"""[P1-SELF-CRITIQUE-CANARY · 2026-07-09] Infra de canario para validar (Fase 2, Paso 6) apagar
self_critique en producción. El harness determinista (`macro_sizing_replay`) es CIEGO a cambios de la
capa de crítica (replaya solo el motor de sizing sobre un corpus post-crítica) → la única señal válida
es un canario live-cohort. Knob `MEALFIT_SELF_CRITIQUE_CANARY_PCT` (0-100, default 0) rutea ese % de
planes a critique-OFF con bucketing determinista sha256(user_id|session_id) — estable por usuario. La
métrica `clinical_band` se etiqueta con el cohorte para poder comparar OFF vs ON (hoy no tiene esa
dimensión). Default 0 = todo el tráfico critique-ON (comportamiento actual).
"""
import os

import graph_orchestrator as go

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return f.read()


def _src():
    return _read(_BACKEND, "graph_orchestrator.py")


def test_knob_default_zero():
    assert go.SELF_CRITIQUE_CANARY_PCT == 0, "el canario debe nacer en 0% (todo critique-ON)"
    assert 'MEALFIT_SELF_CRITIQUE_CANARY_PCT' in _src()


def test_marker_present():
    assert 'P1-SELF-CRITIQUE-CANARY' in _src()


def test_cohort_helper_pct0_always_on():
    """Con PCT=0 (default) ningún plan cae a critique-OFF."""
    _saved = go.SELF_CRITIQUE_CANARY_PCT
    try:
        go.SELF_CRITIQUE_CANARY_PCT = 0
        for uid in ("user-a", "user-b", "user-c", "guest-xyz"):
            st = {"form_data": {"user_id": uid}}
            assert go._self_critique_canary_cohort(st) == "on"
    finally:
        go.SELF_CRITIQUE_CANARY_PCT = _saved


def test_cohort_helper_pct100_always_off():
    _saved = go.SELF_CRITIQUE_CANARY_PCT
    try:
        go.SELF_CRITIQUE_CANARY_PCT = 100
        for uid in ("user-a", "user-b", "guest-xyz"):
            st = {"form_data": {"user_id": uid}}
            assert go._self_critique_canary_cohort(st) == "off"
    finally:
        go.SELF_CRITIQUE_CANARY_PCT = _saved


def test_cohort_helper_deterministic_per_user():
    """El mismo id cae siempre en el mismo bucket (A/B estable por usuario)."""
    _saved = go.SELF_CRITIQUE_CANARY_PCT
    try:
        go.SELF_CRITIQUE_CANARY_PCT = 50
        st = {"form_data": {"user_id": "stable-user-42"}}
        first = go._self_critique_canary_cohort(st)
        for _ in range(5):
            assert go._self_critique_canary_cohort(st) == first
    finally:
        go.SELF_CRITIQUE_CANARY_PCT = _saved


def test_metric_emit_tagged_with_cohort():
    """La métrica clinical_band debe llevar el tag de cohorte (única forma de sliceear OFF vs ON)."""
    src = _src()
    emit = src[src.find('"node": "clinical_band"'):]
    emit = emit[:1500]  # [P1-BAND-TELEMETRY-PER-DAY · 2026-07-10] +per_day empujó el offset
    assert "self_critique_cohort" in emit, "el emit de clinical_band debe etiquetar self_critique_cohort"


def test_node_entry_routes_off_cohort():
    """El nodo debe consultar el cohorte y early-return {'_self_critique_cohort':'off'} para OFF."""
    src = _src()
    node = src[src.find("async def self_critique_node"):]
    node = node[: node.find("\nasync def ", 1) if node.find("\nasync def ", 1) > 0 else 6000]
    assert "_self_critique_canary_cohort(" in node, "el nodo debe consultar el cohorte del canario"
    assert '"_self_critique_cohort"' in node or "'_self_critique_cohort'" in node, (
        "el nodo debe devolver el cohorte OFF en el early-return"
    )
