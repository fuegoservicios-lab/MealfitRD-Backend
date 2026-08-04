"""[P3-UNDERSUPPLY-VISIBILITY · 2026-08-04] El default OFF de
`MEALFIT_GUARD_UNDERSUPPLY_SEVERE` (P2-GUARD-UNDERSUPPLY-CANONICAL) exige "medir volumen
en `_shopping_coherence_block_history`" antes de encender la escalada — pero nadie tenía
ese SELECT programado. Un default que solo se revisa "cuando alguien se acuerde" se
vuelve permanente por inercia, y la propia doctrina del repo (I2-MISS, cron P3-B, tick
P2-LIVE-9) es que la observabilidad de un gate no puede depender de que alguien conozca
la clave a mano.

## Baseline medido (2026-08-04, ~1,4h post-deploy, contra prod)

22 planes con history, 186 entries, hipótesis históricas
`{recipe_unquantified: 411, unknown: 143, cap_swallowed_modifier: 5, pantry_overdeduct: 4}`,
**`magnitude_undersupply`: 0** (aún sin tráfico de listas nuevas construidas con el sello
`pantry_deduction_applied` que la hipótesis necesita).

## El fix

El cron diario `_shopping_coherence_alert_job` (cron_tasks.py) YA agrega TODAS las
hipótesis en un `Counter` (`by_hypothesis`) — `magnitude_undersupply` ya estaba ahí, mezclada
con las demás. Pero solo `cap_swallowed_modifier` tenía un campo EXPLÍCITO (`cap_count`/
`cap_ratio`) en el tick persistido y una condición de alerta dedicada. Este P-fix:

  1. añade `undersupply_count` (y su ratio) como campo EXPLÍCITO en el tick
     `_shopping_coherence_alert_job_tick` (mismo patrón que `cap_count`);
  2. añade una línea de log DEDICADA con el marker `[P3-UNDERSUPPLY-VISIBILITY]`, siempre
     que el cron corre con planes evaluados — la visibilidad no puede depender de que
     alguien conozca la clave `magnitude_undersupply` dentro de un dict genérico;
  3. documenta junto al knob `MEALFIT_GUARD_UNDERSUPPLY_SEVERE`
     (`shopping_calculator._get_guard_undersupply_severe_knob`) el CRITERIO de encendido
     con el baseline: encender cuando la serie diaria muestre `magnitude_undersupply`
     estable y bajo (p.ej. <5% de las entries diarias, sin ráfagas por plan); baseline
     2026-08-04: 0 sobre 186 entries históricas.

tooltip-anchor: P3-UNDERSUPPLY-VISIBILITY
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_CRON_SRC = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
_SC_SRC = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")


def _extract_coh_alert_body(src: str) -> str:
    import re
    anchor = re.search(r"def _shopping_coherence_alert_job\b[^\n]*\n", src)
    assert anchor is not None
    start = anchor.end()
    rest = src[start:]
    next_decl = re.search(r"\n(?:def |class |# ---)", rest)
    end = start + (next_decl.start() if next_decl else len(rest))
    return src[start:end]


# ---------------------------------------------------------------------------
# 0. Marker
# ---------------------------------------------------------------------------
def test_marker_present():
    assert "P3-UNDERSUPPLY-VISIBILITY" in _CRON_SRC
    assert "P3-UNDERSUPPLY-VISIBILITY" in _SC_SRC


# ---------------------------------------------------------------------------
# 1. Parser: el tick lleva el campo explícito + la línea de log dedicada
# ---------------------------------------------------------------------------
def test_tick_tiene_campo_explicito_undersupply_count():
    body = _extract_coh_alert_body(_CRON_SRC)
    finally_idx = body.find("finally:")
    tick_block = body[finally_idx:] if finally_idx != -1 else ""
    assert "undersupply_count" in tick_block, (
        "el tick `_shopping_coherence_alert_job_tick` debe exponer `undersupply_count` "
        "explícito — mismo patrón que `cap_count`, o la visibilidad depende de que "
        "alguien conozca la clave `magnitude_undersupply` dentro del Counter genérico."
    )


def test_log_dedicado_con_marker():
    body = _extract_coh_alert_body(_CRON_SRC)
    assert "P3-UNDERSUPPLY-VISIBILITY" in body, (
        "falta una línea de log DEDICADA con el marker — el Counter genérico ya cubre "
        "la clave, pero la visibilidad no puede depender de que alguien la conozca."
    )
    assert "undersupply_count" in body


# ---------------------------------------------------------------------------
# 2. Funcional offline: el cron real, con DB mockeada, expone el campo + loguea
# ---------------------------------------------------------------------------
class _FakeDivergence(dict):
    pass


def _fake_guard_factory(hyp_counts_per_plan):
    """Fábrica de un stub de `run_shopping_coherence_guard_and_append_history` que
    devuelve, para cada plan (en orden), una lista de divergencias con las hipótesis
    pedidas."""
    calls = {"i": 0}

    def _fake(plan_data, **kw):
        i = calls["i"]
        calls["i"] += 1
        hyps = hyp_counts_per_plan[i] if i < len(hyp_counts_per_plan) else []
        divs = [{"food": f"Alimento{i}{j}", "hypothesis": h, "magnitude": True,
                 "delta_pct": 0.6, "expected_qty": 100.0, "actual_qty": 40.0}
                for j, h in enumerate(hyps)]
        return divs, bool(divs)

    return _fake


@pytest.fixture
def _coh_env(monkeypatch):
    """Neutraliza TODA la IO real de `_shopping_coherence_alert_job`: pool, SELECT de
    planes, persist y el SELECT/INSERT del contador de fallos consecutivos (finally).
    Mismo patrón de mock por SQL-sniffing que el resto de la suite usa para crons
    con imports lazy (ver `_shopping_coherence_alert_job` L1020+: `connection_pool`,
    `run_shopping_coherence_guard_and_append_history` y `update_meal_plan_data` se
    importan DENTRO de la función, así que se parchea el módulo FUENTE, no `cron_tasks`)."""
    import cron_tasks as ct
    import db_core
    import db_plans

    monkeypatch.setattr(db_core, "connection_pool", object())

    plans = [
        {"id": "plan-1", "user_id": "user-1", "plan_data": {"days": []}},
        {"id": "plan-2", "user_id": "user-2", "plan_data": {"days": []}},
    ]

    writes = []

    def _fake_query(query, params=None, fetch_one=False, fetch_all=False):
        if "FROM public.meal_plans" in query:
            return list(plans)
        if "FROM app_kv_store" in query:
            return None  # sin contador previo de fallos consecutivos
        return None

    def _fake_write(query, params=None, **kw):
        writes.append((query, params))
        return True

    monkeypatch.setattr(ct, "execute_sql_query", _fake_query)
    monkeypatch.setattr(ct, "execute_sql_write", _fake_write)
    monkeypatch.setattr(db_plans, "update_meal_plan_data", lambda *a, **k: True)
    monkeypatch.setenv("MEALFIT_COH_ALERT_MIN_PLANS", "1")

    return ct, writes


def _tick_payload(writes):
    for query, params in writes:
        if "_shopping_coherence_alert_job_tick" in str(params):
            return json.loads(params[1])
    raise AssertionError(f"no se encontró el INSERT del tick entre los writes: {writes}")


def test_tick_payload_incluye_undersupply_count(_coh_env, monkeypatch):
    ct, writes = _coh_env
    import shopping_calculator as sc

    monkeypatch.setattr(
        sc, "run_shopping_coherence_guard_and_append_history",
        _fake_guard_factory([
            ["magnitude_undersupply", "cap_swallowed_modifier"],
            ["magnitude_undersupply"],
        ]),
    )

    ct._shopping_coherence_alert_job()

    payload = _tick_payload(writes)
    assert payload.get("undersupply_count") == 2, (
        f"esperaba 2 divergencias `magnitude_undersupply` en el tick, payload={payload}"
    )
    assert payload.get("cap_count") == 1, "el campo hermano cap_count no debe romperse"


def test_tick_payload_cero_cuando_no_hay_undersupply(_coh_env, monkeypatch):
    ct, writes = _coh_env
    import shopping_calculator as sc

    monkeypatch.setattr(
        sc, "run_shopping_coherence_guard_and_append_history",
        _fake_guard_factory([["unknown"], ["recipe_unquantified"]]),
    )

    ct._shopping_coherence_alert_job()

    payload = _tick_payload(writes)
    assert payload.get("undersupply_count") == 0


def test_log_dedicado_reporta_el_conteo(_coh_env, monkeypatch, caplog):
    ct, writes = _coh_env
    import shopping_calculator as sc

    monkeypatch.setattr(
        sc, "run_shopping_coherence_guard_and_append_history",
        _fake_guard_factory([
            ["magnitude_undersupply", "magnitude_undersupply"],
            ["magnitude_undersupply"],
        ]),
    )

    with caplog.at_level(logging.INFO):
        ct._shopping_coherence_alert_job()

    marker_lines = [r.message for r in caplog.records if "P3-UNDERSUPPLY-VISIBILITY" in r.message]
    assert marker_lines, "no se emitió ninguna línea de log con el marker dedicado"
    assert any("3" in ln for ln in marker_lines), (
        f"la línea dedicada debe reportar el conteo real (3): {marker_lines}"
    )


# ---------------------------------------------------------------------------
# 3. El criterio de encendido del knob queda documentado con el baseline
# ---------------------------------------------------------------------------
def test_criterio_de_encendido_documentado_junto_al_knob():
    i = _SC_SRC.index("def _get_guard_undersupply_severe_knob")
    doc = _SC_SRC[i:i + 3200]
    assert "P3-UNDERSUPPLY-VISIBILITY" in doc, (
        "el docstring del knob debe referenciar el P-fix que cerró la visibilidad"
    )
    for pieza in ("2026-08-04", "186 entries", "magnitude_undersupply", "5%"):
        assert pieza in doc, (
            f"el criterio de encendido del knob debe citar {pieza!r} (baseline medido)"
        )


# ---------------------------------------------------------------------------
# 4. Marker bump — patrón fecha-floor (NO literal, ver higiene de este mismo lote:
#    test_p3_agg_num_days_propagate.py::test_last_known_pfix_bumpeado anclaba el
#    marker exacto y se rompía con cada P-fix posterior).
# ---------------------------------------------------------------------------
def test_last_known_pfix_bumpeado():
    import re
    from datetime import date, datetime

    app_src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', app_src)
    assert m, "No se encontró _LAST_KNOWN_PFIX en app.py."
    marker = m.group(1)
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    assert fecha, f"Marker sin fecha ISO: {marker!r}"
    marker_date = datetime.strptime(fecha.group(1), "%Y-%m-%d").date()
    floor = date(2026, 8, 4)
    assert marker_date >= floor, (
        f"_LAST_KNOWN_PFIX={marker!r} (fecha={marker_date}) anterior al floor {floor} "
        f"de cierre de P3-UNDERSUPPLY-VISIBILITY."
    )
