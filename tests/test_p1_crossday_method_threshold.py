"""[P1-CROSSDAY-METHOD-THRESHOLD · 2026-07-11] Los métodos de cocción tienen umbral
cross-día propio (≥5 días), separado de los platos-base (≥3).

Caso vivo (renovación del owner, 15:26): el gate rechazó el intento #1 por
"'plancha' en 3 días" — pollo a la plancha + res a la plancha + pescado a la plancha
son platos DISTINTOS (la plancha es LA preparación saludable es-DO por excelencia, y
la propia directiva del gate la sugiere como variación). La monotonía real de método
(plancha 5+ días) sigue gateada.

tooltip-anchor: P1-CROSSDAY-METHOD-THRESHOLD
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))


def _meal(name):
    return {"meal": "Almuerzo", "name": name, "ingredients": ["100 g de algo"],
            "protein": 25, "carbs": 20, "fats": 9}


def _plan(names_by_day):
    return {"days": [{"day": i + 1, "meals": [_meal(n)]} for i, n in enumerate(names_by_day)]}


def test_method_3_days_distinct_bases_passes():
    from graph_orchestrator import build_variety_report
    rep = build_variety_report(_plan([
        "Pollo a la Plancha con Ensalada",
        "Res a la Plancha con Batata",
        "Pescado a la Plancha con Yuca",
        "Cerdo Guisado con Arroz",
    ]))
    assert "plancha" not in (rep.get("cross_day_dishes") or {}), (
        "3 días de plancha con bases DISTINTAS es variedad legítima — el umbral "
        "compartido rechazó el intento #1 de la renovación del owner"
    )


def test_method_5_days_still_gates():
    from graph_orchestrator import build_variety_report
    rep = build_variety_report(_plan([
        "Pollo a la Plancha", "Res a la Plancha", "Pescado a la Plancha",
        "Pavo a la Plancha", "Cerdo a la Plancha",
    ]))
    assert (rep.get("cross_day_dishes") or {}).get("plancha") == 5, (
        "la monotonía REAL de método (5+ días) sigue gateada"
    )


def test_dish_base_keeps_threshold_3():
    from graph_orchestrator import build_variety_report
    rep = build_variety_report(_plan([
        "Panqueques de Avena", "Panqueques de Guineo", "Panqueques Integrales",
        "Pollo Guisado",
    ]))
    assert (rep.get("cross_day_dishes") or {}).get("panqueque") == 3, (
        "los platos-base reales (panqueques ×3) conservan el umbral 3"
    )


def test_knob_and_tokens_defined():
    import graph_orchestrator as go
    assert go.CROSS_DAY_METHOD_GATE_MIN_DAYS == 5
    assert set(go._PREP_METHOD_TOKENS) == {"plancha", "horneado"}, (
        "solo modificadores puros — guiso/salteado son cabezas de plato renombrables "
        "(el diversificador las resuelve) y conservan umbral 3"
    )


def test_marker_anchored_in_source():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert src.count("P1-CROSSDAY-METHOD-THRESHOLD") >= 2
