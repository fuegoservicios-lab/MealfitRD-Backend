# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-214 · 2026-09-24] Compra única: el duradero que sustituye al fresco es SEGURO y rota.

`_single_trip_fresh_substitute` corre en el escudo final (pre-INSERT y merge de bloques), después del revisor y sin
guarda de alérgenos detrás. Cambiaba toda carne y pescado por «atún en agua»: a un alérgico al pescado le metía atún del
día 8 al 30, y a todos, 23 días de atún. Ahora la decisión por línea vive en `compra_unica.sustituir_linea`: rota la
proteína por día y comida, pasa cada candidato por el backstop clínico y, si ninguno es seguro, no sustituye.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
import compra_unica as cu  # noqa: E402

SINGLE = {"shopping": {"main_cycle_days": 30, "fresh_topup_days": None, "freezer_mode": "none", "batch_cooking": "never"},
          "diet": {"type": "balanced"}}


class _NoopDB:
    def macros_from_ingredient_string(self, s):
        return None

    def lookup(self, s):
        return None


def _plan(lineas_por_dia: dict, total: int = 30):
    """Días 0..total-1 con arroz; en los días indicados, una cena con esas líneas."""
    days = []
    for i in range(total):
        ings = list(lineas_por_dia.get(i) or ["1 taza de arroz"])
        days.append({"day": i + 1, "meals": [{"meal": "Cena", "name": "Plato", "ingredients": list(ings),
                                              "ingredients_raw": list(ings)}]})
    return days


def _sin_truth_up(monkeypatch):
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)


def test_alergico_al_pescado_no_recibe_atun_ni_sardinas(monkeypatch):
    _sin_truth_up(monkeypatch)
    days = _plan({9: ["150 g de pechuga de pollo"], 10: ["150 g de pechuga de pollo"], 11: ["150 g de filete de pescado"]})
    n = go._single_trip_fresh_substitute(days, db=_NoopDB(), effective=SINGLE, diet="balanced",
                                         contexto={"allergies": ["Pescado"]})
    assert n == 3
    for i in (9, 10, 11):
        linea = days[i]["meals"][0]["ingredients"][0]
        assert "atun" not in linea and "sardina" not in linea, linea
        assert linea == "150 g de garbanzos cocidos", linea
        assert days[i]["meals"][0]["ingredients_raw"] == [linea]


def test_la_proteina_rota_por_dia(monkeypatch):
    _sin_truth_up(monkeypatch)
    days = _plan({9: ["150 g de pechuga de pollo"], 10: ["150 g de pechuga de pollo"], 11: ["150 g de pechuga de pollo"]})
    go._single_trip_fresh_substitute(days, db=_NoopDB(), effective=SINGLE, diet="balanced", contexto={})
    subs = [days[i]["meals"][0]["ingredients"][0] for i in (9, 10, 11)]
    assert len(set(subs)) == 3, subs
    assert set(subs) == {"150 g de atun en agua", "150 g de sardinas en lata", "150 g de garbanzos cocidos"}


def test_vegetariano_solo_legumbres(monkeypatch):
    _sin_truth_up(monkeypatch)
    veg = {"shopping": SINGLE["shopping"], "diet": {"type": "vegetarian"}}
    days = _plan({9: ["150 g de pechuga de pollo"], 10: ["150 g de pechuga de pollo"]})
    go._single_trip_fresh_substitute(days, db=_NoopDB(), effective=veg, diet="vegetarian")
    subs = {days[i]["meals"][0]["ingredients"][0] for i in (9, 10)}
    assert subs == {"150 g de garbanzos cocidos", "150 g de lentejas cocidas"}, subs


def test_el_peso_aproximado_manda_sobre_la_pieza(monkeypatch):
    _sin_truth_up(monkeypatch)
    days = _plan({12: ["1 pechuga de pollo (≈200 g)"]})
    go._single_trip_fresh_substitute(days, db=_NoopDB(), effective=SINGLE, diet="balanced")
    assert days[12]["meals"][0]["ingredients"][0].startswith("200 g de "), days[12]["meals"][0]["ingredients"]
    assert cu.cantidad_de("150 g de filete de pescado") == "150 g de "
    assert cu.cantidad_de("½ pedazo mediano de yuca (≈200 g)") == "200 g de "
    assert cu.cantidad_de("Cilantro al gusto") == ""


def test_viveres_y_pan_que_no_llegan_al_fin_del_ciclo(monkeypatch):
    _sin_truth_up(monkeypatch)
    days = _plan({12: ["130 g de plátano verde"], 8: ["2 rebanadas de pan integral"], 22: ["200 g de yuca"],
                  10: ["200 g de yuca"]})
    go._single_trip_fresh_substitute(days, db=_NoopDB(), effective=SINGLE, diet="balanced")
    assert days[12]["meals"][0]["ingredients"] == ["130 g de batata"]
    assert days[8]["meals"][0]["ingredients"][0].endswith("casabe"), days[8]["meals"][0]["ingredients"]
    assert days[22]["meals"][0]["ingredients"] == ["200 g de batata"]
    assert days[10]["meals"][0]["ingredients"] == ["200 g de yuca"], "la yuca aguanta 21 días"


def test_idempotente_y_la_semana_de_frescos_no_se_toca(monkeypatch):
    _sin_truth_up(monkeypatch)
    # [P1-PLAN-LOTE-218] sin congelador, los días 1-3 (índices 0-2) no se tocan
    days = _plan({2: ["150 g de pechuga de pollo"], 15: ["300 ml de leche descremada"]})
    go._single_trip_fresh_substitute(days, db=_NoopDB(), effective=SINGLE, diet="balanced")
    assert days[2]["meals"][0]["ingredients"] == ["150 g de pechuga de pollo"]
    assert days[15]["meals"][0]["ingredients"] == ["300 ml de leche UHT"]
    assert go._single_trip_fresh_substitute(days, db=_NoopDB(), effective=SINGLE, diet="balanced") == 0


def test_sin_duradero_seguro_no_se_sustituye(monkeypatch):
    """Si el backstop rechaza TODOS los candidatos, el fresco se queda con su aviso: nunca un alérgeno."""
    monkeypatch.setattr(go, "clinical_backstop_for_meal", lambda meal, **kw: ["alérgeno"])
    assert cu.sustituto_seguro(cu.PROTEINA_TABLA, 9, False, ["x"]) is None
    assert cu.sustituir_linea("150 g de pechuga de pollo", 9, {"need_days": 10, "allow_frozen": False},
                              alergias=["x"]) is None
    # fail-closed: un backstop que revienta tampoco deja pasar
    def _boom(meal, **kw):
        raise RuntimeError("scan")
    monkeypatch.setattr(go, "clinical_backstop_for_meal", _boom)
    assert cu.es_seguro("atun en agua", ["pescado"]) is False
    assert cu.es_seguro("atun en agua", []) is True, "sin nada que verificar no se consulta"


def test_la_tabla_vive_en_compra_unica_y_go_la_usa():
    assert go._FRESH_SUBSTITUTES is cu.SUSTITUTOS
    assert go._FRESH_SUB_SKIP is cu.SIN_SUSTITUTO
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("def _single_trip_fresh_substitute(")
    cuerpo = src[i:src.index("\ndef ", i + 10)]
    assert "_cu_fs.sustituir_linea(" in cuerpo and "semilla=i + mi" in cuerpo
    assert 'contexto=locals().get("_clin_ctx")' in (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
    assert "tooltip-anchor: P1-PLAN-LOTE-214-SUSTITUIR-LINEA" in (_BACKEND / "compra_unica.py").read_text(encoding="utf-8")
