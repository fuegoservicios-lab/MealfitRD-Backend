"""[REVISION-ESCANER-14NOV / REVISION-DIARIO-14NOV] Fallos encontrados al revisar los commits 0f95df60, 5484ee1d y 99ae360f.

Cada uno se reprodujo ejecutando el código real antes de arreglarlo.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
_BACKEND = Path(__file__).resolve().parent.parent

_SUP = lambda t: [{"texto": t, "supuesta": True}, {"texto": "Con sirope", "supuesta": False}]


def test_el_pronombre_te_no_desactiva_el_filtro_de_preguntas_tontas():
    import dudas_del_plato_inteligentes as d

    assert d._es_duda_tonta({"pregunta": "¿Te pusiste sirope en los pancakes?", "opciones": _SUP("Sin sirope")}) is True
    assert d._es_duda_tonta({"pregunta": "¿Los pancakes llevan sirope?", "opciones": _SUP("Sin sirope")}) is True
    # el té de verdad sigue siendo bebida (lo disuelto no se ve)
    assert d._es_duda_tonta({"pregunta": "¿Le pusiste miel al té?", "opciones": _SUP("Sin miel")}) is False
    assert d._es_duda_tonta({"pregunta": "¿El café lleva azúcar?", "opciones": _SUP("Sin azúcar")}) is False


def test_repollo_no_es_pollo():
    import dudas_del_plato_inteligentes as d

    assert d._relleno_asumido_en_plato({"meal_name": "Sándwich de repollo y zanahoria", "description": ""}) is None
    assert d._relleno_asumido_en_plato({"meal_name": "Sándwich de pollo", "description": ""}) == "pollo"
    assert d._relleno_asumido_en_plato({"meal_name": "Sándwich de jamón y queso", "description": ""}) == "jamon_queso"


def test_un_acompanante_de_pan_no_es_un_habito_de_relleno():
    import habitos_de_la_foto as h

    falsos = [{"meal_name": "Arroz con pollo y pan", "ingredients": []}] * 3
    assert h.extraer_habitos_desde_registros(falsos)["rellenos_pan"] == []
    falsos = [{"meal_name": "Arroz con pollo y pan de agua", "ingredients": []}] * 3
    assert h.extraer_habitos_desde_registros(falsos)["rellenos_pan"] == []
    dentro = [{"meal_name": "Huevo frito con tostadas de mantequilla de maní", "ingredients": []}] * 2
    assert h.extraer_habitos_desde_registros(dentro)["rellenos_pan"] != []
    reales = [{"meal_name": "Sándwich de pollo", "ingredients": ["pan integral", "pollo"]}] * 2
    assert [r["id"] for r in h.extraer_habitos_desde_registros(reales)["rellenos_pan"]] == ["pollo"]
    assert h.extraer_habitos_desde_registros([{"meal_name": "2 tostadas con aguacate", "ingredients": []}])["rellenos_pan"][0]["id"] == "aguacate"


def test_la_meta_con_calculadora_no_exige_condiciones_medicas_pero_si_la_unidad_de_peso():
    src = (_BACKEND / "diary_targets.py").read_text(encoding="utf-8")
    assert '"medicalConditions")' not in src
    assert src.count('req = ("gender", "age", "height", "weight", "weightUnit", "activityLevel", "mainGoal")') == 2


def test_un_perfil_sin_condiciones_medicas_guarda_su_meta(monkeypatch):
    import db_core
    import diary_targets as dt

    escrito = {}
    monkeypatch.setattr(db_core, "connection_pool", object())
    monkeypatch.setattr(db_core, "execute_sql_query", lambda *a, **k: None)          # sin plan guardado
    monkeypatch.setattr(dt, "record_daily_target_snapshot", lambda uid, **kw: escrito.update(kw) or True)
    hp = {"gender": "male", "age": 30, "height": 175, "weight": 80, "weightUnit": "kg", "activityLevel": "moderate",
          "mainGoal": "lose_fat", "medicalConditions": []}
    assert dt.record_calculator_snapshot_if_tracking_only("u-1", hp) is True
    assert escrito["calories"] > 0
