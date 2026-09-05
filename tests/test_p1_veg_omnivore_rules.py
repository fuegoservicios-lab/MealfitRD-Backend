# -*- coding: utf-8 -*-
"""[P1-VEG-OMNIVORE-RULES · 2026-09-05] Dos reglas calibradas para omnívoros castigaban a los vegetarianos en el
plan vivo 18326457 (vegetariano, ganancia muscular, 30 días):

  · `_detect_gainmuscle_dinner_issues` pedía «pollo/pavo/pescado/res/atún como plato» —inexistente en esa
    dieta—, disparó 3 avisos, forzó la corrección de los 3 días del bloque y la cena del día 3 bajó de 54 g a
    34 g de proteína persiguiendo una regla imposible.
  · El tope duro de huevo (25 % de las comidas) rechazó el intento entero por «huevo en 5 de 12»: una
    replanificación completa por una diversidad que la dieta no permite, siendo el huevo una de las pocas
    proteínas completas que le quedan.

El pescetariano NO se exime de la primera (el pescado sí cuenta) y el vegano NO se exime del tope (ahí el huevo
es una violación de dieta que caza otro guard)."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import graph_orchestrator as go  # noqa: E402


def _dias_con_cena_de_queso():
    return [
        {"meals": [{"meal": "Cena", "name": "Bulgur horneado con lentejas y mozzarella fresca",
                    "ingredients": ["150 g de mozzarella", "60 g de lentejas"], "protein": 54}]},
        {"meals": [{"meal": "Cena", "name": "Arepa rellena de queso fresco con lentejas",
                    "ingredients": ["120 g de queso fresco"], "protein": 40}]},
    ]


@pytest.mark.parametrize("dieta", ["vegetarian", "vegetariana", "vegan", "vegana"])
def test_la_cena_no_se_juzga_por_carne_que_la_dieta_prohibe(dieta):
    issues = go._detect_gainmuscle_dinner_issues(
        _dias_con_cena_de_queso(), {"mainGoal": "gain_muscle", "dietType": dieta})
    assert issues == [], issues


def test_el_omnivoro_sigue_recibiendo_la_senal():
    """La regla se creó para un caso real (plan b4316db6): sin dieta que lo impida, sigue viva."""
    issues = go._detect_gainmuscle_dinner_issues(
        _dias_con_cena_de_queso(), {"mainGoal": "gain_muscle", "dietType": "omnivora"})
    assert len(issues) >= 2, issues
    assert any("queso" in i.lower() for i in issues)


def test_el_pescetariano_no_se_exime():
    """En pescetariano SÍ existe proteína animal magra (el pescado): la señal es satisfacible."""
    issues = go._detect_gainmuscle_dinner_issues(
        _dias_con_cena_de_queso(), {"mainGoal": "gain_muscle", "dietType": "pescetariana"})
    assert issues, "sin exención, el pescetariano debe seguir recibiendo el aviso"


def test_tope_de_huevo_mas_ancho_solo_en_vegetariano():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index('_egg = int(_vr.get("egg_meals", 0))')
    bloque = src[i:i + 1200]
    assert "VARIETY_HARD_GATE_EGG_PCT_VEG" in bloque, "el gate usa el tope vegetariano"
    assert '== "vegetarian"' in bloque, "solo vegetariano: el vegano conserva el tope estrecho"
    assert "else 0.25" in bloque, "el omnívoro conserva su 25 %"


def test_el_tope_vegetariano_deja_pasar_cinco_de_doce_y_frena_siete():
    """La aritmética del caso vivo: 5/12 pasaba a ser rechazo con el 25 %; con 0,40 pasa, y 7 sigue cayendo."""
    pct, slack = go.VARIETY_HARD_GATE_EGG_PCT_VEG, go.VARIETY_HARD_GATE_EGG_SLACK
    cap_veg = max(3, round(12 * pct))
    cap_omni = max(3, round(12 * 0.25))
    assert 5 <= cap_veg + slack, f"5 de 12 debe pasar en vegetariano (cap {cap_veg}+{slack})"
    assert 7 > cap_veg + slack, f"7 de 12 debe seguir cayendo (cap {cap_veg}+{slack})"
    assert 5 > cap_omni + slack, "en omnívoro 5 de 12 sigue siendo rechazo (la regla original no cambia)"
