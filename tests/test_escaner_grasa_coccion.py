# backend/tests/test_escaner_grasa_coccion.py
# [ESCANER-GRASA-COCCION · 2026-11-13] El dueño: «debería preguntar por cosas más puntuales, como el aceite que se usa:
# el plátano maduro pudo hacerse con manteca de cerdo». El escáner pregunta CON QUÉ grasa se cocinó (frituras y
# majados), con opciones concretas, y esa duda sobrevive entera al saneado de la respuesta del modelo.
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from vision_agent import _MEAL_VISION_PROMPT, _coerce_meal_scan


def test_el_prompt_pide_la_grasa_con_opciones_concretas():
    p = _MEAL_VISION_PROMPT
    assert "CON QUE GRASA SE COCINO" in p
    for opcion in ("Aceite vegetal", "Manteca de cerdo", "Mantequilla", "Freidora de aire / sin grasa", "Sin grasa"):
        assert opcion in p, opcion
    for alimento in ("platano maduro frito", "tostones", "mangu", "pure"):
        assert alimento in p, alimento


def test_la_regla_no_sube_el_tope_de_dos_dudas_ni_desplaza_cantidad_o_identidad():
    p = _MEAL_VISION_PROMPT
    assert "como MAXIMO 2 cosas" in p
    regla = p[p.index("CON QUE GRASA SE COCINO"):p.index("PLATOS COMPUESTOS CRIOLLOS")]
    assert "te queda sitio entre las 2 dudas" in regla
    assert "la de la grasa no la desplaza" in regla


def test_la_duda_de_la_grasa_sobrevive_al_saneado_con_sus_cuatro_opciones():
    r = _coerce_meal_scan(dict(
        photo_kind="plato", is_food=True, meal_name="Plátano maduro frito", calories=310, protein=2, carbs=48,
        healthy_fats=13, description="Seis tajadas de plátano maduro frito.",
        items=[{"name": "plátano maduro frito", "quantity": 6, "unit": "tajadas"}],
        dudas=[{
            "sobre": "plátano maduro frito", "pregunta": "¿Con qué grasa lo freíste?",
            "opciones": [
                {"texto": "Aceite vegetal", "supuesta": True,
                 "ajuste": {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}},
                {"texto": "Manteca de cerdo",
                 "ajuste": {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}},
                {"texto": "Mantequilla",
                 "ajuste": {"calories": -10, "protein": 0, "carbs": 0, "healthy_fats": -1}},
                {"texto": "Freidora de aire / sin grasa",
                 "ajuste": {"calories": -110, "protein": 0, "carbs": 0, "healthy_fats": -12}},
            ],
        }],
    ))
    assert len(r["dudas"]) == 1
    ops = r["dudas"][0]["opciones"]
    assert [o["texto"] for o in ops] == ["Aceite vegetal", "Manteca de cerdo", "Mantequilla",
                                         "Freidora de aire / sin grasa"]
    assert [o["supuesta"] for o in ops] == [True, False, False, False]
    assert ops[3]["ajuste"]["calories"] < 0 and ops[3]["ajuste"]["healthy_fats"] < 0
    # el coach recibe la pregunta con las opciones tal cual
    assert "Manteca de cerdo" in r["description"]
