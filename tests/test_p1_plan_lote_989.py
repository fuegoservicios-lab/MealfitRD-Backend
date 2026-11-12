# backend/tests/test_p1_plan_lote_989.py
# [P1-PLAN-LOTE-989 · 2026-11-12] Foto de jugo de lechoza: las preguntas no ofrecían la lechoza y la respuesta
# «Con azúcar» renombraba el plato a «Jugo con azúcar» (perdía la fruta).
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from bebidas_de_la_foto import asegurar_supuesta_del_modelo, bebida_y_fruta, es_bebida
from vision_agent import _MEAL_VISION_PROMPT, _coerce_meal_scan

_BASE = dict(photo_kind="plato", is_food=True, calories=120, protein=1, carbs=28, healthy_fats=0)


def _textos(d):
    return [o["texto"] for o in d["opciones"]]


def test_bebida_y_fruta():
    assert bebida_y_fruta("Jugo de lechoza") == ("jugo", "lechoza")
    assert bebida_y_fruta("Jugo de piña") == ("jugo", "piña")
    assert bebida_y_fruta("Batida de guineo y avena")[0] == "batida"
    assert bebida_y_fruta("Jugo de fruta") is None
    assert bebida_y_fruta("Jugo natural") is None


def test_fruta_conocida_pregunta_preparacion_sin_perder_la_fruta():
    r = _coerce_meal_scan(dict(_BASE, meal_name="Jugo de lechoza", description="Vaso de jugo de lechoza, anaranjado.",
                               items=[{"name": "jugo de lechoza", "quantity": 1, "unit": "vaso"}]))
    assert r["meal_name"] == "Jugo de lechoza"
    d = r["dudas"][0]
    assert "lechoza" in d["pregunta"]
    # ninguna opción borra la fruta del nombre del plato
    assert all("lechoza" in o["nombre_plato"].lower() for o in d["opciones"])
    assert "Con azúcar" in _textos(d)


def test_fruta_desconocida_ofrece_las_frutas_probables_por_color():
    r = _coerce_meal_scan(dict(_BASE, meal_name="Jugo natural", description="Un vaso de jugo natural de color anaranjado.",
                               items=[{"name": "jugo", "quantity": 1, "unit": "vaso"}]))
    frutas = r["dudas"][0]
    assert {"Lechoza", "Chinola", "Naranja"} <= set(_textos(frutas))
    assert r["meal_name"] == frutas["opciones"][0]["nombre_plato"]
    # la segunda pregunta (preparación) no pisa la fruta elegida
    assert all(not o.get("nombre_plato") for o in r["dudas"][1]["opciones"])


def test_la_duda_del_modelo_siempre_incluye_su_propia_respuesta():
    r = _coerce_meal_scan(dict(_BASE, meal_name="Jugo de lechoza", description="Vaso de jugo de lechoza, anaranjado y espeso.",
                               items=[{"name": "jugo", "quantity": 1, "unit": "vaso"}],
                               dudas=[{"sobre": "jugo", "pregunta": "¿Qué bebida tienes en el vaso?",
                                       "opciones": [{"texto": "Café con leche", "supuesta": True},
                                                    {"texto": "Té o infusión"}, {"texto": "Chocolate"}]}]))
    d = r["dudas"][0]
    assert d["opciones"][0]["texto"] == "Jugo de lechoza" and d["opciones"][0]["supuesta"]
    assert sum(o["supuesta"] for o in d["opciones"]) == 1


def test_no_duplica_si_ya_esta():
    dudas = [{"pregunta": "¿Qué bebida es?", "opciones": [{"texto": "Café negro", "supuesta": True},
                                                         {"texto": "Té verde"}, {"texto": "Chocolate"}]}]
    assert asegurar_supuesta_del_modelo(dudas, "Café negro") == dudas


def test_lo_de_dentro_de_la_bebida_no_es_un_solido():
    assert es_bebida("Jugo de fruta", "Vaso de jugo de fruta natural") is True
    assert es_bebida("Batida de avena", "Vaso de batida de avena",
                     [{"name": "avena", "quantity": 30, "unit": "g"}]) is True
    # un plato con jugo al lado sigue sin ser «solo bebida»
    assert es_bebida("Arroz con pollo", "Arroz con pollo y un vaso de jugo de naranja") is False


def test_batida_pregunta_la_preparacion_de_la_batida_no_el_te():
    r = _coerce_meal_scan(dict(_BASE, meal_name="Batida de guineo", description="Vaso de batida de guineo con avena.",
                               items=[{"name": "guineo", "quantity": 1, "unit": "u"},
                                      {"name": "avena", "quantity": 30, "unit": "g"}]))
    d = r["dudas"][0]
    assert "batida de guineo" in d["pregunta"]
    assert "Con leche" in _textos(d)


def test_prompt_pide_frutas_probables():
    assert "JUGOS Y BATIDAS" in _MEAL_VISION_PROMPT
    assert "EXACTAMENTE la que pusiste en 'meal_name'" in _MEAL_VISION_PROMPT
