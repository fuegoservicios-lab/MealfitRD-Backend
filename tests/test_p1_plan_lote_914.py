# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-914 · 2026-09-29] La leche de una cucharadita sale también cuando la frase trae una coma antes.

Batería real de embarazo rdv801 (día 2, desayuno): «5 ml de leche pasteurizada» llegó al plato con el 614 vivo. La regla de
la enumeración («A, B y la leche» → «A y B») se tragaba una cláusula con verbos: «…derretida, espolvorea la canela y
acompaña con la guayaba fresca y la leche pasteurizada» quedaba «…derretida y espolvorea la canela y acompaña con la
guayaba fresca», una frase coja más, y el TODO O NADA del 614 dejaba el plato como estaba. Medido sobre 8.574 comidas: 152
con leche de 10 ml o menos; el 614 saca 60, 80 se quedan a propósito (sin otra base líquida) y sólo ésta caía por la regla.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import leche_infima as li  # noqa: E402

_PLATANO = {
    "meal": "Desayuno", "name": "Plátano maduro asado con mantequilla de maní, guayaba y yogurt griego entero",
    "ingredients": ["½ plátano maduro mediano (75 g)", "1 cda de mantequilla de maní natural (16 g)", "145 g de guayaba",
                    "5 ml de leche pasteurizada", "¼ cdta de canela en polvo", "¾ taza de yogurt griego entero pasteurizado"],
    "ingredients_raw": ["75 g de plátano maduro", "16 g de mantequilla de maní natural", "145 g de guayaba",
                        "5 ml de leche pasteurizada", "0.25 cdta de canela en polvo", "185 g de Yogurt griego entero"],
    "recipe": ["Mise en place: lava el plátano maduro y ábrelo a lo largo sin pelarlo; mide 1 cda de mantequilla de maní "
               "natural, 5 ml de leche pasteurizada y ¼ cdta de canela; lava y corta 145 g de guayaba en cuartos.",
               "El Toque de Fuego: asa el plátano con cáscara en sartén u horno a fuego medio-alto durante 12-15 minutos, "
               "volteándolo, hasta que la piel esté negra y la pulpa tierna y caramelizada.",
               "Montaje: sirve el plátano asado con la mantequilla de maní derretida, espolvorea la canela y acompaña con la "
               "guayaba fresca y la leche pasteurizada. Acompaña con yogurt griego entero.",
               "🤰 Seguridad alimentaria (embarazo/lactancia): lava y desinfecta las frutas antes de usarlas."],
}


def test_la_leche_sale_y_la_frase_con_coma_queda_entera():
    m = copy.deepcopy(_PLATANO)
    assert li.quitar(m) == 1
    assert "5 ml de leche pasteurizada" not in m["ingredients"] and "5 ml de leche pasteurizada" not in m["ingredients_raw"]
    assert m["recipe"][0] == ("Mise en place: lava el plátano maduro y ábrelo a lo largo sin pelarlo; mide 1 cda de "
                              "mantequilla de maní natural y ¼ cdta de canela; lava y corta 145 g de guayaba en cuartos.")
    assert m["recipe"][2] == ("Montaje: sirve el plátano asado con la mantequilla de maní derretida, espolvorea la canela y "
                              "acompaña con la guayaba fresca. Acompaña con yogurt griego entero.")
    assert m["recipe"][1] == _PLATANO["recipe"][1] and m["recipe"][3] == _PLATANO["recipe"][3]
    assert li.quitar(m) == 0, "idempotente"


def test_la_enumeracion_de_alimentos_sigue_cerrandose_con_y():
    """Lo que la regla SÍ tiene que seguir haciendo: «A, B y la leche» → «A y B»."""
    m = {"name": "Batido de mango con yogur griego",
         "ingredients": ["150 g de mango", "80 g de guineo", "5 ml de leche descremada", "1 taza de yogurt griego"],
         "ingredients_raw": ["150 g de mango", "80 g de guineo", "5 ml de leche descremada", "240 g de Yogurt griego"],
         "recipe": ["Mise en place: pela el mango y el guineo.",
                    "Montaje: licúa el yogurt, el mango, el guineo y la leche descremada hasta que quede cremoso."]}
    assert li.quitar(m) == 1
    assert m["recipe"][1] == "Montaje: licúa el yogurt, el mango y el guineo hasta que quede cremoso."


def test_ancla():
    src = (_BACKEND / "leche_infima.py").read_text(encoding="utf-8")
    assert "tooltip-anchor: P1-PLAN-LOTE-914" in src


def test_una_orden_tras_la_coma_no_es_un_alimento_de_la_enumeracion():
    """«…, espolvorea la canela y la leche»: lo que sigue a la coma es una ORDEN, no el segundo alimento de una lista. La
    primera versión del arreglo llevaba el límite de palabra roto (un heredoc lo volvió el carácter 0x08) y este caso
    seguía saliendo «…un tazón y espolvorea la canela»: frase coja, TODO O NADA, la leche se quedaba."""
    m = {"name": "Bowl de yogurt con guineo y canela",
         "ingredients": ["1 taza de yogurt griego", "80 g de guineo", "5 ml de leche descremada", "¼ cdta de canela en polvo"],
         "ingredients_raw": ["240 g de Yogurt griego", "80 g de guineo", "5 ml de leche descremada", "0.25 cdta de canela"],
         "recipe": ["Mise en place: pela el guineo y córtalo en rodajas.",
                    "Montaje: sirve el yogurt con el guineo en un tazón, espolvorea la canela y la leche descremada."]}
    assert li.quitar(m) == 1
    assert m["recipe"][1] == "Montaje: sirve el yogurt con el guineo en un tazón, espolvorea la canela."


def test_el_fichero_no_lleva_caracteres_de_control():
    crudo = (_BACKEND / "leche_infima.py").read_bytes()
    assert b"\x08" not in crudo and b"\r" not in crudo
