# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-379 · 2026-09-26] La licuadora va antes del fuego, y ni la carne ni el huevo van a ella.

Batería REAL sobre el 376 (perfil del dueño, día 1): «…cocina los panqueques en sartén… 2-3 minutos por lado… Agrega
queso cottage a la licuadora y licúa hasta integrar». Corpus: «Agrega pechuga de pavo a la licuadora» en unas arepitas y
«Agrega huevo a la licuadora» en un bowl frío."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_el_lacteo_sobre_la_masa_ya_cocinada_va_al_lado():
    m = {"name": "Panqueques de avena con mango, lechosa, queso cottage y yogurt",
         "ingredients": ["15 g de avena", "20 g de queso cottage", "60 g de mango"],
         "recipe": ["Mise en place: mide 15 g de avena.",
                    "El Toque de Fuego: licúa la avena, el yogurt y la leche; cocina los panqueques en sartén antiadherente "
                    "a fuego medio 2-3 minutos por lado. Agrega queso cottage a la licuadora y licúa hasta integrar.",
                    "Montaje: sirve los panqueques con el mango y la lechosa por encima."]}
    assert pc.licuadora_a_tiempo(m) == 1
    assert m["recipe"][1].endswith("2-3 minutos por lado. Sirve queso cottage al lado para acompañar."), m["recipe"][1]
    # si el plato ya lo sirve aparte, la frase sobra
    t = {"name": "Tarta fina de calabacín y queso fresco batido con yogurt griego entero",
         "ingredients": ["66 g de yogurt griego entero"],
         "recipe": ["El Toque de Fuego: Hornea 18-20 minutos. Agrega yogurt griego entero a la licuadora y licúa hasta integrar.",
                    "Montaje: córtala en porciones. Acompaña con yogurt griego entero."]}
    assert pc.licuadora_a_tiempo(t) == 1 and t["recipe"][0] == "El Toque de Fuego: Hornea 18-20 minutos."


def test_ni_la_carne_ni_el_huevo_van_a_la_licuadora():
    a = {"name": "Arepitas tiernas de maíz con coles de Bruselas y pechuga de pavo",
         "ingredients": ["18 g de pechuga de pavo"],
         "recipe": ["El Toque de Fuego: cocina las coles 10-12 min. Agrega pechuga de pavo a la licuadora y licúa hasta "
                    "integrar. Añade Pechuga de pavo al guiso y cocínala a fuego medio 12-15 minutos."]}
    assert pc.licuadora_a_tiempo(a) == 1
    assert a["recipe"][0] == ("El Toque de Fuego: cocina las coles 10-12 min. Añade Pechuga de pavo al guiso y cocínala a "
                              "fuego medio 12-15 minutos."), a["recipe"][0]
    b = {"name": "Bowl frío de guanábana con semillas y huevo", "ingredients": ["3 huevos", "100 g de guanábana"],
         "recipe": ["El Toque de Fuego: Tuesta el maní 2-3 minutos. Agrega huevo a la licuadora y licúa hasta integrar.",
                    "Montaje: Licúa la guanábana con la leche de coco. Sirve bien frío."]}
    assert pc.licuadora_a_tiempo(b) == 1
    assert b["recipe"][0].endswith("Cocina el huevo en la sartén a fuego medio hasta que la clara y la yema estén firmes "
                                   "y sírvelo al lado."), b["recipe"][0]
    c = {"name": "Batido de piña", "ingredients": ["65 g de atún claro en agua"],
         "recipe": ["Montaje: licúa la piña. 💪 Agrega atún en agua a la licuadora y licúa hasta integrar."]}
    assert pc.licuadora_a_tiempo(c) == 1 and c["recipe"][0].endswith("Sirve atún en agua al lado para acompañar.")


def test_el_batido_y_el_bowl_frio_conservan_su_licuadora():
    casos = [
        {"name": "Batido de guineo y avena", "ingredients": ["20 g de queso cottage"],
         "recipe": ["Montaje: licúa el guineo con la avena. Agrega queso cottage a la licuadora y licúa hasta integrar."]},
        {"name": "Yogurt natural batido con guineo y maní tostado", "ingredients": ["20 g de queso cottage"],
         "recipe": ["El Toque de Fuego: licúa el yogurt con el guineo; tuesta el maní en una sartén 2-3 minutos. Agrega "
                    "queso cottage a la licuadora y licúa hasta integrar."]},
        # el batido se licúa DESPUÉS de cocinar las arepitas: la licuadora está a tiempo (replay, owner_like h2)
        {"name": "Batido proteico frío de lechosa con maní, arepita de maíz y yogurt natural entero",
         "ingredients": ["80 g de yogurt griego entero"],
         "recipe": ["El Toque de Fuego: cocina las arepitas en sartén 3-4 minutos por lado. Licua lechosa, hielo y un "
                    "chorrito de agua hasta obtener un batido cremoso. Agrega yogurt natural entero a la licuadora y licúa "
                    "hasta integrar."]},
    ]
    for m in casos:
        antes = list(m["recipe"])
        assert pc.licuadora_a_tiempo(m) == 0 and m["recipe"] == antes, m["name"]


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").licuadora_a_tiempo(meal)  # [P1-PLAN-LOTE-379]' in src
