# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-427 · 2026-09-26] El víver que reemplaza al arroz de noche, con su artículo y su técnica.

Batería REAL sobre el 424 (familia de 4, cena del día 2): «lava y mide el Yuca», «cocina el Yuca en agua con sal según
el paquete hasta que quede suelto», «Incorpora el Yuca cocido». Corpus: 10 comidas con «el Batata/el Yuca» o la
mayúscula del catálogo."""
from __future__ import annotations

import pathlib

import pasos_cerrador as pc
import pasos_sustitucion as ps

_BACKEND = pathlib.Path(__file__).resolve().parents[1]

_CENA = ["Mise en place: lava y mide el Yuca; corta el tomate y la cebolla en cubos.",
         "El Toque de Fuego: cocina el Yuca en agua con sal según el paquete hasta que quede suelto. Sofríe la cebolla 4 "
         "minutos. Incorpora el Yuca cocido y cocina 3 minutos.",
         "Montaje: sirve el Yuca con su salsa."]


def test_la_yuca_con_su_articulo_y_su_hervor():
    m = {"ingredients": ["100 g de yuca", "½ cebolla"], "recipe": list(_CENA)}
    assert pc.tuberculo_del_arroz(m) == 3
    assert m["recipe"] == [
        "Mise en place: pela y corta la yuca en trozos; corta el tomate y la cebolla en cubos.",
        "El Toque de Fuego: hierve la yuca en agua con sal 20-25 minutos, hasta que el cuchillo entre sin fuerza. Sofríe "
        "la cebolla 4 minutos. Incorpora la yuca cocida y cocina 3 minutos.",
        "Montaje: sirve la yuca con su salsa."]
    assert pc.tuberculo_del_arroz(m) == 0


def test_en_origen_el_cambio_del_arroz_de_noche_ya_sale_bien():
    pasos, n = ps.tecnica_del_sustituto(list(_CENA), "Yuca")
    assert n == 3 and pasos[1].startswith("El Toque de Fuego: hierve la yuca en agua con sal 20-25 minutos")
    b, _ = ps.tecnica_del_sustituto(["El Toque de Fuego: calienta el Batata cocido en el microondas."], "Batata")
    assert b == ["El Toque de Fuego: calienta la batata cocida en el microondas."]
    n1, _ = ps.tecnica_del_sustituto(["Mise en place: enjuaga el Ñame."], "Ñame")
    assert n1 == ["Mise en place: pela y corta el ñame en trozos."]


def test_lo_que_no_viene_del_arroz_no_se_toca():
    pasos = ["El Toque de Fuego: hierve la yuca 20-25 minutos. Cocina el arroz según el paquete hasta que quede suelto."]
    m = {"ingredients": ["100 g de yuca", "60 g de arroz"], "recipe": list(pasos)}
    assert pc.tuberculo_del_arroz(m) == 0 and m["recipe"] == pasos
    assert pc.tuberculo_del_arroz({"ingredients": ["2 huevos"], "recipe": list(_CENA)}) == 0


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cerrador").tuberculo_del_arroz(meal)  # [P1-PLAN-LOTE-427]' in src
