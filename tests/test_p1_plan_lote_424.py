# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-421..424 · 2026-09-26] (la prueba del 424 cubre los cuatro) Cuatro defectos de texto de la batería real sobre el 409 (perfil del dueño).

421 «Cocina 3 huevos y 2 claras de huevo a la plancha o hervidos» (una clara suelta no se hierve; corpus 21) · 422 «corta
5 g, exprime 1 limón» (el alimento se fue; 14) · 423 «mide 1½ tortas pequeñas de casabe» con 1 en la lista (4) · 424
«bate 4 claras de huevo con 4 claras de huevo (130 g)» (3)."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_421_el_cerrador_cuaja_las_claras():
    m = {"recipe": ["El Toque de Fuego: dora las arepitas 3-4 min. Cocina 3 huevos y 2 claras de huevo a la plancha o "
                    "hervidos y sírvelos como proteína del plato."]}
    assert pc.claras_del_cerrador(m) == 1
    assert m["recipe"][0].endswith("Cocina 3 huevos y 2 claras de huevo en la sartén, revueltos, hasta que cuajen por "
                                   "completo, y sírvelos como proteína del plato."), m["recipe"][0]
    s = {"recipe": ["Cocina 4 claras de huevo a la plancha o hervidas y sírvelas como proteína del plato."]}
    assert pc.claras_del_cerrador(s) == 1 and "en la sartén, revueltas," in s["recipe"][0] and "sírvelas" in s["recipe"][0]
    h = {"recipe": ["Cocina 2 huevos a la plancha o hervidos y sírvelos como proteína del plato."]}
    antes = list(h["recipe"])
    assert pc.claras_del_cerrador(h) == 0 and h["recipe"] == antes                      # sin claras, se queda


def test_422_la_migaja_sin_alimento_sale():
    m = {"recipe": ["Mise en place: corta 1 pepino en cubos finos; corta 5 g, exprime 1 limón y pica 1 cda de cilantro.",
                    "Mise en place: desmenuza 45 g de queso blanco fresco y mide 30 g. Pela y corta 130 g de yuca.",
                    "Mise en place: pica ½ cebolla, lava y corta 5 g; bate 3 huevos."]}
    assert pc.migaja_sin_alimento(m) == 3
    assert m["recipe"] == ["Mise en place: corta 1 pepino en cubos finos, exprime 1 limón y pica 1 cda de cilantro.",
                           "Mise en place: desmenuza 45 g de queso blanco fresco. Pela y corta 130 g de yuca.",
                           "Mise en place: pica ½ cebolla; bate 3 huevos."], m["recipe"]
    ok = {"recipe": ["Mise en place: corta 5 g de cilantro, mide 30 g de queso."]}
    antes = list(ok["recipe"])
    assert pc.migaja_sin_alimento(ok) == 0 and ok["recipe"] == antes


def test_423_las_tortas_de_casabe_de_la_lista():
    m = {"ingredients": ["1 torta pequeña de casabe", "25 g de maní"],
         "recipe": ["Mise en place: corta la lechosa y mide 1½ tortas pequeñas de casabe y 25 g de maní.",
                    "El Toque de Fuego: tuesta el casabe 1-2 min."]}
    assert pc.tortas_de_casabe(m) == 1
    assert m["recipe"][0] == "Mise en place: corta la lechosa y mide 1 torta pequeña de casabe y 25 g de maní."
    assert pc.tortas_de_casabe(m) == 0
    d = {"ingredients": ["2 tortas pequeñas de casabe"], "recipe": ["Mise en place: mide 1 torta pequeña de casabe."]}
    assert pc.tortas_de_casabe(d) == 1 and d["recipe"][0] == "Mise en place: mide 2 tortas pequeñas de casabe."


def test_424_las_claras_una_vez():
    m = {"recipe": ["Mise en place: bate 4 claras de huevo con 4 claras de huevo (130 g) y sal al gusto."]}
    assert pc.claras_una_vez(m) == 1
    assert m["recipe"][0] == "Mise en place: bate 4 claras de huevo (130 g) y sal al gusto."
    ok = {"recipe": ["Mise en place: bate 2 claras de huevo con 1 huevo."]}
    assert pc.claras_una_vez(ok) == 0


def test_anclas():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    for fn, n in (("claras_del_cerrador", 421), ("migaja_sin_alimento", 422), ("tortas_de_casabe", 423),
                  ("claras_una_vez", 424)):
        assert f'__import__("pasos_cantidades").{fn}(meal)  # [P1-PLAN-LOTE-{n}]' in src, fn
