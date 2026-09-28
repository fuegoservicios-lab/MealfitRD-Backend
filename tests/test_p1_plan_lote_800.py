# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-800 · 2026-09-28] V7f no acredita a un víver la cocción de otro víver ni la del huevo.

Corpus de la cola 744 (autocorrector de fruta dulce + salado → batata): «corta ¼ batata mediana en cubos … hierve el
plátano en agua con sal durante 15-18 min, hasta que un cuchillo entre sin fuerza … revuelve hasta que estén
completamente cuajados … sirve el mangú … y batata fresco aparte». V7f la daba por cocida y la batata llegaba cruda.
"""
from __future__ import annotations

import json
import pathlib

import culinary_coherence as cc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def _catalogo():
    return json.loads((_BACKEND / "scripts" / "data" / "culinary_corpus_2026_09_12.json")
                      .read_text(encoding="utf-8"))["catalogo_filas"]


_IDX = cc.build_culinary_index(_catalogo())


def _sin(ingredientes, pasos):
    return [f for f, _c in cc.alimentos_sin_coccion({"name": "Plato", "ingredients": ingredientes, "recipe": pasos}, _IDX)]


def test_la_batata_del_mangu_no_la_cuece_el_platano():
    sin = _sin(["1 plátano verde", "3 huevos", "½ cebolla morada", "¼ batata mediana"],
               ["Mise en place: pela y corta 1 plátano verde en trozos; pica ½ cebolla morada y corta ¼ batata mediana "
                "(60 g) en cubos.",
                "El Toque de Fuego: hierve el plátano en agua con sal durante 15-18 min, hasta que un cuchillo entre sin "
                "fuerza; escúrrelos y májalos. En una sartén a fuego medio, cocina la cebolla con el aceite 3-4 min y "
                "añade los huevos; revuelve hasta que estén completamente cuajados.",
                "Montaje: sirve el mangú con la cebolla y los huevos encima, y batata fresca aparte."])
    assert "Batata" in sin, sin


def test_hornea_sin_nombrar_nada_sigue_contando():
    sin = _sin(["1 batata mediana"],
               ["Mise en place: corta la batata en bastones.",
                "El Toque de Fuego: colócalos en una bandeja; hornea unos 20-25 minutos, hasta que estén tiernos.",
                "Montaje: sirve."])
    assert "Batata" not in sin, sin


def test_poner_agua_a_hervir_no_cuece_la_batata():
    sin = _sin(["1 guineo verde", "50 g de queso de hoja", "½ batata mediana"],
               ["Mise en place: pela y corta en trozos 1 guineo verde; mide 50 g de queso de hoja, ½ batata mediana "
                "(85 g) y 1 cdta de aceite de oliva; prepara agua para hervir y sal al gusto.",
                "El Toque de Fuego: hierve el guineo verde en el agua con sal durante 15-18 min, hasta que esté tierno; "
                "escúrrelos y májalos hasta obtener un mangú suave; dora el queso de hoja en una sartén 2-3 min por lado.",
                "Montaje: sirve el mangú con el queso dorado y batata fresca cortada en cubos."])
    assert "Batata" in sin, sin
