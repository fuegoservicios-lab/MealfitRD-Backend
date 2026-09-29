# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-886 · 2026-09-29] «suma pechuga de pollo desmenuzado» → «desmenuzada»: el participio concuerda con el
núcleo del alimento, no con su complemento.

Replay del escudo sobre las baterías (rdv592 celíaco, D1 cena): «sofríe el ajo… suma pechuga de pollo desmenuzado… arma el
pastelón alternando capas de papa, pechuga de pollo guisado». Corpus (426 planes): 33 pasos así («guisado», «desmenuzado»,
«horneado», «pechuga de pavo cocido»), leídos uno a uno junto a las 42 correcciones del replay.
"""
from __future__ import annotations

import pathlib

import participio_concuerda as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_el_participio_concuerda_con_el_corte_no_con_el_animal():
    casos = {
        "El Toque de Fuego: sofríe el ajo 30 s, suma pechuga de pollo desmenuzado y guisa 4-5 min.":
            "El Toque de Fuego: sofríe el ajo 30 s, suma pechuga de pollo desmenuzada y guisa 4-5 min.",
        "Montaje: sirve pechuga de pavo guisado con su jugo sobre las arepitas de maíz.":
            "Montaje: sirve pechuga de pavo guisada con su jugo sobre las arepitas de maíz.",
        "El Toque de Fuego: añade pechuga de pavo cocido, el limón y el cilantro.":
            "El Toque de Fuego: añade pechuga de pavo cocida, el limón y el cilantro.",
        "Montaje: reparte las pechugas de pollo horneados sobre el mapuey.":
            "Montaje: reparte las pechugas de pollo horneadas sobre el mapuey.",
        "Montaje: sirve la carne de res molido con el arroz.": "Montaje: sirve la carne de res molida con el arroz.",
        "Montaje: sirve la pierna de cerdo frito con yuca.": "Montaje: sirve la pierna de cerdo frita con yuca.",
    }
    for antes, despues in casos.items():
        assert pc.concordar_nucleo(antes) == despues, pc.concordar_nucleo(antes)
        m = {"recipe": [antes]}
        assert pc.concordar_pasos(m) == 1 and m["recipe"] == [despues]


def test_lo_que_ya_concuerda_o_no_es_un_corte_no_se_toca():
    for t in ("Montaje: sirve la pechuga de pollo desmenuzada con el arroz cocido.",
              "Montaje: coloca 2 lonjas de pavo ahumado sobre el pan.",
              "Montaje: sirve 1 rebanada de pan tostado.",
              "Montaje: sirve el caldo de pollo desmenuzado bien caliente.",
              "El Toque de Fuego: cocina la pechuga de pollo a la plancha 6-7 minutos por lado.",
              "Montaje: sirve la carne de res molida con el arroz."):
        assert pc.concordar_nucleo(t) == t, t


def test_notas_knob_y_ancla(monkeypatch):
    nota = "⚠️ Seguridad alimentaria: pechuga de pollo guisado bien caliente."
    m = {"recipe": [nota, "Montaje: sirve pechuga de pollo guisado."]}
    assert pc.concordar_pasos(m) == 1 and m["recipe"] == [nota, "Montaje: sirve pechuga de pollo guisada."]
    assert pc.concordar_pasos(m) == 0, "idempotente"
    monkeypatch.setenv("MEALFIT_PARTICIPLE_HEAD_AGREEMENT", "false")
    m = {"recipe": ["Montaje: sirve pechuga de pollo guisado."]}
    assert pc.concordar_pasos(m) == 0 and m["recipe"] == ["Montaje: sirve pechuga de pollo guisado."]
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("participio_concuerda").concordar_pasos(meal)  # [P1-PLAN-LOTE-880]' in src
    assert "tooltip-anchor: P1-PLAN-LOTE-886" in (_BACKEND / "participio_concuerda.py").read_text(encoding="utf-8")
