# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-911 · 2026-09-29] Lo que el paso escurre no viene «escurrido».

El paso copia el nombre de la LISTA con su descriptor: «escurre 1 taza de lentejas de lata escurridas (192 g)» (batería real
de embarazo rdv801) y «escurre ⅔ taza de habichuelas rojas de lata, escurridas y enjuagadas (≈120 g)» (rdv864). Corpus del
VPS (5.336 comidas, salida de la cola con el 888): 9 cláusulas con «escurre» y el participio en el mismo objeto.
"""
from __future__ import annotations

import pathlib

import escurre_una_vez as euv

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def _meal(mise):
    return {"ingredients": ["1 taza de lentejas de lata escurridas"],
            "recipe": [mise, "El Toque de Fuego: sofríe la cebolla 5 min e incorpora las lentejas.",
                       "Montaje: sirve el guiso sobre el arroz."]}


def test_las_lentejas_de_lata_escurridas_se_escurren_una_vez():
    m = _meal("Mise en place: enjuaga 40 g de arroz blanco; escurre 1 taza de lentejas de lata escurridas (192 g); pela y "
              "corta la zanahoria en cubitos.")
    assert euv.limpiar(m) == 1
    assert m["recipe"][0] == ("Mise en place: enjuaga 40 g de arroz blanco; escurre 1 taza de lentejas de lata (192 g); "
                              "pela y corta la zanahoria en cubitos.")
    assert euv.limpiar(m) == 0, "idempotente"


def test_lo_enjuagado_pasa_al_verbo():
    m = _meal("Mise en place: corta 105 g de plátano verde en trozos pequeños y escurre ⅔ taza de habichuelas rojas de "
              "lata, escurridas y enjuagadas (≈120 g); pica ½ tomate.")
    assert euv.limpiar(m) == 1
    assert m["recipe"][0] == ("Mise en place: corta 105 g de plátano verde en trozos pequeños y escurre y enjuaga ⅔ taza "
                              "de habichuelas rojas de lata (≈120 g); pica ½ tomate.")


def test_cocidas_y_escurridas_se_queda_en_cocidas():
    m = _meal("Mise en place: enjuaga y escurre ¾ taza de habichuelas negras cocidas y escurridas (155 g); pica el ajo.")
    assert euv.limpiar(m) == 1
    assert m["recipe"][0] == "Mise en place: enjuaga y escurre ¾ taza de habichuelas negras cocidas (155 g); pica el ajo."


def test_el_verbo_que_ya_enjuaga_no_se_repite():
    m = _meal("Mise en place: escurre y enjuaga 2¼ tazas de garbanzos en lata escurridos y enjuagados (425 g); ralla el "
              "repollo.")
    assert euv.limpiar(m) == 1
    assert m["recipe"][0] == "Mise en place: escurre y enjuaga 2¼ tazas de garbanzos en lata (425 g); ralla el repollo."


def test_dentro_del_parentesis_de_la_lata():
    m = _meal("Mise en place: enjuaga y escurre 1½ tazas de garbanzos cocidos (de lata, enjuagados y escurridos) (235 g); "
              "pica la cebolla.")
    assert euv.limpiar(m) == 1
    assert m["recipe"][0] == ("Mise en place: enjuaga y escurre 1½ tazas de garbanzos cocidos (de lata) (235 g); pica la "
                              "cebolla.")


def test_lo_que_no_se_toca(monkeypatch):
    intactos = [
        # el verbo es «mide»: el descriptor dice cómo viene, no repite una orden
        "Mise en place: mide ½ taza de habichuelas rojas cocidas (de lata, enjuagadas y escurridas) (90 g); exprime ½ limón.",
        # el participio es de OTRA orden de la misma frase
        "Mise en place: escurre la pasta; mezcla el atún con las habichuelas escurridas.",
        # sin participio
        "Mise en place: escurre 1 lata de atún y pica la cebolla.",
    ]
    for mise in intactos:
        m = _meal(mise)
        assert euv.limpiar(m) == 0 and m["recipe"][0] == mise, mise
    nota = {"ingredients": [], "recipe": ["🌱 Nota del Nutricionista AI: escurre las lentejas escurridas de la lata."]}
    assert euv.limpiar(nota) == 0
    monkeypatch.setenv("MEALFIT_DRAIN_ONCE", "false")
    m = _meal("Mise en place: escurre 1 taza de lentejas de lata escurridas (192 g).")
    assert euv.limpiar(m) == 0


def test_ancla_tras_el_442():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    i442 = src.index('__import__("pasos_cantidades").fresco_no_se_escurre(meal)  # [P1-PLAN-LOTE-442]')
    i911 = src.index('__import__("escurre_una_vez").limpiar(meal)  # [P1-PLAN-LOTE-911]')
    i634 = src.index('__import__("doble_punto").limpiar(meal)  # [P1-PLAN-LOTE-634]')
    assert i442 < i911 < i634
    assert "tooltip-anchor: P1-PLAN-LOTE-911" in (_BACKEND / "escurre_una_vez.py").read_text(encoding="utf-8")
