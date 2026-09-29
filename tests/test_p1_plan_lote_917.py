# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-917 · 2026-09-29] La línea de la lista que perdió su cifra la recupera.

Batería real de embarazo rdv864 (día 1, cena): la lista visible decía «g de nabo pelado y cortado en rodajas de 1 cm» y la
del motor «264.35 g de nabo pelado y cortado en rodajas de 1 cm». La cifra se pierde dentro del grafo, antes del escudo (ya
falta en `pipeline_result`), y sin ella nadie puede leer la línea: el contrato no sincroniza el paso («corta 250 g de
nabo») y el escudo no la reescala. Red de seguridad al principio del contrato: la cifra vuelve de la línea del motor del
mismo alimento; si el motor tampoco la tiene, del paso que lo mide.
"""
from __future__ import annotations

import pathlib

import linea_con_su_cifra as lcc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def _nabo():
    return {"name": "Nabo asado a la parrilla con queso blanco fresco, aguacate y edamame",
            "ingredients": ["g de nabo pelado y cortado en rodajas de 1 cm", "20 g de queso blanco fresco pasteurizado",
                            "½ aguacate", "60 g de edamame cocido"],
            "ingredients_raw": ["264.35 g de nabo pelado y cortado en rodajas de 1 cm",
                                "20 g de queso blanco fresco pasteurizado", "0.48 aguacate", "60 g de edamame cocido"],
            "recipe": ["Mise en place: pela y corta 250 g de nabo en rodajas de 1 cm; mide 20 g de queso blanco fresco.",
                       "Montaje: sirve el nabo asado con el queso y el aguacate."]}


def test_la_cifra_vuelve_de_la_linea_del_motor_redondeada_a_5_g():
    m = _nabo()
    assert lcc.restaurar(m) == 1
    assert m["ingredients"][0] == "265 g de nabo pelado y cortado en rodajas de 1 cm"
    assert m["ingredients_raw"][0] == "264.35 g de nabo pelado y cortado en rodajas de 1 cm", "el motor no se toca"
    assert m["ingredients"][1:] == _nabo()["ingredients"][1:]
    assert lcc.restaurar(m) == 0, "idempotente"


def test_si_el_motor_tampoco_la_tiene_la_da_el_paso_que_lo_mide():
    m = {"name": "Batata asada al microondas con atún al limón",
         "ingredients": ["1 batata mediana", "G de queso blanco fresco (extensor opcional)", "245 g de atún"],
         "ingredients_raw": ["1 batata mediana", "G de queso blanco fresco (extensor opcional)", "245 g de atún"],
         "recipe": ["Mise en place: pincha 275 g de batata; escurre 245 g de atún.",
                    "Montaje: abre la batata y sirve el atún encima; añade 30 g de queso blanco fresco solo si necesitas "
                    "extender la porción."]}
    assert lcc.restaurar(m) == 1
    assert m["ingredients"][1] == "30 g de queso blanco fresco (extensor opcional)"
    assert m["ingredients_raw"][1] == "30 g de queso blanco fresco (extensor opcional)"


def test_sin_de_donde_sacarla_la_linea_se_queda():
    m = {"name": "Ensalada", "ingredients": ["g de nabo pelado", "1 tomate"], "ingredients_raw": ["g de nabo pelado", "1 tomate"],
         "recipe": ["Montaje: mezcla el nabo con el tomate."]}
    assert lcc.restaurar(m) == 0 and m["ingredients"][0] == "g de nabo pelado"


def test_lo_que_no_es_una_linea_sin_cifra_no_se_toca(monkeypatch):
    intactas = ["Pizca de sal y pimienta", "Hojas de menta fresca (opcional)", "Sal al gusto", "Gofio de maíz 30 g",
                "250 g de nabo", "Granola 20 g"]
    m = {"name": "x", "ingredients": list(intactas), "ingredients_raw": list(intactas), "recipe": ["Montaje: sirve."]}
    assert lcc.restaurar(m) == 0 and m["ingredients"] == intactas
    monkeypatch.setenv("MEALFIT_LIST_LINE_KEEPS_NUMBER", "false")
    m = _nabo()
    assert lcc.restaurar(m) == 0 and m["ingredients"][0].startswith("g de nabo")


def test_dos_lineas_del_motor_del_mismo_alimento_no_se_adivina():
    m = _nabo()
    m["ingredients_raw"].append("100 g de nabo pelado y cortado en rodajas de 1 cm")
    assert lcc.restaurar(m) == 0 and m["ingredients"][0].startswith("g de nabo")


def test_ancla_al_principio_del_contrato():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    i917 = src.index('__import__("linea_con_su_cifra").restaurar(meal)  # [P1-PLAN-LOTE-917]')
    i319 = src.index('__import__("pasos_cantidades").quitar_trazas(meal)  # [P1-PLAN-LOTE-319]')
    assert i917 < i319, "antes de todo lo que lee la lista"
    modulo = (_BACKEND / "linea_con_su_cifra.py")
    assert "tooltip-anchor: P1-PLAN-LOTE-917" in modulo.read_text(encoding="utf-8") and b"\x08" not in modulo.read_bytes()
