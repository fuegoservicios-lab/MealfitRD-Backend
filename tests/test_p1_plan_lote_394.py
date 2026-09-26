# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-394 · 2026-09-26] Un hervor no lo mide la plancha de otra frase.

Batería REAL sobre el 379 (estatina, día 3): «cocina la cebada en agua según las instrucciones del paquete, hasta que
esté tierna, aproximadamente 3-4 min por lado a fuego medio-alto» — el recorte de tiempos eligió la técnica con el paso
entero (la otra frase dice «sartén»). Corpus: «cocina el bulgur en agua hirviendo 1-2 min de licuado a velocidad altautos»."""
from __future__ import annotations

import ast
import pathlib
import re

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_el_hervor_recupera_su_tiempo():
    m = {"recipe": [
        "El Toque de Fuego: cocina la cebada en agua según las instrucciones del paquete, hasta que esté tierna, "
        "aproximadamente 3-4 min por lado a fuego medio-alto. En una sartén a fuego medio, calienta el aceite; cocina la "
        "cebolla 2 min.",
        "El Toque de Fuego: cocina el bulgur en agua hirviendo 1-2 min de licuado a velocidad altautos hasta que esté tierno; "
        "licúa la salsa.",
        "El Toque de Fuego: cocina el arroz blanco en agua con sal hasta que esté suelto, unos 2-3 min por lado a fuego "
        "medioutos.",
        "El Toque de Fuego: cocina la cebada en agua con sal a fuego bajo durante 6-8 min hasta dorar, hasta que esté tierna."]}
    assert pc.hervor_con_su_tiempo(m) == 4
    assert m["recipe"][0].startswith("El Toque de Fuego: cocina la cebada en agua según las instrucciones del paquete, hasta "
                                     "que esté tierna, aproximadamente 30-40 min. En una sartén"), m["recipe"][0]
    assert m["recipe"][1].startswith("El Toque de Fuego: cocina el bulgur en agua hirviendo 10-12 min hasta que esté tierno;")
    assert m["recipe"][2] == ("El Toque de Fuego: cocina el arroz blanco en agua con sal hasta que esté suelto, unos 15-20 "
                              "min.")
    assert m["recipe"][3] == ("El Toque de Fuego: cocina la cebada en agua con sal a fuego bajo durante 30-40 min, hasta "
                              "que esté tierna.")
    assert pc.hervor_con_su_tiempo(m) == 0


def test_la_plancha_de_verdad_no_se_toca():
    pasos = ["El Toque de Fuego: sella el pollo 3-4 min por lado a fuego medio-alto.",
             "El Toque de Fuego: hierve la yautía 15-18 min. Sella el pescado 3-4 min por lado a fuego medio-alto.",
             "El Toque de Fuego: licúa la avena 1-2 min de licuado a velocidad alta."]
    m = {"recipe": list(pasos)}
    assert pc.hervor_con_su_tiempo(m) == 0 and m["recipe"] == pasos


def test_el_recorte_ya_no_parte_minutos():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    linea = next(l for l in src.splitlines() if l.startswith("_TT_MIN_RE = "))
    literal = linea.split("_re.compile(", 1)[1].split(", _re.I")[0]          # un literal r"…" del propio repo
    rx = re.compile(ast.literal_eval(literal), re.I)
    assert rx.search("cocina 35-40 minutos").group(0) == "35-40 minutos"


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").hervor_con_su_tiempo(meal)  # [P1-PLAN-LOTE-394]' in src
