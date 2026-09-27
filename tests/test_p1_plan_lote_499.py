# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-499 · 2026-09-27] El guineo no tiene semillas que separar.

La IA escribe «separa 60 g de semillas de guineo… distribuye encima las semillas de guineo» (tres planes de adulto mayor
con HTA re-encadenados) y «extrae las semillas de 1 guineo» (batería real del dueño, día 13).
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pasos_cerrador as pc  # noqa: E402


def test_el_guineo_va_en_rodajas():
    m = {"recipe": ["Mise en place: mide ½ taza de yogurt natural sin azúcar (160 g) y separa 60 g de semillas de guineo.",
                    "Montaje: coloca yogurt natural en un tazón y distribuye encima las semillas de guineo. Termina con "
                    "queso cottage."]}
    assert pc.guineo_sin_semillas(m) == 2
    assert m["recipe"][0] == ("Mise en place: mide ½ taza de yogurt natural sin azúcar (160 g) y corta 60 g de guineo "
                              "en rodajas."), m["recipe"][0]
    assert "distribuye encima las rodajas de guineo" in m["recipe"][1], m["recipe"][1]


def test_extrae_las_semillas_de_1_guineo():
    m = {"recipe": ["Mise en place: mide 15 g de yogurt natural, extrae las semillas de 1 guineo y pesa 10 g de maní."]}
    pc.guineo_sin_semillas(m)
    assert m["recipe"][0] == "Mise en place: mide 15 g de yogurt natural, corta 1 guineo en rodajas y pesa 10 g de maní.", \
        m["recipe"][0]


def test_la_hoja_del_guiso_entra_al_final():
    """Replay de la cola sobre el 498: «Incorpora la espinaca al guiso y cocínala a fuego medio 8-10 minutos»."""
    m = {"recipe": ["El Toque de Fuego: cocina la cebolla 3 minutos. Incorpora espinaca al guiso y cocínalos a fuego medio "
                    "12-15 minutos, hasta que estén cocidos por dentro; incorpóralos con cuidado para no deshacer el resto."]}
    pc.guiso_sin_proteina(m)
    assert m["recipe"][0].endswith("Incorpora la espinaca al guiso en los últimos 2-3 minutos, hasta que se ablande."), \
        m["recipe"][0]


def test_la_hierba_va_al_final_y_la_remolacha_cuece_como_viver():
    """Replay de la cola sobre el 498: «Añade el cilantro al guiso y cocínalo a fuego medio 8-10 minutos, hasta que esté
    tierno» (3) y la remolacha cruda en 8-10 minutos."""
    cola = (" al guiso y cocínalo a fuego medio 12-15 minutos, hasta que esté cocido por dentro; incorpóralo con cuidado "
            "para no deshacer el resto.")
    m = {"recipe": ["El Toque de Fuego: sofríe la cebolla 3 minutos. Añade cilantro" + cola]}
    pc.guiso_sin_proteina(m)
    assert m["recipe"][0].endswith("Añade el cilantro al guiso al final, justo antes de servir."), m["recipe"][0]
    m = {"recipe": ["El Toque de Fuego: sofríe la cebolla 3 minutos. Añade remolacha" + cola]}
    pc.guiso_sin_proteina(m)
    assert "Añade la remolacha al guiso y cocínala a fuego medio 25-30 minutos, hasta que esté tierna." in \
        m["recipe"][0], m["recipe"][0]


def test_otras_semillas_no_se_tocan():
    m = {"recipe": ["Montaje: espolvorea 5 g de semillas de linaza y 10 g de semillas de chía."]}
    antes = list(m["recipe"])
    assert pc.guineo_sin_semillas(m) == 0 and m["recipe"] == antes
