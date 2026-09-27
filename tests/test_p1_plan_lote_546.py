# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-546 · 2026-09-27] El huevo cuaja; y el caldo de hidratar un grano no hace guiso.

Batería real (rechaza pescado y berenjena, «Nada» de tiempo): el bulgur hidratado con «agua caliente (o caldo)» se leyó
como olla y el cerrador escribió «Añade 3 huevos y 2 claras de huevo al guiso y cocínalo a fuego medio 12-15 minutos,
hasta que esté cocido por dentro».
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pasos_cerrador as pc  # noqa: E402

_COLA = (" al guiso y cocínalo a fuego medio 12-15 minutos, hasta que esté cocido por dentro; incorpóralo con cuidado "
         "para no deshacer el resto.")


def _bulgur():
    return {"name": "Bulgur tibio con queso fresco, pepino al limón y huevo",
            "recipe": ["Mise en place: mide 30 g de bulgur; corta 20 g de queso blanco fresco en cubos y 1 pepino en dados.",
                       "El Toque de Fuego: hidrata el bulgur con agua caliente (o caldo) en proporción 1:2 durante 8-10 "
                       "min, tapado, hasta que esté tierno y haya absorbido el líquido. Añade 3 huevos y 2 claras de huevo"
                       + _COLA,
                       "Montaje: sirve el bulgur con el pepino, el queso y el huevo."]}


def test_sin_guiso_el_huevo_va_revuelto_a_la_sarten():
    m = _bulgur()
    assert pc.guiso_sin_proteina(m) == 1
    paso = m["recipe"][1]
    assert "al guiso" not in paso and "12-15" not in paso, paso
    assert paso.endswith("Cocina 3 huevos y 2 claras de huevo revueltos en una sartén antiadherente a fuego medio 3-4 "
                         "minutos, hasta que cuajen."), paso


def test_el_caldo_de_hidratar_el_grano_no_es_olla():
    import graph_orchestrator as go
    from constants import strip_accents
    m = _bulgur()
    m["recipe"][1] = m["recipe"][1].split(" Añade ")[0]
    assert go._meal_is_stewy(m, strip_accents) is False
    guiso = {"name": "Pollo guisado", "recipe": ["El Toque de Fuego: sofríe y añade agua o caldo."]}
    assert go._meal_is_stewy(guiso, strip_accents) is True
    olla = {"name": "Arroz con pollo", "recipe": ["El Toque de Fuego: añade 2 tazas de agua o caldo, tapa y cocina a "
                                                  "fuego bajo 20 minutos."]}
    assert go._meal_is_stewy(olla, strip_accents) is True


def test_en_un_guiso_de_verdad_el_huevo_se_cuaja_tapado():
    m = {"name": "Habichuelas guisadas con huevo",
         "recipe": [f"El Toque de Fuego: sofríe la cebolla 3 minutos y añade las habichuelas. Añade 2 huevos{_COLA}"]}
    assert pc.guiso_sin_proteina(m) == 1
    assert m["recipe"][0].endswith("Añade 2 huevos al guiso, tapa y cocina a fuego bajo 5-6 minutos, hasta que cuajen."), \
        m["recipe"][0]


def test_el_pollo_y_las_claras_siguen_como_estaban():
    m = {"name": "Pollo guisado", "recipe": [f"El Toque de Fuego: sofríe. Añade pechuga de pollo{_COLA}"]}
    antes = list(m["recipe"])
    assert pc.guiso_sin_proteina(m) == 0 and m["recipe"] == antes


def test_ancla_en_el_criterio_del_guiso():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert 'blob = __import__("pasos_cerrador").sin_caldo_de_grano(blob)  # [P1-PLAN-LOTE-546]' in src
