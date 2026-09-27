# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-490 · 2026-09-27] La hierba fresca pasa a orégano SECO en su medida.

Replay forzado de los días 21+ de la compra única (322 planes, 1.061 hierbas sustituidas): «¼ taza de cilantro picado»
salía «¼ taza de orégano», «75 g de cilantro» «75 g de orégano» y «½ ramita de cilantro» «½ de orégano», sin unidad. El
seco es unas tres veces más fuerte: 1 cda de fresca = 1 cdta de seca, tope 1 cda por línea. Y el seco no se pica.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import compra_unica as cu  # noqa: E402
import sustitucion_fresca as sf  # noqa: E402

_REQ = {"need_days": 21, "allow_frozen": False}


def test_la_medida_del_seco():
    for viejo, nuevo in (("¼ taza de cilantro picado", "1 cda de orégano"),
                         ("2 cdas de cilantro picado", "2 cdtas de orégano"),
                         ("1 cda de perejil picado", "1 cdta de orégano"),
                         ("1 cdta de cilantro picado", "¼ cdta de orégano"),
                         ("½ ramita de cilantro", "¼ cdta de orégano"),
                         ("2 ramitas de cilantro", "1 cdta de orégano"),
                         ("75 g de cilantro", "1 cda de orégano"),
                         ("1½ cucharadas de cilantro picado", "1½ cdtas de orégano")):
        r = cu.sustituir_linea(viejo, 20, _REQ, gramos_de=lambda _t: None)
        assert r and r[0] == nuevo, (viejo, r)


def test_sin_cantidad_queda_el_nombre():
    r = cu.sustituir_linea("Cilantro fresco", 20, _REQ, gramos_de=lambda _t: None)
    assert r and r[1] == "oregano" and "taza" not in r[0], r


def test_el_seco_se_mide_y_no_va_picado():
    m = {"name": "Pollo guisado con cilantro", "ingredients": ["2 cdas de cilantro picado"],
         "ingredients_raw": ["2 cdas de cilantro picado"],
         "recipe": ["Mise en place: pica 2 cdas de cilantro y reserva.",
                    "Montaje: termina con el cilantro picado por encima."]}
    sf.sustituir_en_plato(m, 0, "2 cdas de cilantro picado", "2 cdtas de orégano", "oregano")
    assert m["recipe"][0] == "Mise en place: mide 2 cdtas de orégano y reserva.", m["recipe"][0]
    assert m["recipe"][1] == "Montaje: termina con el orégano por encima.", m["recipe"][1]
