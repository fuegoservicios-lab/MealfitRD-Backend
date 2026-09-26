# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-363 · 2026-09-26] «4 huevos con 3 yemas»: la cantidad del ingrediente Y el resto de lo dicho.

El dueño: «si escribo "4 huevos con 3 yemas" tiene que entender que me comí 4 huevos pero le quité una yema». El 362
leía el número del texto («4») y ponía el huevo en 4: la yema quitada se perdía. Ahora el ajuste por texto devuelve
DOS cosas: `cantidad` (cuántas unidades del ingrediente de la duda: 4) y el `ajuste` TOTAL del plato con todo lo dicho
(+1 huevo −1 yema). El frontend pone el ingrediente en 4 y aplica aparte lo que no cabe en «4 unidades».
Tooltip-anchor: P1-PLAN-LOTE-363
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]

_REQ = {
    "plato": "Tortilla con panecillo",
    "macros": {"calories": 490},
    "pregunta": "¿De cuántos huevos?",
    "opciones": [{"texto": "2 huevos", "supuesta": True, "ajuste": {}}],
    "respuesta": "4 huevos con 3 yemas",
    "ingrediente": {"nombre": "Huevo revuelto", "cantidad": 2, "unidad": "unidad"},
}


def test_el_modelo_ve_el_ingrediente_y_se_le_pide_la_cantidad():
    import ajuste_de_duda as ad
    msg = ad.mensaje_para_el_modelo(ad.PeticionAjuste(**_REQ))
    assert "Ingrediente de la duda: Huevo revuelto — 2 unidad" in msg
    assert "cantidad" in ad._SISTEMA and "yema" in ad._SISTEMA


def test_la_cantidad_viaja_acotada_y_es_opcional():
    import ajuste_de_duda as ad
    assert ad.normalizar({"calories": 135, "cantidad": 4})["cantidad"] == 4
    assert "cantidad" not in ad.normalizar({"calories": 10, "cantidad": None})
    assert "cantidad" not in ad.normalizar({"calories": 10, "cantidad": 0})
    assert "cantidad" not in ad.normalizar({"calories": 10, "cantidad": 5000})


def test_el_ingrediente_es_opcional_en_la_peticion():
    import ajuste_de_duda as ad
    req = dict(_REQ)
    req.pop("ingrediente")
    assert "Ingrediente de la duda" not in ad.mensaje_para_el_modelo(ad.PeticionAjuste(**req))


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 363
