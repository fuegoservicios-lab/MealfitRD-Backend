# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-793 · 2026-09-28] (G95) La oferta del JSON-LD de la app va en USD, igual que el apex.

`frontend/index.html` declaraba `"priceCurrency": "DOP"` en el `Offer` del `SoftwareApplication`, mientras
`bioboros.com` (repo del landing, `build.py` desde `contract/commercial.json`) declara la MISMA oferta en `USD`.
Dos datos estructurados del mismo producto con dos monedas distintas: un buscador no sabe cuál creer, y la
app se vende en seis países (el cobro es en USD vía PayPal). El dueño decidió el 28-sep: USD, como el apex.

`test_p2_seo_hreflang.py` fijaba DOP a propósito («P1-28 es decisión del dueño, no de este P-fix»); esa
decisión ya está tomada, así que aquel test se invirtió y éste ancla la forma: el bloque se PARSEA como JSON
(un `priceCurrency` bien escrito dentro de un JSON roto no le sirve a nadie).

tooltip-anchor: P1-PLAN-LOTE-793
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_INDEX = Path(__file__).resolve().parent.parent.parent / "frontend" / "index.html"


@pytest.fixture(scope="module")
def bloques() -> list:
    if not _INDEX.is_file():
        pytest.skip("index.html no está en este árbol")
    html = _INDEX.read_text(encoding="utf-8")
    crudos = re.findall(r'<script type="application/ld\+json">(.*?)</script>', html, re.S)
    assert crudos, "index.html ya no tiene datos estructurados"
    return [json.loads(c) for c in crudos]


def _ofertas(bloques):
    return [b["offers"] for b in bloques if b.get("@type") == "SoftwareApplication" and "offers" in b]


def test_la_oferta_existe_y_es_json_valido(bloques):
    assert len(_ofertas(bloques)) == 1


def test_la_oferta_va_en_usd_como_el_apex(bloques):
    (oferta,) = _ofertas(bloques)
    assert oferta["@type"] == "Offer"
    assert oferta["priceCurrency"] == "USD", (
        "La oferta de la app volvió a una moneda distinta de la del apex (USD): dos datos estructurados "
        "del mismo producto con dos monedas."
    )
    assert oferta["price"] == "0", "el plan de entrada sigue siendo gratis"


def test_ninguna_moneda_dominicana_en_los_datos_estructurados(bloques):
    assert "DOP" not in json.dumps(bloques)
