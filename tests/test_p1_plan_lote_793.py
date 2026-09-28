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


# ── [P1-PLAN-LOTE-793 · ronda 1] (revisión, defecto 8) El test invertido no puede seguir diciendo lo contrario ──

_HREFLANG = Path(__file__).resolve().parent / "test_p2_seo_hreflang.py"


def test_el_test_de_hreflang_ya_no_dice_que_el_dop_no_se_toca():
    """`test_p2_seo_hreflang.py` se invirtió (ahora exige USD) pero su docstring seguía diciendo «No se
    toca el precio en DOP» y el test invertido seguía bajo la cabecera «Lo que NO se toca». Un test cuyo
    texto dice lo contrario de lo que comprueba invita a «arreglarlo» devolviendo el DOP."""
    src = _HREFLANG.read_text(encoding="utf-8")
    assert "No se toca el precio en DOP" not in src
    cabecera = src.find("# ── Lo que NO se toca")
    usd = src.find("def test_la_moneda_del_offer_es_la_del_apex_usd")
    assert usd != -1, "desapareció el test que fija USD en el Offer"
    assert not (cabecera != -1 and cabecera < usd and "# ──" not in src[cabecera + 5: usd]), (
        "el test que exige USD sigue bajo la cabecera «Lo que NO se toca»"
    )
