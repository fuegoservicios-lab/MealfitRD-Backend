# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-624 · 2026-09-27] El durazno FRESCO se le ofrece a los cinco países beta, no sólo a Estados Unidos.

G68 de la auditoría de países: en el catálogo vivo `normalize_name('melocotón'/'durazno')` resuelve a «Durazno en
almíbar» (78 kcal y 14 g de azúcar por ración, contra 39 kcal y 8,4 g del fresco), y la fila fresca «Duraznos» sólo
entraba en el catálogo verificado de US. Justo los dos perfiles que el producto vende como su fuerte —diabetes tipo 2
y déficit— recibían el de almíbar en España, México, Colombia y Puerto Rico. Es un cambio de DATOS en la partición
que ya existe (la misma que da la fila a US); no se mueve ningún alias ni se renombra nada, y RD no cambia.

tooltip-anchor: P1-PLAN-LOTE-624
"""
import pytest


@pytest.mark.parametrize("cc", ["ES", "MX", "CO", "PR", "US"])
def test_el_durazno_fresco_es_comprable_en_cada_pais_beta(cc):
    from shopping_calculator import is_country_catalog_unpriced_item
    assert is_country_catalog_unpriced_item("Duraznos", country=cc)


def test_rd_no_cambia():
    from shopping_calculator import is_country_catalog_unpriced_item, _COUNTRY_CATALOG_UNPRICED_BY_COUNTRY
    assert _COUNTRY_CATALOG_UNPRICED_BY_COUNTRY["DO"] == ()
    assert not is_country_catalog_unpriced_item("Duraznos", country="DO")


def test_la_tupla_plana_no_gana_tokens():
    # «duraznos» ya estaba (en US): la vista plana que usa el agregador queda igual
    from shopping_calculator import _COUNTRY_CATALOG_UNPRICED_TOKENS
    assert _COUNTRY_CATALOG_UNPRICED_TOKENS.count("duraznos") == 1
