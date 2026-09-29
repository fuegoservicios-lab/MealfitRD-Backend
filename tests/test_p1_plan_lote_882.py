# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-882 · 2026-09-29] «1.17 tallo de Apio» → «2 tallos de Apio»: una pieza natural se compra entera.

Batería real (mujer que pierde grasa, 29-sep, código 868): la lista decía «1.17 tallo de Apio»; 6d vio «4.67 tallo de
Apio» en PR. Corpus: 62 de 825 listas con «X.XX tallo de Apio» y 14 con «X.XX hojas» de repollo o lechuga. El bloque 4
(último recurso, sin peso aplicable) sólo redondeaba hacia arriba los envases.
"""
from __future__ import annotations

import shopping_calculator as sc


def _lista(nombre, unidad, qty):
    return sc.apply_smart_market_units(nombre, 0.0, unidad, qty, master_item=None).get("display_string")


def test_las_piezas_naturales_salen_enteras_y_en_su_numero():
    assert _lista("Apio", "tallo", 1.17) == "2 tallos de Apio"
    assert _lista("Apio", "tallos", 4.67) == "5 tallos de Apio"
    assert _lista("Repollo", "hojas", 3.5) == "4 hojas de Repollo"
    assert _lista("Apio", "tallo", 0.5) == "1 tallo de Apio"
    assert _lista("Apio", "tallo", 2.0) == "2 tallos de Apio"
