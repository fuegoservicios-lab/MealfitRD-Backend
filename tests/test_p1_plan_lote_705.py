# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-705 · 2026-09-28] (G59) El registro de catálogo-país ya no afirma que sus nombres «NUNCA aparecen en un plan DO».

La justificación de byte-identidad DO era falsa: Adobo, Sofrito, Kétchup y Aderezo ranch son palabras dominicanas
corrientes y siguen en `_COUNTRY_CATALOG_UNPRICED_TOKENS`. Medido el 28-sep en prod: 13 planes en 21 días, 0 ítems
sin precio (con precio gana `_is_verified_for_shopping`), y el PDF ya avisa «estimado parcial (N % con precio)» vía
`budget_reconciliation.price_coverage`. No se estrecha el keep: hacerlo vuelve a quitar comida de la lista en silencio.

tooltip-anchor: P1-PLAN-LOTE-705
"""
import shopping_calculator as sc


def test_el_comentario_no_repite_la_afirmacion_falsa():
    src = open(sc.__file__, encoding="utf-8").read()
    assert "NUNCA aparecen en un plan DO" not in src
    assert "[G59 · P1-PLAN-LOTE-705] NO byte-idéntico en DO" in src


def test_las_palabras_dominicanas_siguen_con_keep():
    # el keep las conserva sin precio en vez de soltarlas: estrecharlo es el modo de fallo P1-BAKING-STAPLES
    for nombre in ("Adobo", "Sofrito", "Ketchup", "Aderezo ranch"):
        assert sc.is_country_catalog_unpriced_item(nombre), nombre
