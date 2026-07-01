"""[P1-VERIFIED-ONLY-OBSERVABILITY · 2026-06-21] Visibilidad del drop de
VERIFIED_INGREDIENTS_ONLY.

Contexto (audit presupuesto↔calidad 2026-06-21): con MEALFIT_VERIFIED_INGREDIENTS_ONLY
ON (prod), un ingrediente de receta fuera del catálogo verificado (~202) se dropea de la lista de
compras Y el coherence guard se filtra a sí mismo el mismo ingrediente (espejo
expected_raw) → cero divergencia → cero retry → cero señal. Resultado: "lista de compras
incompleta entregada sin aviso" — el escenario exacto que preocupa al owner.

El prompt upstream (_get_verified_catalog_instruction) ya prohíbe fuertemente inventar
ingredientes, así que el drop debería ser raro. Pero cuando el LLM desobedece, ANTES era
100% silencioso. Este P-fix NO cambia el comportamiento (no fuerza retry — evita un
retry-storm por condimentos raros como laurel/comino); hace VISIBLE el drop vía WARNING
grep-able en ambos puntos, para medir la tasa real de desobediencia en prod y decidir el
siguiente paso (ampliar catálogo / forzar retry / avisar al usuario) con datos.

Tests parser-based: anclan que la observabilidad existe en el source de prod. El
comportamiento funcional (el WARNING se emite ante un ingrediente no-verificado) se valida
en vivo en el VPS.
"""
import re

import shopping_calculator


def _src():
    return open(shopping_calculator.__file__, encoding="utf-8").read()


def test_marker_presente():
    assert _src().count("P1-VERIFIED-ONLY-OBSERVABILITY") >= 2, (
        "El marker debe anclar AMBOS puntos de observabilidad (guard + aggregator)."
    )


def test_guard_captura_lo_filtrado_antes_de_descartar():
    src = _src()
    # El guard DEBE capturar el set esperado ANTES de filtrar, para poder reportar
    # qué ingredientes de receta cayeron fuera del catálogo.
    assert "_expected_before_filter" in src, (
        "El guard debe snapshot del expected ANTES del filtro verified-only."
    )
    assert "_dropped_recipe_ingredients" in src
    assert "[VERIFIED-ONLY-GUARD-BLIND]" in src, (
        "El guard debe emitir el WARNING grep-able cuando filtra ingredientes de receta."
    )


def test_aggregator_drop_es_warning_no_info():
    src = _src()
    assert "[VERIFIED-ONLY-DROP]" in src, (
        "El drop del aggregator debe emitir un WARNING grep-able (tag VERIFIED-ONLY-DROP)."
    )
    # Regresión: el drop antes era logging.info (silencioso en prod, nivel info no se
    # surfacea). Debe ser logging.warning ahora.
    drop_region = src[src.index("[VERIFIED-ONLY-DROP]") - 200: src.index("[VERIFIED-ONLY-DROP]") + 50]
    assert "logging.warning" in drop_region, (
        "El drop del aggregator debe ser logging.warning, no logging.info."
    )


def test_filtro_verified_sigue_activo_para_no_retry_storm():
    # El fix es observabilidad-only: el filtro del espejo DEBE seguir existiendo
    # (no bloquea por condimentos raros). Si alguien lo quita sin un closer/retry
    # gateado, este test obliga a reconsiderar el retry-storm.
    src = _src()
    assert "_is_verified_for_shopping(k)" in src, (
        "El espejo verified-only del guard debe seguir filtrando expected_raw "
        "(la observabilidad NO cambia el comportamiento de retry)."
    )
