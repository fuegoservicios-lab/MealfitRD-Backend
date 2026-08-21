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
    """[RE-ANCLADO por P1-COHERENCE-MIRROR-KEEP · 2026-08-21] El filtro y su WARN se movieron a
    `_filter_expected_to_shopping_survivors`, así que los nombres locales `_expected_before_filter`
    y `_dropped_recipe_ingredients` ya no existen. La INVARIANTE es la misma y sigue anclada: hay
    que fotografiar el esperado ANTES de filtrar, o no hay nada que reportar. Se re-ancla a la
    propiedad (el snapshot + el WARN), no a la grafía de dos variables locales — que es lo que
    este repo pide de un test parser-based."""
    src = _src()
    i = src.find("def _filter_expected_to_shopping_survivors")
    assert i > 0, "el filtro del lado esperado desapareció"
    _fin = src.find("\ndef ", i + 1)
    cuerpo = src[i:_fin if _fin > 0 else len(src)]
    assert "_antes = set(expected_raw.keys())" in cuerpo, (
        "El guard debe fotografiar el expected ANTES de filtrar: sin snapshot no hay nada que "
        "reportar en el WARN."
    )
    assert "_caidos = _antes - set(" in cuerpo, "el WARN dejó de derivarse del snapshot"
    assert "[VERIFIED-ONLY-GUARD-BLIND]" in cuerpo, (
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
    """[RE-ANCLADO por P1-COHERENCE-MIRROR-KEEP · 2026-08-21] Este test anclaba la grafía exacta
    `_is_verified_for_shopping(k)` — y esa llamada es justamente el defecto que P1-COHERENCE-
    MIRROR-KEEP quitó: el lado esperado replicaba UNA de las tres ramas del agregador, así que
    toda fila conservada-sin-precio (staples de horneado desde 2026-07-01, catálogo-país desde
    F2-T5) quedaba como fantasma `unknown` en el guard, 1:1 con los ítems sin precio del plan.

    Lo que este test protege de verdad —que el filtro del espejo SIGA EXISTIENDO, para no abrir
    un retry-storm por condimentos raros— no cambió: sigue filtrando, sólo que ahora pregunta por
    `_survives_shopping_list`, que responde por las tres ramas. Se re-ancla a la PROPIEDAD."""
    src = _src()
    assert "expected_raw = _filter_expected_to_shopping_survivors(" in src, (
        "El espejo del guard debe seguir filtrando expected_raw (quitarlo abriría el retry-storm "
        "por condimentos que el agregador dropea)."
    )
    assert "_survives_shopping_list" in src, (
        "El filtro debe preguntar por el SSOT de las tres ramas, no por el predicado de precio."
    )
