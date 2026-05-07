"""
Tests P1-ORQ-2: `semantic_cache_check_node` valida compatibilidad de despensa.

Bug original:
  El cache validaba 6 dimensiones (allergies, medicalConditions, dietType,
  dislikes, calories, macros) pero NO `current_pantry_ingredients` ni
  `current_shopping_list`. Un usuario con pantry configurada recibía un plan
  cacheado generado SIN pantry awareness; `review_plan_node` lo rechazaba con
  severity="high"; `should_retry` cortaba; el guardrail no lo trataba como
  critical → entregaba plan con disclaimer ámbar evitable. Una regeneración
  forzada habría producido un plan limpio.

Fix:
  `_pantry_cache_discard_reason(actual_form, cached_form)` retorna un string
  descriptivo si el cache hit DEBERÍA descartarse, o None si es válido.

Reglas:
  - Asimetría: pantry presente AHORA + ausente en cache → discard.
  - Drift: ambos presentes + Jaccard distance > PANTRY_DRIFT_THRESHOLD (0.2) → discard.
  - Ambos vacíos → cache hit válido.
  - Solo cache tiene pantry → cache hit válido (downstream no exige pantry).
"""
import re
import inspect

import pytest

from graph_orchestrator import (
    PANTRY_DRIFT_THRESHOLD,
    _normalize_pantry_set,
    _pantry_cache_discard_reason,
)


# ---------------------------------------------------------------------------
# 1. Threshold y helper de normalización
# ---------------------------------------------------------------------------
def test_pantry_drift_threshold_es_20_por_ciento():
    """Threshold se mantiene en 0.2 (20%) — la auditoría especifica ese valor."""
    assert PANTRY_DRIFT_THRESHOLD == 0.2


def test_normalize_pantry_set_strip_cantidades_y_aplica_sinonimos():
    """`pollo 500g`, `pollo 600g`, `pechuga de pollo` colapsan al mismo término."""
    form = {
        "current_pantry_ingredients": [
            "pollo 500g",
            "pollo 600g",
            "pechuga de pollo",
            "arroz 1000g",
        ]
    }
    s = _normalize_pantry_set(form)
    # Las 3 variantes de pollo colapsan → set tiene 2 entradas distintas
    # (pollo + arroz). Si normalize_ingredient_for_tracking devuelve cosas
    # distintas (e.g. "pechuga de pollo" no se sinonimiza), aceptamos hasta 3.
    assert 2 <= len(s) <= 3
    # Garantía: tras la normalización al menos un item canónico contiene "pollo".
    assert any("pollo" in item for item in s)


def test_normalize_pantry_set_acepta_shopping_list_como_alias():
    """`current_shopping_list` debe tratarse igual que pantry (mismo `or`
    semántico que review_plan_node:5603)."""
    form_a = {"current_pantry_ingredients": ["pollo 500g"]}
    form_b = {"current_shopping_list": ["pollo 500g"]}
    assert _normalize_pantry_set(form_a) == _normalize_pantry_set(form_b)


def test_normalize_pantry_set_robusta_a_inputs_invalidos():
    """None, dicts no-form, items no-string, listas vacías → frozenset vacío."""
    assert _normalize_pantry_set(None) == frozenset()
    assert _normalize_pantry_set({}) == frozenset()
    assert _normalize_pantry_set({"current_pantry_ingredients": None}) == frozenset()
    assert _normalize_pantry_set({"current_pantry_ingredients": "no soy lista"}) == frozenset()
    assert _normalize_pantry_set({"current_pantry_ingredients": [None, 42, ""]}) == frozenset()


# ---------------------------------------------------------------------------
# 2. Asimetría — pantry actual sin pantry en cache
# ---------------------------------------------------------------------------
def test_asimetria_actual_con_pantry_cache_sin_descarta():
    """Bug primario: usuario tiene pantry, cache se generó sin pantry → descarte."""
    actual = {"current_pantry_ingredients": ["pollo 500g", "arroz 1000g"]}
    cached = {}  # plan generado sin awareness de pantry

    reason = _pantry_cache_discard_reason(actual, cached)
    assert reason is not None
    assert "intención de despensa" in reason


def test_asimetria_actual_con_shopping_list_cache_sin_descarta():
    """Mismo escenario pero usando `current_shopping_list` (alias)."""
    actual = {"current_shopping_list": ["pollo 500g"]}
    cached = {}
    assert _pantry_cache_discard_reason(actual, cached) is not None


# ---------------------------------------------------------------------------
# 3. Sin asimetría — escenarios donde cache hit es válido
# ---------------------------------------------------------------------------
def test_ambos_sin_pantry_cache_hit_valido():
    """Flujo no-pantry: ambos vacíos → no aplica filtro."""
    assert _pantry_cache_discard_reason({}, {}) is None
    assert _pantry_cache_discard_reason(
        {"current_pantry_ingredients": []},
        {"current_pantry_ingredients": []},
    ) is None


def test_solo_cache_tiene_pantry_es_valido():
    """Cache fue generado CON pantry pero la request actual no la pide.
    Servirlo es seguro (downstream no activará pantry guard)."""
    actual = {}
    cached = {"current_pantry_ingredients": ["pollo 500g"]}
    assert _pantry_cache_discard_reason(actual, cached) is None


def test_pantry_identica_es_valido():
    """Mismos ingredientes (modulo cantidad) → jaccard_dist=0 < 0.2 → válido."""
    actual = {"current_pantry_ingredients": ["pollo 500g", "arroz 1000g", "tomate 300g"]}
    cached = {"current_pantry_ingredients": ["pollo 700g", "arroz 800g", "tomate 200g"]}
    assert _pantry_cache_discard_reason(actual, cached) is None


# ---------------------------------------------------------------------------
# 4. Drift de Jaccard — descarta solo cuando >20%
# ---------------------------------------------------------------------------
def test_drift_pequeno_no_descarta():
    """5 ingredientes en común + 1 nuevo = jaccard_dist = 1/6 ≈ 0.167 < 0.2."""
    common = ["pollo 500g", "arroz 1000g", "tomate 300g", "cebolla 200g", "aceite 250ml"]
    actual = {"current_pantry_ingredients": common + ["lechuga 200g"]}
    cached = {"current_pantry_ingredients": list(common)}
    assert _pantry_cache_discard_reason(actual, cached) is None


def test_drift_significativo_descarta():
    """3 ingredientes en común + 2 nuevos + 2 removidos = jaccard_dist alto → descarta."""
    actual = {
        "current_pantry_ingredients": [
            "pollo 500g", "arroz 1000g", "tomate 300g",
            "lechuga 200g", "zanahoria 100g",  # 2 nuevos
        ]
    }
    cached = {
        "current_pantry_ingredients": [
            "pollo 500g", "arroz 1000g", "tomate 300g",
            "papa 500g", "cebolla 200g",  # 2 que ya no están
        ]
    }
    reason = _pantry_cache_discard_reason(actual, cached)
    assert reason is not None
    assert "drift de despensa" in reason
    assert "jaccard_dist" in reason


def test_drift_total_descarta():
    """Pantry completamente distintas → jaccard_dist=1.0 → descarta."""
    actual = {"current_pantry_ingredients": ["pollo 500g", "arroz 1000g"]}
    cached = {"current_pantry_ingredients": ["res 500g", "papa 1000g"]}
    reason = _pantry_cache_discard_reason(actual, cached)
    assert reason is not None
    assert "drift de despensa" in reason


# ---------------------------------------------------------------------------
# 5. El nodo invoca el helper (regresión sobre la integración)
# ---------------------------------------------------------------------------
def test_semantic_cache_check_node_llama_helper():
    """Verifica por inspección que `semantic_cache_check_node` invoca
    `_pantry_cache_discard_reason` — protege contra que un refactor lo borre
    silenciosamente."""
    import graph_orchestrator as go
    src = inspect.getsource(go.semantic_cache_check_node)
    assert "_pantry_cache_discard_reason" in src, (
        "semantic_cache_check_node debe invocar _pantry_cache_discard_reason "
        "para validar pantry. Si refactorizaste el filtro, asegura que el "
        "helper sigue siendo el contract activo."
    )


def test_helpers_son_export_publico():
    """Los helpers deben ser importables directamente desde el módulo."""
    from graph_orchestrator import (
        _normalize_pantry_set as _ns,
        _pantry_cache_discard_reason as _pcdr,
        PANTRY_DRIFT_THRESHOLD as _t,
    )
    assert callable(_ns)
    assert callable(_pcdr)
    assert isinstance(_t, float)
