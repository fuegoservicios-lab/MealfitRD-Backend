"""[P1-PDF-2] Tests para `is_perishable_category` y la persistencia del flag
`is_perishable` en cada item de `aggregated_shopping_list`.

Bug original cubierto:
  La heurística "perecedero vs estable" vivía SOLO en `Dashboard.jsx` con
  substring-match contra la categoría:
    `cat.toLowerCase().includes('proteína'|'lácteo'|'vegetal'|'fruta')`
  Si `_get_display_category` devolvía una variante con typo o sin tilde
  (ej. "Proteinas" sin acento — caso de drift histórico), el item caía a
  la sección estable del PDF y el usuario asumía que duraba >7 días.

Fix:
  Backend es ahora SSOT. `aggregate_and_deduct_shopping_list` añade
  `is_perishable: bool` a cada item. `is_perishable_category(cat, shelf_life)`
  aplica reglas en orden de precedencia:
    1. shelf_life_days ≤ 7 → True
    2. category contiene "urgente" → True
    3. category lowercased ⊇ uno de PERISHABLE_CATEGORY_PREFIXES → True
    4. Default → False

Cobertura:
  - test_perishable_categories_match_canonical_set
  - test_stable_categories_default_to_false
  - test_shelf_life_overrides_category_classification
  - test_urgent_category_is_always_perishable
  - test_helper_handles_edge_inputs
  - test_aggregator_persists_is_perishable_in_each_item
"""
import pytest

from shopping_calculator import (
    PERISHABLE_CATEGORY_PREFIXES,
    PERISHABLE_SHELF_LIFE_THRESHOLD_DAYS,
    is_perishable_category,
)


# ---------------------------------------------------------------------------
# 1. Categorías canónicas perecederas — el frontend las marcaba via substring;
#    el backend canónicamente debe coincidir.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("category", [
    # Forma DB cruda (singular, con tilde, capitalizado)
    "Proteínas", "Lácteos", "Vegetales", "Frutas",
    # Forma display canónica (mayúsculas, plurales)
    "PROTEÍNAS", "LÁCTEOS", "VEGETALES", "FRUTAS",
    # Variantes case + plural (mismas que el frontend tolera)
    "proteínas", "lácteos", "vegetales", "frutas",
])
def test_perishable_categories_classified_true(category):
    """Cada categoría perecedera (sin shelf_life_days) → True."""
    assert is_perishable_category(category, None) is True, (
        f"Categoría {category!r} debe clasificarse como perecedera"
    )


# ---------------------------------------------------------------------------
# 2. Categorías estables — default conservador para todo lo no listado.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("category", [
    "DESPENSA",          # arroz, pasta, legumbres
    "ESPECIAS",          # condimentos secos
    "SUPLEMENTOS",       # proteína whey, etc.
    "Despensa y Granos", # raw DB variant
    "OTROS",             # fallback
    # [2026-05-06] VÍVERES removido de aquí — ahora es perecedero (tubérculos
    # frescos como Batata, Yautía, Plátano duran 7-14 días en clima tropical).
    # Ver `test_viveres_and_hierbas_now_perishable` abajo.
])
def test_stable_categories_default_to_false(category):
    """Categorías estables (sin shelf_life_days) → False."""
    assert is_perishable_category(category, None) is False, (
        f"Categoría {category!r} debe clasificarse como estable"
    )


@pytest.mark.parametrize("category", [
    "VÍVERES", "Víveres", "víveres",
    "HIERBAS", "Hierbas", "hierbas",
])
def test_viveres_and_hierbas_now_perishable(category):
    """[2026-05-06] Tubérculos frescos (Víveres) y hierbas frescas
    (cilantro, perejil) son perecederos, no estables. Antes caían al
    fallback de shelf=14 → estable; ahora cat-first los clasifica
    correctamente como perecederos vía PERISHABLE_CATEGORY_PREFIXES."""
    assert is_perishable_category(category, None) is True, (
        f"Categoría {category!r} debe clasificarse como perecedera"
    )


# ---------------------------------------------------------------------------
# 3. CATEGORÍA tiene precedencia sobre shelf_life_days (REORDENADO 2026-05-06).
# ---------------------------------------------------------------------------
# [2026-05-06] La precedencia original (shelf-first) era teóricamente sólida
# pero rompía con el estado real de master_ingredients: shelf_life_days=14
# está poblado como default genérico para casi TODOS los items frescos
# (cerdo, lechosa, mango, queso, brócoli, tomate, yautía, batata…). Con
# threshold=7, shelf>7 devolvía False (stable) y la sección "DESPENSA —
# ESTABLES" del PDF weekly se llenaba de carnes y frutas que el usuario
# debe consumir en 1 semana. Este es el mismo bug que arreglamos en
# `_classify_perishability` (cat→shelf reorder); replicamos aquí para
# que el path weekly (no-hybrid) quede consistente con biweekly/monthly.
# Ver `is_perishable_category` docstring para rationale.
@pytest.mark.parametrize("category,shelf_life,expected", [
    # Categorías STABLE explícitas → False, independiente de shelf:
    # Despensa con shelf bajo (improbable en data real, pero posible si master
    # se actualiza para "Pan fresco artesanal") → seguimos clasificando como
    # estable porque la curación humana de cat='Despensa' es señal fuerte.
    ("Despensa", 1, False),
    ("Despensa", 5, False),
    ("Despensa", 14, False),
    ("Despensa", 365, False),
    ("Conservas", 365, False),
    ("Especias", 14, False),
    # Categorías PERISHABLE con shelf típico → True (cat-first):
    # Proteínas con shelf=14 default DB (caso más común: cerdo/pollo/res)
    # debe quedar perecedero — el cat 'Proteínas' es la señal humana.
    ("Proteínas", 1, True),
    ("Proteínas", 7, True),
    ("Proteínas", 14, True),
    ("Lácteos", 14, True),
    ("Vegetales", 14, True),
    ("Frutas", 14, True),
    # [2026-05-07] Override "shelf largo (≥30d) → staple" para enlatados
    # /curados que viven en cat=Proteínas/Lácteos por origen alimentario
    # pero realmente son estables: Atún en agua (shelf=730), leche UHT
    # (shelf=180), queso parmesano (shelf=120). Antes esto regresaba True
    # (perecedero) y enviaba latas a la sección "compra cada 7 días".
    ("Proteínas", 30, False),   # canned tuna threshold
    ("Proteínas", 180, False),  # ej. fiambre curado / canned chicken
    ("Proteínas", 730, False),  # ej. atún en lata 2 años
    ("Lácteos", 180, False),    # leche UHT
    ("Lácteos", 120, False),    # queso curado
    # Categoría ambigua/desconocida → fallback a shelf:
    ("Otros", 5, True),    # shelf<=7 → perecedero
    ("Otros", 14, False),  # shelf>7 → estable
    ("Suplementos", 30, False),
])
def test_category_overrides_shelf_when_explicit(category, shelf_life, expected):
    """Categoría explícita (stable o perishable) gana sobre shelf_life_days.

    Solo cuando la categoría es desconocida ("Otros", "Suplementos", etc.)
    el shelf_life actúa como tiebreaker.
    """
    assert is_perishable_category(category, shelf_life) is expected, (
        f"cat={category!r} shelf={shelf_life} debe dar is_perishable={expected}"
    )


def test_shelf_life_threshold_constant_matches_documented_value():
    """El umbral está documentado como 7 días en multiple sites (frontend
    fallback, db_inventory _infer_shelf_life_days). Anchor en el valor."""
    assert PERISHABLE_SHELF_LIFE_THRESHOLD_DAYS == 7


# ---------------------------------------------------------------------------
# 4. Items urgentes ("🚨 Compra Urgente") — siempre perecederos.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("category", [
    "🚨 Compra Urgente",
    "Compra Urgente",
    "URGENTE",
    "Urgente — refrigerar",
])
def test_urgent_category_is_always_perishable(category):
    """Sin shelf_life_days, una categoría con substring "urgente" → perecedero.

    Cubre el caso de items urgent_items (líneas 1530+ del aggregator) que se
    añaden con `category='🚨 Compra Urgente'` y deben ir en la sección
    "compra inmediata" del PDF independientemente de su categoría base.
    """
    assert is_perishable_category(category, None) is True


# ---------------------------------------------------------------------------
# 5. Edge inputs.
# ---------------------------------------------------------------------------
def test_none_category_returns_false():
    """`category=None` no clasifica como perecedero (default conservador)."""
    assert is_perishable_category(None, None) is False


def test_empty_category_returns_false():
    """String vacío → estable."""
    assert is_perishable_category("", None) is False


def test_shelf_life_string_parseable_works():
    """El helper acepta `shelf_life_days` como string parseable a int.
    Defensa contra payloads serializados que perdieron el tipo.

    [2026-05-06] Tras el reorder cat-first, este test usa cat='Otros'
    (ambigua) para que shelf actúe como fallback. Si usáramos 'Despensa'
    o 'Proteínas', la categoría ganaría y el shelf no se evaluaría.
    """
    assert is_perishable_category("Otros", "5") is True
    assert is_perishable_category("Otros", "10") is False


def test_shelf_life_unparseable_falls_through():
    """`shelf_life_days` no parseable → caer a regla 2 (categoría)."""
    # Cae a categoría: Despensa no contiene perecedero → False
    assert is_perishable_category("Despensa", "no-int") is False
    # Cae a categoría: Proteínas SÍ → True
    assert is_perishable_category("Proteínas", "no-int") is True


def test_unknown_category_with_no_shelf_life_returns_false():
    """Categoría desconocida sin shelf_life → estable (conservador)."""
    assert is_perishable_category("Categoría futura", None) is False


def test_perishable_prefixes_set_immutable():
    """`PERISHABLE_CATEGORY_PREFIXES` debe ser frozenset; mutación runtime
    evadiría el invariante."""
    assert isinstance(PERISHABLE_CATEGORY_PREFIXES, frozenset)


def test_perishable_prefixes_canonical_set():
    """Snapshot del set canónico — guard contra cambios silenciosos.
    Si añades un prefijo, actualiza este test Y `_infer_shelf_life_days`
    en `db_inventory.py` para coherencia."""
    assert PERISHABLE_CATEGORY_PREFIXES == frozenset({
        "proteína", "lácteo", "vegetal", "fruta",
        "víver", "hierba",
    })


# ---------------------------------------------------------------------------
# 6. Sanity: el aggregator persiste `is_perishable` en cada item structured.
# ---------------------------------------------------------------------------
def _build_simple_plan_ingredient_strings():
    """Construye una lista mínima que dispara el path estructurado del
    aggregator sin requerir DB real (las llamadas a master_ingredients
    vuelven con dict vacío y el helper maneja category="Otros")."""
    return [
        "200g pollo",
        "1 unidad manzana",
    ]


def test_aggregator_emits_is_perishable_field_in_structured_items(monkeypatch):
    """Cada item devuelto por `aggregate_and_deduct_shopping_list(..., structured=True)`
    debe traer la key `is_perishable` (bool).

    Mocking de `get_master_ingredients` para evitar dependencia DB."""
    import shopping_calculator as sc

    # Mock master_ingredients: pollo es proteína; manzana es fruta.
    fake_master = [
        {"name": "pollo", "category": "Proteínas", "shelf_life_days": 5},
        {"name": "manzana", "category": "Frutas", "shelf_life_days": 7},
    ]
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: fake_master)
    # Bypass el cache local de master.
    sc.invalidate_master_cache()

    items = sc.aggregate_and_deduct_shopping_list(
        _build_simple_plan_ingredient_strings(),
        consumed_ingredients=[],
        structured=True,
        multiplier=1.0,
    )

    assert isinstance(items, list)
    assert items, "El aggregator debió devolver al menos 1 item para el payload de prueba"
    for it in items:
        assert isinstance(it, dict), f"Item structured debe ser dict, vio {type(it).__name__}"
        assert "is_perishable" in it, (
            f"Item {it.get('name')!r} sin field `is_perishable`. "
            f"Keys actuales: {sorted(it.keys())}"
        )
        assert isinstance(it["is_perishable"], bool), (
            f"`is_perishable` debe ser bool, vio {type(it['is_perishable']).__name__} "
            f"en item {it.get('name')!r}"
        )
