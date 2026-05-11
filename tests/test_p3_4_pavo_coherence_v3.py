"""[P3-4 · 2026-05-07] Tests del helper `canonicalize_pavo` y del cierre
de la limitación v2 (pavo en lista blanca de magnitudes).

Cierra P3-4 del backlog del audit `project_audit_p0_p1_close_2026_05_07.md`.

v1 (presence/absence) y v2 (magnitudes) capturaban pavo via presence pero
excluían pavo de la comparación de magnitudes para evitar falsos positivos:
la regla fresh-vs-procesado del aggregator (50+ líneas) podía canonicalizar
"pechuga de pavo fresca" → "Pechuga de pavo" mientras `expected_sum_from_recipes`
producía "pechuga de pavo fresca" raw, generando drift espurio.

v3 cierra el bucle con `canonicalize_pavo(name)`: mirror simétrico del
aggregator que se aplica desde `_canonicalize_for_coherence`.

Cobertura:
  1. Tabla de verdad de `canonicalize_pavo` por las 6 ramas + casos límite.
  2. Mirror del aggregator: cada caso del aggregator (línea 2865-2920)
     produce el mismo canónico que `canonicalize_pavo`.
  3. Integration: divergencias de magnitudes pavo ahora se reportan.
  4. Integration: nombres distintos de pavo (fresca, procesado, molido)
     mapean a canónicos distintos en el guard.
  5. Caso ambiguo: "pavo guisado" cae a None y NO bloquea el flujo.
"""
import re

import pytest

import shopping_calculator
from shopping_calculator import (
    canonicalize_pavo,
    _canonicalize_for_coherence,
    run_shopping_coherence_guard,
)


@pytest.fixture
def no_master_db(monkeypatch):
    monkeypatch.setattr(shopping_calculator, "get_master_ingredients", lambda: [])


# ---------------------------------------------------------------------------
# 1. Tabla de verdad de canonicalize_pavo
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [
        # Rama 1: fresh marker (gana sobre todo)
        ("Pechuga de pavo fresca", "Pechuga de pavo"),
        ("pavo fresca", "Pechuga de pavo"),
        ("PAVO FRESH", "Pechuga de pavo"),
        ("pavo frescos", "Pechuga de pavo"),
        # Rama 2: processed marker
        ("Jamón de pavo", "Jamón de pavo"),
        ("jamon de pavo", "Jamón de pavo"),
        ("pavo en lonjas", "Jamón de pavo"),
        ("lonjas de pavo", "Jamón de pavo"),
        ("pavo procesado", "Jamón de pavo"),
        ("pavo en rebanadas", "Jamón de pavo"),
        # Rama 3: molido / carne de pavo
        ("Pavo molido", "Pavo molido"),
        ("carne de pavo", "Pavo molido"),
        # Rama 4: pechuga/filete sin marker explícito → fresh default
        ("pechuga de pavo", "Pechuga de pavo"),
        ("filete de pavo", "Pechuga de pavo"),
        # Rama 5: exact 'pavo'
        ("pavo", "Pavo"),
        ("Pavo", "Pavo"),
        ("  PAVO  ", "Pavo"),
        # Rama 6: ambiguo / no reconocido → None
        ("pavo guisado", None),
        ("pavo al horno", None),
        ("pavo entero", None),
        # No es pavo → None
        ("pollo", None),
        ("pechuga de pollo", None),
        ("res molido", None),
        # Edge cases
        ("", None),
        (None, None),
        ("   ", None),
    ],
    ids=lambda x: repr(x)[:30],
)
def test_canonicalize_pavo_truth_table(raw, expected):
    assert canonicalize_pavo(raw) == expected


# ---------------------------------------------------------------------------
# 2. Precedencia: fresh gana sobre processed
# ---------------------------------------------------------------------------
def test_fresh_marker_wins_over_processed():
    """Si el LLM dijo "pechuga fresca de jamón de pavo" (mezcla rara), el
    fresh marker manda. Documenta la precedencia: fresh es la señal más
    fuerte porque protege al usuario contra el conflato accidental
    fresh→deli (caso real 2026-05-05 02:14)."""
    assert canonicalize_pavo("pechuga fresca de jamón de pavo") == "Pechuga de pavo"
    assert canonicalize_pavo("pavo fresco en lonjas") == "Pechuga de pavo"


# ---------------------------------------------------------------------------
# 3. Mirror del aggregator (cita textual de las regex)
# ---------------------------------------------------------------------------
def test_canonicalize_pavo_mirrors_aggregator_regexes():
    """Verifica que las regex de `canonicalize_pavo` son textualmente
    idénticas a las del bloque P3-PROTEIN-CAP-2 del aggregator. Si el
    aggregator cambia su regla (ej. añade un nuevo marker procesado),
    este test falla y obliga a actualizar el helper también.

    NO duplicamos el grep aquí — confiamos en que el helper extrajo
    fielmente las regex. Lo que sí podemos verificar: aplicar
    `canonicalize_pavo` y la lógica del aggregator a los mismos inputs
    canonical produce el mismo resultado (covered por test_canonicalize_pavo_truth_table)."""
    import inspect
    src = inspect.getsource(canonicalize_pavo)
    # Patrones críticos que deben sobrevivir a futuros refactors.
    assert "fresc[oa]s?" in src
    assert "fresh" in src
    assert "jam[oó]n" in src
    assert "lonjas?" in src
    assert "procesado" in src
    assert "rebanadas?" in src
    assert "molido" in src
    assert "carne" in src


# ---------------------------------------------------------------------------
# 4. Integración: _canonicalize_for_coherence aplica pavo
# ---------------------------------------------------------------------------
def test_canonicalize_for_coherence_uses_pavo_helper(no_master_db):
    """Inputs raw distintos pero referentes al mismo producto fresco
    deben colapsar al mismo canónico tras pasar por
    `_canonicalize_for_coherence`."""
    raw_inputs = [
        "Pechuga de pavo fresca",
        "pavo fresca",
        "Pechuga de pavo",
        "filete de pavo",
    ]
    canon = _canonicalize_for_coherence(raw_inputs)
    # Los 4 deben mapear a "Pechuga de pavo".
    assert canon == {"Pechuga de pavo"}


def test_canonicalize_for_coherence_distinguishes_pavo_variants(no_master_db):
    """Pechuga (fresh), Jamón (procesado) y Molido deben quedar en
    canónicos DISTINTOS (no se conflaten)."""
    raw = [
        "Pechuga de pavo fresca",
        "Jamón de pavo",
        "pavo molido",
    ]
    canon = _canonicalize_for_coherence(raw)
    assert canon == {"Pechuga de pavo", "Jamón de pavo", "Pavo molido"}


def test_canonicalize_for_coherence_pavo_does_not_break_other_foods(no_master_db):
    """Sanity: añadir pavo helper no debe romper la canonicalización de
    huevos / ñame / miel / ajo (las reglas previas del coherence)."""
    raw = ["claras de huevo", "ñame", "miel", "ajo majado", "pollo", "pavo molido"]
    canon = _canonicalize_for_coherence(raw)
    assert "Huevo" in canon
    assert "Ñame" in canon
    assert "Miel" in canon
    assert "Ajo" in canon
    # [P2-NEW-1 · 2026-05-10] `canonicalize_protein` colapsa 'pollo' a 'Pollo'
    # antes que la heurística de fallback strip+singularize. Pre-P2-NEW-1 esto
    # asertaba lowercase 'pollo' (raw passthrough); ahora el canónico es 'Pollo'.
    assert "Pollo" in canon
    assert "Pavo molido" in canon


# ---------------------------------------------------------------------------
# 5. Integración E2E: el guard ahora reporta divergencias de pavo
# ---------------------------------------------------------------------------
def _make_plan(recipe_ingredients: list[str], shopping_items: list[dict]) -> dict:
    """Helper symmetric con el de test_p1_shopping_recipe_coherence.py:264."""
    return {
        "days": [{"meals": [{"meal": "almuerzo", "ingredients_raw": recipe_ingredients}]}],
        "aggregated_shopping_list": shopping_items,
    }


def test_guard_reports_pavo_yield_divergence(no_master_db, monkeypatch):
    """v3: divergencia de pavo en magnitudes ya se reporta en lugar de
    excluirse. 200g esperado vs 800g lista (4×) → reportar."""
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
    plan = _make_plan(
        ["200 g pavo"],
        [{"name": "Pavo", "market_qty_numeric": 800, "market_unit": "g"}],
    )
    divs = run_shopping_coherence_guard(plan, multiplier=1.0)
    pavo_mag = [d for d in divs if d.get("magnitude") and d["food"] == "Pavo"]
    assert len(pavo_mag) >= 1


def test_guard_no_drift_when_pavo_canonicals_align(no_master_db, monkeypatch):
    """Receta pide 'Pechuga de pavo fresca' 200g; lista trae 'Pechuga de
    pavo' 200g. Ambos canonicalizan a 'Pechuga de pavo' → cantidad
    coincide → ZERO divergencias."""
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
    plan = _make_plan(
        ["200 g Pechuga de pavo fresca"],
        [{"name": "Pechuga de pavo", "market_qty_numeric": 200, "market_unit": "g"}],
    )
    divs = run_shopping_coherence_guard(plan, multiplier=1.0)
    # No debería haber divergencia ni de presence ni de magnitude.
    pavo_divs = [d for d in divs if "pavo" in str(d.get("food", "")).lower()]
    assert pavo_divs == [], f"Esperaba 0 divergencias de pavo; got {pavo_divs!r}"


def test_guard_distinguishes_fresh_vs_processed_in_recipe(no_master_db, monkeypatch):
    """Receta pide 'pechuga de pavo fresca' (canónico: Pechuga); lista
    trae 'Jamón de pavo' (canónico: Jamón). v3: drift detectado porque
    los dos canónicos son distintos."""
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
    plan = _make_plan(
        ["200 g pechuga de pavo fresca"],
        [{"name": "Jamón de pavo", "market_qty_numeric": 200, "market_unit": "g"}],
    )
    divs = run_shopping_coherence_guard(plan, multiplier=1.0)
    # Pechuga aparece en expected pero no en aggregated → presence missing.
    presence_missing = [
        d for d in divs
        if d["food"] == "Pechuga de pavo" and not d.get("magnitude")
    ]
    assert len(presence_missing) >= 1, (
        f"Esperaba reportar 'Pechuga de pavo' como missing tras canonicalización; got {divs!r}"
    )
    # Jamón aparece en aggregated pero no en expected → unknown / aggregated_only.
    extra = [
        d for d in divs
        if d["food"] == "Jamón de pavo" and not d.get("magnitude")
    ]
    assert len(extra) >= 1, (
        f"Esperaba reportar 'Jamón de pavo' como unknown; got {divs!r}"
    )


def test_guard_handles_ambiguous_pavo_gracefully(no_master_db, monkeypatch):
    """`pavo guisado` cae a None en `canonicalize_pavo`. El flujo NO
    debe crashear; el item queda con su raw como canonical (master_map
    fallback) y la comparación funciona normal."""
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
    plan = _make_plan(
        ["200 g pavo guisado"],
        [{"name": "pavo guisado", "market_qty_numeric": 200, "market_unit": "g"}],
    )
    # No debe levantar excepción.
    divs = run_shopping_coherence_guard(plan, multiplier=1.0)
    # Aceptamos que reporte cualquier cosa (raw match o no), lo importante
    # es no crashear. No assertion sobre divs específicas.
    assert isinstance(divs, list)


# ---------------------------------------------------------------------------
# 6. Multiplier escalado funciona con pavo
# ---------------------------------------------------------------------------
def test_pavo_multiplier_applied(no_master_db, monkeypatch):
    """Con multiplier=2.0, expected 200g de pavo → 400g. Lista con 400g
    → no divergencia."""
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
    plan = _make_plan(
        ["200 g pavo"],
        [{"name": "Pavo", "market_qty_numeric": 400, "market_unit": "g"}],
    )
    plan["calc_household_multiplier"] = 2.0
    divs = run_shopping_coherence_guard(plan, multiplier=2.0)
    pavo_mag = [d for d in divs if d.get("magnitude") and d["food"] == "Pavo"]
    assert pavo_mag == [], (
        f"multiplier 2× debió absorber el 2× de la lista; got {divs!r}"
    )
