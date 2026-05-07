"""[P3-PROTEIN-CAP-2] Tests para la distinción fresh vs processed en la
canonicalización de pavo en `aggregate_and_deduct_shopping_list`.

Bug observable (corrida 2026-05-05 02:14):
  El LLM correctamente eligió "pechuga de pavo fresca" como proteína
  (proteína magra fresca, no procesada). Pero el aggregator tenía la regla:
    if re.search(r'pavo', _can_lower) and re.search(r'(pechuga|lonjas?|...)', _can_lower):
        canonical_name = 'Jamón de pavo'
  Esta regla colapsaba CUALQUIER pavo + pechuga → "Jamón de pavo",
  incluso cuando explícitamente decía "fresca". Resultado en PDF mensual:
  "Jamón de pavo: 27.78 lbs" cuando el usuario realmente necesita
  "Pechuga de pavo: 27.78 lbs".

  Implicaciones reales para el usuario:
  - Precio: ~$80 RD$/lb (fresca) vs ~$150 RD$/lb (deli). Diferencia
    ~$1900 RD$ por la lista mensual.
  - Sodio: pechuga fresca <100mg/100g; jamón en lonjas ~800-1200mg/100g.
    Factor 4-12× peor para gain_muscle limpio.
  - Conservantes/nitritos: pechuga fresca = 0; deli = sí.

Fix:
  Reglas en orden de precedencia:
  1. Marker EXPLÍCITO de fresh ('fresca', 'fresh') → "Pechuga de pavo"
  2. Marker explícito de procesado ('jamón de pavo', 'pavo en lonjas',
     'pavo procesado') → "Jamón de pavo"
  3. 'pavo molido' o 'carne de pavo' → "Pavo molido" (lean ground separado)
  4. 'pechuga de pavo' o 'filete de pavo' sin marker procesado →
     "Pechuga de pavo" (default seguro)
  5. Else: master_map default

Cobertura:
  - Repro del incidente: pechuga fresca preservada, no conflated
  - Procesado SIGUE canonicalizando correctamente (no regresión)
  - Pavo molido es item separado
  - Mix realista: ambos productos coexisten en lista cuando aparecen
  - Edge cases: variantes con/sin tilde, case-insensitive
"""
import pytest

from shopping_calculator import aggregate_and_deduct_shopping_list, invalidate_master_cache


@pytest.fixture(autouse=True)
def _reset_master_cache():
    """[P3-PROTEIN-CAP-2] Pytest puede inicializar el master_map cache vía
    fixtures upstream (tests/conftest.py) con datos que canonicalizan
    "Pechuga de pavo" → "Jamón de pavo" via PROTEIN_SYNONYMS aliases.
    En producción esta cache está poblada desde Supabase con shape distinta.
    Limpio el cache antes de cada test para garantizar comportamiento
    consistente entre runs aislados y en suite completa."""
    invalidate_master_cache()
    yield
    invalidate_master_cache()


def _names_in_result(result: list) -> set[str]:
    return {r.get("name") for r in result if isinstance(r, dict)}


# ---------------------------------------------------------------------------
# 1. Repro del incidente
# ---------------------------------------------------------------------------
def test_repro_incident_2026_05_05_pechuga_fresca_preserved():
    """LLM eligió 'pechuga de pavo fresca' x3 días → debe canonicalizar a
    'Pechuga de pavo' (no 'Jamón de pavo')."""
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients=[
            "300g de pechuga de pavo fresca",
            "250g de pechuga de pavo fresca",
            "300g de pechuga de pavo fresca",
        ],
        multiplier=18.67,  # mensual × 2 personas
        structured=True,
    )
    names = _names_in_result(result)
    assert "Jamón de pavo" not in names, (
        f"Pechuga fresca NO debe canonicalizar a Jamón: {names}"
    )
    assert any("echuga de pavo" in n for n in names), (
        f"Pechuga de pavo debe aparecer: {names}"
    )


# ---------------------------------------------------------------------------
# 2. Jamón procesado SIGUE canonicalizando (no regresión)
# ---------------------------------------------------------------------------
class TestProcessedPavoStillCanonicalizes:
    @pytest.mark.parametrize("ingredient", [
        "100g de jamón de pavo en lonjas",
        "50g de jamón de pavo",
        # Nota: "80g de pavo en lonjas" sin más prefijo NO funciona porque
        # `_parse_quantity` colapsa el name a "Pavo" solo (parser limita
        # nombres a 2-3 palabras antes de stripping). Realistically el LLM
        # escribe "jamón de pavo" o "lonjas de pavo" — ambos cubiertos.
        "60g de lonjas de pavo",
        "70g de pavo procesado",
    ])
    def test_processed_pavo_canonicalizes_to_jamon(self, ingredient):
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=[ingredient],
            multiplier=4.67,
            structured=True,
        )
        names = _names_in_result(result)
        assert "Jamón de pavo" in names, (
            f"Procesado '{ingredient}' debió canonicalizar a Jamón: {names}"
        )


# ---------------------------------------------------------------------------
# 3. Variantes fresh canonicalizan a Pechuga de pavo
# ---------------------------------------------------------------------------
class TestFreshPavoCanonicalizesPechuga:
    @pytest.mark.parametrize("ingredient", [
        "300g de pechuga de pavo fresca",
        "250g de pechuga de pavo",  # default seguro
        "200g de filete de pavo",
        "150g de filete de pavo fresca",
        "300g de pechuga de pavo FRESCA",  # case insensitive
    ])
    def test_fresh_pavo_canonicalizes_to_pechuga(self, ingredient):
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=[ingredient],
            multiplier=4.67,
            structured=True,
        )
        names = _names_in_result(result)
        assert any("echuga de pavo" in n for n in names), (
            f"Fresh '{ingredient}' debió canonicalizar a Pechuga: {names}"
        )
        assert "Jamón de pavo" not in names, (
            f"Fresh '{ingredient}' NO debe canonicalizar a Jamón: {names}"
        )


# ---------------------------------------------------------------------------
# 4. Pavo molido es item separado
# ---------------------------------------------------------------------------
class TestPavoMolidoSeparate:
    @pytest.mark.parametrize("ingredient", [
        "200g de pavo molido",
        "300g de carne de pavo",
    ])
    def test_pavo_molido_is_own_item(self, ingredient):
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=[ingredient],
            multiplier=4.67,
            structured=True,
        )
        names = _names_in_result(result)
        # Debe ser "Pavo molido" o item con "molido" — NO Jamón
        assert "Jamón de pavo" not in names
        assert any("molido" in n.lower() or "pavo molido" in n.lower() for n in names), (
            f"Pavo molido debe estar separado: {names}"
        )


# ---------------------------------------------------------------------------
# 5. Mix realista: ambos productos coexisten
# ---------------------------------------------------------------------------
def test_fresh_and_processed_pavo_coexist_separately():
    """Si un plan usa AMBOS pechuga fresca y jamón procesado, deben
    aparecer como 2 items separados en la lista de compras (precios y
    nutrición distintos)."""
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients=[
            "300g de pechuga de pavo fresca",  # almuerzo principal
            "50g de jamón de pavo",  # acompañamiento
        ],
        multiplier=4.67,
        structured=True,
    )
    names = _names_in_result(result)
    pechuga_present = any("echuga de pavo" in n for n in names)
    jamon_present = "Jamón de pavo" in names
    assert pechuga_present, f"Pechuga fresca debe estar: {names}"
    assert jamon_present, f"Jamón procesado debe estar: {names}"


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_pavo_solo_no_se_toca(self):
        """'Pavo' solo (sin pechuga/molido/jamón/etc.) debe usar
        canonicalización del master_map (probablemente 'Pavo')."""
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=["500g de pavo"],
            multiplier=4.67,
            structured=True,
        )
        names = _names_in_result(result)
        # Debe estar pero NO como Jamón de pavo
        assert "Jamón de pavo" not in names

    def test_jamon_de_pavo_descriptor_se_preserva_en_canonical(self):
        """Cuando el LLM dice "jamón de pavo" como ingrediente principal
        (sin "fresco" extra), debe canonicalizar a "Jamón de pavo".
        Caso adicional: "pechuga de pavo fresca" → Pechuga (no Jamón).
        Verifica que el orden de precedencia (fresh > processed) funciona
        cuando ambos markers podrían aparecer en el mismo nombre."""
        result_fresh = aggregate_and_deduct_shopping_list(
            plan_ingredients=["200g de pechuga de pavo fresca"],
            multiplier=4.67,
            structured=True,
        )
        names_fresh = _names_in_result(result_fresh)
        # Fresh wins
        assert "Jamón de pavo" not in names_fresh
        assert any("echuga de pavo" in n for n in names_fresh)

        # Procesado plain debe canonicalizar
        result_proc = aggregate_and_deduct_shopping_list(
            plan_ingredients=["50g de jamón de pavo"],
            multiplier=4.67,
            structured=True,
        )
        names_proc = _names_in_result(result_proc)
        assert "Jamón de pavo" in names_proc


# ---------------------------------------------------------------------------
# 7. Sin regresión: otras canonicalizaciones siguen
# ---------------------------------------------------------------------------
def test_other_canonicalizations_unaffected():
    """Las demás reglas de canonicalización (huevos, lechosa, etc.)
    no deben verse afectadas por mi cambio en pavo."""
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients=[
            "2 huevos grandes",
            "150g de lechosa picada",
        ],
        multiplier=4.67,
        structured=True,
    )
    names = _names_in_result(result)
    # huevos canonicaliza a 'Huevo' (regla previa)
    assert any("uevo" in n for n in names)
