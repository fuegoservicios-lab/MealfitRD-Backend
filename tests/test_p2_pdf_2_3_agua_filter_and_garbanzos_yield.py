"""[P2-PDF-2 + P2-PDF-3] Tests para dos bugs encontrados en el PDF
2026-05-05 (post P0/P1/P2 fixes anteriores):

P2-PDF-2 — "Agua fría" entró a la lista de compras:
  El set `IGNORE_SHOPPING` en `aggregate_and_deduct_shopping_list` era
  match LITERAL ('agua', 'hielo', 'agua potable', 'cubos de hielo').
  Variantes que el LLM usó: "agua fría", "agua tibia", "agua mineral",
  "agua filtrada" — ninguna estaba listada → entraron al PDF como items
  a comprar (caso real: "Agua fría — 3 lbs" en sección OTROS).

  Fix: cambio a match por PALABRA-PREFIX normalizada. Cualquier nombre
  que sea exactamente "agua"/"hielo" o empiece con "agua "/"hielo " se
  ignora. Boundary check evita falso-skip de "aguaymanto" (fruta real).

P2-PDF-3 — Garbanzos cocidos sin yield 0.35× cocido→seco:
  En P2-PDF-1 expandimos la regla `_calculate_yield_multiplier` para
  legumbres/granos cocidos, pero el regex incluía `lentejas|habichuelas|
  frijoles|guandules` y olvidé `garbanzos`. Resultado en PDF: 11
  paquetes (1 lb c/u) = 11 lbs secas para un plan que solo necesitaba
  ~5 lbs (over-buy 2×). `soya` y `tofu` añadidos por simetría.

Cobertura:
  - Filter agua: 6 variantes filtradas + boundary check (aguaymanto)
  - Filter hielo: legacy + variantes
  - Yield garbanzos / soya / tofu cocidos
  - Regresión: legumbres anteriores siguen funcionando
  - Regresión: items legítimos no se filtran
"""
import pytest

from shopping_calculator import (
    _calculate_yield_multiplier,
    aggregate_and_deduct_shopping_list,
)


# ---------------------------------------------------------------------------
# 1. P2-PDF-2: filtro de agua/hielo expandido
# ---------------------------------------------------------------------------
class TestAguaHieloFilter:
    def _ingredient_names(self, items: list) -> set[str]:
        """Extrae nombres canónicos lower-case del aggregator output."""
        result = aggregate_and_deduct_shopping_list(items, [], structured=True)
        return {r.get("name", "").lower() for r in result if isinstance(r, dict)}

    @pytest.mark.parametrize("water_variant", [
        "agua fría",
        "agua fria",          # sin tilde
        "agua tibia",
        "agua caliente",
        "agua mineral",
        "agua filtrada",
        "agua del grifo",
        "agua",               # legacy literal
    ])
    def test_agua_variants_filtered(self, water_variant):
        names = self._ingredient_names([
            "200g de pollo",
            f"500ml de {water_variant}",
        ])
        # Pollo debe estar; cualquier name que empiece con "agua" debe NO estar
        assert any("pollo" in n for n in names), f"pollo missing — names={names}"
        agua_leaks = [n for n in names if n == "agua" or n.startswith("agua ")]
        assert agua_leaks == [], (
            f"Variante '{water_variant}' no fue filtrada: {agua_leaks}"
        )

    def test_hielo_legacy_filtered(self):
        names = self._ingredient_names([
            "200g de pollo",
            "5 cubos de hielo",
            "1 unidad de hielo",
        ])
        hielo_leaks = [n for n in names if n == "hielo" or n.startswith("hielo ")
                       or n == "cubos de hielo"]
        assert hielo_leaks == [], f"hielo no filtrado: {hielo_leaks}"

    def test_aguaymanto_NOT_filtered(self):
        """Boundary check: 'aguaymanto' empieza con 'agua' pero es una fruta
        real (NO debe ser tratada como agua)."""
        names = self._ingredient_names([
            "1 unidad de aguaymanto",
            "200g de pollo",
        ])
        assert any("aguaymanto" in n for n in names), (
            f"aguaymanto fue filtrado erróneamente — names={names}"
        )

    def test_aguacate_NOT_filtered(self):
        """'Aguacate' empieza con 'agua' — NO debe ser filtrado."""
        names = self._ingredient_names([
            "1 unidad de aguacate",
            "200g de pollo",
        ])
        assert any("aguacate" in n for n in names), (
            f"aguacate fue filtrado erróneamente — names={names}"
        )

    def test_pollo_legitimo_preservado(self):
        """Sanity check: items normales pasan al output."""
        names = self._ingredient_names(["200g de pollo", "1 cebolla"])
        assert any("pollo" in n for n in names)
        assert any("cebolla" in n for n in names)


# ---------------------------------------------------------------------------
# 2. P2-PDF-3: yield para garbanzos / soya / tofu cocidos
# ---------------------------------------------------------------------------
class TestGarbanzosSoyaTofuYield:
    @pytest.mark.parametrize("text,expected", [
        # Garbanzos — cocido (todas las formas)
        ("garbanzos cocidos", 0.35),
        ("garbanzo cocido", 0.35),
        ("garbanzos hervidos", 0.35),
        # Soya
        ("soya cocida", 0.35),
        ("soyas cocidas", 0.35),
        ("soya hervida", 0.35),
        # Tofu
        ("tofu cocido", 0.35),
        ("tofu hervido", 0.35),
    ])
    def test_yield_aplica_a_legumbres_extendidas(self, text, expected):
        got = _calculate_yield_multiplier(text, only_legumbres_grains=True)
        assert got == pytest.approx(expected), (
            f"{text!r}: esperado yield {expected}, recibido {got}"
        )

    @pytest.mark.parametrize("text", [
        # No tiene 'cocido'/'hervido' — yield = 1.0
        "garbanzos secos",
        "garbanzos crudos",
        "tofu firme",
        "soya texturizada",
        # No es legumbre/grano (proteína animal)
        "pollo cocido",
        "pavo hervido",
    ])
    def test_yield_NO_aplica_cuando_no_corresponde(self, text):
        got = _calculate_yield_multiplier(text, only_legumbres_grains=True)
        assert got == 1.0, (
            f"{text!r}: esperado yield 1.0 (no aplica), recibido {got}"
        )

    def test_legumbres_anteriores_siguen_funcionando_regression(self):
        """Sanity: lentejas, habichuelas, frijoles, guandules siguen
        recibiendo yield 0.35 (no regresión del fix P2-PDF-1)."""
        for legumbre in ["lentejas cocidas", "habichuelas cocidas",
                         "frijoles cocidos", "guandules cocidos",
                         "arroz integral cocido", "pasta cocida"]:
            got = _calculate_yield_multiplier(legumbre, only_legumbres_grains=True)
            assert got == 0.35, f"Regresión: {legumbre!r} → yield {got}, esperado 0.35"


# ---------------------------------------------------------------------------
# 3. Repro del PDF 2026-05-05: garbanzos en plan real
# ---------------------------------------------------------------------------
def test_repro_pdf_2026_05_05_garbanzos_correctly_yielded():
    """Plan real:
       - Día 1: 250g garbanzos cocidos
       - Día 3: 125g garbanzos cocidos (post-consolidación)
       Total ciclo: 375g cocidos × 18.67 (mensual × 2 personas) = 7000g cocidos.
       Pre-fix: 7000g secas → ~7 paquetes (1 lb) = 11 paquetes con buffering.
       Post-fix: 7000g × 0.35 = 2450g secas ≈ 5.4 lbs ≈ 6 paquetes (1 lb)."""
    from shopping_calculator import _parse_quantity

    plan_items = [
        "250g de garbanzos cocidos",
        "125g de garbanzos cocidos",
    ]
    multiplier = (2 * 4) * (7 / 3)  # 18.67

    total_g_secas = 0.0
    for item in plan_items:
        qty, unit, name = _parse_quantity(
            item, apply_yield_multiplier=False, apply_legumbres_yield_only=True,
        )
        assert unit == "g"
        # name puede ser "Garbanzos" o variante canonicalizada
        assert "garbanzo" in name.lower(), f"name inesperado: {name!r}"
        total_g_secas += qty * multiplier

    total_lbs_secas = total_g_secas / 453.592
    # Pre-fix: ~15 lbs cocidas (yield 1.0). Post-fix: ~5 lbs secas (yield 0.35).
    assert 4.0 <= total_lbs_secas <= 6.5, (
        f"Garbanzos mensual debe estar en rango realista 4-6.5 lbs secas, "
        f"recibido {total_lbs_secas:.2f} lbs (g_total={total_g_secas:.0f}g)"
    )


# ---------------------------------------------------------------------------
# 4. Repro del PDF: agua fría NO debe entrar a la lista
# ---------------------------------------------------------------------------
def test_repro_pdf_2026_05_05_agua_fria_filtered():
    """Plan tenía 'agua fría' como ingrediente. Pre-fix entró al PDF como
    'OTROS: Agua fría — 3 lbs'. Post-fix: filtrado, no aparece."""
    plan_items = [
        "200g de pollo",
        "500ml de agua fría",
        "200ml de agua fría",  # múltiples menciones se aggregan
    ]
    result = aggregate_and_deduct_shopping_list(plan_items, [], structured=True)
    names = [r.get("name", "").lower() for r in result if isinstance(r, dict)]
    agua_leaks = [n for n in names if n == "agua" or n.startswith("agua ")]
    assert agua_leaks == [], (
        f"Repro fallido: 'agua fría' apareció en lista final: {agua_leaks}"
    )
    assert any("pollo" in n for n in names), "Pollo debe estar en lista"
