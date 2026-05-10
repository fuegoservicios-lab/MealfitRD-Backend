"""[P1-PDF-5] Tests para `_format_pkg_suffix` y la disambiguación per-package
vs total en `display_qty`.

Bug original (UX, observado en PDF 2026-05-04):
  El formato literal `f"({sku_label})"` para todos los casos hacía que
  cuando count > 1 el usuario no pudiera distinguir si el valor en
  paréntesis era TOTAL o POR EMPAQUE:
    - "16 paquetes (1 lb)"   ¿16 paquetes que SUMAN 1 lb? ¿O 16 paquetes
                              de 1 lb c/u = 16 lbs?
    - "13 sobres (14g)"      Físicamente imposible: 13 sobres no caben en
                              14g; el `14g` es POR sobre. Lectura errónea
                              llevaba al usuario a comprar de menos.
    - "9 potes (16 oz)"      Tras P1-PDF-4 fix, sin sufijo seguía ambiguo.
    - "10 Uds. (~33.1 lbs)"  Mega-frutas: ¿cada lechosa pesa ~33 lbs? ¿O
                              10 lechosas suman ~33 lbs?

Fix:
  Helper `_format_pkg_suffix(count, label)`:
    - count <= 1          → `(label)`           (sin ambigüedad)
    - count >  1 + ~      → `(label total)`     (mega-frutas: peso TOTAL)
    - count >  1 + exacto → `(label c/u)`       (containers: POR EMPAQUE)

  Convención dominicana: "c/u" = "cada uno" (etiquetas de góndola).

Cobertura:
  - Helper unitario (4 casos)
  - Block 1 SKU path (yogurt mensual)
  - Block 1 standard path (lentejas, oregano)
  - Block 2 mega-frutas (aguacate, lechosa)
  - Casos count=1 que NO deben llevar c/u (preservar legibilidad)
  - Cadena `display_string` final con name (formato cliente-facing)
"""
import pytest

from shopping_calculator import (
    _format_pkg_suffix,
    _has_pkg_suffix,
    apply_smart_market_units,
)


# ---------------------------------------------------------------------------
# 1. Helper unitario
# ---------------------------------------------------------------------------
class TestFormatPkgSuffix:
    def test_count_1_returns_plain_label(self):
        assert _format_pkg_suffix(1, "150g") == "(150g)"
        assert _format_pkg_suffix(1, "16 oz") == "(16 oz)"

    def test_count_0_treated_as_singular(self):
        # MARKET_MINIMUMS bumps a 0.5/0.25 → int(float()) = 0 → no c/u
        assert _format_pkg_suffix(0, "1 lb") == "(1 lb)"
        assert _format_pkg_suffix(0.5, "1 lb") == "(1 lb)"

    def test_count_gt_1_exact_label_adds_cu(self):
        assert _format_pkg_suffix(9, "16 oz") == "(16 oz c/u)"
        assert _format_pkg_suffix(13, "14g") == "(14g c/u)"
        assert _format_pkg_suffix(2, "453g") == "(453g c/u)"

    def test_count_gt_1_approx_label_adds_total(self):
        assert _format_pkg_suffix(10, "~5.5 lbs") == "(~5.5 lbs total)"
        assert _format_pkg_suffix(10, "~33.1 lbs") == "(~33.1 lbs total)"

    def test_empty_label_returns_empty(self):
        assert _format_pkg_suffix(5, "") == ""
        assert _format_pkg_suffix(5, None) == ""

    def test_string_count_coerced(self):
        assert _format_pkg_suffix("9", "16 oz") == "(16 oz c/u)"
        assert _format_pkg_suffix("1", "150g") == "(150g)"

    def test_invalid_count_degrades_to_singular(self):
        assert _format_pkg_suffix("abc", "1 lb") == "(1 lb)"
        assert _format_pkg_suffix(None, "1 lb") == "(1 lb)"


class TestHasPkgSuffix:
    def test_recognizes_legacy_format(self):
        assert _has_pkg_suffix("9 potes (16 oz)", "16 oz") is True

    def test_recognizes_cu_variant(self):
        assert _has_pkg_suffix("9 potes (16 oz c/u)", "16 oz") is True

    def test_recognizes_total_variant(self):
        assert _has_pkg_suffix("10 Uds. (~5 lbs total)", "~5 lbs") is True

    def test_returns_false_when_absent(self):
        assert _has_pkg_suffix("9 potes", "16 oz") is False

    def test_empty_inputs_return_false(self):
        assert _has_pkg_suffix("", "16 oz") is False
        assert _has_pkg_suffix("9 potes", "") is False
        assert _has_pkg_suffix("9 potes (X)", None) is False


# ---------------------------------------------------------------------------
# 2. End-to-end via apply_smart_market_units — escenarios reales del PDF
# ---------------------------------------------------------------------------
class TestRealPDFScenariosDisambiguated:
    def test_yogurt_monthly_2_personas_has_cu(self):
        """Caso real producción: 8.23 lbs de yogurt → 9 potes (16 oz c/u)."""
        master = {
            "market_container": "pote",
            "container_weight_g": 453,
            "available_sizes_g": [150, 227, 453],
            "category": "Lácteos",
            "shelf_life_days": 14,
        }
        result = apply_smart_market_units(
            "Yogurt griego sin azúcar", 8.23, "lb", 0.0, master
        )
        assert "c/u" in result["display_qty"], (
            f"Yogurt mensual debe llevar 'c/u', recibido: {result['display_qty']!r}"
        )

    def test_oregano_13_sobres_has_cu_not_misleading_total(self):
        """Caso PDF: '13 sobres (14g)' físicamente imposible si 14g fuera total.
        Ahora '13 sobres (14g c/u)' deja claro que cada sobre pesa 14g."""
        master = {
            "market_container": "sobre",
            "container_weight_g": 14,
            "category": "Despensa",
            "shelf_life_days": 14,
        }
        result = apply_smart_market_units(
            "Orégano dominicano", 0.4, "lb", 0.0, master
        )
        # 0.4 lb ≈ 181g → 13 sobres de 14g
        assert "c/u" in result["display_qty"]
        assert "sobres" in result["display_qty"]
        # `14g` debe estar presente (peso por sobre)
        assert "14g" in result["display_qty"]

    def test_lentejas_paquetes_has_cu(self):
        """'X paquetes (1 lb)' → 'X paquetes (1 lb c/u)'."""
        master = {
            "market_container": "paquete",
            "container_weight_g": 453,
            "category": "Despensa",
            "shelf_life_days": 14,
        }
        result = apply_smart_market_units("Lentejas", 10.0, "lb", 0.0, master)
        assert "c/u" in result["display_qty"]
        assert "paquetes" in result["display_qty"]

    def test_almendras_multiple_paquetes_has_cu(self):
        master = {
            "market_container": "paquete",
            "container_weight_g": 170,
            "available_sizes_g": [100, 170, 227],
            "category": "Despensa",
            "shelf_life_days": 14,
        }
        result = apply_smart_market_units("Almendras fileteadas", 1.0, "lb", 0.0, master)
        # ≈ 453g → 2-3 paquetes
        if int(result["market_qty"]) > 1:
            assert "c/u" in result["display_qty"]

    # ── Mega-frutas: sufijo " total" ──
    def test_aguacate_multiple_uds_has_total(self):
        master = {"density_g_per_unit": 250, "category": "Frutas", "shelf_life_days": 5}
        result = apply_smart_market_units("Aguacate", 5.5, "lb", 0.0, master)
        # ~10 unidades aguacate
        assert "total" in result["display_qty"], (
            f"Mega-fruta count>1 debe llevar 'total', recibido: {result['display_qty']!r}"
        )
        assert "~" in result["display_qty"]

    def test_lechosa_multiple_uds_has_total(self):
        master = {"density_g_per_unit": 1500, "category": "Frutas", "shelf_life_days": 5}
        result = apply_smart_market_units("Lechosa", 33.1, "lb", 0.0, master)
        if int(result["market_qty"]) > 1:
            assert "total" in result["display_qty"]


# ---------------------------------------------------------------------------
# 3. Casos donde NO debe añadirse sufijo (count == 1)
# ---------------------------------------------------------------------------
class TestSingleUnitNoCu:
    def test_single_pote_no_cu(self):
        """1 pote (16 oz) — sin ambigüedad porque solo hay 1 unidad."""
        master = {
            "market_container": "pote",
            "container_weight_g": 453,
            "available_sizes_g": [150, 227, 453],
            "category": "Lácteos",
            "shelf_life_days": 14,
        }
        # 1 lb ≈ 453g → 1 pote 16oz
        result = apply_smart_market_units(
            "Yogurt griego sin azúcar", 1.0, "lb", 0.0, master
        )
        if int(result["market_qty"]) == 1:
            assert "c/u" not in result["display_qty"], (
                f"count=1 NO debe llevar c/u, recibido: {result['display_qty']!r}"
            )

    def test_single_aguacate_no_total(self):
        """1 aguacate no necesita 'total' en el sufijo."""
        master = {"density_g_per_unit": 250, "category": "Frutas", "shelf_life_days": 5}
        result = apply_smart_market_units("Aguacate", 0.55, "lb", 0.0, master)
        if int(result["market_qty"]) == 1:
            assert " total" not in result["display_qty"]


# ---------------------------------------------------------------------------
# 4. Bloques NO afectados por el fix (carnes, weighable lbs)
# ---------------------------------------------------------------------------
class TestUnaffectedBlocks:
    def test_meat_lbs_unchanged(self):
        """Block 3 (carnes en lbs): formato '8 ½ lbs' no usa sufijo per-pkg."""
        master = {"category": "Proteínas", "shelf_life_days": 3}
        result = apply_smart_market_units("Pechuga de pollo", 8.5, "lb", 0.0, master)
        assert "c/u" not in result["display_qty"]
        assert "total" not in result["display_qty"]

    def test_native_weighable_keeps_hybrid_format(self):
        """Yuca/batata: formato 'X lbs (~Y Uds.)' — Y siempre es approx total."""
        master = {"density_g_per_unit": 400, "category": "Víveres", "shelf_life_days": 14}
        result = apply_smart_market_units("Yuca", 12.25, "lb", 0.0, master)
        # Format esperado: "12 ¼ lbs (~14 Uds.)" — el "~Uds." ya es self-explanatory.
        # NO debe duplicar sufijo c/u.
        assert "c/u" not in result["display_qty"]


# ---------------------------------------------------------------------------
# 5. display_string final: confirma que el name no se duplica con "de"
# ---------------------------------------------------------------------------
def test_display_string_contains_name_after_suffix():
    master = {
        "market_container": "pote",
        "container_weight_g": 453,
        "available_sizes_g": [150, 227, 453],
        "category": "Lácteos",
        "shelf_life_days": 14,
    }
    result = apply_smart_market_units(
        "Yogurt griego sin azúcar", 8.23, "lb", 0.0, master
    )
    # "9 potes (16 oz c/u) de Yogurt griego sin azúcar"
    assert "Yogurt griego sin azúcar" in result["display_string"]
    assert "c/u" in result["display_string"]
    # No debe haber "de de" por doble preposición
    assert "de de" not in result["display_string"]
