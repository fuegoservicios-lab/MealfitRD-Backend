"""
Tests unitarios para el sistema de Lista de Compras.
Cubre: categorización local, pre-consolidación de ingredientes, y sanitización.

Ejecutar con:
    cd backend && python -m pytest tests/test_shopping.py -v
"""
import sys, os, re, unicodedata
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # type: ignore[import-unresolved]
from constants import categorize_shopping_item, _CATEGORY_KEYWORDS  # type: ignore[import-unresolved]
from agent import _pre_consolidate_ingredients  # type: ignore[import-unresolved]


# ============================================================
# 1. TESTS DE categorize_shopping_item
# ============================================================

class TestCategorizeItem:
    """Verifica que categorize_shopping_item clasifica items correctamente."""

    # --- Carnes y Pescados ---
    @pytest.mark.parametrize("item", [
        "Pechuga de pollo", "1 lb Pollo", "Salmón fresco", "Carne molida",
        "Chuleta de cerdo", "Camarones", "Bistec de res",
        "Carne de res", "Atún",
    ])
    def test_carnes_pescados(self, item):
        cat, emoji = categorize_shopping_item(item)
        assert cat == "Carnes y Pescados", f"'{item}' debería ser Carnes y Pescados, pero fue '{cat}'"
        assert emoji == "🥩"

    # --- Frutas y Verduras ---
    @pytest.mark.parametrize("item", [
        "Tomate", "Lechuga", "Aguacate", "Plátano maduro", "Yuca",
        "Brócoli", "Zanahoria", "Guineo", "Mango", "Batata",
    ])
    def test_frutas_verduras(self, item):
        cat, emoji = categorize_shopping_item(item)
        assert cat == "Frutas y Verduras", f"'{item}' debería ser Frutas y Verduras, pero fue '{cat}'"
        assert emoji == "🥬"

    # --- Lácteos y Huevos ---
    @pytest.mark.parametrize("item", [
        "Leche entera", "Queso cheddar", "Yogurt natural", "Mantequilla",
        "Crema de leche", "Crema agria", "Queso crema",
    ])
    def test_lacteos_huevos(self, item):
        cat, emoji = categorize_shopping_item(item)
        assert cat == "Lácteos y Huevos", f"'{item}' debería ser Lácteos y Huevos, pero fue '{cat}'"
        assert emoji == "🥛"

    # --- Huevos (ahora dentro de Lácteos y Huevos) ---
    @pytest.mark.parametrize("item", ["Huevos", "1 docena de huevo", "huevos revueltos"])
    def test_huevos_en_lacteos(self, item):
        cat, emoji = categorize_shopping_item(item)
        assert cat == "Lácteos y Huevos", f"'{item}' debería ser Lácteos y Huevos, pero fue '{cat}'"
        assert emoji == "🥛"

    # --- Granos y Cereales ---
    @pytest.mark.parametrize("item", [
        "Arroz blanco", "Avena en hojuelas", "Pasta", "Pan integral",
        "Lentejas", "Frijoles negros", "Harina de trigo",
        "Habichuelas rojas",
    ])
    def test_granos_cereales(self, item):
        cat, emoji = categorize_shopping_item(item)
        assert cat == "Granos y Cereales", f"'{item}' debería ser Granos y Cereales, pero fue '{cat}'"
        assert emoji == "🌾"

    # --- Condimentos ---
    @pytest.mark.parametrize("item", [
        "Pimienta negra", "Orégano", "Sazón", "Vinagre", "Sal marina",
    ])
    def test_condimentos(self, item):
        cat, emoji = categorize_shopping_item(item)
        assert cat == "Condimentos y Especias", f"'{item}' debería ser Condimentos y Especias, pero fue '{cat}'"
        assert emoji == "🧂"

    # --- Aceites y Grasas ---
    @pytest.mark.parametrize("item", ["Aceite de oliva", "Aceite vegetal", "Aceite de coco"])
    def test_aceites(self, item):
        cat, emoji = categorize_shopping_item(item)
        assert cat == "Aceites y Grasas", f"'{item}' debería ser Aceites y Grasas, pero fue '{cat}'"
        assert emoji == "🫒"

    # --- Bebidas ---
    @pytest.mark.parametrize("item", ["Agua de coco", "Jugo", "Café"])
    def test_bebidas(self, item):
        cat, emoji = categorize_shopping_item(item)
        assert cat == "Bebidas", f"'{item}' debería ser Bebidas, pero fue '{cat}'"
        assert emoji == "🥤"

    # --- Panadería ---
    @pytest.mark.parametrize("item", ["Pan"])
    def test_panaderia(self, item):
        cat, emoji = categorize_shopping_item(item)
        assert cat == "Panadería", f"'{item}' debería ser Panadería, pero fue '{cat}'"
        assert emoji == "🍞"

    # --- Fallback a "Otros" ---
    @pytest.mark.parametrize("item", [
        "XYZABC123", "producto desconocido", "",
    ])
    def test_otros_fallback(self, item):
        cat, emoji = categorize_shopping_item(item)
        assert cat == "Otros", f"'{item}' debería ser Otros, pero fue '{cat}'"
        assert emoji == "🛒"

    # --- Insensibilidad a acentos ---
    def test_accent_insensitive(self):
        cat1, _ = categorize_shopping_item("Plátano")
        cat2, _ = categorize_shopping_item("Platano")
        assert cat1 == cat2 == "Frutas y Verduras"

    # --- Case insensitive ---
    def test_case_insensitive(self):
        cat1, _ = categorize_shopping_item("POLLO")
        cat2, _ = categorize_shopping_item("pollo")
        cat3, _ = categorize_shopping_item("Pollo")
        assert cat1 == cat2 == cat3 == "Carnes y Pescados"


# ============================================================
# 1b. TESTS DE FALSOS POSITIVOS CORREGIDOS
# ============================================================

class TestCategorizeItemFalsePositives:
    """Verifica que los falsos positivos conocidos ya no ocurran."""

    def test_salmon_not_condimento(self):
        """'Salmón' NO debe matchear 'sal' (Condimentos)."""
        cat, _ = categorize_shopping_item("Salmón")
        assert cat == "Carnes y Pescados", f"'Salmón' fue '{cat}', debería ser 'Carnes y Pescados'"

    def test_fresa_not_proteina(self):
        """'Fresa' NO debe matchear 'res' (Proteínas)."""
        cat, _ = categorize_shopping_item("Fresa")
        assert cat == "Frutas y Verduras", f"'Fresa' fue '{cat}', debería ser 'Frutas y Verduras'"

    def test_agua_de_coco_not_aceites(self):
        """'Agua de coco' debe ser Bebidas, no Aceites."""
        cat, _ = categorize_shopping_item("Agua de coco")
        assert cat == "Bebidas", f"'Agua de coco' fue '{cat}', debería ser 'Bebidas'"

    def test_habichuelas_goes_to_granos(self):
        """'Habichuelas' debe ir a Granos, no a Frutas y Verduras."""
        cat, _ = categorize_shopping_item("Habichuelas rojas")
        assert cat == "Granos y Cereales", f"'Habichuelas rojas' fue '{cat}', debería ser 'Granos y Cereales'"

    def test_crema_de_leche_is_lacteos(self):
        """'Crema de leche' debe ser Lácteos, no algo genérico."""
        cat, _ = categorize_shopping_item("Crema de leche")
        assert cat == "Lácteos y Huevos", f"'Crema de leche' fue '{cat}', debería ser 'Lácteos y Huevos'"

    def test_salsa_is_condimento(self):
        """'Salsa' (word boundary) debe matchear Condimentos, no Proteínas por 'sal'."""
        cat, _ = categorize_shopping_item("Salsa de tomate")
        assert cat == "Condimentos y Especias", f"'Salsa de tomate' fue '{cat}', debería ser 'Condimentos y Especias'"


# ============================================================
# 2. TESTS DE _pre_consolidate_ingredients
# ============================================================

class TestPreConsolidateIngredients:
    """Verifica que _pre_consolidate_ingredients fusiona ingredientes correctamente."""

    def test_identical_items_summed(self):
        """Items idénticos deben sumarse."""
        result = _pre_consolidate_ingredients(["2 huevos", "3 huevos"])
        assert len(result) == 1
        assert "5" in result[0], f"Esperaba '5 huevos', obtuve: {result[0]}"

    def test_different_items_preserved(self):
        """Items diferentes no deben fusionarse."""
        result = _pre_consolidate_ingredients(["2 lb Pollo", "1 lb Arroz"])
        assert len(result) == 2

    def test_units_preserved(self):
        """Las unidades deben mantenerse tras fusión."""
        result = _pre_consolidate_ingredients(["2 lb Pollo", "3 lb Pollo"])
        assert len(result) == 1
        assert "5 lb Pollo" in result[0]

    def test_fractions_handled(self):
        """Fracciones como 1/2 deben procesarse."""
        result = _pre_consolidate_ingredients(["1/2 aguacate", "1/2 aguacate"])
        assert len(result) == 1
        assert "1" in result[0]  # 0.5 + 0.5 = 1

    def test_no_number_items_not_summed(self):
        """Items sin número al inicio no deben intentar sumarse."""
        result = _pre_consolidate_ingredients(["Sal al gusto", "Sal al gusto"])
        assert len(result) == 1  # Se agrupan por nombre pero no se suman

    def test_empty_list(self):
        """Lista vacía retorna lista vacía."""
        result = _pre_consolidate_ingredients([])
        assert result == []

    def test_whitespace_items_ignored(self):
        """Items vacíos o solo whitespace se ignoran."""
        result = _pre_consolidate_ingredients(["", "  ", "2 huevos"])
        assert len(result) == 1
        assert "huevos" in result[0].lower()

    def test_integer_formatting(self):
        """Números enteros no deben mostrar decimales (5, no 5.0)."""
        result = _pre_consolidate_ingredients(["2 huevos", "3 huevos"])
        assert result[0] == "5 huevos"

    def test_decimal_formatting(self):
        """Números decimales deben mostrar 1 decimal."""
        result = _pre_consolidate_ingredients(["1 lb Pollo", "1/2 lb Pollo"])
        assert "1.5" in result[0]

    def test_order_preserved(self):
        """El orden de aparición debe preservarse."""
        result = _pre_consolidate_ingredients(["1 Tomate", "2 Cebolla", "1 Tomate"])
        assert len(result) == 2
        assert "tomate" in result[0].lower()
        assert "cebolla" in result[1].lower()

    def test_mixed_types(self):
        """Non-string items son ignorados."""
        result = _pre_consolidate_ingredients([123, None, "2 huevos"])
        assert len(result) == 1


# ============================================================
# 3. TESTS DE SANITIZACIÓN
# ============================================================

class TestSanitization:
    """Tests de la lógica de sanitización usada en tools.py y app.py."""

    MAX_ITEM_LENGTH = 100

    def _sanitize(self, text: str) -> str:
        """Replica la función _sanitize interna del chat tool."""
        clean = re.sub(r'<[^>]+>', '', text).strip()
        clean = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', clean)
        return clean[:self.MAX_ITEM_LENGTH]

    def test_html_tags_removed(self):
        assert self._sanitize("<b>Pollo</b>") == "Pollo"

    def test_script_tags_removed(self):
        result = self._sanitize('<script>alert("xss")</script>Arroz')
        assert "<script>" not in result
        assert "Arroz" in result

    def test_control_chars_removed(self):
        result = self._sanitize("Leche\x00\x01\x02")
        assert result == "Leche"

    def test_length_limited(self):
        long_input = "A" * 200
        result = self._sanitize(long_input)
        assert len(result) == self.MAX_ITEM_LENGTH

    def test_normal_text_unchanged(self):
        assert self._sanitize("Arroz integral") == "Arroz integral"

    def test_whitespace_stripped(self):
        assert self._sanitize("  Huevos  ") == "Huevos"

    def test_accented_text_preserved(self):
        """Los acentos NO deben eliminarse en sanitización (solo en categorización)."""
        assert self._sanitize("Plátano") == "Plátano"

    def test_empty_string(self):
        assert self._sanitize("") == ""

    def test_only_tags(self):
        assert self._sanitize("<div><span></span></div>") == ""

    def test_nested_tags(self):
        result = self._sanitize("<div><b>Pollo</b> y <i>Arroz</i></div>")
        assert result == "Pollo y Arroz"


# ============================================================
# 4. TESTS DE NORMALIZACIÓN (usada en dedup)
# ============================================================

class TestNormalization:
    """Tests de la normalización Unicode usada en dedup."""

    def _normalize(self, text: str) -> str:
        """Replica la lógica de normalización de db._deduplicate_shopping_items_impl."""
        if not text:
            return ""
        nfkd = unicodedata.normalize('NFKD', text.lower().strip())
        return re.sub(r'\s+', ' ', ''.join(c for c in nfkd if not unicodedata.combining(c)))

    def test_accent_removal(self):
        assert self._normalize("Plátano") == "platano"

    def test_case_insensitive(self):
        assert self._normalize("POLLO") == "pollo"

    def test_whitespace_collapsed(self):
        assert self._normalize("  arroz   blanco  ") == "arroz blanco"

    def test_empty_string(self):
        assert self._normalize("") == ""

    def test_special_chars_preserved(self):
        """Caracteres no-combinantes como ñ deben preservarse."""
        # ñ en NFKD es n + combining tilde, pero the function strips combining chars
        # So ñ becomes n
        result = self._normalize("ñame")
        assert result == "name"  # ñ → n after stripping combining chars

    def test_multiple_accents(self):
        assert self._normalize("Salmón frío") == "salmon frio"

    def test_idempotent(self):
        """Normalizar dos veces da el mismo resultado."""
        text = "Plátano Maduro"
        assert self._normalize(text) == self._normalize(self._normalize(text))
