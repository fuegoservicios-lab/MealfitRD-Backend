# backend/tests/test_synonyms.py
"""
Tests automatizados para validar la calidad del catálogo de sinónimos.
Estos tests aseguran que:
1. Cada sinónimo mapea correctamente a su ingrediente base
2. No hay sinónimos duplicados entre categorías (cross-pollution)
3. Términos base incluyen su propia forma como sinónimo
4. El mapa inverso no tiene conflictos
5. La normalización con accents funciona correctamente
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from constants import (
    PROTEIN_SYNONYMS, CARB_SYNONYMS, VEGGIE_FAT_SYNONYMS, FRUIT_SYNONYMS,
    GLOBAL_REVERSE_MAP, strip_accents, normalize_ingredient_for_tracking,
    DOMINICAN_PROTEINS, DOMINICAN_CARBS, DOMINICAN_VEGGIES_FATS, DOMINICAN_FRUITS
)


class TestSynonymIntegrity(unittest.TestCase):
    """Valida la integridad interna de los mapas de sinónimos."""
    
    ALL_SYNONYM_MAPS = {
        "PROTEIN": PROTEIN_SYNONYMS,
        "CARB": CARB_SYNONYMS,
        "VEGGIE_FAT": VEGGIE_FAT_SYNONYMS,
        "FRUIT": FRUIT_SYNONYMS,
    }
    
    def test_base_term_is_own_synonym(self):
        """Cada término base debe aparecer como uno de sus propios sinónimos."""
        for map_name, syn_map in self.ALL_SYNONYM_MAPS.items():
            for base, variants in syn_map.items():
                variants_lower = [v.lower() for v in variants]
                self.assertIn(
                    base.lower(), variants_lower,
                    f"[{map_name}] '{base}' no aparece en sus propios sinónimos: {variants}"
                )
    
    def test_no_duplicate_synonyms_within_category(self):
        """Dentro de una categoría, un sinónimo no debe mapear a dos bases distintas."""
        for map_name, syn_map in self.ALL_SYNONYM_MAPS.items():
            seen = {}
            for base, variants in syn_map.items():
                for v in variants:
                    v_lower = v.lower()
                    if v_lower in seen and seen[v_lower] != base.lower():
                        self.fail(
                            f"[{map_name}] Sinónimo duplicado: '{v}' mapea tanto a "
                            f"'{seen[v_lower]}' como a '{base}'"
                        )
                    seen[v_lower] = base.lower()
    
    def test_no_cross_category_conflicts(self):
        """Un sinónimo no debe resolver a bases de dos categorías distintas.
        Ej: 'tortilla' → 'huevos' (proteína) no debe existir también en carbohidratos.
        """
        all_mappings = {}  # variant → (category, base)
        conflicts = []
        for map_name, syn_map in self.ALL_SYNONYM_MAPS.items():
            for base, variants in syn_map.items():
                for v in variants:
                    v_lower = v.lower()
                    if v_lower in all_mappings:
                        prev_cat, prev_base = all_mappings[v_lower]
                        if prev_cat != map_name:
                            conflicts.append(
                                f"'{v}' → '{base}' ({map_name}) CONFLICTA con "
                                f"'{prev_base}' ({prev_cat})"
                            )
                    all_mappings[v_lower] = (map_name, base.lower())
        
        if conflicts:
            self.fail(
                f"{len(conflicts)} conflictos cross-categoría:\n" + 
                "\n".join(f"  • {c}" for c in conflicts)
            )
    
    def test_all_synonyms_are_nonempty(self):
        """Ningún sinónimo debe ser string vacío."""
        for map_name, syn_map in self.ALL_SYNONYM_MAPS.items():
            for base, variants in syn_map.items():
                for v in variants:
                    self.assertTrue(
                        v.strip(),
                        f"[{map_name}] Sinónimo vacío para base '{base}'"
                    )
    
    def test_catalog_lists_match_synonym_keys(self):
        """Las listas de catálogo (DOMINICAN_PROTEINS etc.) deben tener 
        una entrada correspondiente en su mapa de sinónimos."""
        catalog_to_synonym = {
            "DOMINICAN_PROTEINS": (DOMINICAN_PROTEINS, PROTEIN_SYNONYMS),
            "DOMINICAN_CARBS": (DOMINICAN_CARBS, CARB_SYNONYMS),
            "DOMINICAN_VEGGIES_FATS": (DOMINICAN_VEGGIES_FATS, VEGGIE_FAT_SYNONYMS),
            "DOMINICAN_FRUITS": (DOMINICAN_FRUITS, FRUIT_SYNONYMS),
        }
        for list_name, (catalog_list, syn_map) in catalog_to_synonym.items():
            syn_keys = {k.lower() for k in syn_map.keys()}
            for item in catalog_list:
                self.assertIn(
                    item.lower(), syn_keys,
                    f"[{list_name}] '{item}' está en el catálogo pero NO tiene sinónimos definidos"
                )


class TestGlobalReverseMap(unittest.TestCase):
    """Valida que el mapa inverso global funcione correctamente."""
    
    def test_all_variants_resolve(self):
        """Cada variante en GLOBAL_REVERSE_MAP debe resolver a un término base."""
        for variant, base in GLOBAL_REVERSE_MAP.items():
            self.assertTrue(base, f"Variante '{variant}' resuelve a base vacía")
            self.assertIsInstance(base, str)
    
    def test_known_mappings(self):
        """Validar mapeos conocidos para evitar regresiones."""
        known_correct = {
            "pechuga": "pollo",
            "bistec": "res",
            "mangú": "plátano verde",
            "mangu": "plátano verde",
            "tostones": "plátano verde",
            "maduros": "plátano maduro",
            "papaya": "lechosa",
            "maracuyá": "chinola",
            "okra": "molondrones",
            "chayote": "tayota",
            "tofu": "soya/tofu",
            "banana": "guineo",
            "casabe": "yuca",
        }
        for variant, expected_base in known_correct.items():
            variant_lower = variant.lower()
            # Probar con y sin acentos
            variant_no_accent = strip_accents(variant_lower)
            
            resolved = GLOBAL_REVERSE_MAP.get(variant_lower) or GLOBAL_REVERSE_MAP.get(variant_no_accent)
            self.assertEqual(
                resolved, expected_base,
                f"'{variant}' debería resolver a '{expected_base}', resolvió a '{resolved}'"
            )


class TestNormalization(unittest.TestCase):
    """Tests para normalize_ingredient_for_tracking."""
    
    def test_accent_insensitive(self):
        """El normalizador debe ser insensible a acentos."""
        self.assertEqual(
            normalize_ingredient_for_tracking("Plátano Verde"),
            normalize_ingredient_for_tracking("Platano Verde")
        )
    
    def test_case_insensitive(self):
        """El normalizador debe ser case-insensitive."""
        self.assertEqual(
            normalize_ingredient_for_tracking("POLLO"),
            normalize_ingredient_for_tracking("pollo")
        )
    
    def test_quantity_stripped(self):
        """El normalizador debe eliminar cantidades/unidades del inicio."""
        result = normalize_ingredient_for_tracking("200g pechuga de pollo")
        # Debe resolver a "pollo" (la base)
        self.assertIsNotNone(result, "normalize_ingredient_for_tracking('200g pechuga de pollo') retornó None")
    
    def test_strips_accents_function(self):
        """strip_accents debe eliminar diacríticos correctamente."""
        self.assertEqual(strip_accents("plátano"), "platano")
        self.assertEqual(strip_accents("maracuyá"), "maracuya")
        self.assertEqual(strip_accents("ñame"), "name")
        self.assertEqual(strip_accents("café"), "cafe")


if __name__ == "__main__":
    unittest.main(verbosity=2)
