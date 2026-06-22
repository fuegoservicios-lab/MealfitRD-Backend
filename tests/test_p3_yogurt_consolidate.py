"""[P3-YOGURT-CONSOLIDATE · 2026-06-22] Todo yogurt resuelve a UN solo ítem de
compra "Yogurt" (pedido del owner: una sola línea en la lista, sin "griego").

El guard en `normalize_name` (mismo patrón que el guard de pavo) short-circuita
ANTES del alias lookup → todas las variantes ("yogurt griego entero", "yogurt
griego sin azúcar", "yogur natural", etc.) colapsan a "Yogurt" → el aggregator,
que keya por `normalize_name`, produce UNA sola línea.

La distinción nutricional entero (fat ~4g/100g) vs nonfat (~0.37g) NO se pierde:
nutrition_db resuelve las variantes por sus aliases en Tier-1/2 ANTES de delegar
a este `normalize_name` (Tier-3). El master row "Yogurt" provee precio/envase
(pote). Simétrico con canonicalize_lacteo (coherencia recetas↔lista → 'Yogur').
"""
import pytest

import shopping_calculator as sc


@pytest.mark.parametrize("variant", [
    "yogurt griego entero",
    "Yogurt griego sin azúcar",
    "yogur griego",
    "yogurt griego natural",
    "yogurt natural",
    "yogurt",
    "yogur",
    "Yogur descremado",
    "yogurt griego 0%",
    "yogur griego sin grasa",
    "150g de yogurt griego light",
])
def test_yogurt_variant_resolves_to_single_canonical(variant):
    # P3-YOGURT-CONSOLIDATE: el guard devuelve 'Yogurt' sin tocar el master list.
    assert sc.normalize_name(variant) == "Yogurt", variant


@pytest.mark.parametrize("non_yogurt", [
    "leche entera", "Queso blanco", "Mantequilla", "Crema de leche", "Helado de vainilla",
])
def test_non_yogurt_NOT_collapsed_to_yogurt(non_yogurt):
    # El guard solo dispara con \\byogur(t)?\\b — no debe tragarse otros lácteos.
    assert sc.normalize_name(non_yogurt) != "Yogurt", non_yogurt


def test_coherence_symmetry_yogurt_maps_to_yogur():
    # La línea de lista "Yogurt" y las variantes de receta canonicalizan al MISMO
    # bucket de coherencia → cero divergencia recetas↔lista que fuerce retry.
    assert sc.canonicalize_lacteo("Yogurt") == "Yogur"
    assert sc.canonicalize_lacteo("yogurt griego entero") == "Yogur"
    assert sc.canonicalize_lacteo("yogurt griego sin azúcar") == "Yogur"
