"""[P1-CITRUS-JUICE-YIELD · 2026-07-24] 75 g de jugo se compraban como 75 g de limón entero.

El catálogo **no tiene ninguna fila de jugo** (verificado en Neon): `jugo de limón` es un
ALIAS de `Limón`, la fruta entera. Así que la lista agregaba "2 cdas de jugo de limón" como
30 g de limón, cuando exprimir 30 g de jugo exige ~86 g de fruta.

Plan vivo `732588f8` — cuatro comidas piden jugo:

    D1 Bowl Poke          2 cdas de jugo de limón
    D1 Wrap de Pechuga    1 cda de jugo de limón
    D3 Filete de pescado  1 cda de jugo de limón fresco
    D3 Bollos de Harina   1 cda de jugo de limón fresco

5 cucharadas ≈ 75 g de jugo, que necesitan ~214 g de fruta ≈ 3.2 limones. La lista entregada
compró **2 unidades**: el usuario se queda sin limón a mitad del ciclo.

Dónde vive la regla: en el tramo compartido con la regla #1 (cocido→seco), ANTES del
early-return del modo aggregator. Es el mismo tipo de desajuste — la receta habla en una forma
y el SKU se vende en otra — y no reintroduce la asimetría P1-2 que motivó ese early-return,
porque el inventario del usuario también habla de limones enteros (nace de esta misma lista).
"""
import pytest

import shopping_calculator as sc


# ───────────── 1. el caso vivo ─────────────

@pytest.mark.parametrize("linea", [
    "2 cdas de jugo de limón",
    "1 cda de jugo de limón",
    "1 cda de jugo de limón fresco",
    "zumo de limon",
])
def test_jugo_de_limon_convierte_a_fruta_entera(linea):
    assert sc._calculate_yield_multiplier(linea, only_legumbres_grains=True) > 2.0, (
        "el jugo se compra exprimiendo fruta entera, no se vende como fila propia"
    )


def test_el_caso_del_plan_en_numeros():
    """5 cdas ≈ 75 g de jugo. Antes: 1.1 limones. Ahora: ~3.2."""
    jugo_g = 75.0
    fruta_g = jugo_g * sc._calculate_yield_multiplier("jugo de limón", only_legumbres_grains=True)
    limones = fruta_g / 67.0        # density_g_per_unit de `Limón` en el catálogo
    assert 2.8 <= limones <= 3.8, f"{limones:.1f} limones para 75 g de jugo"
    assert jugo_g / 67.0 < 1.2, "el comportamiento viejo compraba ~1 limón"


def test_aplica_en_el_modo_aggregator():
    """El aggregator llama con `only_legumbres_grains=True`, que ignora las reglas #2-4.
    Si la regla viviera después de ese early-return, la lista de compras no la vería —
    o sea, no arreglaría nada."""
    con_gate = sc._calculate_yield_multiplier("jugo de limón", only_legumbres_grains=True)
    sin_gate = sc._calculate_yield_multiplier("jugo de limón", only_legumbres_grains=False)
    assert con_gate == sin_gate > 1.0


# ───────────── 2. lo que NO debe tocar ─────────────

@pytest.mark.parametrize("linea", [
    "1 limón", "2 limones", "1 naranja",
    "ralladura de limón",        # es cáscara, no jugo
    "jugo de manzana",           # no es cítrico: se compra embotellado
    "150 g de pollo",
])
def test_sin_conversion(linea):
    assert sc._calculate_yield_multiplier(linea, only_legumbres_grains=True) == 1.0, linea


def test_las_reglas_previas_siguen_intactas():
    assert sc._calculate_yield_multiplier("200g de habichuelas cocidas", only_legumbres_grains=True) == 0.35
    assert sc._calculate_yield_multiplier("pollo cocido") == 1.35
    assert sc._calculate_yield_multiplier("yuca pelada") == 1.30


def test_se_autodesactiva_si_el_jugo_llega_a_ser_comprable(monkeypatch):
    """Si algún día el catálogo incorpora 'Jugo de limón' como producto, ya no hay nada que
    convertir. Se apaga sola en vez de doble-contar en silencio — mismo patrón que el factor
    cocido→seco, que se desactiva si la fila del catálogo pasa a estar en cocido."""
    monkeypatch.setattr(sc, "_CITRUS_JUICE_BUYABLE_CACHE", None, raising=False)
    monkeypatch.setattr(sc, "get_master_ingredients",
                        lambda *a, **k: [{"name": "Jugo de limón"}, {"name": "Limón"}])
    try:
        assert sc._calculate_yield_multiplier("jugo de limón", only_legumbres_grains=True) == 1.0
    finally:
        sc._CITRUS_JUICE_BUYABLE_CACHE = None


# ───────────── 3. knob ─────────────

def test_knob_y_multiplicador_coherentes():
    assert 0.15 <= sc.CITRUS_JUICE_YIELD <= 1.0
    assert abs(sc.CITRUS_JUICE_YIELD_MULT - 1.0 / sc.CITRUS_JUICE_YIELD) < 0.001
    from pathlib import Path
    src = Path(sc.__file__).read_text(encoding="utf-8")
    assert '_knob_env_float("MEALFIT_CITRUS_JUICE_YIELD", 0.35' in src
