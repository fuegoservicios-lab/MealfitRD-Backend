"""[P1-OLIVE-CAP-REAL-CONTAINER · 2026-07-08] El cap de aceitunas (P5-OLIVE-CAP) calibraba su
peso-por-frasco a una constante HARDCODEADA (340.194g, 12oz) que quedó desincronizada del
catálogo real: SQL forense en vivo confirmó `master_ingredients.container_weight_g = 142` para
'Aceitunas' (frasco más chico que el asumido en 2026-05-05, cuando P5-OLIVE-CAP se escribió).

El cap SEGUÍA reduciendo el peso total correctamente (`_olive_cap_g` se calculaba y aplicaba), pero
`apply_smart_market_units` (BLOQUE 1, container REAL de la DB) reconvertía ese peso a frascos
usando el tamaño REAL (142g) → un cap calibrado a "3 frascos" (1021g con la constante vieja)
terminaba comprando ~8 frascos reales de 142g (2.7x de sobre-compra en UNIDADES de compra, aunque
el peso en gramos "parecía" correcto). 13 tests de `test_p6_olive_veg_citrus_caps_fixes.py`
fallaban por esto — no por un bug en la lógica del cap, sino por el drift catálogo↔constante.

Fix: resolver `container_weight_g` desde `master_map` (ya indexado en la función, línea ~7002)
en vez de hardcodear — cae al default 340.194g SOLO si el catálogo no tiene el dato para ese
ingrediente (fail-safe, preserva comportamiento histórico).
"""
import inspect

import pytest

from shopping_calculator import (
    aggregate_and_deduct_shopping_list,
    get_master_ingredients,
    invalidate_master_cache,
)


@pytest.fixture(autouse=True)
def _reset_master_cache():
    invalidate_master_cache()
    yield
    invalidate_master_cache()


def test_marker_and_default_fallback_present():
    import shopping_calculator as sc
    src = inspect.getsource(sc.aggregate_and_deduct_shopping_list)
    assert "P1-OLIVE-CAP-REAL-CONTAINER" in src
    assert "_OLIVE_FRASCO_GRAMS_DEFAULT" in src, (
        "el default hardcodeado debe seguir existiendo como fallback fail-safe"
    )
    assert 'master_map.get(_name)' in src or "master_map.get(_name.lower())" in src, (
        "debe resolver el container REAL desde master_map, no hardcodear"
    )


def test_olive_cap_calibrates_to_real_catalog_container_weight():
    """El cap final (en gramos) debe corresponder a N frascos del tamaño REAL del catálogo
    (`container_weight_g`), NO del default hardcodeado 340.194g — a menos que coincidan."""
    real_container_g = None
    for m in get_master_ingredients() or []:
        if str(m.get("name", "")).strip().lower() == "aceitunas":
            real_container_g = m.get("container_weight_g")
            break
    if not real_container_g:
        pytest.skip("catálogo local sin 'Aceitunas' — nada que verificar sin DB")
    real_container_g = float(real_container_g)

    result = aggregate_and_deduct_shopping_list(
        plan_ingredients=["500g de aceitunas"],
        multiplier=2 * 4 * 7 / 3,  # 2p mensual → person_weeks=8 → cap 3 frascos
        structured=True,
    )
    from constants import strip_accents
    row = next((r for r in result if isinstance(r, dict)
                and "aceitun" in strip_accents(r.get("name", "")).lower()), None)
    assert row is not None, f"Aceitunas debe seguir presente en la lista: {result}"
    qty = float(row.get("market_qty", 0))
    unit = (row.get("market_unit") or "").lower()

    # Cap esperado calibrado al tamaño REAL del catálogo — con margen 30% por redondeo a
    # frascos enteros (espejo del margen que ya usa test_p6_olive_veg_citrus_caps_fixes.py).
    expected_cap_g = 3 * real_container_g * 1.30
    # Cap con el default STALE — si el bug siguiera vivo, el resultado caería en esta banda.
    stale_cap_g = 3 * 340.194 * 0.95

    if unit in ("frasco", "frascos", "pote", "potes", "botella", "botellas"):
        total_g = qty * real_container_g
    elif unit == "g":
        total_g = qty
    else:
        total_g = qty * real_container_g  # fallback conservador

    assert total_g <= expected_cap_g, (
        f"cap debe calibrar a 3 frascos DEL TAMAÑO REAL ({real_container_g:.0f}g c/u ≈ "
        f"{expected_cap_g:.0f}g), no a 3×340.194g: recibido {qty} {unit} = {total_g:.0f}g"
    )
    if abs(real_container_g - 340.194) > 20:  # catálogo real difiere del default hardcodeado
        assert total_g < stale_cap_g, (
            f"si el bug del container hardcodeado siguiera vivo, el total caería cerca de "
            f"{stale_cap_g:.0f}g (3 frascos de 340.194g) — recibido {total_g:.0f}g, "
            f"sugiere que SIGUE usando la constante vieja en vez del catálogo real"
        )
