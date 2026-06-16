"""[P6-EGGS-AGGREGATE-CAP-FIX] Tests para el fix del bug de comparación
unidad-vs-cartones en el eggs cap.

Bug observable (PDF 2026-05-05 15:34):
  Lista mostró "Huevo: 22 cartón (30 uds.) = 660 huevos" para 2p × mes,
  cuando el cap de P6-EGGS-AGGREGATE-CAP debería limitar a 8 cartones (240
  huevos). Cap nunca disparó.

Causa raíz:
  El loop comparaba el VALOR de cualquier unidad contra el threshold-de-
  unidades (240). Para `units['cartón'] = 22`, el chequeo era `22 > 240`
  → False → no cap. Mismatch de unidades en el comparador.

Fix:
  Splits el loop en 2:
    - Loop 1: 'unidad'/'unidades' contra `_eggs_cap_units` (240 huevos)
    - Loop 2: 'cartón'/'carton'/'cartones' contra `_eggs_cap_cartones` (8)

Cobertura:
  - Unit count grande → cap a 240
  - Cartón count alto → cap a 8 (NUEVO, era el bug)
  - Cap g sigue funcionando (no afectado)
  - Cap escalado por person_weeks (cartones también escalan)
  - Light proteins / no-eggs no se afectan
"""
import pytest


def _state(name, units_dict):
    return {name: dict(units_dict)}


# ---------------------------------------------------------------------------
# 1. Repro PDF: cartón=22 debe ser cap a 8 (era el bug)
# ---------------------------------------------------------------------------
def test_cartones_count_capped_to_eggs_cap_cartones():
    """[P6-EGGS-AGGREGATE-CAP-FIX] Caso del bug: aggregator pre-cap tiene
    units['cartón']=22. Cap debe reducir a 8 (8 person_weeks = 8 cartones)."""
    from shopping_calculator import aggregate_and_deduct_shopping_list, invalidate_master_cache
    invalidate_master_cache()

    # Simular el caso: usar gramos altos que el aggregator convierta a cartones
    # vía container_weight_g. Más simple: pedir 22 cartones explícitos.
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients=["22 cartones de huevos"],
        multiplier=1.0,  # raw, sin amplificar
        structured=True,
    )
    # Buscar Huevo en resultados
    from constants import strip_accents
    for r in result:
        if not isinstance(r, dict):
            continue
        name = strip_accents(r.get("name", "")).lower()
        if "huevo" not in name:
            continue
        qty = float(r.get("market_qty", 0))
        unit = (r.get("market_unit") or "").lower()
        # Cap es 8 cartones para 1 person_week (max(2, 1) = 2 actually)
        # multiplier 1.0 → person_weeks = 1.0 × 3/7 = 0.428 → max(2, 0) = 2
        # Cap = 2 cartones = 60 huevos
        if "cartón" in unit or "cartones" in unit:
            assert qty <= 8, f"Cartones cap fallido: {qty} > 8"
        elif unit in ("ud.", "uds.", "unidad", "unidades"):
            # Si display es unidades, debe ser ≤ 240 (8 cartones × 30)
            assert qty <= 240


# ---------------------------------------------------------------------------
# 2. Cap por unidades sigue funcionando (no roto por el fix)
# ---------------------------------------------------------------------------
def test_unidad_count_still_capped_correctly():
    """Sanity: el path original de cap por 'unidad' sigue operando."""
    from shopping_calculator import aggregate_and_deduct_shopping_list, invalidate_master_cache
    invalidate_master_cache()
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients=["999 huevos enteros"],
        multiplier=18.666666,  # 2p × mes
        structured=True,
    )
    from constants import strip_accents
    for r in result:
        if not isinstance(r, dict):
            continue
        name = strip_accents(r.get("name", "")).lower()
        if "huevo" not in name:
            continue
        qty = float(r.get("market_qty", 0))
        unit = (r.get("market_unit") or "").lower()
        # Cap mensual×2p = 8 cartones = 240 huevos
        if "cartón" in unit or "cartones" in unit:
            assert qty <= 8 * 1.10, f"Cartones {qty} > 8"
        elif unit in ("ud.", "uds.", "unidad", "unidades"):
            assert qty <= 240 * 1.10, f"Unidades {qty} > 240"


# ---------------------------------------------------------------------------
# 3. Cap escalado: cartones threshold escala con person_weeks
# ---------------------------------------------------------------------------
class TestCartonesThresholdScaling:
    @pytest.mark.parametrize("scenario,multiplier,expected_cartones_cap", [
        ("4p mensual", 4 * 4 * 7 / 3, 16),
        ("2p mensual", 2 * 4 * 7 / 3, 8),
        ("1p mensual", 1 * 4 * 7 / 3, 4),
        ("2p quincenal", 2 * 2 * 7 / 3, 4),
        ("2p semanal", 2 * 1 * 7 / 3, 2),
    ])
    def test_cartones_cap_scales(self, scenario, multiplier, expected_cartones_cap):
        from shopping_calculator import aggregate_and_deduct_shopping_list, invalidate_master_cache
        invalidate_master_cache()
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=["999 cartones de huevos"],
            multiplier=multiplier,
            structured=True,
        )
        from constants import strip_accents
        for r in result:
            if not isinstance(r, dict):
                continue
            name = strip_accents(r.get("name", "")).lower()
            if "huevo" not in name:
                continue
            qty = float(r.get("market_qty", 0))
            unit = (r.get("market_unit") or "").lower()
            if "cartón" in unit or "cartones" in unit:
                assert qty <= expected_cartones_cap * 1.10, (
                    f"{scenario}: cartones {qty} > {expected_cartones_cap}"
                )


# ---------------------------------------------------------------------------
# 4. Sanity: source code refleja el split
# ---------------------------------------------------------------------------
def test_source_has_split_loops():
    """Sanity guard: el fix debe tener 2 loops separados, no 1 combinado."""
    import inspect
    import shopping_calculator as sc
    src = inspect.getsource(sc.aggregate_and_deduct_shopping_list)
    # Localizar la sección eggs. Anclar en la forma con brackets
    # `[P6-EGGS-AGGREGATE-CAP]` (header de la sección real del cap) en lugar
    # del bare token, porque comentarios posteriores (P1-CAPS-COHERENCE-RECONCILE
    # al inicio de la función) listan `P6-EGGS-AGGREGATE-CAP,` entre los caps
    # acumulados en `_CAPS_APPLIED_LAST_RUN` ~1600 líneas antes del fix real,
    # lo que envenenaba el `find` y dejaba la ventana de 6000 chars fuera del fix.
    idx = src.find("[P6-EGGS-AGGREGATE-CAP]")
    assert idx >= 0, "P6-EGGS-AGGREGATE-CAP debe existir en el código"
    section = src[idx:idx + 6000]
    # Marker del fix
    assert "P6-EGGS-AGGREGATE-CAP-FIX" in section or "_eggs_cap_cartones" in section, (
        "El fix debe usar threshold separado para cartones"
    )
    # [P6-EGGS-AGGREGATE-CAP-FIX-2] Detección de cartón keys con suffix
    # ('cartón (30 uds.)') vía substring + parse del tamaño.
    assert "P6-EGGS-AGGREGATE-CAP-FIX-2" in section, (
        "Fix-2 debe estar marcado para alertar regresión silenciosa"
    )
    assert "_huevos_per_unit" in section, (
        "Cap debe parsear el tamaño del cartón del suffix"
    )
