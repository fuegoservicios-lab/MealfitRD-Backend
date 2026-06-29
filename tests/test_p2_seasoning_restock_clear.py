"""[P2-SEASONING-RESTOCK-CLEAR · 2026-06-29] Un condimento del catálogo verificado que el usuario YA
tiene en su Nevera (consumed/inventario) NO debe re-listarse tras un restock.

Bug reportado en testing en vivo (2026-06-29): tras un swap + "Sí, ya compré — llenar mi Nevera",
la lista de compras quedaba con UN solo ítem: "Vainilla 1 botella RD$20". Causa raíz: el plan emite
los condimentos de forma NOMINAL ("al gusto"/"pizca", sin peso) → la deducción de consumed
(`aggregated[name][unit] -= qty`, basada en peso/unidad) NO los resta; luego `SEASONING-CATALOG-KEEP`
los re-inyecta como "1 empaque" aunque el usuario ya los compró. Es la clase del bug
`P3-RESTOCK-LECHE-UNIT` (asimetría de unidad lista↔inventario), pero para condimentos del catálogo.

Fix (`aggregate_and_deduct_shopping_list`): construir un set de nombres normalizados de lo que el
usuario YA tiene (consumed_ingredients) y, en la rama SEASONING-CATALOG-KEEP, saltar el keep si el
condimento ya está en ese set. Match por `normalize_name`, fail-open.

Test parser-based (Neon-free): ancla el set de consumed + el chequeo en el seasoning-keep + el orden.
"""
import shopping_calculator as sc


def _src():
    return open(sc.__file__, encoding="utf-8").read()


def test_marker_present():
    assert "P2-SEASONING-RESTOCK-CLEAR" in _src()


def test_consumed_name_set_built_in_aggregator():
    src = _src()
    # El set se inicializa y se puebla con normalize_name dentro del loop de consumed.
    assert "_consumed_name_set = set()" in src, "Falta el set de nombres del consumed/inventario."
    assert "_consumed_name_set.add(normalize_name(name))" in src, (
        "El set debe poblarse con el nombre NORMALIZADO de cada ítem consumed."
    )


def test_seasoning_keep_skips_when_in_consumed():
    src = _src()
    # En la rama del keep, se anula _keep_seasoning si el condimento ya está en la Nevera.
    assert "normalize_name(name) in _consumed_name_set" in src, (
        "El SEASONING-CATALOG-KEEP debe saltar si el condimento ya está en el inventario del usuario."
    )
    # El log de telemetría del skip.
    assert "P2-SEASONING-RESTOCK-CLEAR] '" in src or "[P2-SEASONING-RESTOCK-CLEAR]" in src


def test_consumed_set_built_before_seasoning_check():
    """El set debe construirse (loop de consumed, temprano) ANTES de la rama del seasoning-keep
    (mucho más abajo en la misma función) — si no, estaría vacío al consultarlo."""
    src = _src()
    idx_build = src.find("_consumed_name_set.add(normalize_name(name))")
    idx_check = src.find("normalize_name(name) in _consumed_name_set")
    assert idx_build != -1 and idx_check != -1
    assert idx_build < idx_check, "El set de consumed debe poblarse antes de consultarse en el seasoning-keep."


def test_skip_is_fail_open():
    """El chequeo va dentro de try/except → un error de normalize NO debe tumbar el agregador
    (degrada al comportamiento previo: mantener el condimento)."""
    src = _src()
    idx = src.find("normalize_name(name) in _consumed_name_set")
    region = src[max(0, idx - 300): idx + 500]
    assert "try:" in region and "except Exception" in region, (
        "El skip del seasoning-keep debe ser fail-open (try/except)."
    )
