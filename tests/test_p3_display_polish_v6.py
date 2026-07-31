"""[P3-DISPLAY-POLISH-V6 · 2026-07-31] Dos pulidos de display que no llegaban al usuario.

(1) P3-UNICODE-FRACTION-POLISH — la regla "¼ cucharada es impráctica → 1 cdta" está INERTE para
    exactamente la forma que produce el pipeline. Su parser hace `float(qty_str)`, y `float("¼")`
    lanza ValueError → early return. Medido:

        "0.25 cda de ajonjolí" -> "1 cdta de ajonjolí"   ✓
        "1/4 cda de ajonjolí"  -> "1 cdta de ajonjolí"   ✓
        "¼ cda de ajonjolí"    -> "¼ cda de ajonjolí"    ✗  ← lo que ve el usuario

    El docstring de la función dice literalmente que cubre `"¼ cda"`. Un comentario que promete lo
    que el código no hace es peor que ninguno: invita a dar el caso por cubierto. El plan real
    fe788498 entrega 6 líneas de "¼ cda".

(2) P3-NAME-CONNECTOR-ENUM — el conector del renombrado mira si el nombre ya trae " con " pero no si
    ya termina en una enumeración, así que produce "Tostadas de Pan Integral con Ricotta y Mango y
    Yogurt" (doble "y"). Caso real del plan fe788498.

Anchor de producción: P3-DISPLAY-POLISH-V6.
"""
import pytest


# ═════════════ (1) fracciones unicode ═════════════

@pytest.mark.parametrize("entrada,esperado", [
    ("¼ cda de ajonjolí", "1 cdta de ajonjolí"),
    ("0.25 cda de ajonjolí", "1 cdta de ajonjolí"),
    ("1/4 cda de ajonjolí", "1 cdta de ajonjolí"),
])
def test_el_cuarto_de_cucharada_se_pule_venga_como_venga(entrada, esperado):
    """Las tres notaciones son el MISMO cuarto de cucharada; el pulido no puede depender de cuál llegue."""
    from humanize_ingredients import humanize_ingredient
    assert humanize_ingredient(entrada) == esperado


@pytest.mark.parametrize("frac,dec", [("¼", 0.25), ("½", 0.5), ("¾", 0.75), ("⅓", 1 / 3), ("⅔", 2 / 3)])
def test_el_parser_entiende_las_fracciones_unicode(frac, dec):
    from humanize_ingredients import _qty_str_to_float
    got = _qty_str_to_float(frac)
    assert got is not None and abs(got - dec) < 1e-6, f"{frac!r} -> {got!r}"


def test_el_parser_tolera_mixtos_y_basura():
    from humanize_ingredients import _qty_str_to_float
    assert abs(_qty_str_to_float("1½") - 1.5) < 1e-6
    assert abs(_qty_str_to_float("2") - 2.0) < 1e-6
    assert abs(_qty_str_to_float("1,5") - 1.5) < 1e-6
    assert _qty_str_to_float("") is None
    assert _qty_str_to_float("hola") is None
    assert _qty_str_to_float(None) is None


def test_no_toca_media_cucharada():
    """Control: ½ cda SÍ es practicable — el umbral (≤0.34) no puede ampliarse por accidente."""
    from humanize_ingredients import humanize_ingredient
    assert "cdta" not in humanize_ingredient("½ cda de aceite de oliva")


# ═════════════ (2) el conector del nombre ═════════════

@pytest.mark.parametrize("nombre,anadido,prohibido", [
    ("Tostadas de Pan Integral con Ricotta y Mango", "Yogurt", " y Mango y Yogurt"),
    ("Canoa de Pan Integral con Queso y Molondrones", "Soya", " y Molondrones y Soya"),
])
def test_el_nombre_no_encadena_dos_conjunciones(nombre, anadido, prohibido):
    """'…con Ricotta y Mango y Yogurt' — el conector solo miraba si ya había ' con '."""
    from graph_orchestrator import _name_connector_for
    conector = _name_connector_for(nombre)
    resultado = f"{nombre}{conector}{anadido}"
    assert prohibido not in resultado, f"doble conjunción: {resultado!r}"


def test_el_conector_sigue_usando_y_tras_un_solo_con():
    """Control: el caso que la regla original resolvía bien no puede romperse."""
    from graph_orchestrator import _name_connector_for
    assert _name_connector_for("Revoltillo con Kale") == " y "


def test_el_conector_usa_con_cuando_no_hay_ninguno():
    from graph_orchestrator import _name_connector_for
    assert _name_connector_for("Mangú de Plátano Verde") == " con "
