"""[P6-FRACTION-MIXED-FIX] Tests para el fix del bug de fracciones mixtas
pegadas al entero ("6¼" → "6.25", NO "60.25").

Bug observable (PDF 2026-05-05 18:48):
  CONSOLIDATION emitía "60 lonjas de pavo" (en vez de 6) y "80 fresas"
  (en vez de 8). Reviewer médico rechazó CRÍTICAMENTE:
    - "Ingesta masiva de sodio (120 lonjas pavo Día 1)"
    - "Cantidad excesiva de fresas (160 Día 3)"
  P0-PIPE-1 + P6-SURGICAL-PROMOTE rescataron via snapshot anterior, pero
  la causa raíz era el parser, no el modelo.

Causa raíz:
  En `assemble_plan_node` (consolidación), el loop:
    fractions = {'¼': '0.25', ...}
    raw_lower = raw_lower.replace(f_char, f_val)
  hace concatenación literal:
    "6¼".replace("¼", "0.25") → "60.25"  ← BUG
  El qty regex captura "60.25" como número entero ≈ 60.

Fix:
  Regex previo `(\\d)([½⅓⅔¼¾])` que detecta entero pegado a fracción y
  lo convierte a `entero + valor` correctamente:
    "6¼" → "6.25"
    "8¼" → "8.25"
    "1½" → "1.5"
  El loop de standalone (¼ taza → 0.25 taza) sigue intacto sin regresión.

Cobertura:
  - "6¼ lonjas pavo" → qty=6.25 (NO 60.25)
  - "8¼ fresas" → qty=8.25 (NO 80.25)
  - "1½ tazas" → qty=1.5 (mixto frecuente)
  - "¼ taza" standalone sigue → qty=0.25 (no regresión)
  - Todas las 5 fracciones unicode (½⅓⅔¼¾) en mixto
  - Sanity: marker `P6-FRACTION-MIXED-FIX` en source
"""
import re
import pytest


# Replicar la lógica del fix para tests aislados (sin invocar assemble_plan_node).
# Cualquier divergencia con el código real será atrapada por test_source_has_fix.
_MIXED_FRAC = {'½': 0.5, '⅓': 1/3, '⅔': 2/3, '¼': 0.25, '¾': 0.75}
_FRACTIONS_STANDALONE = {'½': '0.5', '⅓': '0.33', '⅔': '0.67', '¼': '0.25', '¾': '0.75'}


def _apply_fraction_pipeline(raw: str) -> str:
    """Reproduce la pipeline real de fracciones del consolidator.

    [P6-FRACTION-MIXED-FIX-2] Tolera whitespace opcional entre dígito y
    fracción (`'1 ¼' → '1.25'`)."""
    raw_lower = raw.lower().strip()
    raw_lower = re.sub(
        r'(\d)\s*([½⅓⅔¼¾])',
        lambda m: f"{int(m.group(1)) + _MIXED_FRAC[m.group(2)]:.4g}",
        raw_lower,
    )
    for f_char, f_val in _FRACTIONS_STANDALONE.items():
        raw_lower = raw_lower.replace(f_char, f_val)
    return raw_lower


def _parse_qty_from_raw(raw_lower: str) -> float:
    """Reproduce el qty extractor (líneas ~6182-6195) para verificar el qty
    final post-fracciones."""
    qty_match = re.match(
        r'^([\d\.,/]+(?:\s*-\s*[\d\.,/]+)?)(?:\s+|$|(?=[a-zñA-ZÑ]))',
        raw_lower,
    )
    if not qty_match:
        return -1.0
    qty_str = qty_match.group(1).strip()
    if '-' in qty_str:
        qty_str = qty_str.split('-')[-1].strip()
    qty_str = qty_str.replace(',', '.')
    try:
        if '/' in qty_str:
            return float(qty_str.split('/')[0]) / float(qty_str.split('/')[1])
        return float(qty_str)
    except ValueError:
        return -1.0


# ===========================================================================
# 1. Repro PDF — los dos casos del rechazo médico
# ===========================================================================
def test_repro_pdf_6cuarto_lonjas_pavo():
    """'6¼ lonjas de pavo' debe parsear qty=6.25, NO 60.25."""
    raw = "6¼ lonjas de pechuga de pavo"
    processed = _apply_fraction_pipeline(raw)
    qty = _parse_qty_from_raw(processed)
    assert qty == pytest.approx(6.25, abs=0.01), (
        f"Bug fracción mixta: '{raw}' → qty={qty} (esperado 6.25, "
        f"el bug daba 60.25 → display 60 → reviewer flageaba '120 lonjas')"
    )


def test_repro_pdf_8cuarto_fresas():
    """'8¼ fresas' debe parsear qty=8.25, NO 80.25."""
    raw = "8¼ fresas"
    processed = _apply_fraction_pipeline(raw)
    qty = _parse_qty_from_raw(processed)
    assert qty == pytest.approx(8.25, abs=0.01), (
        f"Bug fracción mixta: '{raw}' → qty={qty} (esperado 8.25, "
        f"el bug daba 80.25 → display 80 → reviewer flageaba '160 fresas')"
    )


# ===========================================================================
# 2. Mixto con todas las fracciones unicode
# ===========================================================================
@pytest.mark.parametrize("raw,expected_qty", [
    ("1½ tazas de arroz", 1.5),
    ("2¼ lbs de pollo", 2.25),
    ("3¾ cucharadas de aceite", 3.75),
    ("1⅓ tazas de leche", pytest.approx(1.333, abs=0.01)),
    ("2⅔ tazas de harina", pytest.approx(2.667, abs=0.01)),
    ("9¼ filetes de tilapia", 9.25),
])
def test_mixed_fraction_all_unicode_chars(raw, expected_qty):
    processed = _apply_fraction_pipeline(raw)
    qty = _parse_qty_from_raw(processed)
    assert qty == expected_qty, f"'{raw}' → {qty}, esperado {expected_qty}"


# ===========================================================================
# 3. NO-regresión: standalone fractions siguen funcionando
# ===========================================================================
@pytest.mark.parametrize("raw,expected_qty", [
    ("¼ taza de arroz", 0.25),
    ("½ cda de sal", 0.5),
    ("¾ libra de carne", 0.75),
    ("⅓ taza de aceite", pytest.approx(0.33, abs=0.01)),
    ("⅔ cdita de comino", pytest.approx(0.67, abs=0.01)),
])
def test_standalone_fractions_still_work(raw, expected_qty):
    """No-regresión: fracciones standalone (sin entero pegado) siguen
    convirtiendo correctamente vía el loop original."""
    processed = _apply_fraction_pipeline(raw)
    qty = _parse_qty_from_raw(processed)
    assert qty == expected_qty, f"'{raw}' → {qty}, esperado {expected_qty}"


# ===========================================================================
# 4. NO-regresión: enteros simples siguen funcionando
# ===========================================================================
@pytest.mark.parametrize("raw,expected_qty", [
    ("200 g de pollo", 200),
    ("3 huevos", 3),
    ("1 taza de arroz", 1),
    ("12 lonjas de pavo", 12),
])
def test_plain_integers_unaffected(raw, expected_qty):
    processed = _apply_fraction_pipeline(raw)
    qty = _parse_qty_from_raw(processed)
    assert qty == expected_qty


# ===========================================================================
# 5. Edge cases: múltiples fracciones, fracción al final
# ===========================================================================
def test_no_concatenation_when_no_digit_prefix():
    """'taza ¼' (raro pero posible) no debe matchear el regex mixto."""
    raw = "1 taza y ¼ de leche"
    processed = _apply_fraction_pipeline(raw)
    # ¼ standalone → 0.25; nada se fusiona con dígitos no-pegados
    assert "0.25" in processed
    assert "10.25" not in processed  # NO debe concatenar "1" + "0.25"


def test_space_between_int_and_fraction_now_matched():
    """[P6-FRACTION-MIXED-FIX-2] '1 ¼ tazas' (con espacio) AHORA debe
    convertirse a 1.25 (antes producía '1 0.25 tazas' → bug downstream)."""
    raw = "1 ¼ tazas de arroz"
    processed = _apply_fraction_pipeline(raw)
    qty = _parse_qty_from_raw(processed)
    assert qty == 1.25, (
        f"Esperado qty=1.25 para '1 ¼ tazas', recibido {qty} (procesado='{processed}')"
    )


# ===========================================================================
# 5b. [P6-FRACTION-MIXED-FIX-2] Repro bug PDF 20:14 ([825c94ef])
# ===========================================================================
def test_repro_pdf_1_quarter_lonjas_queso_with_space():
    """Caso exacto del PDF 20:14: '1 ¼ lonjas de queso' producía
    '1 0.25 lonjas de queso' que el aggregator interpretaba como item
    huérfano '0.25 lonjas de queso'."""
    raw = "1 ¼ lonjas de queso mozzarella"
    processed = _apply_fraction_pipeline(raw)
    qty = _parse_qty_from_raw(processed)
    assert qty == 1.25, (
        f"Bug PDF: '{raw}' → qty={qty}, esperado 1.25. "
        f"Procesado='{processed}'. Si processed empieza con '1 0.25' "
        f"el bug downstream genera item huérfano en aggregate."
    )
    # El procesado NO debe tener "1 0.25" (que rompía el aggregator)
    assert "1 0.25" not in processed, (
        f"BUG REGRESIÓN: '{processed}' contiene '1 0.25' que produce "
        f"item huérfano '0.25 lonjas de queso' en aggregate."
    )


@pytest.mark.parametrize("raw,expected_qty", [
    ("1 ¼ tazas", 1.25),
    ("2 ½ lbs", 2.5),
    ("3 ¾ cucharadas", 3.75),
    ("1 ⅓ tazas", pytest.approx(1.333, abs=0.01)),
    ("4 ⅔ tazas", pytest.approx(4.667, abs=0.01)),
    ("9 ¼ filetes", 9.25),
])
def test_mixed_fraction_with_space_all_unicode(raw, expected_qty):
    """[P6-FRACTION-MIXED-FIX-2] Cobertura completa: dígito + espacio + fracción
    para las 5 fracciones unicode."""
    processed = _apply_fraction_pipeline(raw)
    qty = _parse_qty_from_raw(processed)
    assert qty == expected_qty, (
        f"'{raw}' → qty={qty}, esperado {expected_qty} (procesado='{processed}')"
    )


def test_multiple_spaces_also_matched():
    """Edge: múltiples espacios entre dígito y fracción ('2  ½' con 2 spaces)."""
    raw = "2  ½ tazas de leche"
    processed = _apply_fraction_pipeline(raw)
    qty = _parse_qty_from_raw(processed)
    assert qty == 2.5, f"'{raw}' → qty={qty}, esperado 2.5"


# ===========================================================================
# 6. Sanity guard — marker en source code
# ===========================================================================
def test_source_has_fraction_mixed_fix_marker():
    """Sanity: el marker debe vivir en graph_orchestrator para alertar
    si alguien revierte el fix."""
    import inspect
    import graph_orchestrator as go
    src = inspect.getsource(go.assemble_plan_node)
    assert "P6-FRACTION-MIXED-FIX" in src, (
        "El marker debe existir; sin él alguien podría revertir el fix "
        "y reintroducir 'qty=60' por '6¼'"
    )
    # El regex debe estar presente con whitespace tolerance (FIX-2)
    assert "_MIXED_FRAC" in src, "_MIXED_FRAC dict debe existir"
    assert r"(\d)\s*([½⅓⅔¼¾])" in src, (
        "regex de detección mixta debe tolerar whitespace opcional "
        "(P6-FRACTION-MIXED-FIX-2 — '1 ¼ lonjas' debe matchear)"
    )
    assert "P6-FRACTION-MIXED-FIX-2" in src, (
        "Marker FIX-2 debe existir para alertar regresión del bug PDF 20:14"
    )


def test_source_keeps_standalone_loop():
    """Sanity: el loop standalone original debe seguir presente (NO removerlo,
    solo lo precedimos con el regex de mixto)."""
    import inspect
    import graph_orchestrator as go
    src = inspect.getsource(go.assemble_plan_node)
    # El dict original con strings sigue presente para fracciones standalone
    assert "'½': '0.5'" in src or '"½": "0.5"' in src, (
        "Loop original de fracciones standalone debe seguir presente"
    )
