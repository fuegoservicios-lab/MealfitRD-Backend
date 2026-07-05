"""[P2-DISPLAY-THIRDS · 2026-07-05] Fracciones de display que el humanizador no cubría —
screenshots del plan vivo 23c958bb:
- "0.33 taza de harina de Negrito (41g)" → el prettify solo redondeaba a CUARTOS → "⅓ taza".
- "jugo de 0.5 limón" → decimal INTERNO sin cantidad líder ("de 0.5 limón") era invisible para
  el lead-prettify → "jugo de ½ limón".
Display-only (POST macros/shopping), fail-safe.
"""


def _pretty(s):
    from humanize_ingredients import _prettify_quantity_display
    return _prettify_quantity_display(s)


def test_lead_third_converted():
    assert _pretty("0.33 taza de harina de Negrito (41g)").startswith("⅓ taza")


def test_lead_mixed_third_converted():
    assert _pretty("1.67 taza de avena").startswith("1⅔ taza")


def test_inner_decimal_half_converted():
    assert _pretty("jugo de 0.5 limón") == "jugo de ½ limón"


def test_inner_decimal_third_converted():
    assert _pretty("Jugo de 0.33 naranja agria") == "Jugo de ⅓ naranja agria"


def test_quarter_path_regression():
    assert _pretty("0.5 papa mediana").startswith("½ papa")
    assert _pretty("1.75 cdta de canela").startswith("1¾ cdta")


def test_untouched_strings():
    for s in ("Sal al gusto", "½ taza de arroz", "2 dientes de ajo", "150g de pechuga de pollo"):
        assert _pretty(s) == s
