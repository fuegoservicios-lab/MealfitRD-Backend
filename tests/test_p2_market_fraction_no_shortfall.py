"""[P2-MARKET-FRACTION-NO-SHORTFALL · 2026-07-31] La lista de compras mandaba comprar MENOS de lo
que las recetas piden.

Medido sobre el plan real fe788498 (46 ítems, 34 evaluables en peso):

    Molondrones   necesita 294 g · compras 227 g (½ lb)    −22,9%
    Pulpo         necesita 513 g · compras 454 g (1 lb)    −11,6%
    Cangrejo      necesita 502 g · compras 454 g (1 lb)     −9,6%
    Cebolla       necesita 600 g · compras 567 g (1¼ lbs)   −5,5%
    Chivo         necesita 700 g · compras 680 g (1½ lbs)   −2,8%

Causa: el paso de presentación redondea al cuarto de libra MÁS CERCANO
(`if frac_w < 0.15: ""` / `<= 0.35: "1/4"` / `<= 0.65: "1/2"` / `<= 0.85: "3/4"`).

Lo importante es de dónde viene el número que redondea: `lbs_for_weighable` sale de `units_count`,
que YA pasó por el anti-desperdicio y un `ceil`. O sea que **la cantidad era correcta y es el
redondeo de presentación el que la baja por debajo del requisito**. No es un problema de cálculo:
es que la última capa deshace el trabajo de la anterior.

En una lista de compras la asimetría importa: que sobre es recuperable (queda en la nevera), que
falte te deja sin poder cocinar el plato. Pero subir SIEMPRE al siguiente cuarto encarece de más una
proteína cara por un déficit trivial — el chivo pasaría de −2,8% a +13%. Por eso el criterio es una
TOLERANCIA: se permite redondear hacia abajo solo si el déficit es despreciable.

Anchor de producción: P2-MARKET-FRACTION-NO-SHORTFALL.
"""
import pytest

LB_G = 453.592


def _comprado_g(whole, frac_str):
    from shopping_calculator import _FRACTION_DECIMAL
    return (whole + _FRACTION_DECIMAL.get(frac_str, 0.0)) * LB_G


# (nombre, gramos que el plan necesita)  — los cinco casos medidos en producción
CASOS_REALES = [
    ("Molondrones", 294.0),
    ("Pulpo", 513.33),
    ("Cangrejo", 502.0),
    ("Cebolla", 600.0),
    ("Chivo", 700.0),
]


@pytest.mark.parametrize("nombre,necesita_g", CASOS_REALES)
def test_ningun_item_se_queda_significativamente_corto(nombre, necesita_g):
    """El caso de producción: comprar menos de lo que la receta pide."""
    from shopping_calculator import _lbs_to_market_fraction, MARKET_FRACTION_SHORTFALL_TOL

    whole, frac = _lbs_to_market_fraction(necesita_g / LB_G)
    comprado = _comprado_g(whole, frac)
    deficit = (necesita_g - comprado) / necesita_g

    assert deficit <= MARKET_FRACTION_SHORTFALL_TOL + 1e-9, (
        f"{nombre}: necesita {necesita_g:.0f} g y manda comprar {comprado:.0f} g "
        f"({-deficit*100:+.1f}%), por encima de la tolerancia "
        f"({MARKET_FRACTION_SHORTFALL_TOL*100:.0f}%)"
    )


def test_no_sobre_compra_cuando_el_deficit_es_trivial():
    """Control anti-sobrecoste: el chivo se queda a −2,8%, dentro de tolerancia → NO se sube.

    Subir siempre al cuarto siguiente encarecería una proteína cara por un déficit que nadie nota.
    """
    from shopping_calculator import _lbs_to_market_fraction

    whole, frac = _lbs_to_market_fraction(700.0 / LB_G)   # 1.543 lbs
    assert (whole, frac) == (1, "1/2"), (
        f"el chivo debía quedarse en 1½ lbs (−2,8%, despreciable) y salió {whole} {frac!r}"
    )


def test_nunca_devuelve_cero_para_un_peso_real():
    """Un peso pequeño pero real no puede convertirse en 'no compres nada'."""
    from shopping_calculator import _lbs_to_market_fraction

    for g in (60.0, 100.0, 200.0):
        whole, frac = _lbs_to_market_fraction(g / LB_G)
        assert _comprado_g(whole, frac) > 0, f"{g} g -> cantidad 0"


@pytest.mark.parametrize("lbs,esperado", [
    (2.0, (2, "")),        # exacto: no inventa fracción
    (1.0, (1, "")),
    (0.25, (0, "1/4")),
    (0.5, (0, "1/2")),
])
def test_los_valores_exactos_no_se_alteran(lbs, esperado):
    """Control: un peso que ya cae en un cuarto exacto se deja como está."""
    from shopping_calculator import _lbs_to_market_fraction
    assert _lbs_to_market_fraction(lbs) == esperado


def test_es_monotono():
    """Más gramos nunca pueden dar menos compra."""
    from shopping_calculator import _lbs_to_market_fraction
    prev = -1.0
    g = 50.0
    while g <= 2000.0:
        whole, frac = _lbs_to_market_fraction(g / LB_G)
        actual = _comprado_g(whole, frac)
        assert actual >= prev - 1e-9, f"a {g:.0f} g la compra BAJÓ ({actual:.0f} < {prev:.0f})"
        prev = actual
        g += 25.0
