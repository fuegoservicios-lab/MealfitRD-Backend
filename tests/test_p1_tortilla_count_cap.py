"""[P1-TORTILLA-COUNT-CAP · 2026-07-24] "4 tortillas de trigo" en un wrap de 1 persona.

Plan vivo `732588f8`, D1 Cena ("Wrap de Pechuga de pollo…"): la línea declaraba 4 tortillas.
`density_g_per_unit` del catálogo = 48 g → 192 g = **525 kcal y 96 g de carbohidratos**, más
carbos que los que la comida entera declaraba. Los pasos de esa MISMA receta dicen "la tortilla"
en singular y el desc "tortilla rellena": el 4 vivía solo en la línea de ingrediente, y por eso
desanclaba la etiqueta de macros (D1 Cena medía 1.81× lo declarado).

Ninguna rama de `_cap_unrealistic_portions` la veía: no es taza (`_REALISM_CUP_CAPS`), no es cdta
(`_REALISM_CDTA_LEAD_RE`), no estaba en `_REALISM_COUNT_CAPS`. Es la tercera vez que aparece la
misma lección (12/07, seed cap; 12/07, spice cap): **cada unidad necesita su rama de cap**.
"""
import graph_orchestrator as go


def _wrap(n_tortillas):
    return [{"day": 1, "meals": [{
        "name": "Wrap de Pechuga de pollo al Grill",
        "ingredients": [f"{n_tortillas} tortillas de trigo", "120 g de pechuga de pollo"],
        "recipe": ["Calienta la tortilla en un sartén y rellénala con el pollo."],
    }]}]


def test_cuatro_tortillas_se_capan_a_dos():
    days = _wrap(4)
    assert go._cap_unrealistic_portions(days) >= 1
    assert days[0]["meals"][0]["ingredients"][0] == "2 tortillas de trigo"
    assert days[0]["meals"][0].get("_portion_realism_capped") is True


def test_dos_tortillas_pasan_intactas():
    """El techo es servible, no restrictivo: un wrap de 2 tortillas (96 g, ~263 kcal) es normal."""
    days = _wrap(2)
    go._cap_unrealistic_portions(days)
    assert days[0]["meals"][0]["ingredients"][0] == "2 tortillas de trigo"


def test_kcal_antes_vs_despues():
    """Las cifras que motivaron el cap (48 g/unidad, 273.81 kcal/100 g del catálogo)."""
    antes = 4 * 48 / 100 * 273.81
    despues = 2 * 48 / 100 * 273.81
    assert round(antes) == 526 and round(despues) == 263


def test_arepa_y_casabe_tambien_tienen_rama():
    """Misma clase de unidad (pan plano contable) — se cierran juntas o el próximo plan
    reproduce el defecto con otro nombre."""
    for food in ("arepa", "casabe"):
        assert go._REALISM_COUNT_CAPS.get(food) == 2.0, f"falta el cap de {food}"
