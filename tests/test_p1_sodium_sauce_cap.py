"""[P1-SODIUM-SAUCE-CAP · 2026-07-08] Cap de salsas alto-sodio en el autofix de sodio por día.

Forense del plan vivo fcb739fa (Día 3 = 2799mg sodio, sobre el techo 2000): el driver era "4 cucharadas de
salsa de soya" en la "Pasta Integral con Muslo de Pollo" (~2400mg) — que ninguna rama del autofix tocaba
(strip=cubito/sazón, salt-to-taste=solo sal cuantificada, swap=solo enlatados/curados). Las salsas alto-sodio
cuantificadas (soya/inglesa/teriyaki/pescado) ahora se acotan a ≤MAX_CDA cuando el día supera el techo.
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


def test_marker_knobs_and_regex():
    assert "P1-SODIUM-SAUCE-CAP" in _GO
    assert 'SODIUM_SAUCE_CAP_ENABLED = _env_bool("MEALFIT_SODIUM_SAUCE_CAP", True)' in _GO
    assert 'SODIUM_SAUCE_CAP_MAX_CDA = _env_int("MEALFIT_SODIUM_SAUCE_CAP_MAX_CDA", 1' in _GO
    assert "_SODIUM_SAUCE_QTY_RX" in _GO
    # corre en el autofix ANTES del swap
    i = _GO.index("def _day_sodium_autofix")
    _next = _GO.find("\ndef ", i + 10)  # cuerpo completo hasta el siguiente def
    body = _GO[i:_next if _next > i else i + 20000]
    assert "SODIUM_SAUCE_CAP_ENABLED" in body
    assert body.index("SODIUM_SAUCE_CAP_ENABLED") < body.index("swap del enlatado")


class _NaDB:
    """salsa de soya = 600mg/cda; pollo 200; resto 50."""

    def micros_from_ingredient_string(self, s):
        low = str(s).lower()
        m = re.match(r"\s*(\d+(?:[.,]\d+)?)", low)
        q = float(m.group(1).replace(",", ".")) if m else 1.0
        if "salsa de soya" in low:
            return {"sodium_mg": 600.0 * q}
        if "pollo" in low:
            return {"sodium_mg": 200.0}
        return {"sodium_mg": 50.0}

    def macros_from_ingredient_string(self, s):
        return {"kcal": 100.0}

    def grams_from_ingredient_string(self, s):
        return 50.0


@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "SODIUM_DAY_AUTOFIX_ENABLED", True)
    monkeypatch.setattr(g, "SODIUM_SAUCE_CAP_ENABLED", True)
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    return g


def _soy_day():
    line = "4 cucharadas de salsa de soya"
    return [{"day": 3, "meals": [{
        "meal": "Almuerzo", "name": "Pasta con Pollo",
        "ingredients": ["200 g de muslo de pollo", "1/4 taza de pasta integral", line],
        "ingredients_raw": ["200 g de muslo de pollo", "1/4 taza de pasta integral", line],
        "recipe": ["Mise en place: corta el pollo.",
                   "El Toque de Fuego: cocina la pasta y baña con 4 cucharadas de salsa de soya.",
                   "Montaje: sirve."]}]}]


def _soy_line(days):
    return next(l for l in days[0]["meals"][0]["ingredients"] if "salsa de soya" in l.lower())


def test_caps_soy_sauce_when_day_over_ceiling(go):
    days = _soy_day()  # 4 cdas × 600 = 2400 + pollo 200 + pasta 50 = 2650 > 2000
    n = go._day_sodium_autofix(days, {}, _NaDB())
    assert n >= 1
    assert _soy_line(days).startswith("1 cda"), f"soya capeada a 1 cda: {_soy_line(days)!r}"
    # el paso también se sincroniza
    step = " ".join(days[0]["meals"][0]["recipe"])
    assert "4 cucharadas de salsa de soya" not in step, "el paso no debe seguir diciendo 4 cucharadas"


def test_under_ceiling_untouched(go):
    """Un día bajo el techo no toca la salsa (aunque tenga 4 cdas)."""
    days = _soy_day()
    # baja el sodio de la soya a casi nada → día bajo techo
    class _LowDB(_NaDB):
        def micros_from_ingredient_string(self, s):
            return {"sodium_mg": 10.0}
    go._day_sodium_autofix(days, {}, _LowDB())
    assert _soy_line(days).startswith("4"), "día bajo techo → salsa intacta"


def test_knob_off_reverts(go, monkeypatch):
    monkeypatch.setattr(go, "SODIUM_SAUCE_CAP_ENABLED", False)
    days = _soy_day()
    go._day_sodium_autofix(days, {}, _NaDB())
    assert _soy_line(days).startswith("4"), "knob OFF → salsa no se acota"


def test_half_tbsp_not_capped(go):
    """½ cda de salsa (ya baja) no se toca aunque el día esté sobre el techo."""
    days = _soy_day()
    days[0]["meals"][0]["ingredients"][2] = "½ cda de salsa de soya"
    days[0]["meals"][0]["ingredients_raw"][2] = "½ cda de salsa de soya"
    # añade un pollo alto para superar el techo por otra vía
    days[0]["meals"][0]["ingredients"].append("filete de pescado")
    days[0]["meals"][0]["ingredients_raw"].append("filete de pescado")
    go._day_sodium_autofix(days, {}, _NaDB())
    assert "½ cda de salsa de soya" in days[0]["meals"][0]["ingredients"], "½ cda no se acota"
