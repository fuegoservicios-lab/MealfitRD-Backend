"""[P1-APPETIT-AUTOFIX · 2026-07-04] Autofixes deterministas cena-fritura y fruta+salado.

Evidencia en vivo (2026-07-04, monitoreo de generaciones reales):
  - "Pollo Frito con Yuca" en CENA → el gate soft P2-SLOT-CENA-FRITURA rechazó y costó
    un RETRY COMPLETO de generación (tokens).
  - Pareo fruta-dulce+base-salada (huevo+mango, revoltillo+guayaba) → el LLM REINCIDIÓ
    tras la directiva del retry y el plan se entregó como advisory-final.

Patrón night-rice: reescritura determinista PRE-reviewer → el gate no dispara → cero
retry. Los gates quedan como backstop. Updates (swap/chat) NO corren estos autofixes
(el deseo explícito del usuario manda — misma doctrina del slot advisory-only).
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)


def _read(rel):
    with open(os.path.join(_BACKEND, rel), encoding="utf-8") as f:
        return f.read()


_GO = _read("graph_orchestrator.py")
_CONST = _read("constants.py")


@pytest.fixture()
def go():
    import graph_orchestrator as g
    return g


# ---------------------------------------------------------------------------
# knobs + wiring
# ---------------------------------------------------------------------------

def test_knobs_default_on():
    assert '_env_bool("MEALFIT_CENA_FRY_AUTOFIX", True)' in _GO
    assert '_env_bool("MEALFIT_FRUIT_SAVORY_AUTOFIX", True)' in _GO


def test_wired_in_assemble_pre_engine():
    i = _GO.index("cena(s) con 'arroz de noche' reescrita(s)")
    win = _GO[i:i + 2500]
    assert "_dinner_fry_autofix(days)" in win, "fry-autofix debe correr en el seam pre-motor de assemble"
    assert "_fruit_savory_autofix(days, form_data)" in win, "fruit-autofix debe correr en el mismo seam"


def test_fry_tokens_parity_with_slot_gate():
    """Todo token de la regla cena-fritura del gate (menos chicharrón, decisión documentada)
    debe tener rewrite — si el gate crece y el autofix no, este test lo detecta."""
    i = _CONST.index('"fritura pesada de proteína como plato de la cena"')
    blk = _CONST[i:i + 400]
    gate_tokens = set(re.findall(r'"([a-zñá-ú ]+)"', blk.split("tokens", 1)[1])) - {"soft", "hardness"}
    import graph_orchestrator as g
    rewrite_tokens = {t for t, _ in g._FRY_DINNER_REWRITES}
    missing = gate_tokens - rewrite_tokens - {"chicharron"}
    assert not missing, f"tokens del gate sin rewrite en el autofix: {missing}"


# ---------------------------------------------------------------------------
# fry-autofix funcional
# ---------------------------------------------------------------------------

def _mk_fry_day(slot="Cena", name="Pollo Frito con Yuca y Ensalada de Repollo"):
    return [{
        "day": 1,
        "meals": [{
            "meal": slot,
            "name": name,
            "ingredients": ["150 g de pollo", "200 g de yuca", "1 cda de aceite"],
            "recipe": [
                "Mise en place: sazona el pollo.",
                "El toque de fuego: fríe el pollo en abundante aceite 6-8 min.",
                "Montaje: sirve con la yuca.",
            ],
        }],
    }]


def test_fry_rewrites_dinner(go):
    days = _mk_fry_day()
    assert go._dinner_fry_autofix(days) == 1
    meal = days[0]["meals"][0]
    assert "frito" not in meal["name"].lower()
    assert "pollo a la plancha" in meal["name"].lower()
    steps = " ".join(meal["recipe"]).lower()
    assert "fríe" not in steps and "abundante aceite" not in steps
    assert "plancha" in steps
    assert meal["_slot_autofix_applied"] == "fry_dinner"
    # ingredientes intactos (macros intactos — cero interacción con el motor).
    assert meal["ingredients"] == ["150 g de pollo", "200 g de yuca", "1 cda de aceite"]
    # idempotente.
    assert go._dinner_fry_autofix(days) == 0


def test_fry_only_dinner_and_skips_chicharron(go):
    days_lunch = _mk_fry_day(slot="Almuerzo")
    assert go._dinner_fry_autofix(days_lunch) == 0, "almuerzo frito es legítimo — solo cena"
    days_chich = _mk_fry_day(name="Chicharrón de Cerdo con Tostones")
    assert go._dinner_fry_autofix(days_chich) == 0, \
        "chicharrón se omite a propósito (plancha destruye la identidad) — lo decide el gate"


def test_fry_knob_off(go, monkeypatch):
    monkeypatch.setattr(go, "CENA_FRY_AUTOFIX_ENABLED", False)
    assert go._dinner_fry_autofix(_mk_fry_day()) == 0


# ---------------------------------------------------------------------------
# fruit-savory-autofix funcional
# ---------------------------------------------------------------------------

def _mk_clash_day():
    return [{
        "day": 1,
        "meals": [{
            "meal": "Desayuno",
            "name": "Revoltillo de Huevos con Mango Fresco",
            "ingredients": ["2 huevos", "100 g de mango"],
            "ingredients_raw": ["2 huevos", "100 g de mango"],
            "recipe": ["Sirve el revoltillo con el mango fresco."],
        }],
    }]


def test_fruit_savory_swaps_to_aguacate(go, monkeypatch):
    _tu_calls = []
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings",
                        lambda meal, db: _tu_calls.append(meal.get("name")))
    days = _mk_clash_day()
    assert go._fruit_savory_autofix(days, {}) == 1
    meal = days[0]["meals"][0]
    assert "mango" not in meal["name"].lower()
    assert "aguacate" in meal["name"].lower()
    assert any("aguacate" in s for s in meal["ingredients"])
    assert any("aguacate" in s for s in meal["ingredients_raw"])
    assert "mango" not in " ".join(meal["recipe"]).lower()
    assert _tu_calls, "el swap fruta→aguacate cambia macros → truth-up obligatorio"
    assert meal["_slot_autofix_applied"] == "fruit_savory"
    # idempotente (el nombre ya no trae la fruta).
    assert go._fruit_savory_autofix(days, {}) == 0


def test_fruit_savory_respects_dislike_ladder(go, monkeypatch):
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    days = _mk_clash_day()
    assert go._fruit_savory_autofix(days, {"dislikes": ["aguacate"]}) == 1
    assert "batata" in days[0]["meals"][0]["name"].lower(), "dislike de aguacate → escalera a batata"


def test_fruit_savory_no_clash_no_touch(go, monkeypatch):
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    # fruta + pescado NO es clash (cocina caribeña legítima — doctrina del detector SSOT).
    days = [{
        "day": 1,
        "meals": [{"meal": "Almuerzo", "name": "Pescado en Salsa de Mango",
                   "ingredients": ["150 g de pescado", "50 g de mango"], "recipe": ["..."]}],
    }]
    assert go._fruit_savory_autofix(days, {}) == 0
    assert "mango" in days[0]["meals"][0]["name"].lower()


def test_fruit_savory_knob_off(go, monkeypatch):
    monkeypatch.setattr(go, "FRUIT_SAVORY_AUTOFIX_ENABLED", False)
    assert go._fruit_savory_autofix(_mk_clash_day(), {}) == 0


# ---------------------------------------------------------------------------
# marker
# ---------------------------------------------------------------------------

def test_marker_anchored_in_source():
    # NO pineamos _LAST_KNOWN_PFIX (cada P-fix posterior lo bumpea y rompería en cadena);
    # el contrato marker↔test lo enforza test_p2_hist_audit_14. Anclamos el marker en el SOURCE.
    assert "P1-APPETIT-AUTOFIX" in _GO
