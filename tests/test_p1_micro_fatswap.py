"""[P1-MICRO-FATSWAP + P2-MICRO-SEED-VITA + P2-MISE-COOK-SPLIT · 2026-07-06]

Los 3 subrayados amarillos del plan vivo 6ffffd39:
- Banner "plan no óptimo (micros)": omega-3 1.2<1.6 y vit E 13.8<15 con las grasas del día YA
  sobre target — los portadores de esos micros SON grasas y el budget kcal compartido se agota
  → FATSWAP: presupuesto extra propio para micros grasa-basados + re-trim ESPEJO inmediato de
  grasas del día (protege portadores, recorta grasa genérica) = swap macro-neutral.
- Chip "Vitamina A ⚠ Día 2" (1541/900 promedio, día 2 corto): el día solo tenía leche como
  portador → nada que escalar → seed de vit A (zanahoria/auyama/espinacas), slot-aware (salado
  → almuerzo/cena, nota "acompaña el plato con").
- Chip "Receta con pasos incompletos" (Tostadas): el LLM puso el tostado DENTRO del Mise →
  advisory "falta El Toque de Fuego" falso-positivo → split verbatim de la cocción a un TdF
  sintetizado + tiempo por técnica del backstop.
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


def test_knobs_default_on():
    m = re.search(r'MICRO_FATSWAP_ENABLED\s*=\s*_env_bool\(\s*"MEALFIT_MICRO_FATSWAP"\s*,\s*(\w+)\)', _GO)
    assert m and m.group(1) == "True"
    m2 = re.search(r'MISE_COOK_SPLIT_ENABLED\s*=\s*_env_bool\(\s*"MEALFIT_MISE_COOK_SPLIT"\s*,\s*(\w+)\)', _GO)
    assert m2 and m2.group(1) == "True"
    assert '"vit_a_mcg": ("50 g de zanahoria rallada"' in _GO, "seed de vit A registrado"


@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "MICRONUTRIENT_CLOSER_ENABLED", True)
    monkeypatch.setattr(g, "MICRO_CLOSER_PERDAY_ENABLED", False)
    monkeypatch.setattr(g, "MICRO_SEED_ENABLED", True)
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    return g


class _FakeDB:
    """linaza porta omega-3 (0.16/g); zanahoria porta vit A (8.35/g); nada más resuelve."""

    def micros_from_ingredient_string(self, s):
        low = str(s).lower()
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", low)
        g = float(m.group(1).replace(",", ".")) if m else 0.0
        if "linaza" in low:
            return {"omega3_g": 0.16 * g}
        if "zanahoria" in low:
            return {"vit_a_mcg": 8.35 * g}
        return {}

    def macros_from_ingredient_string(self, s):
        low = str(s).lower()
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", low)
        g = float(m.group(1).replace(",", ".")) if m else 0.0
        if "linaza" in low:
            return {"kcal": 0.53 * g}
        if "zanahoria" in low:
            return {"kcal": 0.41 * g}
        return {"kcal": 0.0}

    def grams_from_ingredient_string(self, s):
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", str(s).lower())
        return float(m.group(1).replace(",", ".")) if m else None

    def rescale_ingredient_string(self, s, factor):
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", str(s))
        if not m:
            return s
        g = float(m.group(1).replace(",", ".")) * factor
        return re.sub(r"^\s*\d+(?:[.,]\d+)?\s*g", f"{g:.0f} g", s, count=1)


def _mock_report(monkeypatch, key, floor):
    import micronutrients
    monkeypatch.setattr(micronutrients, "build_micronutrient_report", lambda *a, **kw: {
        "panel": [{"key": key, "piso": floor, "valor": 0.0, "status": "bajo"}],
        "gaps": [{"key": key, "piso": floor, "status": "bajo"}],
        "coverage": 1.0, "per_day_floors": {"flagged": False}})


# ───────────────────── FATSWAP ─────────────────────

def _fat_plan():
    return {
        "macros": {"protein": "123g", "carbs": "271g", "fats": "58g"},
        "days": [{"day": 1, "meals": [
            {"meal": "Desayuno", "name": "Avena", "cals": 400,
             "ingredients": ["5 g de semillas de linaza", "1 cda de aceite de oliva"],
             "ingredients_raw": ["5 g de semillas de linaza", "1 cda de aceite de oliva"],
             "recipe": ["Mise en place: mide.", "Montaje: sirve."]},
        ]}],
    }


def test_fatswap_grants_extra_budget_and_fires_mirror_trim(go, monkeypatch):
    _mock_report(monkeypatch, "omega3_g", 1.6)
    # budget mínimo viable (el loop del día hace break con ≤1.0): alcanza para una escala
    # parcial pero NO para cerrar el piso — el fatswap debe terminar el trabajo.
    monkeypatch.setattr(go, "MICRONUTRIENT_CLOSER_MAX_KCAL_PER_DAY", 5.0)
    _trim_calls = []
    monkeypatch.setattr(go, "_trim_day_fats_to_target",
                        lambda meals, tf, db, **kw: _trim_calls.append(tf) or True)
    plan = _fat_plan()
    assert go._close_micro_gaps_for_plan(plan, {}, _FakeDB()) >= 1
    line = plan["days"][0]["meals"][0]["ingredients"][0]
    g = float(re.match(r"^\s*(\d+)", line).group(1))
    assert g > 5.0, f"el fatswap escaló la linaza pese al budget agotado: {line}"
    assert _trim_calls and abs(_trim_calls[0] - 58.0) < 1e-6, \
        "el ESPEJO (re-trim de grasas del día a target 58g) corre INMEDIATO tras el swap"
    raw = plan["days"][0]["meals"][0]["ingredients_raw"][0]
    assert re.match(r"^\s*(\d+)", raw).group(1) == re.match(r"^\s*(\d+)", line).group(1), \
        "lockstep raw (panel/compras ven el cierre)"


def test_fatswap_knob_off_no_extra(go, monkeypatch):
    _mock_report(monkeypatch, "omega3_g", 1.6)
    monkeypatch.setattr(go, "MICRONUTRIENT_CLOSER_MAX_KCAL_PER_DAY", 5.0)
    monkeypatch.setattr(go, "MICRO_FATSWAP_ENABLED", False)
    _trim_calls = []
    monkeypatch.setattr(go, "_trim_day_fats_to_target",
                        lambda meals, tf, db, **kw: _trim_calls.append(tf) or True)
    plan = _fat_plan()
    go._close_micro_gaps_for_plan(plan, {}, _FakeDB())
    assert not _trim_calls, "knob OFF → sin presupuesto extra ni trim espejo (residual honesto)"


def test_fatswap_not_for_nonfat_micros():
    i = _GO.index("[P1-MICRO-FATSWAP · 2026-07-06] (banner vivo plan 6ffffd39) micros GRASA-basados")
    win = _GO[i:i + 900]
    assert "_FAT_BASED_MICROS" in win, "el swap SOLO aplica a omega-3/vit E (portadores = grasas)"
    assert '_FAT_BASED_MICROS = ("omega3_g", "vit_e_mg")' in _GO


# ───────────────────── SEED VIT A (slot-aware) ─────────────────────

def test_vita_seed_lands_in_savory_meal_with_coherent_note(go, monkeypatch):
    _mock_report(monkeypatch, "vit_a_mcg", 900.0)
    plan = {"days": [{"day": 1, "meals": [
        {"meal": "Merienda", "name": "Yogurt con Fruta", "cals": 200,
         "ingredients": ["100 g de yogurt"], "ingredients_raw": ["100 g de yogurt"],
         "recipe": ["Mise en place: mide.", "Montaje: sirve."]},
        {"meal": "Almuerzo", "name": "Pollo Guisado", "cals": 700,
         "ingredients": ["150 g de pollo"], "ingredients_raw": ["150 g de pollo"],
         "recipe": ["Mise en place: pica.", "El Toque de Fuego: guisa 20 min.", "Montaje: sirve."]},
    ]}]}
    assert go._close_micro_gaps_for_plan(plan, {}, _FakeDB()) >= 1
    merienda, almuerzo = plan["days"][0]["meals"]
    assert any("zanahoria" in s.lower() for s in almuerzo["ingredients"]), \
        "seed SALADO (zanahoria) → almuerzo/cena, no la merienda dulce"
    assert not any("zanahoria" in s.lower() for s in merienda["ingredients"])
    _note = next((s for s in almuerzo["recipe"] if "🌱" in str(s)), "")
    assert "acompaña el plato con" in _note and "vitamina A" in _note, _note


# ───────────────────── MISE-COOK-SPLIT ─────────────────────

def _tostadas():
    return {
        "meal": "Desayuno", "name": "Tostadas de Queso Blanco con Aguacate y Limón",
        "ingredients": ["4 lonjas de pan integral familiar", "20 g de queso blanco", "½ aguacate"],
        "recipe": [
            "Mise en place: Tuesta las 2 lonjas de pan integral en un tostador o sartén sin aceite. "
            "Corta el queso blanco en rodajas finas. Parte el aguacate y corta la pulpa en láminas.",
            "Montaje: Coloca las rodajas de queso sobre las tostadas. Sirve inmediatamente.",
        ],
    }


def test_mise_cook_split_synthesizes_tdf_and_clears_advisory():
    import graph_orchestrator as g
    meal = _tostadas()
    assert g._split_cooking_from_mise(meal) is True
    steps = meal["recipe"]
    _tdf = next((s for s in steps if str(s).lower().startswith("el toque de fuego")), None)
    assert _tdf and "Tuesta las 2 lonjas" in _tdf, steps
    assert re.search(r"\d+\s*-?\s*\d*\s*min", _tdf), f"el backstop añade tiempo por técnica: {_tdf}"
    _mise = next(s for s in steps if str(s).lower().startswith("mise en place"))
    assert "Tuesta" not in _mise and "Corta el queso" in _mise, \
        "el Mise retiene la prep; la cocción se movió"
    assert meal.get("_mise_cook_split") is True
    assert g._recipe_step_contract_issues(meal) == [], \
        "el advisory 'falta El Toque de Fuego' (chip amarillo) desaparece"
    assert g._split_cooking_from_mise(meal) is False, "idempotente (TdF ya presente)"


def test_mise_split_noop_cases():
    import graph_orchestrator as g
    # una sola oración → moverla vaciaría el Mise
    m1 = {"name": "X", "recipe": ["Mise en place: Tuesta el pan.", "Montaje: sirve."]}
    assert g._split_cooking_from_mise(m1) is False
    # sin oraciones de cocción (plato frío) → nada que mover
    m2 = {"name": "Yogurt", "recipe": [
        "Mise en place: Corta la fruta. Mide el yogurt.", "Montaje: sirve frío."]}
    assert g._split_cooking_from_mise(m2) is False
    # ya hay TdF → no-op
    m3 = _tostadas()
    m3["recipe"].insert(1, "El Toque de Fuego: calienta 2 min.")
    assert g._split_cooking_from_mise(m3) is False


def test_mise_split_wired_before_boundary_lint():
    i_split = _GO.index("[P2-MISE-COOK-SPLIT · 2026-07-06] cocción atrapada en el Mise sin pilar TdF → split ANTES")
    i_lint = _GO.index("(P2-C) Contract-lint per-meal en el persist boundary")
    assert i_split < i_lint, "el split corre ANTES del lint del boundary (el chip muere ahí)"
