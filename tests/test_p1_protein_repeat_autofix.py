"""[P1-PROTEIN-REPEAT-AUTOFIX · 2026-07-04] Autofix determinista de proteína repetida same-day.

Causa #1 RESTANTE de retries medida en vivo (4 firings en las 2 renovaciones del 2026-07-04):
la MISMA proteína principal en 2+ comidas del MISMO día (gate P1-VARIETY-SAME-DAY-PROTEIN).
Los días se generan en paralelo y a ciegas entre sí → el LLM reincide incluso con la directiva
del retry. Cada retry = un intento completo de generación (costo API DeepSeek).

Diseño: detector ESPEJO del gate (mismos labels/aliases/word-boundary/relax de meal-count);
conserva la comida donde la proteína es identidad (en el NOMBRE) y reescribe las otras a una
proteína alternativa (escalera por cercanía culinaria, alergia/dislike-aware, sin crear
repetición nueva). HUEVO y ATÚN excluidos v1 (platos-identidad / línea enlatada) — gate backstop.

Cubre también [P2-SLOT-CROQUETA-BINDER]: croqueta/albóndiga excluidas del slot-rule de avena
(aglutinante legítimo — falso positivo vivo "Croquetas de Pavo Molido con Avena" en almuerzo).
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)


def _read(rel):
    with open(os.path.join(_BACKEND, rel), encoding="utf-8") as f:
        return f.read()


_GO = _read("graph_orchestrator.py")


@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    return g


def _meal(slot, name, ings, steps=None):
    return {"meal": slot, "name": name, "ingredients": list(ings),
            "ingredients_raw": list(ings), "recipe": list(steps or ["Cocina."])}


def _mk_days_pollo_x2():
    """pollo en almuerzo (identidad, en el NOMBRE) + pollo en cena → la cena se reescribe."""
    return [{"day": 1, "meals": [
        _meal("Desayuno", "Mangú con Huevo", ["2 huevos", "200 g de plátano verde"]),
        _meal("Almuerzo", "Pollo Guisado con Arroz", ["150 g de pechuga de pollo", "150 g de arroz"]),
        _meal("Cena", "Wrap de Pollo", ["120 g de pechuga de pollo", "1 tortilla integral"],
              ["Cocina el pollo.", "Arma el wrap."]),
    ]}]


# ---------------------------------------------------------------------------
# knobs + wiring
# ---------------------------------------------------------------------------

def test_knobs_defaults():
    assert '_env_bool("MEALFIT_PROTEIN_REPEAT_AUTOFIX", True)' in _GO
    assert '_env_int("MEALFIT_PROTEIN_REPEAT_AUTOFIX_MAX_PER_DAY", 2' in _GO


def test_wired_in_assemble_pre_engine_after_sodium():
    # [2026-07-05] window 1000→2200: P1-EGG-CAP-AUTOFIX se insertó en el seam entre el pase
    # de sodio y este autofix (orden deliberado: el egg-fix limpia repeticiones de huevo antes).
    i = _GO.index("acción(es) de sodio per-día \"\n                            f\"aplicada(s) pre-reviewer")
    win = _GO[i:i + 2200]
    assert "_protein_repeat_autofix(days, form_data)" in win, \
        "el autofix corre en el mismo seam pre-motor, después del pase de sodio"
    assert "_egg_cap_autofix(days, form_data)" in win, \
        "el egg-fix vive en el mismo seam, antes del protein-repeat"


def test_wired_after_budget_passes():
    """[2026-07-05] Los pases de presupuesto REESCRIBEN proteínas después del pase pre-motor y
    pueden COLAPSAR dos proteínas distintas del mismo día en el mismo equivalente económico
    (caso vivo corr=2f37b6b4: camarones+pulpo→'Filete de pescado blanco' → 2 retries seguidos
    del gate de variedad = 2 generaciones cobradas). Re-fire espejo en AMBOS callsites."""
    # 1) cheapen pre-engine (P1-BUDGET-TIER-LEVERS)
    i = _GO.index("por equivalentes económicos (presupuesto económico/ajustado)")
    win = _GO[i:i + 700]
    assert "_protein_repeat_autofix(days, form_data)" in win, \
        "re-fire tras el cheapen-pass pre-engine"
    # 2) convergencia driver-aware (P1-BUDGET-CONVERGENCE) — ANTES del truth-up/re-banda/rebuild
    j = _GO.index("_apply_budget_cheapen_pass(_bc_days, form_data, force=True)")
    win2 = _GO[j:j + 1400]
    assert "_protein_repeat_autofix(_bc_days, form_data)" in win2, \
        "re-fire tras la convergencia de presupuesto"
    assert win2.index("_protein_repeat_autofix(_bc_days, form_data)") < win2.index(
        "truth-up post-sustitución"), \
        "el re-fire debe correr ANTES del truth-up/re-banda para que downstream vea la corrección"


# ---------------------------------------------------------------------------
# funcional
# ---------------------------------------------------------------------------

def test_rewrites_second_meal_keeps_identity_meal(go):
    days = _mk_days_pollo_x2()
    n = go._protein_repeat_autofix(days, {}, db=object())
    assert n == 1
    lunch, dinner = days[0]["meals"][1], days[0]["meals"][2]
    # el almuerzo (primera aparición con la proteína en el nombre) se conserva…
    assert "pollo" in lunch["name"].lower()
    assert any("pollo" in s for s in lunch["ingredients"])
    # …y la cena se reescribe al primer destino de la escalera (pavo), qty intacta.
    assert dinner["_protein_autofix_applied"] == "pollo->pavo"
    assert "pavo" in dinner["name"].lower() and "pollo" not in dinner["name"].lower()
    assert "120 g de pechuga de pavo" in dinner["ingredients"]
    assert dinner["ingredients"] == dinner["ingredients_raw"]
    assert "pollo" not in " ".join(dinner["recipe"]).lower()


def test_target_skips_protein_already_in_day(go):
    days = _mk_days_pollo_x2()
    # el día ya tiene pavo en la merienda → el destino salta a pescado (2º de la escalera).
    days[0]["meals"].insert(2, _meal("Merienda", "Rollitos de Pavo", ["60 g de pechuga de pavo"]))
    n = go._protein_repeat_autofix(days, {}, db=object())
    assert n == 1
    dinner = days[0]["meals"][3]
    assert dinner["_protein_autofix_applied"] == "pollo->pescado"
    assert "pescado" in dinner["name"].lower()


def test_dislike_and_allergy_ladder(go, monkeypatch):
    days = _mk_days_pollo_x2()
    monkeypatch.setattr(
        go, "_scan_allergen_violations",
        lambda plan, allergies: (["pescado"] if any(
            "pescado" in str(i).lower() for d in plan["days"]
            for m in d["meals"] for i in m["ingredients"]) else []),
    )
    # dislike pavo + alergia pescado → cae al 3º de la escalera (cerdo).
    n = go._protein_repeat_autofix(
        days, {"dislikes": ["pavo"], "allergies": ["pescado"]}, db=object())
    assert n == 1
    dinner = days[0]["meals"][2]
    assert dinner["_protein_autofix_applied"] == "pollo->cerdo"


def test_no_safe_target_leaves_gate_as_backstop(go, monkeypatch):
    days = _mk_days_pollo_x2()
    monkeypatch.setattr(go, "_scan_allergen_violations", lambda plan, allergies: ["x"])
    n = go._protein_repeat_autofix(
        days, {"dislikes": ["pavo", "cerdo", "pescado"], "allergies": ["todo"]}, db=object())
    assert n == 0
    assert "pollo" in days[0]["meals"][2]["name"].lower(), \
        "sin destino seguro NO se reescribe (el gate de variedad decide)"


def test_molido_form_and_no_repollo_false_positive(go):
    days = [{"day": 1, "meals": [
        _meal("Almuerzo", "Tacos de Carne Molida", ["120 g de carne molida", "2 tortillas"]),
        _meal("Cena", "Bistec Encebollado con Repollo", ["150 g de bistec", "50 g de repollo"]),
    ]}]
    n = go._protein_repeat_autofix(days, {}, db=object())
    assert n == 1
    dinner = days[0]["meals"][1]
    # res→cerdo: 'bistec' (compuesto) → forma canónica; 'repollo' NO debe corromperse (\b).
    assert dinner["_protein_autofix_applied"] == "res->cerdo"
    assert "repollo" in " ".join(dinner["ingredients"]).lower()
    assert "repollo" in dinner["name"].lower()
    assert "lomo de cerdo" in " ".join(dinner["ingredients"]).lower()


def test_huevo_protagonists_and_atun_now_fixed(go):
    # [P1-EGG-INTRINSIC-DEDUP · 2026-07-11] huevo protagonista ×2 (revoltillo + tortilla
    # de claras) YA NO queda para el gate: la premisa "decide el gate" produjo 3 rechazos
    # + entrega degradada en el primer plan modo-Nevera del owner (corr=9cc4317e). El pase
    # (3) conserva el desayuno y transplanta la cabeza del otro plato. Contrato completo:
    # test_p1_egg_intrinsic_dedup.py.
    days = [{"day": 1, "meals": [
        _meal("Desayuno", "Revoltillo de Huevo", ["2 huevos"]),
        _meal("Cena", "Tortilla de Claras", ["4 claras de huevo"]),
    ]}]
    assert go._protein_repeat_autofix(days, {}, db=object()) >= 1, (
        "el pase intrínseco debe corregir revoltillo+tortilla same-day"
    )
    _d1 = days[0]["meals"]
    assert "huevos" in " ".join(_d1[0]["ingredients"]).lower(), "keeper = desayuno"
    assert "clara" not in " ".join(_d1[1]["ingredients"]).lower(), "cena reasignada"
    assert not _d1[1]["name"].lower().startswith("tortilla"), "cabeza transplantada"
    # [P2-PROTEIN-LADDER-GAPS · 2026-07-11] atún YA NO está excluido: la exclusión v1
    # ("lata de pollo en agua") se resolvió con compuestos largos-primero; el caso vivo
    # corr=c0a950c6 (no_ladder_for_label → rechazo de plan completo) exigió la escalera.
    days2 = [{"day": 1, "meals": [
        _meal("Almuerzo", "Ensalada de Atún", ["1 lata de atún en agua"]),
        _meal("Cena", "Wrap de Atún", ["1 lata de atún en agua"]),
    ]}]
    fixed = go._protein_repeat_autofix(days2, {}, db=object())
    assert fixed >= 1, "atún ×2 same-day debe corregirse (escalera + compounds)"
    _blob = " ".join(str(x) for m in days2[0]["meals"] for x in (m["ingredients"] + [m["name"]]))
    assert "lata de pollo" not in _blob.lower(), "el compound reescribe la frase enlatada ENTERA"


def test_high_mealcount_relax_mirrors_gate(go):
    days = _mk_days_pollo_x2()
    days[0]["meals"] += [
        _meal("Merienda", "Yogurt con Fruta", ["1 yogurt"]),
        _meal("Merienda 2", "Batido de Lechosa", ["1 taza de lechosa"]),
    ]
    assert len(days[0]["meals"]) >= 5
    assert go._protein_repeat_autofix(days, {}, db=object()) == 0, \
        "con 5+ comidas el gate degrada a advisory → el autofix tampoco toca"


def test_knob_off_and_gate_off(go, monkeypatch):
    monkeypatch.setattr(go, "PROTEIN_REPEAT_AUTOFIX_ENABLED", False)
    assert go._protein_repeat_autofix(_mk_days_pollo_x2(), {}, db=object()) == 0
    monkeypatch.setattr(go, "PROTEIN_REPEAT_AUTOFIX_ENABLED", True)
    monkeypatch.setattr(go, "VARIETY_GATE_SAME_DAY_PROTEIN", False)
    assert go._protein_repeat_autofix(_mk_days_pollo_x2(), {}, db=object()) == 0, \
        "si el gate está apagado la repetición no causa retry → no reescribir el plato del usuario"


def test_idempotent(go):
    days = _mk_days_pollo_x2()
    assert go._protein_repeat_autofix(days, {}, db=object()) == 1
    assert go._protein_repeat_autofix(days, {}, db=object()) == 0


def test_budget_collision_two_pescado_same_day(go):
    """Estado post-budget del caso vivo: dos mariscos del mismo día colapsados al mismo
    'filete de pescado blanco' → el re-fire diversifica el segundo a pollo (escalera)."""
    days = [{"day": 1, "meals": [
        _meal("Almuerzo", "Filete de Pescado Blanco al Ajillo",
              ["150 g de filete de pescado blanco", "150 g de arroz"]),
        _meal("Cena", "Filete de Pescado Blanco Guisado",
              ["120 g de filete de pescado blanco", "100 g de batata"]),
    ]}]
    n = go._protein_repeat_autofix(days, {}, db=object())
    assert n == 1
    lunch, dinner = days[0]["meals"]
    assert "pescado" in lunch["name"].lower(), "la primera aparición (identidad) se conserva"
    assert dinner["_protein_autofix_applied"] == "pescado->pollo"
    assert "pechuga de pollo" in " ".join(dinner["ingredients"]).lower()
    assert "pescado" not in dinner["name"].lower()


# ---------------------------------------------------------------------------
# [P2-SLOT-CROQUETA-BINDER] avena-aglutinante fuera del slot-rule
# ---------------------------------------------------------------------------

def test_croqueta_binder_not_flagged():
    from constants import slot_violations_for_meal_name
    for slot in ("almuerzo", "cena"):
        assert slot_violations_for_meal_name("Croquetas de Pavo Molido con Avena", slot) == [], \
            f"croqueta con avena-aglutinante es plato fuerte legítimo en {slot} (falso positivo vivo)"
        assert slot_violations_for_meal_name("Albóndigas de Res con Avena", slot) == []
    # la regla sigue viva para el caso real que caza:
    assert slot_violations_for_meal_name("Avena con Frutas", "almuerzo"), \
        "avena como plato principal del almuerzo debe seguir flaggeada"


def test_marker_anchored_in_source():
    assert "P1-PROTEIN-REPEAT-AUTOFIX" in _GO
    assert "P2-SLOT-CROQUETA-BINDER" in _read("constants.py")
