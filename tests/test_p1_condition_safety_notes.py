# [P1-CONDITION-SAFETY-NOTES · 2026-08-09] Generalización del patrón embarazo (anotar
# determinista + hacer visible al reviewer) a las demás condiciones. Los 10 fallos de la corrida
# 31304538636 eran la misma clase «no lo especifica» fuera de embarazo: HTA «versiones bajas en
# sodio» (perfil 4), dislipidemia «descremados»/yemas (5), gastritis «irritantes» (6), SOP
# porciones fruta alto-IG (7), hipotiroidismo interacciones levotiroxina (8). + guanábana
# (annonacina) excluida del POOL de generación de embarazo (whack-a-mole → exclusión en fuente).
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("MEALFIT_DB_BACKEND", "neon")
os.environ.setdefault("NEON_DATABASE_URL", "postgresql://stub:stub@localhost:5432/stub")
os.environ.setdefault("NEON_DATABASE_URL_UNPOOLED", "postgresql://stub:stub@localhost:5432/stub")

import graph_orchestrator as go  # noqa: E402

_SRC = open(os.path.join(os.path.dirname(__file__), "..", "graph_orchestrator.py"),
            encoding="utf-8").read()


def _plan(name, ingredients, recipe=None):
    return {"days": [{"day": 1, "meals": [{"name": name, "ingredients": list(ingredients),
                                           "recipe": list(recipe or [])}]}]}


def _note(plan):
    return next((s for s in plan["days"][0]["meals"][0]["recipe"] if "Nota clínica" in str(s)), "")


def test_dislipidemia_lacteos_y_yemas():
    p = _plan("Revoltillo con queso", ["2 huevos", "40 g de queso"])
    n = go._apply_condition_safety_annotations(p, {"medicalConditions": ["Colesterol Alto"]})
    assert n == 1
    nota = _note(p)
    assert "DESCREMADOS" in nota and "yemas" in nota, "los rechazos medidos del perfil 5"


def test_hta_sodio_y_version_baja_absuelve():
    p = _plan("Sandwich", ["30 g de jamon", "1 pan de agua"])
    go._apply_condition_safety_annotations(p, {"medicalConditions": ["Hipertensión"]})
    assert "BAJAS EN SODIO" in _note(p), "perfil 4: «versiones bajas en sodio»"
    p2 = _plan("Sandwich", ["30 g de queso bajo en sodio"])
    go._apply_condition_safety_annotations(p2, {"medicalConditions": ["Hipertensión"]})
    assert _note(p2) == "", "el ingrediente ya declara bajo en sodio — la cláusula sobra"


def test_hipotiroidismo_levotiroxina():
    p = _plan("Batido de espinacas", ["50 g de espinacas", "200 ml de leche"])
    go._apply_condition_safety_annotations(p, {"medicalConditions": ["Hipotiroidismo"]})
    assert "levotiroxina" in _note(p) and "4 horas" in _note(p), "perfil 8: absorción"


def test_sop_fruta_y_gastritis_suave():
    p = _plan("Bowl de mango", ["150 g de mango"])
    go._apply_condition_safety_annotations(p, {"medicalConditions": ["SOP (PCOS)"]})
    assert "PEQUEÑA" in _note(p), "perfil 7: porción de fruta alto-IG"
    p2 = _plan("Pescado al limón", ["150 g de pescado", "1 limon"])
    go._apply_condition_safety_annotations(p2, {"medicalConditions": ["Gastritis"]})
    assert "SUAVE" in _note(p2), "perfil 6: irritantes"


def test_sin_condicion_o_sin_tokens_cero():
    p = _plan("Pollo guisado", ["150 g de pollo"])
    assert go._apply_condition_safety_annotations(p, {"medicalConditions": ["Ninguna"]}) == 0
    p2 = _plan("Pollo guisado", ["150 g de pollo"])
    assert go._apply_condition_safety_annotations(
        p2, {"medicalConditions": ["Hipotiroidismo"]}) == 0, "sin tokens gatillo → sin nota"


def test_idempotente_y_absolution():
    fd = {"medicalConditions": ["Colesterol Alto"]}
    p = _plan("Revoltillo", ["2 huevos"])
    go._apply_condition_safety_annotations(p, fd)
    go._apply_condition_safety_annotations(p, fd)
    meal = p["days"][0]["meals"][0]
    assert sum(1 for s in meal["recipe"] if "Nota clínica" in s) == 1
    meal["ingredients"] = ["150 g de pechuga de pollo"]
    meal["name"] = "Pollo a la plancha"
    go._apply_condition_safety_annotations(p, fd)
    assert not any("Nota clínica" in s for s in meal["recipe"]), "absolución: el riesgo salió"


def test_knob_off():
    import pytest  # noqa: F401
    old = go.CONDITION_SAFETY_NOTES_ENABLED
    go.CONDITION_SAFETY_NOTES_ENABLED = False
    try:
        p = _plan("Revoltillo", ["2 huevos"])
        assert go._apply_condition_safety_annotations(
            p, {"medicalConditions": ["Colesterol Alto"]}) == 0
    finally:
        go.CONDITION_SAFETY_NOTES_ENABLED = old


def test_summary_incluye_nota_clinica():
    meal = {"recipe": ["Paso 1.", "⚕️ Nota clínica: usa los lácteos DESCREMADOS."]}
    assert "DESCREMADOS" in go._meal_safety_notes_for_summary(meal), (
        "una nota que el reviewer no ve no previene nada (lección corr=0b4ca77c)")


def test_capa_clinica_invoca_la_pasada():
    i_def = _SRC.index("def _apply_deterministic_clinical_layer")
    i_end = _SRC.index("def _apply_macro_engine", i_def)
    assert "_apply_condition_safety_annotations(plan, form_data)" in _SRC[i_def:i_end]


def test_pool_scrub_embarazo_guanabana():
    # la exclusión en FUENTE: guanábana/cundeamor no deben poder entrar al pool del day-gen
    # de un perfil de embarazo. Estructural: el scrub existe, usa los triggers SSOT de las
    # subs (sin lista paralela) y vive junto al scrub de alérgenos del skeleton.
    i = _SRC.index("SKELETON PREGNANCY SCRUB")
    win_start = _SRC.rindex("_is_pregnancy_or_lactation", 0, i)
    win = _SRC[win_start:i]
    assert "_PREGNANCY_UTEROTONIC_SUBS" in win and "_PREGNANCY_AVOID_FRUIT_SUBS" in win, (
        "los tokens del scrub deben venir de las subs SSOT — una lista paralela driftea")


def test_guanabana_sub_en_condition_rules():
    from condition_rules import _RULES_BY_ID
    preg = _RULES_BY_ID["pregnancy"]
    _all = [t for row in preg.substitutions for t in row[0]]
    assert "guanabana" in _all
    assert any(str(row[1]).lower() == "mango" for row in preg.substitutions)
