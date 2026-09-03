# [P1-PREGNANCY-SAFETY-NOTES · 2026-08-09] El reviewer (Sol, perfiles difíciles) rechaza planes de
# embarazo/lactancia porque las recetas no DICEN las palabras de seguridad, aunque el plato sea
# seguro: corr=86482b8c acumuló 5 rechazos en UNA review (huevo sin «yema y clara firmes 71°C»,
# deli sin «74°C», hojas sin «lavado», lechosa sin «completamente madura», champiñones sin «cocción
# completa»); corr=d395f5c8 pidió lácteos pasteurizados; corr=140dfe19 (lactancia) mariscos
# «completamente cocidos». P1-REVIEWER-VERIFICATION-ADVISORY dejó esta clase FUERA a propósito:
# a diferencia de «verificar la etiqueta», la receta SÍ puede decir la instrucción — el fix
# correcto es anotar upstream, no amordazar al reviewer. La pasada es note-only (macro-preservante,
# shopping-safe: no muta tokens), recomputada por corrida (absolution-aware: si el huevo salió del
# plato, la nota sale con él — clase P1-RAW-COOKED-ABSOLUTION).
import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("MEALFIT_DB_BACKEND", "neon")
os.environ.setdefault("NEON_DATABASE_URL", "postgresql://stub:stub@localhost:5432/stub")
os.environ.setdefault("NEON_DATABASE_URL_UNPOOLED", "postgresql://stub:stub@localhost:5432/stub")

import graph_orchestrator as go  # noqa: E402

_PREG_FD = {"medicalConditions": ["Embarazo / Lactancia"]}


def _plan(meals):
    return {"days": [{"day": 1, "meals": meals}]}


def _meal(name, ingredients, recipe=None):
    return {"name": name, "ingredients": list(ingredients), "recipe": list(recipe or []),
            "protein": 30, "carbs": 40, "fats": 10, "calories": 370}


def _notes(meal):
    return [s for s in (meal.get("recipe") or []) if "Seguridad alimentaria (embarazo" in str(s)]


def test_huevo_gana_nota_yema_y_clara_firmes():
    p = _plan([_meal("Revoltillo criollo", ["2 huevos", "1 tomate"])])
    n = go._apply_pregnancy_food_safety_annotations(p, _PREG_FD)
    assert n == 1
    nota = _notes(p["days"][0]["meals"][0])
    assert len(nota) == 1 and "yema y clara firmes" in nota[0], (
        "el reviewer exige cocción completa EXPLÍCITA del huevo (corr=86482b8c)")


def test_deli_gana_74_pero_pavo_fresco_no():
    p = _plan([
        _meal("Wrap de pechuga de pavo tipo deli", ["60 g de pechuga de pavo tipo deli"]),
        _meal("Pechuga de pavo a la plancha", ["150 g de pechuga de pavo"]),
    ])
    go._apply_pregnancy_food_safety_annotations(p, _PREG_FD)
    m_deli, m_fresh = p["days"][0]["meals"]
    assert any("74" in s for s in _notes(m_deli)), "deli frío = riesgo listeria (corr=86482b8c)"
    assert not any("74" in s for s in _notes(m_fresh)), (
        "pechuga de pavo FRESCA a la plancha no es fiambre — anotarla sería falso positivo")


def test_hojas_lechosa_hongos_mariscos():
    p = _plan([_meal("Ensalada de espinacas con champiñones y mejillones",
                     ["50 g de espinacas", "80 g de champiñones", "150 g de mejillones",
                      "100 g de lechosa"])])
    go._apply_pregnancy_food_safety_annotations(p, _PREG_FD)
    nota = _notes(p["days"][0]["meals"][0])
    assert len(nota) == 1, "UNA nota combinada por comida, no una pila de 4"
    blob = nota[0]
    for frag in ("lava y desinfecta", "madura", "champiñones", "POR COMPLETO"):
        assert frag in blob, f"falta la cláusula '{frag}' (las 4 son rechazos medidos)"


def test_atun_en_lata_no_gana_clausula_de_coccion():
    p = _plan([_meal("Ensalada de atún", ["120 g de atún en agua", "1 tomate"])])
    go._apply_pregnancy_food_safety_annotations(p, _PREG_FD)
    assert not any("POR COMPLETO" in s for s in _notes(p["days"][0]["meals"][0])), (
        "el atún enlatado YA está cocido — la cláusula sería ruido")


def test_lacteos_pasteurizados_pero_leche_vegetal_no():
    p = _plan([
        _meal("Batido de lechosa", ["200 ml de leche de coco", "100 g de lechosa"]),
        _meal("Tostada con queso fresco", ["40 g de queso fresco", "1 pan integral"]),
    ])
    go._apply_pregnancy_food_safety_annotations(p, _PREG_FD)
    m_coco, m_queso = p["days"][0]["meals"]
    assert not any("PASTEURIZADOS" in s for s in _notes(m_coco)), (
        "la leche de coco no es láctea — pedirla pasteurizada sería absurdo")
    assert any("PASTEURIZADOS" in s for s in _notes(m_queso)), "corr=d395f5c8"


def test_no_embarazo_cero_notas_y_knob_off_cero(monkeypatch):
    p = _plan([_meal("Revoltillo", ["2 huevos"])])
    assert go._apply_pregnancy_food_safety_annotations(p, {"medicalConditions": ["DM2"]}) == 0
    assert _notes(p["days"][0]["meals"][0]) == []
    monkeypatch.setattr(go, "PREGNANCY_SAFETY_NOTES_ENABLED", False)
    assert go._apply_pregnancy_food_safety_annotations(p, _PREG_FD) == 0


def test_idempotente_y_absolution_aware():
    p = _plan([_meal("Revoltillo", ["2 huevos"])])
    go._apply_pregnancy_food_safety_annotations(p, _PREG_FD)
    go._apply_pregnancy_food_safety_annotations(p, _PREG_FD)
    meal = p["days"][0]["meals"][0]
    assert len(_notes(meal)) == 1, "dos pasadas no duplican la nota"
    # absolution: el swap sacó el huevo → la nota debe salir con él (nota stale = clase
    # P1-RAW-COOKED-ABSOLUTION, el step-rewriter renombraba notas viejas)
    meal["ingredients"] = ["150 g de pechuga de pollo"]
    meal["name"] = "Pollo guisado"
    go._apply_pregnancy_food_safety_annotations(p, _PREG_FD)
    assert _notes(meal) == [], "el plato ya no tiene huevo/riesgo — la nota vieja debe removerse"


def test_clausula_ya_cubierta_por_el_llm_no_se_duplica():
    p = _plan([_meal("Huevos duros", ["2 huevos"],
                     recipe=["Hierve 12 min hasta que la yema y clara firmes queden."])])
    go._apply_pregnancy_food_safety_annotations(p, _PREG_FD)
    meal = p["days"][0]["meals"][0]
    assert not any("yema y clara firmes" in s for s in _notes(meal)), (
        "si el LLM ya escribió la instrucción, la cláusula sobra")


def test_macro_preservante():
    p = _plan([_meal("Revoltillo", ["2 huevos"])])
    before = {k: p["days"][0]["meals"][0][k] for k in ("protein", "carbs", "fats", "calories")}
    before_ings = copy.deepcopy(p["days"][0]["meals"][0]["ingredients"])
    go._apply_pregnancy_food_safety_annotations(p, _PREG_FD)
    meal = p["days"][0]["meals"][0]
    assert {k: meal[k] for k in before} == before
    assert meal["ingredients"] == before_ings, (
        "note-only: mutar tokens de ingredientes rompería la coherencia receta↔lista")


def test_capa_clinica_invoca_la_pasada():
    src = open(os.path.join(os.path.dirname(__file__), "..", "graph_orchestrator.py"),
               encoding="utf-8").read()
    i_def = src.index("def _apply_deterministic_clinical_layer")
    i_end = src.index("\ndef ", src.index("\ndef ", i_def) + 5) if False else src.index(
        "def _apply_macro_engine", i_def)
    body = src[i_def:i_end]
    assert "_apply_pregnancy_food_safety_annotations(plan, form_data)" in body, (
        "la pasada debe correr en la capa clínica (generación + fallbacks) — fuera de ella "
        "los planes de embarazo llegan al reviewer sin las anotaciones y la clase revive")


def test_cundeamor_sustituido_en_condition_rules():
    # corr=d395f5c8: cundeamor (melón amargo) = uterotónico contraindicado — el reviewer lo
    # rechazó como CRITICAL. Debe salir por el MISMO mecanismo SSOT que el mercurio.
    from condition_rules import CONDITION_RULES, _PREGNANCY_MERCURY_SUBS
    rule = next(r for r in CONDITION_RULES if r.id == "pregnancy")
    _all_terms = [t for row in rule.substitutions for t in row[0]]
    assert "cundeamor" in _all_terms, "cundeamor debe tener sub determinista (uterotónico)"
    # el scanner de mercurio indexa _PREGNANCY_MERCURY_SUBS[0][0] — la tupla del mercurio NO
    # debe mutarse al añadir el sub nuevo (index-sensitive, graph_orchestrator:_scan_mercury_*)
    assert "tiburon" in _PREGNANCY_MERCURY_SUBS[0][0]
    assert not any("cundeamor" in row[0] for row in _PREGNANCY_MERCURY_SUBS), (
        "el sub uterotónico vive en su PROPIA tupla, no dentro de la de mercurio")
