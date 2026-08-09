# [P1-REVIEWER-SEES-SAFETY-NOTES · 2026-08-09] La corrida dirigida 31299769707 (post-deploy de
# P1-PREGNANCY-SAFETY-NOTES) probó que las anotaciones SOLAS no bastan: corr=0b4ca77c anotó 9
# comidas y el reviewer RECHAZÓ critical con «el plan no lo especifica» — porque el resumen que
# el reviewer recibe (`all_meals_summary`, review_plan_node) serializa SOLO nombre+ingredientes,
# nunca los pasos de receta donde viven las notas. Un reviewer que no ve las recetas no puede
# ser satisfecho por NINGUNA nota en las recetas: rechazo estructural garantizado para embarazo.
# Fix: las líneas «Seguridad alimentaria» del plato viajan en su línea del resumen (solo comidas
# anotadas — costo de tokens acotado y por perfil). + cláusula nueva medida en la misma corrida:
# habichuelas/leguminosas SECAS sin proceso de cocción explícito (PHA).
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("MEALFIT_DB_BACKEND", "neon")
os.environ.setdefault("NEON_DATABASE_URL", "postgresql://stub:stub@localhost:5432/stub")
os.environ.setdefault("NEON_DATABASE_URL_UNPOOLED", "postgresql://stub:stub@localhost:5432/stub")

import graph_orchestrator as go  # noqa: E402

_SRC = open(os.path.join(os.path.dirname(__file__), "..", "graph_orchestrator.py"),
            encoding="utf-8").read()


def test_helper_extrae_notas_de_seguridad():
    meal = {"name": "Revoltillo", "ingredients": ["2 huevos"],
            "recipe": ["Bate los huevos.",
                       "🤰 Seguridad alimentaria (embarazo/lactancia): cocina el huevo POR "
                       "COMPLETO (yema y clara firmes, sin puntos líquidos)."]}
    out = go._meal_safety_notes_for_summary(meal)
    assert "yema y clara firmes" in out
    assert "Bate los huevos" not in out, "solo las notas de seguridad viajan, no la receta entera"


def test_helper_vacio_sin_notas():
    assert go._meal_safety_notes_for_summary({"recipe": ["Hierve 10 min."]}) == ""
    assert go._meal_safety_notes_for_summary({}) == ""
    assert go._meal_safety_notes_for_summary({"recipe": "no-lista"}) == ""


def test_resumen_del_reviewer_incluye_las_notas():
    # Estructural: la línea de `all_meals_summary` debe llevar el resultado del helper —
    # sin esto, el reviewer no VE las prácticas y «no lo especifica» es rechazo garantizado
    # (medido: corr=0b4ca77c, 9 comidas anotadas → rechazo critical igual).
    i = _SRC.index("all_meals_summary.append(")
    win = _SRC[i:i + 400]
    assert "_meal_safety_notes_for_summary(meal)" in win, (
        "el resumen del reviewer debe incluir las notas de seguridad del plato; las notas "
        "que el reviewer no ve no previenen nada"
    )


def test_clausula_legumbre_seca():
    # medido en la corrida 31299769707: «Las habichuelas rojas deben cocinarse completamente;
    # el Día 2 se indican secas y el plan no especifica el proceso de cocción» (PHA).
    plan = {"days": [{"meals": [{"name": "Bowl con habas",
                                 "ingredients": ["40 g de habichuelas rojas secas"],
                                 "recipe": []}]}]}
    go._apply_pregnancy_food_safety_annotations(
        plan, {"medicalConditions": ["Embarazo"]})
    rec = plan["days"][0]["meals"][0]["recipe"]
    assert any("remoja" in s.lower() and "tiernas" in s.lower() for s in rec), (
        "leguminosa SECA sin cocción explícita = demanda medida del reviewer (fitohemaglutinina)"
    )


def test_canela_gana_ceilan_y_ceilan_declarada_absuelve():
    # residual medido corr=9909fb32: cumarina de la Cassia — «reducir a pizca o Ceilán».
    p1 = {"days": [{"meals": [{"name": "Avena con canela",
                               "ingredients": ["30 g de avena", "1 cdta de canela"],
                               "recipe": []}]}]}
    go._apply_pregnancy_food_safety_annotations(p1, {"medicalConditions": ["Embarazo"]})
    assert any("Ceilán" in str(s) for s in p1["days"][0]["meals"][0]["recipe"])
    p2 = {"days": [{"meals": [{"name": "Avena",
                               "ingredients": ["1 pizca de canela de Ceilán"],
                               "recipe": []}]}]}
    go._apply_pregnancy_food_safety_annotations(p2, {"medicalConditions": ["Embarazo"]})
    assert not any("Cassia" in str(s) for s in p2["days"][0]["meals"][0]["recipe"]), (
        "el ingrediente ya nombra Ceilán — la cláusula sobra")


def test_fruta_y_hierba_ganan_lavado():
    # residual medido corr=9909fb32: «no explicitadas para TODOS los platos» — la versión
    # hojas-only dejaba mango/cilantro fuera y el reviewer generalizaba el rechazo.
    p = {"days": [{"meals": [{"name": "Pollo guisado con mango y cilantro",
                              "ingredients": ["150 g de pollo", "80 g de mango",
                                              "5 g de cilantro"],
                              "recipe": []}]}]}
    go._apply_pregnancy_food_safety_annotations(p, {"medicalConditions": ["Embarazo"]})
    assert any("lava y desinfecta" in str(s) for s in p["days"][0]["meals"][0]["recipe"])


def test_legumbre_cocida_no_gana_clausula():
    plan = {"days": [{"meals": [{"name": "Habichuelas guisadas",
                                 "ingredients": ["100 g de habichuelas rojas guisadas"],
                                 "recipe": []}]}]}
    go._apply_pregnancy_food_safety_annotations(
        plan, {"medicalConditions": ["Embarazo"]})
    rec = plan["days"][0]["meals"][0]["recipe"]
    assert not any("remoja" in str(s).lower() for s in rec), (
        "guisada ya está cocida — la cláusula sería ruido")
