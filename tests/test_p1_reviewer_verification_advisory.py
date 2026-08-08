# [P1-REVIEWER-VERIFICATION-ADVISORY · 2026-08-08] Iteración 3 del issue #9 — tres palancas:
# 1. Las demandas de VERIFICACIÓN del reviewer LLM («requiere certificación sin gluten»,
#    «contaminación cruzada», «vigilancia tiroidea») NO pueden ser rechazo crítico guest:
#    son irrefutables por construcción (no existe dato de marca) y además ENVENENAN el retry
#    (viajan en la directiva). Se degradan a advisory DETERMINISTA post-parse; los guards
#    deterministas (alérgeno/dieta/piso) conservan la última palabra.
# 2. El scrub de pools del skeleton era diet-only: asignó Huevos+Queso a un alérgico a
#    huevo/lácteos declarando «limpio» (corrida 31232856541).
# 3. FPs de subcadena del scanner de alérgenos (16ª de la clase): «mantequilla»⊂«mantequilla
#    de maní», «leche»⊂«leche de coco» — la excusa _plant_adj existía SOLO en el scan de dieta.
#    Dirección segura intacta: maní-alérgico sigue matcheando «mantequilla de maní» vía «maní».
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# 1. Downgrade de demandas de verificación del reviewer
# ---------------------------------------------------------------------------

def test_demandas_de_verificacion_se_degradan_a_advisory():
    import graph_orchestrator as go
    issues = [
        "Debe verificarse que el tofu y el casabe estén certificados sin gluten y libres de contaminación cruzada.",
        "El plan incluye atún en lata sin indicar certificación sin gluten.",
        "El volumen de soya requiere vigilancia de función tiroidea.",
    ]
    approved, kept, severity, advisories = go._downgrade_reviewer_verification_demands(
        False, issues, "critical")
    assert approved is True, "todas eran demandas de verificación → el plan queda aprobado por el LLM"
    assert kept == [], "las demandas NO pueden quedarse en issues (envenenarían el retry)"
    assert severity == "low"
    assert len(advisories) == 3, "las demandas se conservan como advisories observables"


def test_issue_real_sobrevive_y_bloquea():
    import graph_orchestrator as go
    issues = [
        "Debe verificarse la certificación sin gluten del casabe.",
        "Día 2: el plan contiene 2 huevos enteros, contraindicados por alergia al huevo.",
    ]
    approved, kept, severity, advisories = go._downgrade_reviewer_verification_demands(
        False, issues, "critical")
    assert approved is False, "queda un issue REAL → sigue rechazado"
    assert len(kept) == 1 and "huevos" in kept[0]
    assert severity == "critical", "la severidad del rechazo real no se toca"
    assert len(advisories) == 1


def test_approved_true_pasa_intacto():
    import graph_orchestrator as go
    approved, kept, severity, advisories = go._downgrade_reviewer_verification_demands(
        True, [], "low")
    assert approved is True and kept == [] and advisories == []


def test_pasteurizado_embarazo_NO_se_degrada():
    # Food-safety de embarazo («pasteurizado») es defendible clínicamente — fuera del patrón.
    import graph_orchestrator as go
    issues = ["El queso fresco debe ser pasteurizado por el embarazo declarado."]
    approved, kept, severity, _ = go._downgrade_reviewer_verification_demands(
        False, issues, "critical")
    assert approved is False and len(kept) == 1


def test_wiring_downgrade_en_review_node():
    src = open(os.path.join(os.path.dirname(__file__), "..", "graph_orchestrator.py"),
               encoding="utf-8").read()
    i = src.find("approved = result.approved")
    assert i > 0
    blk = src[i: i + 900]
    assert "_downgrade_reviewer_verification_demands(" in blk, (
        "el downgrade debe correr sobre el output del reviewer ANTES de los guards deterministas")


# ---------------------------------------------------------------------------
# 2. Scrub de ALÉRGENOS en pools del skeleton
# ---------------------------------------------------------------------------

def test_allergen_pool_item_banned():
    import graph_orchestrator as go
    assert go._allergen_pool_item_banned("Huevos", ["Huevo"]) is True
    assert go._allergen_pool_item_banned("Queso blanco fresco", ["Lácteos"]) is True
    assert go._allergen_pool_item_banned("Lentejas", ["Lácteos", "Gluten", "Huevo"]) is False
    assert go._allergen_pool_item_banned("Pechuga de pollo", ["Mariscos"]) is False


def test_wiring_allergen_scrub_en_skeleton():
    src = open(os.path.join(os.path.dirname(__file__), "..", "graph_orchestrator.py"),
               encoding="utf-8").read()
    i = src.find("SKELETON DIET SCRUB")
    assert i > 0
    assert "_allergen_pool_item_banned(" in src[i - 3000: i + 3000], (
        "el scrub de pools debe filtrar también ALÉRGENOS (era diet-only)")


# ---------------------------------------------------------------------------
# 3. Excusa plant-adjacent en el scanner de alérgenos
# ---------------------------------------------------------------------------

def _scan(ings, allergies):
    import graph_orchestrator as go
    plan = {"days": [{"meals": [{"name": "m", "ingredients": ings}]}]}
    return go._scan_allergen_violations(plan, allergies)


def test_leche_de_coco_no_es_lacteo():
    assert _scan(["200 ml de leche de coco"], ["Lácteos"]) == []
    assert _scan(["1 taza de yogur de soya"], ["Lácteos"]) == []
    assert _scan(["1 cdta de mantequilla de maní"], ["Lácteos"]) == []


def test_lacteos_reales_siguen_matcheando():
    assert _scan(["580 ml de leche"], ["Lácteos"]) != []
    assert _scan(["1 cda de mantequilla"], ["Lácteos"]) != []
    assert _scan(["1 taza de leche entera"], ["Lácteos"]) != []


def test_mani_alergico_sigue_cazando_mantequilla_de_mani():
    # La dirección segura: el maní-alérgico matchea vía el término «maní» directo.
    assert _scan(["1 cdta de mantequilla de maní"], ["Maní"]) != []


def test_coco_alergico_sigue_cazando_leche_de_coco():
    assert _scan(["200 ml de leche de coco"], ["Coco"]) != []
