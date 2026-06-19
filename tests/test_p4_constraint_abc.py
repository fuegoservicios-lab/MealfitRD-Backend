"""[P4-CONSTRAINT-ABC · 2026-06-14] Refactor behavior-preserving: el patrón renal 3-capas (ajuste-
fuente → enforce-per-comida → red-salida) + las sustituciones se generalizan a una jerarquía
`ClinicalConstraint` + `ClinicalConstraintEngine`. CLAVE: los constraints DELEGAN a las funciones
validadas de graph_orchestrator (NO reimplementan el cap renal — iatrogénico si se rompe). Estos tests
verifican: (a) el engine construye los constraints activos correctos por precedencia; (b) el hook
adjust_targets vía engine == llamada directa (equivalencia); (c) los constraints delegan (parser-anchor
— un rewrite que deje de delegar falla CI); (d) la capa clínica enruta sin cambiar comportamiento.
"""
import copy
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import clinical_constraints as cc
import graph_orchestrator as go


# ── (a) Engine: constraints activos por precedencia ──
def test_engine_active_constraints_by_profile():
    # [P1-RENAL-SODIUM-SUBS · 2026-06-19] ERC-puro ahora también lleva sustituciones de sodio (la fila renal
    # reusa _HTA_SODIUM_SUBS) → renal(10) + substitutions(30), antes solo ["renal"].
    assert [c.id for c in cc.ClinicalConstraintEngine({"medicalConditions": ["Enfermedad renal crónica"]}).active] == ["renal", "substitutions"]
    # HTA + dislipidemia → ambas son substitution-bearing → UN solo constraint 'substitutions'
    assert [c.id for c in cc.ClinicalConstraintEngine({"medicalConditions": ["Hipertensión", "Colesterol alto"]}).active] == ["substitutions"]
    # renal + DM2 → renal(10) antes que substitutions(30)
    assert [c.id for c in cc.ClinicalConstraintEngine({"medicalConditions": ["Enfermedad renal crónica", "Diabetes T2"]}).active] == ["renal", "substitutions"]
    assert cc.ClinicalConstraintEngine({"medicalConditions": ["Ninguna"]}).active == []


# ── (b) Equivalencia: adjust_targets vía engine == _apply_renal_cap_to_nutrition directo ──
def _renal_nutrition(p=160, c=200, f=60):
    m = {"protein_g": p, "carbs_g": c, "fats_g": f,
         "protein_str": f"{p}g", "carbs_str": f"{c}g", "fats_str": f"{f}g"}
    return {"macros": dict(m), "total_daily_macros": dict(m)}


def test_renal_adjust_targets_equivalent_to_direct():
    form = {"medicalConditions": ["Enfermedad renal crónica"], "weight": 80, "weightUnit": "kg", "gender": "male"}
    nut_direct = _renal_nutrition()
    nut_engine = copy.deepcopy(nut_direct)
    go._apply_renal_cap_to_nutrition(nut_direct, form)            # camino directo (pre-refactor)
    cc.ClinicalConstraintEngine(form).run_adjust_targets(nut_engine)  # vía el engine
    assert nut_direct == nut_engine                              # byte-idéntico
    assert nut_engine["renal_protein_cap"]["applied"] is True
    assert nut_engine["renal_protein_cap"]["protein_g"] == round(go.RENAL_PROTEIN_GKG_CEILING * 80)  # 64


def test_non_renal_adjust_targets_is_noop():
    form = {"medicalConditions": ["Hipertensión"], "weight": 80, "weightUnit": "kg"}
    nut = _renal_nutrition()
    before = copy.deepcopy(nut)
    cc.ClinicalConstraintEngine(form).run_adjust_targets(nut)
    assert nut == before   # HTA no tiene adjust_targets → no-op


# ── (c) Parser-anchor: los constraints DELEGAN (no reimplementan) ──
def test_renal_constraint_delegates_not_reimplements():
    src = inspect.getsource(cc.RenalProteinCapConstraint)
    assert "_apply_renal_cap_to_nutrition" in src     # capa 1
    assert "_enforce_renal_per_meal" in src           # capa 2
    assert "_renal_exit_safety_net" in src            # capa 3
    # NO debe contener la aritmética del cap (multiply por el ceiling) — eso vive en la fn validada
    assert "RENAL_PROTEIN_GKG_CEILING" not in src
    assert "_trim_day_protein_to_ceiling" not in src  # eso lo llama la fn delegada, no el constraint


def test_substitution_constraint_delegates():
    src = inspect.getsource(cc.SubstitutionEngineConstraint)
    assert "_apply_condition_substitutions" in src
    assert "collect_substitutions" in src   # solo para applies()


# ── (d) El SubstitutionEngineConstraint aplica el pase único de sustitución ──
def test_substitution_constraint_applies_subs():
    eng = cc.ClinicalConstraintEngine({"medicalConditions": ["Colesterol alto"]})
    plan = {"days": [{"meals": [{"ingredients": ["2 cda de mantequilla"]}]}]}
    n = eng.enforce_one("substitutions", plan, {}, cc.ClinicalContext())
    assert n == 1
    assert "aceite de oliva" in " ".join(plan["days"][0]["meals"][0]["ingredients"]).lower()


def test_enforce_one_noop_for_inactive_constraint():
    eng = cc.ClinicalConstraintEngine({"medicalConditions": ["Ninguna"]})
    plan = {"days": [{"meals": [{"ingredients": ["2 cda de mantequilla"]}]}]}
    assert eng.enforce_one("substitutions", plan, {}, cc.ClinicalContext()) is None
    assert eng.enforce_one("renal", plan, {}, cc.ClinicalContext()) is None


# ── Cableado: la capa clínica enruta vía el engine ──
def test_clinical_layer_wires_engine():
    src = inspect.getsource(go._apply_deterministic_clinical_layer)
    assert 'ClinicalConstraintEngine(form_data)' in src
    assert '_eng.enforce_one("renal"' in src
    assert '_eng.enforce_one("substitutions"' in src


def test_renal_helpers_extracted_verbatim():
    # los cuerpos verbatim existen como funciones módulo-nivel (delegación del constraint)
    assert callable(getattr(go, "_enforce_renal_per_meal", None))
    assert callable(getattr(go, "_renal_exit_safety_net", None))


def test_renal_trim_fallback_when_engine_import_fails(monkeypatch):
    """[review adversaria] SEGURIDAD iatrogénica: si el engine no carga (_eng=None), el trim renal
    per-comida NO debe saltarse silenciosamente — corre vía el fallback directo `_enforce_renal_per_meal`
    (simétrico al fallback de sustituciones)."""
    import clinical_constraints

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("engine down (simulado)")

    monkeypatch.setattr(clinical_constraints, "ClinicalConstraintEngine", _Boom)
    _called = {}
    _orig = go._enforce_renal_per_meal

    def _spy(plan, pg, dc, db):
        _called["yes"] = True
        return _orig(plan, pg, dc, db)

    monkeypatch.setattr(go, "_enforce_renal_per_meal", _spy)
    plan = {"calories": 2000, "macros": {"protein": "64g", "carbs": "250g", "fats": "60g"},
            "renal_protein_cap": {"applied": True, "protein_g": 64, "gkg": 0.8, "comorbid_diabetes": False},
            "days": [{"meals": [{"protein": 80, "carbs": 50, "fats": 20, "cals": 600,
                                 "ingredients": ["200g de pechuga de pollo"]}]}]}
    nut = {"total_daily_calories": 2000,
           "total_daily_macros": {"protein_str": "64g", "carbs_str": "250g", "fats_str": "60g"}}
    form = {"medicalConditions": ["Enfermedad renal crónica"], "weight": 80, "weightUnit": "kg", "gender": "male"}
    go._apply_deterministic_clinical_layer(plan, form, nut)
    assert _called.get("yes") is True                               # el trim renal corrió por el fallback
    assert plan["renal_protein_cap"]["meals_enforced"] is True
