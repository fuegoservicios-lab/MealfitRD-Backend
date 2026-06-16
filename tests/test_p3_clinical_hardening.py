"""[P3-CLINICAL-HARDENING · 2026-06-13] Las 3 mejoras hacia uso clínico de producción:
- FS8 (P3-SUPPLEMENT-ADVICE): recomendaciones de suplementación accionables para cerrar gaps.
- FS9 (P3-PRO-REVIEW-FLAG): flag de revisión profesional si hay condiciones médicas.
- FS7 (P3-VARIETY-HARD-GATE): cap de huevo como restricción dura (rechazo+retry) en review.
"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from micronutrients import build_supplement_recommendations

_GO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "graph_orchestrator.py")


def _report(*gaps):
    return {"gaps": list(gaps)}


def _gap(key, nutriente, valor, piso, unidad, status="bajo"):
    return {"key": key, "nutriente": nutriente, "valor": valor, "piso": piso,
            "unidad": unidad, "status": status}


# ---------- FS8: supplement advice ----------

def test_supp_genera_recs_para_floors_bajos():
    rep = _report(
        _gap("vit_d_mcg", "Vitamina D", 4.0, 15.0, "mcg"),
        _gap("iron_mg", "Hierro", 12.0, 18.0, "mg"),
    )
    out = build_supplement_recommendations(rep, sex="female")
    assert out["count"] == 2
    keys = {i["key"] for i in out["items"]}
    assert keys == {"vit_d_mcg", "iron_mg"}
    assert all(i["dosis_sugerida"] and i["precaucion"] and i["primero_alimentos"] for i in out["items"])
    assert "no sustituye" in out["disclaimer"].lower() or "no una prescripción" in out["disclaimer"].lower()


def test_supp_hierro_es_sex_aware():
    rep = _report(_gap("iron_mg", "Hierro", 10.0, 18.0, "mg"))
    f = build_supplement_recommendations(rep, sex="female")["items"][0]["dosis_sugerida"]
    m = build_supplement_recommendations(rep, sex="male")["items"][0]["dosis_sugerida"]
    assert "18 mg" in f and "8 mg" in m and f != m


def test_supp_ignora_ceilings():
    # Sodio/azúcar son techos (reducir), NO suplemento.
    rep = _report(
        {"key": "sodium_mg", "nutriente": "Sodio", "valor": 3000, "techo": 2000, "unidad": "mg", "status": "alto"},
        _gap("vit_d_mcg", "Vitamina D", 3.0, 15.0, "mcg"),
    )
    out = build_supplement_recommendations(rep, sex="female")
    assert out["count"] == 1 and out["items"][0]["key"] == "vit_d_mcg"


def test_supp_sin_gaps_no_recs():
    assert build_supplement_recommendations({"gaps": []}, sex="male")["count"] == 0


# ---------- FS9 + FS7: parser-anchors de la lógica inline ----------

def _func_src(name):
    src = open(_GO, encoding="utf-8").read()
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, ast.AsyncFunctionDef) and n.name == name:
            return ast.get_source_segment(src, n)
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return ast.get_source_segment(src, n)
    raise AssertionError(f"{name} no encontrada")


def test_fs9_pro_review_flag_en_assemble():
    # [P3-FALLBACK-CLINICAL-LAYER Fase B] FS9 se movió del inline de assemble al SSOT
    # `_apply_deterministic_clinical_layer` (que assemble + fallback heredan). El test verifica el wiring
    # en el SSOT + que assemble delegue a él.
    src = _func_src("_apply_deterministic_clinical_layer")
    assert "requires_professional_review" in src
    assert "_has_real_medical_flags" in src
    assert "PRO_REVIEW_FLAG_ENABLED" in src
    assert "_apply_deterministic_clinical_layer(" in _func_src("assemble_plan_node"), \
        "assemble debe delegar la capa clínica al SSOT (Fase B)"


def test_fs7_variety_hard_gate_en_review():
    src = _func_src("review_plan_node")
    assert "VARIETY_HARD_GATE_ENABLED" in src
    assert "variety_report" in src
    # debe rechazar (approved=False) y escalar severity ante sobreuso de huevo
    assert "SOBREUSO DE HUEVO" in src


def test_fs8_supplement_wiring_en_assemble():
    # [P3-FALLBACK-CLINICAL-LAYER Fase B] FS8 (suplementación) se movió al SSOT
    # `_apply_deterministic_clinical_layer` (heredado por assemble + fallback).
    src = _func_src("_apply_deterministic_clinical_layer")
    assert "build_supplement_recommendations" in src
    assert "micronutrient_supplement_advice" in src
    assert "_apply_deterministic_clinical_layer(" in _func_src("assemble_plan_node"), \
        "assemble debe delegar la capa clínica al SSOT (Fase B)"
