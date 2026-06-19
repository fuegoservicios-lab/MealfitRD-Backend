"""[P2 audit fresco 2026-06-19] Batch clínico/precisión: P2-1/2/3/4/5/7/8/9/13.

Ancla los P2 implementables-limpios del audit fresco 2026-06-19:
  P2-2 renal+HTA potasio (no piso DASH 4700 en ERC) · P2-4 dri_targets pregnancy-aware (hierro 27/B12 2.6
  + condition_target folato) · P2-7 dosis hierro age-aware (post-menopausia) · P2-8 vit-K terms genéricos ·
  P2-1 food-safety seafood (ceviche/sushi detectados, veg crudo NO; 'ceviche' fuera del prompt) ·
  P2-3 Guard 8e embarazo FS9 · P2-5 cap renal peso ajustado · P2-9 telemetría clamp solver · P2-13 marker-regen.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import micronutrients as mn
import medication_rules as mr

_GO_PATH = Path(__file__).resolve().parent.parent / "graph_orchestrator.py"


class _StubMicroDB:
    def micros_from_ingredient_string(self, s):
        return {}


# ════════════════════════════════════════════════════════════════════════════════════════════════
# P2-2 · renal+HTA: el piso DASH de potasio (4700) NO se aplica en ERC comórbida
# ════════════════════════════════════════════════════════════════════════════════════════════════
def _report(conditions, **kw):
    plan = {"days": [{"meals": [{"ingredients": ["100g de arroz"]}]}]}
    return mn.build_micronutrient_report(plan, _StubMicroDB(), sex="F", conditions=conditions, **kw)


def test_p2_2_hta_alone_has_dash_potassium():
    cts = _report(["hipertension"]).get("condition_targets", [])
    assert any(ct["condicion"] == "Hipertensión (patrón DASH)" for ct in cts)


def test_p2_2_renal_hta_drops_dash_potassium():
    cts = _report(["hipertension", "enfermedad renal"]).get("condition_targets", [])
    assert not any(ct["condicion"] == "Hipertensión (patrón DASH)" for ct in cts), \
        "ERC comórbida → no debe imponerse el piso DASH de potasio (contraindicado)"
    assert any(ct["condicion"] == "Enfermedad renal crónica" for ct in cts), "debe quedar el target renal (P1-4)"


# ════════════════════════════════════════════════════════════════════════════════════════════════
# P2-4 · dri_targets pregnancy-aware
# ════════════════════════════════════════════════════════════════════════════════════════════════
def test_p2_4_dri_targets_pregnant_iron_b12():
    t = mn.dri_targets(sex="F", pregnant=True)
    assert t["iron_mg"]["floor"] == 27.0
    assert t["b12_mcg"]["floor"] == 2.6
    # No-embarazada (joven) sigue en 18.
    assert mn.dri_targets(sex="F", age=30)["iron_mg"]["floor"] == 18.0


def test_p2_4_pregnancy_condition_target():
    cts = _report([], pregnant=True).get("condition_targets", [])
    assert any(ct["condicion"] == "Embarazo / lactancia" and "Folato" in ct["regla"] for ct in cts)


# ════════════════════════════════════════════════════════════════════════════════════════════════
# P2-7 · dosis de hierro age-aware
# ════════════════════════════════════════════════════════════════════════════════════════════════
def _iron_gap_report():
    return {"gaps": [{"key": "iron_mg", "status": "bajo", "nutriente": "Hierro",
                      "valor": 6, "piso": 18, "unidad": "mg"}]}


def test_p2_7_iron_dose_postmenopausal():
    items = mn.build_supplement_recommendations(_iron_gap_report(), sex="F", age=60)["items"]
    assert any("post-menopausia" in it["dosis_sugerida"] for it in items)


def test_p2_7_iron_dose_menstruating_when_young():
    items = mn.build_supplement_recommendations(_iron_gap_report(), sex="F", age=30)["items"]
    assert any("menstrúas" in it["dosis_sugerida"] for it in items)


def test_p2_7_iron_dose_pregnant_overrides(  # [review P2] coherente con el piso 27 del panel, no la contradictoria 18/8
):
    items = mn.build_supplement_recommendations(_iron_gap_report(), sex="F", age=30, pregnant=True)["items"]
    assert any("27 mg" in it["dosis_sugerida"] and "embarazo" in it["dosis_sugerida"].lower() for it in items)


# ════════════════════════════════════════════════════════════════════════════════════════════════
# P2-8 · vit-K monitor cuenta términos genéricos de hoja verde
# ════════════════════════════════════════════════════════════════════════════════════════════════
def test_p2_8_vitk_counts_generic_greens():
    plan = {"days": [
        {"meals": [{"ingredients": ["1 ensalada verde"]}]},
        {"meals": [{"ingredients": ["1/2 aguacate"]}]},
    ]}
    per_day = mr._high_vit_k_count_per_day(plan)
    assert per_day == [1, 1], "ensalada verde y aguacate deben contar como hoja verde / vit K"


# ════════════════════════════════════════════════════════════════════════════════════════════════
# graph_orchestrator: P2-1 / P2-3 / P2-5 / P2-9 / P2-13
# ════════════════════════════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


@pytest.fixture(scope="module")
def src() -> str:
    return _GO_PATH.read_text(encoding="utf-8")


# P2-1 food-safety seafood
def test_p2_1_seafood_scan_detects_raw_not_veg(go):
    plan = {"days": [{"meals": [
        {"name": "Ceviche de pescado", "ingredients": ["200g de pescado", "limón"]},
        {"name": "Ensalada", "ingredients": ["zanahoria cruda", "lechuga"]},
    ]}]}
    viol = go._scan_raw_seafood_meat_violations(plan)
    names = {v[2] for v in viol}
    assert "Ceviche de pescado" in names, "ceviche (pescado crudo) debe detectarse"
    assert "Ensalada" not in names, "vegetal crudo NO debe flagearse (zanahoria cruda es segura)"


def test_p2_1_tartar_ambiguous_requires_animal_protein(go):
    # [review P2] tartar/carpaccio de VEGETAL NO debe flagearse; el de proteína animal SÍ.
    plan = {"days": [{"meals": [
        {"name": "Tartar de remolacha", "ingredients": ["remolacha", "alcaparras"]},
        {"name": "Tartar de atún", "ingredients": ["atún fresco", "soya"]},
        {"name": "Carpaccio de calabacín", "ingredients": ["calabacín", "limón"]},
    ]}]}
    names = {v[2] for v in go._scan_raw_seafood_meat_violations(plan)}
    assert "Tartar de atún" in names
    assert "Tartar de remolacha" not in names, "tartar de vegetal no es pescado/carne crudos"
    assert "Carpaccio de calabacín" not in names


def test_p2_1_ceviche_removed_from_prompt(src):
    assert "al horno→ceviche" not in src, "el prompt no debe inducir ceviche (pescado crudo)"
    assert "al horno→al vapor" in src


# P2-3 Guard 8e embarazo FS9 (funcional)
def test_p2_3_pregnancy_fs9_gate(go):
    plan = {"days": [{"meals": [{"name": "A", "ingredients": ["100g de arroz", "150g de pollo"]}]}]}
    out = go._apply_deterministic_clinical_layer(plan, {"medicalConditions": ["Embarazo"], "gender": "female"}, {})
    rpr = out.get("requires_professional_review")
    assert isinstance(rpr, dict) and rpr.get("flag") is True and rpr.get("pregnancy") is True


# P2-5 cap renal peso ajustado
def test_p2_5_renal_adjusted_weight(go, monkeypatch):
    form = {"weight": 120, "weightUnit": "kg", "height": 170, "gender": "male"}  # IMC ~41.5
    monkeypatch.setattr(go, "RENAL_ADJUSTED_WEIGHT_ENABLED", False)
    assert go._renal_weight_basis_kg(form) == pytest.approx(120.0), "knob OFF → peso real"
    monkeypatch.setattr(go, "RENAL_ADJUSTED_WEIGHT_ENABLED", True)
    adj = go._renal_weight_basis_kg(form)
    assert adj < 120.0, "knob ON + IMC>30 → peso ajustado (menor que el real, cap más conservador)"
    # No-obeso (IMC<30) → peso real aún con knob ON.
    assert go._renal_weight_basis_kg({"weight": 70, "weightUnit": "kg", "height": 175, "gender": "male"}) == pytest.approx(70.0)


# P2-9 / P2-13 parser-anchors
def test_p2_9_solver_clamp_telemetry_present(src):
    assert "P2-SOLVER-CLAMP-TELEMETRY" in src
    assert '_solver_clamp_saturated' in src


def test_p2_13_marker_regen_strips_clinical_flag(src):
    assert "P2-MARKER-REGEN-CLINICAL-RECHECK" in src
    assert '_best_snap.pop("_clinical_layer_applied", None)' in src


def test_p2_13_reapplies_clinical_layer_after_swap(src):
    # [review P1 fix] el strip solo sirve si la capa se RE-APLICA tras el swap (el seam _is_fallback no lo cubre).
    i_swap = src.find("_swapped = _swap_to_best_attempt_if_better(final_state)")
    i_reapply = src.find("Capa clínica re-aplicada al snapshot")
    assert i_swap != -1 and i_reapply != -1, "debe existir la re-aplicación post-swap"
    assert i_swap < i_reapply, "la re-aplicación de la capa clínica debe ir DESPUÉS del swap"
    # …y gateada por flag ausente (idempotente para snapshots normales).
    assert 'not _swapped_plan.get("_clinical_layer_applied")' in src
