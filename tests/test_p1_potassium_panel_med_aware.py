"""[P1-POTASSIUM-PANEL-MED-AWARE · 2026-06-19] (audit fresco P1-1) El panel de micros era CIEGO a los
medicamentos: para un perfil HTA-sin-ERC subía el piso DASH de potasio a 4700 mg y emitía la nota "come
más guineo/aguacate/leguminosas" — INCLUSO si el usuario tomaba un ahorrador de potasio (espironolactona)
o un IECA/ARA-II, fármacos que ELEVAN el potasio sérico (hiperkalemia → arritmia). En paralelo el
`medication_review` decía lo contrario ("NO maximices potasio"): señales OPUESTAS en el mismo PDF, con el
panel determinista empujando la dirección peligrosa.

El fix pasa un flag `k_elevating_med` a `build_micronutrient_report` (derivado de
`detect_potassium_elevating_med`): cuando hay un fármaco-K, NO se eleva el piso DASH de potasio ni se emite
la nota de maximización — espejo exacto del guard renal `not _has_renal` (la ERC ya suprime el piso DASH-K).
El sodio bajo y el magnesio de DASH sí se mantienen. Este test ancla ambas mitades.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import medication_rules as mr
import micronutrients as mn


class _StubDB:
    """Devuelve micros fijos por ingrediente → control total del test (K bajo el piso a propósito)."""
    def __init__(self, **micros):
        self._m = micros

    def micros_from_ingredient_string(self, s):
        return dict(self._m)


def _plan(n_ings=3):
    return {"days": [{"meals": [{"ingredients": [f"ing{i}" for i in range(n_ings)]}]}]}


# ════════════════════════════════════════════════════════════════════════════════════════════════
# detect_potassium_elevating_med — los fármacos que SUBEN el potasio sérico
# ════════════════════════════════════════════════════════════════════════════════════════════════
def test_detect_kelev_spironolactone():
    assert mr.detect_potassium_elevating_med({"medications": ["Espironolactona"]}) is True


def test_detect_kelev_potassium_sparing_family():
    for med in ("aldactone", "eplerenona", "amilorida", "triamtereno"):
        assert mr.detect_potassium_elevating_med({"medications": [med]}) is True, med


def test_detect_kelev_ace_arb():
    # IECA/ARA-II también elevan el potasio (la fila ace_arb está en el set).
    for med in ("losartan", "lisinopril", "enalapril", "valsartan"):
        assert mr.detect_potassium_elevating_med({"medications": [med]}) is True, med


def test_detect_kelev_false_for_non_k_meds():
    # Metformina/warfarina/estatina NO elevan el potasio → no deben suprimir el piso DASH.
    assert mr.detect_potassium_elevating_med({"medications": ["metformina"]}) is False
    assert mr.detect_potassium_elevating_med({"medications": ["warfarina"]}) is False
    assert mr.detect_potassium_elevating_med({"medications": ["atorvastatina"]}) is False


def test_detect_kelev_false_when_no_meds():
    assert mr.detect_potassium_elevating_med({"medications": []}) is False
    assert mr.detect_potassium_elevating_med({"medications": ["Ninguno"]}) is False
    assert mr.detect_potassium_elevating_med({}) is False


def test_detect_kelev_free_text_backstop():
    # Un ahorrador-K escrito en texto libre ("otra condición") también cuenta (defensa en profundidad).
    assert mr.detect_potassium_elevating_med({"otherConditions": "tomo espironolactona para el corazon"}) is True


# ════════════════════════════════════════════════════════════════════════════════════════════════
# build_micronutrient_report — el piso DASH de potasio cede ante un fármaco-K (espejo del guard renal)
# ════════════════════════════════════════════════════════════════════════════════════════════════
def _report(k_elevating_med, sex="M"):
    db = _StubDB(grams=100, potassium_mg=200, magnesium_mg=30, sodium_mg=100)
    return mn.build_micronutrient_report(_plan(), db, sex=sex, conditions=["Hipertensión"],
                                         k_elevating_med=k_elevating_med)


def test_hta_no_kmed_keeps_dash_4700_floor():
    """Sin fármaco-K: comportamiento clásico DASH (piso 4700, fila estándar)."""
    rep = _report(k_elevating_med=False)
    panel = {p["key"]: p for p in rep["panel"]}
    assert panel["potassium_mg"]["piso"] == 4700.0
    cts = {c["condicion"] for c in rep["condition_targets"]}
    assert "Hipertensión (patrón DASH)" in cts


def test_hta_with_kmed_does_not_maximize_potassium():
    """Con fármaco-K: el piso DASH de potasio NO se eleva (queda en el DRI), pero magnesio/sodio sí."""
    rep = _report(k_elevating_med=True, sex="M")
    panel = {p["key"]: p for p in rep["panel"]}
    # Potasio NO sube a 4700 — queda en el DRI male (3400), evitando el nudge a hiperkalemia.
    assert panel["potassium_mg"]["piso"] == 3400.0
    # Magnesio DASH SÍ se mantiene (un ahorrador-K no contraindica el magnesio).
    assert panel["magnesium_mg"]["piso"] == 500.0


def test_hta_with_kmed_replaces_dash_row_no_maximize_copy():
    """Con fármaco-K: la fila DASH NO instruye maximizar potasio; lo modera por medicación."""
    rep = _report(k_elevating_med=True)
    cts = {c["condicion"]: c for c in rep["condition_targets"]}
    # La fila estándar "Hipertensión (patrón DASH)" (que pide ≥4700) NO está.
    assert "Hipertensión (patrón DASH)" not in cts
    # En su lugar, una fila que modera el potasio por la medicación.
    moderada = [c for k, c in cts.items() if "potasio moderado por medicación" in k]
    assert moderada, "debe haber una fila DASH que modere el potasio por el fármaco"
    regla = moderada[0]["regla"]
    assert "4700" not in regla, "no debe pedir ≥4700 mg de potasio con un fármaco que lo eleva"
    assert "NO maximizar el potasio" in regla


def test_hta_with_kmed_no_potassium_gap_note_pushing_up():
    """Coherencia con medication_review: ningún condition_target empuja a ≥4700 de potasio."""
    rep = _report(k_elevating_med=True)
    for c in rep["condition_targets"]:
        assert "≥4700" not in c.get("regla", ""), c


def test_female_hta_with_kmed_keeps_dri_floor():
    rep = _report(k_elevating_med=True, sex="F")
    panel = {p["key"]: p for p in rep["panel"]}
    assert panel["potassium_mg"]["piso"] == 2600.0   # DRI female, sin maximización DASH
    assert panel["magnesium_mg"]["piso"] == 500.0


def test_hta_with_kmed_potassium_note_moderates_not_pushes_up():
    """El gap del piso de potasio NO debe decir 'come más guineo' con un fármaco-K (PDF coherente con el aviso)."""
    rep = _report(k_elevating_med=True, sex="M")
    kgap = [g for g in rep["gaps"] if g["key"] == "potassium_mg"]
    assert kgap, "K bajo el DRI → gap de potasio presente"
    nota = kgap[0]["nota"].lower()
    assert "guineo" not in nota and "aumenta" not in nota, "no debe nudgear a subir el potasio"
    assert "modera" in nota, "debe moderar el potasio"


def test_hta_without_kmed_potassium_note_is_standard_dash():
    """Sin fármaco-K: la nota estándar (sube potasio con guineo/legumbres) se mantiene."""
    rep = _report(k_elevating_med=False, sex="M")
    kgap = [g for g in rep["gaps"] if g["key"] == "potassium_mg"]
    assert kgap
    assert "guineo" in kgap[0]["nota"].lower()
