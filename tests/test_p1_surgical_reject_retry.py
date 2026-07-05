"""[P1-SURGICAL-REJECT-RETRY · 2026-07-05] Retry quirúrgico sobre rechazo del reviewer.

Pedido explícito del owner: "cuando generas el día y el desayuno está mal pero los platos de
los otros horarios están correctos, debería dejar activos los otros y modificar solo el horario
que tiene el plato problema". Caso vivo corr=2f37b6b4: 2 regeneraciones COMPLETAS cobradas por
una repetición de proteína atribuible a días concretos.

Diseño:
  - `_surgical_reject_targets(state)` (helper compartido router↔nodo, patrón
    `_collect_unresolved_marker_days`): TODAS las rejection_reasons deben matchear la whitelist
    determinista (`_SURGICAL_REJECT_SAFE_PREFIXES`) Y los detectores puros (variety report +
    slot appropriateness) deben atribuir un subconjunto ESTRICTO de días. Si no → None (retry
    completo normal).
  - `should_retry` enruta a "marker_regen" (mismo nodo quirúrgico) ANTES del check de
    MAX_ATTEMPTS (sirve también como reparación de último intento). Una pasada por attempt
    (`_surgical_reject_attempted`, reset en retry_reflection).
  - El nodo en modo rechazo lleva al corrector los issues específicos del día + regla de
    MÍNIMA INTERVENCIÓN (comidas sanas quedan literales) y NO promueve a 'approved' (el plan
    venía rechazado — la re-review fresca decide).
"""
import os
import time

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)


def _read(rel):
    with open(os.path.join(_BACKEND, rel), encoding="utf-8") as f:
        return f.read()


_GO = _read("graph_orchestrator.py")

_PROTEIN_GATE_MSG = (
    "MISMA PROTEÍNA REPETIDA EL MISMO DÍA (rechazo de variedad): la misma proteína principal "
    "aparece en 2+ comidas del mismo día — comer lo mismo el mismo día fatiga."
)


@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "_emit_plan_quality_degraded_alert", lambda *a, **kw: None)
    monkeypatch.setattr(g, "_mark_plan_result_quality_degraded", lambda *a, **kw: None)
    return g


def _meal(slot, name, ings):
    return {"meal": slot, "name": name, "ingredients": list(ings),
            "ingredients_raw": list(ings), "recipe": ["Cocina."]}


def _mk_plan(bad_days=(1,), total=3):
    """Plan de `total` días; los `bad_days` traen pollo repetido en 2 comidas."""
    days = []
    for n in range(1, total + 1):
        if n in bad_days:
            meals = [
                _meal("Desayuno", "Mangú con Huevo", ["2 huevos", "200 g de plátano"]),
                _meal("Almuerzo", "Pollo Guisado", ["150 g de pechuga de pollo", "150 g de arroz"]),
                _meal("Cena", "Wrap de Pollo", ["120 g de pechuga de pollo", "1 tortilla integral"]),
            ]
        else:
            meals = [
                _meal("Desayuno", "Avena con Guineo", ["40 g de avena", "1 guineo"]),
                _meal("Almuerzo", "Pescado con Batata", ["150 g de filete de pescado blanco", "150 g de batata"]),
                _meal("Cena", "Tortilla de Claras con Yuca", ["4 claras", "100 g de yuca"]),
            ]
        days.append({"day": n, "meals": meals})
    return {"days": days}


def _mk_state(bad_days=(1,), total=3, reasons=None, **over):
    st = {
        "review_passed": False,
        "_rejection_severity": "minor",
        "rejection_reasons": reasons if reasons is not None else [_PROTEIN_GATE_MSG],
        "plan_result": _mk_plan(bad_days, total),
        "attempt": 1,
        "pipeline_start": time.time(),
        "form_data": {},
    }
    st.update(over)
    return st


# ---------------------------------------------------------------------------
# knobs + wiring estructural
# ---------------------------------------------------------------------------

def test_knobs_defaults():
    assert '_env_bool("MEALFIT_SURGICAL_REJECT_RETRY", True)' in _GO
    assert '_env_int("MEALFIT_SURGICAL_REJECT_MIN_BUDGET_S", 150' in _GO


def test_state_flag_declared_and_reset():
    assert "_surgical_reject_attempted: bool" in _GO, "flag declarado en PlanState"
    assert _GO.count('"_surgical_reject_attempted": False') >= 2, \
        "reset en initial_state Y en retry_reflection (mismo patrón que _marker_regen_attempted)"


def test_router_runs_before_max_attempts_check():
    i_router = _GO.index("[P1-SURGICAL-REJECT-RETRY] Rechazo atribuible a")
    i_max = _GO.index("Máximo de {MAX_ATTEMPTS} intentos alcanzado y revisión NO aprobada")
    assert i_router < i_max, \
        "el check quirúrgico corre ANTES del gate de MAX_ATTEMPTS (reparación de último intento)"
    win = _GO[i_router:i_router + 400]
    assert 'return "marker_regen"' in win, "reusa la ruta existente al nodo quirúrgico"


def test_node_reject_mode_no_promote():
    i = _GO.index("En modo RECHAZO el flag propio evita el loop")
    win = _GO[i:i + 700]
    assert '"_surgical_reject_attempted": True' in win
    assert "return state_update" in win, \
        "en modo rechazo retorna ANTES del bloque de promoción a 'approved'"
    # la promoción original sigue existiendo para el modo markers-tras-approved.
    assert "_best_attempt_review_passed" in _GO[i:i + 4000]


def test_node_minimal_intervention_directive():
    assert "REGLA DE MÍNIMA INTERVENCIÓN" in _GO
    i = _GO.index("REGLA DE MÍNIMA INTERVENCIÓN")
    win = _GO[i - 800:i + 500]
    assert "_reject_mode" in win and "EXACTAMENTE iguales" in win, \
        "las comidas sanas deben conservarse literales (pedido del owner)"


# ---------------------------------------------------------------------------
# funcional: clasificador de atribución
# ---------------------------------------------------------------------------

def test_targets_found_for_attributable_protein_repeat(go):
    t = go._surgical_reject_targets(_mk_state(bad_days=(1,)))
    assert t is not None
    assert t["days"] == [1]
    assert any("proteína" in s or "proteina" in s for s in t["issues_by_day"][1])


def test_targets_multiple_bad_days_subset(go):
    t = go._surgical_reject_targets(_mk_state(bad_days=(1, 3), total=3))
    assert t is not None and t["days"] == [1, 3]


def test_no_targets_when_reason_not_whitelisted(go):
    st = _mk_state(reasons=[_PROTEIN_GATE_MSG,
                            "SKELETON FIDELITY: el plan ignoró la asignación del planificador"])
    assert go._surgical_reject_targets(st) is None, \
        "UNA sola razón no-quirúrgica → retry completo (conservador)"


def test_no_targets_when_all_days_bad(go):
    st = _mk_state(bad_days=(1, 2, 3), total=3)
    assert go._surgical_reject_targets(st) is None, \
        "todos los días culpables → la regen completa cuesta lo mismo"


def test_no_targets_without_reasons_or_single_day(go):
    assert go._surgical_reject_targets(_mk_state(reasons=[])) is None
    assert go._surgical_reject_targets(_mk_state(bad_days=(1,), total=1)) is None


def test_slot_violation_is_attributable(go):
    st = _mk_state(reasons=[
        "COMIDA FUERA DE HORARIO (rechazo de coherencia cultural es-DO): Día 2, cena: «Avena con "
        "Frutas» es comida de desayuno en la cena, que no corresponde a la cena dominicana."])
    st["plan_result"]["days"][1]["meals"][2] = _meal(
        "Cena", "Avena con Frutas", ["40 g de avena", "1 guineo"])
    # el día 1 de _mk_plan trae pollo repetido; para aislar el caso slot lo limpiamos.
    st["plan_result"]["days"][0]["meals"][2] = _meal(
        "Cena", "Tortilla de Claras", ["4 claras"])
    t = go._surgical_reject_targets(st)
    assert t is not None and 2 in t["days"]
    assert any("fuera de horario" in s.lower() or "FUERA DE HORARIO" in s
               for s in t["issues_by_day"][2])


# ---------------------------------------------------------------------------
# funcional: enrutado en should_retry
# ---------------------------------------------------------------------------

def test_should_retry_routes_to_surgical(go):
    assert go.should_retry(_mk_state(bad_days=(1,))) == "marker_regen"


def test_should_retry_once_per_attempt(go):
    st = _mk_state(bad_days=(1,), _surgical_reject_attempted=True)
    assert go.should_retry(st) == "retry", \
        "flag seteado → una sola pasada quirúrgica por attempt, luego path normal"


def test_should_retry_knob_off(go, monkeypatch):
    monkeypatch.setattr(go, "SURGICAL_REJECT_RETRY_ENABLED", False)
    assert go.should_retry(_mk_state(bad_days=(1,))) == "retry"


def test_should_retry_non_attributable_falls_through(go):
    st = _mk_state(reasons=["SKELETON FIDELITY: el plan ignoró la asignación"])
    assert go.should_retry(st) == "retry"


def test_should_retry_last_attempt_repair(go):
    st = _mk_state(bad_days=(1,), attempt=go.MAX_ATTEMPTS)
    assert go.should_retry(st) == "marker_regen", \
        "en el intento final reparar > entregar degradado con banner"


def test_should_retry_no_budget_falls_through(go):
    st = _mk_state(bad_days=(1,),
                   pipeline_start=time.time() - (go.GLOBAL_PIPELINE_TIMEOUT_S - 10))
    out = go.should_retry(st)
    assert out != "marker_regen", "sin presupuesto quirúrgico NO se enruta al surgical"


def test_marker_anchored_in_source():
    assert "P1-SURGICAL-REJECT-RETRY" in _GO
