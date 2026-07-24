"""[P1-HARDEN-POOLS-CANARY-GATING · 2026-07-24] El canario A1 etiquetaba pero NO gateaba.

Bug (encontrado al ir a encender A1 en prod, 2026-07-24):
    `_harden_pools_canary_cohort(state)` (graph_orchestrator.py:593) reparte a los usuarios
    en cohortes 'off'/'on' con bucket sha256 y ese tag viaja a la métrica `clinical_band`
    (:36921, `harden_pools_cohort`) — la ÚNICA dimensión que permite sliceear el A/B.

    Pero el enforcer se invocaba SIN la cohorte::

        _harden_counts = harden_day_pools(skeleton, form_data, None)   # <- sin cohorte

    y `harden_day_pools` solo mira el master switch. Con `HARDEN_POOLS_ENABLED=True` y
    `HARDEN_POOLS_CANARY_PCT=50`, el enforcer corría para el **100%** de los planes mientras
    la telemetría etiquetaba al 50% como grupo de control. El A/B no quedaba incompleto:
    quedaba INVERTIDO en la mitad de la muestra — peor que no medir, porque el resultado
    parecería decir que la restricción dura no sirve.

    Con el default de hoy (`PCT=0` → todos 'on') el tag es honesto por accidente, así que
    el bug solo se manifiesta al configurar un canario real. Que es exactamente lo que hay
    que hacer para poder apagar `self_critique` con datos.

Fix:
    `harden_day_pools(..., cohort=...)` — early-return cuando la cohorte es 'off'. El call
    site calcula la cohorte UNA vez, se la pasa al enforcer y reporta ESA MISMA en el state:
    lo que la telemetría dice que pasó es lo que pasó.

    `cohort=None` (default) = sin gating por cohorte → preserva la semántica de los tests y
    de cualquier caller que no participe del canario.
"""
import re
from pathlib import Path

import graph_orchestrator as go


_SRC = Path(__file__).resolve().parent.parent / "graph_orchestrator.py"


def _days_with(pool_of):
    return {"days": [{"day": i + 1, "protein_pool": list(p), "carb_pool": [], "fruit_pool": []}
                     for i, p in enumerate(pool_of)]}


def _skel_needing_arity():
    """Día 1 con 2 gate-labels para 3 mains (el caso del forensic corr=d57ffe04)."""
    return _days_with([
        ["Salmón", "Huevos", "Mantequilla de maní"],
        ["Res", "Chivo", "Habichuelas rojas"],
        ["Pollo", "Cerdo", "Habichuelas rojas"],
    ])


def _enable(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    monkeypatch.setattr(go, "HARDEN_MAIN_ARITY", True)
    monkeypatch.setattr(go, "HARDEN_MAIN_ARITY_TARGET", 3)


# ---------------------------------------------------------------------------
# 1. El enforcer respeta la cohorte
# ---------------------------------------------------------------------------
def test_cohorte_off_no_endurece(monkeypatch):
    """Grupo de control: los pools deben quedar EXACTAMENTE como los dejó el LLM."""
    _enable(monkeypatch)
    skel = _skel_needing_arity()
    before = [list(d["protein_pool"]) for d in skel["days"]]
    counts = go.harden_day_pools(skel, {}, None, cohort="off")
    after = [d["protein_pool"] for d in skel["days"]]
    assert before == after, "la cohorte de control NO debe recibir el endurecimiento"
    assert not any(counts.values()), f"cohorte 'off' debe reportar cero mutaciones, dio {counts}"


def test_cohorte_on_si_endurece(monkeypatch):
    _enable(monkeypatch)
    skel = _skel_needing_arity()
    counts = go.harden_day_pools(skel, {}, None, cohort="on")
    assert counts["main_arity_added"] > 0, f"cohorte 'on' debe endurecer, dio {counts}"


def test_sin_cohorte_preserva_semantica_previa(monkeypatch):
    """`cohort=None` = sin canario → aplica (compat con los tests A1 existentes)."""
    _enable(monkeypatch)
    skel = _skel_needing_arity()
    counts = go.harden_day_pools(skel, {}, None)
    assert counts["main_arity_added"] > 0


def test_master_off_gana_sobre_la_cohorte(monkeypatch):
    """El kill-switch sigue siendo el kill-switch: 'on' no lo resucita."""
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", False)
    monkeypatch.setattr(go, "HARDEN_MAIN_ARITY", True)
    skel = _skel_needing_arity()
    before = [list(d["protein_pool"]) for d in skel["days"]]
    go.harden_day_pools(skel, {}, None, cohort="on")
    assert before == [d["protein_pool"] for d in skel["days"]]


# ---------------------------------------------------------------------------
# 2. Telemetría honesta: se reporta la cohorte que se aplicó
# ---------------------------------------------------------------------------
def test_callsite_pasa_la_misma_cohorte_que_reporta():
    """Una sola evaluación de la cohorte: la del enforcer y la del tag deben ser el MISMO
    valor. Si alguien vuelve a llamar `_harden_pools_canary_cohort(state)` dentro del
    `return`, dos evaluaciones podrían divergir si el bucketing dejara de ser puro."""
    src = _SRC.read_text(encoding="utf-8")
    i = src.index("async def plan_skeleton_node")
    j = src.index("def _gate_label_of")
    body = src[i:j]

    m_call = re.search(r"harden_day_pools\(\s*skeleton\s*,\s*form_data\s*,\s*None\s*,\s*cohort\s*=\s*(\w+)", body)
    assert m_call, "el call site debe pasar la cohorte al enforcer"
    var = m_call.group(1)

    m_assign = re.search(rf"{re.escape(var)}\s*=\s*_harden_pools_canary_cohort\(state\)", body)
    assert m_assign, f"`{var}` debe salir de _harden_pools_canary_cohort(state)"
    assert m_assign.start() < m_call.start(), "la cohorte se calcula ANTES de endurecer"

    m_tag = re.search(rf'"_harden_pools_cohort"\s*:\s*{re.escape(var)}\b', body)
    assert m_tag, "el state debe reportar la MISMA variable de cohorte que se le pasó al enforcer"
    assert f'"_harden_pools_cohort": _harden_pools_canary_cohort(' not in body, (
        "no re-evaluar la cohorte en el return — se reporta la que realmente se aplicó"
    )


def test_marker_presente():
    src = _SRC.read_text(encoding="utf-8")
    assert re.search(r"\[P1-HARDEN-POOLS-CANARY-GATING\s*·\s*2026-07-24\]", src)


# ---------------------------------------------------------------------------
# 3. La cohorte tiene que SOBREVIVIR al state (LangGraph filtra por schema)
# ---------------------------------------------------------------------------
def _plan_state_keys() -> set:
    src = _SRC.read_text(encoding="utf-8")
    i = src.index("class PlanState(TypedDict):")
    j = src.index("def _ensure_plan_result_contract", i)
    return set(re.findall(r"^\s{4}(\w+)\s*:", src[i:j], re.M))


def test_langgraph_descarta_claves_no_declaradas():
    """El mecanismo del bug, aislado: si no está en el TypedDict, no llega al final_state.

    Verificado en vivo antes de escribir esto — 80/80 filas de `clinical_band` con tag
    desde 2026-07-09 decían cohorte "on" con el master switch OFF (enforcer sin correr
    nunca): el `.get(...) or "on"` caía siempre al fallback."""
    from typing import TypedDict
    from langgraph.graph import StateGraph, END

    class _S(TypedDict):
        declarada: str

    g = StateGraph(_S)
    g.add_node("n", lambda s: {"declarada": "si", "_no_declarada": "x"})
    g.set_entry_point("n")
    g.add_edge("n", END)
    out = g.compile().invoke({"declarada": "inicial"})
    assert "_no_declarada" not in out, (
        "si LangGraph dejara de filtrar, este test sobra — pero la declaración explícita "
        "en PlanState sigue siendo la forma correcta"
    )


def test_cohortes_declaradas_en_planstate():
    keys = _plan_state_keys()
    for k in ("_harden_pools_cohort", "_self_critique_cohort"):
        assert k in keys, f"`{k}` debe declararse en PlanState o el tag se pierde"


def test_toda_cohorte_leida_por_la_metrica_esta_declarada():
    """Blanket: cualquier `final_state.get("_..._cohort")` que alimente la métrica
    `clinical_band` debe existir en PlanState. Generaliza el defecto a los canarios
    que se añadan después."""
    src = _SRC.read_text(encoding="utf-8")
    declared = _plan_state_keys()
    leidas = set(re.findall(r'final_state\.get\(\s*"(_\w*cohort)"', src))
    assert leidas, "no se encontró ninguna cohorte leída desde final_state"
    faltantes = sorted(leidas - declared)
    assert not faltantes, (
        f"cohortes leídas por la telemetría pero NO declaradas en PlanState: {faltantes}. "
        "LangGraph las descarta → el tag cae al fallback y el A/B mide una constante."
    )
