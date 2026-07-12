"""[P1-CHAT-DAY-REGEN-TOOL · 2026-07-12] El chat puede actualizar un día completo del plan.

Pedido del owner: paridad con los botones del Plan — el agente ya cubría el
plato individual (`modify_single_meal`); faltaba el día completo ("Actualizar
platos"). Diseño: la tool NO re-implementa el motor — invoca DIRECTAMENTE el
handler `api_regenerate_day` (patrón de los tests), heredando el gate clínico
server-side, el gate de suficiencia de nevera, la idempotencia de 45s, el
cobro de 1 crédito y el flag `_day_regen_inflight` (la página Plan muestra la
ola de progreso sola). Corre en thread daemon: el handler es SÍNCRONO
(~1-2 min de swaps LLM) y bloquearía el stream del chat.
tooltip-anchor: P1-CHAT-DAY-REGEN-TOOL
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

from tools import agent_tools, regenerate_full_day  # noqa: E402

with open(os.path.join(_BACKEND, "tools.py"), encoding="utf-8") as f:
    _TL = f.read()


def test_tool_defined_and_gated():
    # [P1-CHAT-PLAN-TOOLS-OFF · 2026-07-12] La tool existe pero su registro en
    # agent_tools sigue el knob MEALFIT_CHAT_PLAN_TOOLS_ENABLED (OFF por
    # decisión del owner — el chat no muta el plan por ahora).
    from tools import _PLAN_MUTATION_TOOLS, _chat_plan_mutation_tools_enabled
    assert "regenerate_full_day" in [t.name for t in _PLAN_MUTATION_TOOLS]
    in_active = "regenerate_full_day" in [t.name for t in agent_tools]
    assert in_active == _chat_plan_mutation_tools_enabled(), \
        "el registro de la tool debe seguir el knob"
    assert {"user_id", "day_number"} <= set(regenerate_full_day.args.keys())


def _tool_body():
    i = _TL.find("def regenerate_full_day(")
    assert i != -1
    return _TL[i:i + 6000]


def test_tool_reuses_endpoint_not_reimplements():
    body = _tool_body()
    assert "from routers.plans import api_regenerate_day" in body, \
        "reusar el handler = heredar gates clínicos/nevera/idempotencia/cobro"
    assert "verified_user_id=user_id" in body
    assert "threading.Thread" in body and "daemon=True" in body, \
        "el handler es síncrono ~1-2 min: en el hilo del chat mataría el stream"


def test_tool_has_paywall_gate():
    body = _tool_body()
    assert "from auth import verify_api_quota" in body, \
        "mismo paywall que el botón — sin esto el chat regeneraría gratis al llegar al cap"


def test_tool_validates_day_against_active_plan():
    body = _tool_body()
    assert "ORDER BY GREATEST(created_at" in body, \
        "plan activo = mismo resolver SSOT de recencia que restore/rename"
    assert "jsonb_array_length" in body, "valida day_number contra los días reales del plan"


def test_prompts_teach_the_tool_with_confirmation():
    with open(os.path.join(_BACKEND, "prompts", "chat_agent.py"), encoding="utf-8") as f:
        prompts = f.read()
    assert prompts.count("regenerate_full_day") >= 2, \
        "ambos builders (inline y stream) deben enseñar la tool"
    assert "CONFIRMA" in prompts, \
        "cuesta 1 crédito + 2 min: el agente debe confirmar antes de disparar"


def test_doc_documents_tool():
    # [P1-CHAT-PLAN-TOOLS-OFF] La tool vive en la sección "Retiradas temporalmente"
    # del doc (fuera de la tabla de paridad) mientras el knob esté OFF.
    with open(os.path.join(_BACKEND, "docs", "agent_tools_user_id_table.md"),
              encoding="utf-8") as f:
        doc = f.read()
    assert "regenerate_full_day" in doc and "Retiradas temporalmente" in doc
