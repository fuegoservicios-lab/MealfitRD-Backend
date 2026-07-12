"""[P1-CHAT-PLAN-TOOLS-OFF · 2026-07-12] El chat NO muta el plan (decisión del owner, reversible).

Verbatim del owner: "quiero que el agente no tenga la capacidad de actualizar
platos de ninguna manera y mucho menos actualizar el plan completo, ya en el
futuro le daremos esa habilidad pero por ahora no".

Implementación reversible: las 3 tools de mutación (generate_new_plan_from_chat,
modify_single_meal, regenerate_full_day) quedan DEFINIDAS pero fuera de
`agent_tools` salvo `MEALFIT_CHAT_PLAN_TOOLS_ENABLED=true` (+ restart). Los
prompts redirigen a los botones de la página Plan y prohíben prometer/afirmar
modificaciones. La intercepción en execute_tools y el expand-fallback quedan
dormidos, intactos para el futuro.
tooltip-anchor: P1-CHAT-PLAN-TOOLS-OFF
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

from tools import agent_tools, _PLAN_MUTATION_TOOLS, _chat_plan_mutation_tools_enabled  # noqa: E402

_MUTATION_NAMES = {"generate_new_plan_from_chat", "modify_single_meal", "regenerate_full_day"}


def test_mutation_tools_out_of_active_set_by_default():
    if _chat_plan_mutation_tools_enabled():
        return  # entorno con el knob ON — el gating se verifica en el otro sentido
    active = {t.name for t in agent_tools}
    assert not (_MUTATION_NAMES & active), \
        f"tools de mutación activas con el knob OFF: {_MUTATION_NAMES & active}"


def test_mutation_tools_preserved_for_future():
    assert {t.name for t in _PLAN_MUTATION_TOOLS} == _MUTATION_NAMES, \
        "las 3 tools siguen definidas — re-habilitar = flip del knob, no re-código"


def test_prompts_redirect_to_buttons(monkeypatch):
    monkeypatch.delenv("MEALFIT_CHAT_PLAN_TOOLS_ENABLED", raising=False)
    from prompts.chat_agent import build_tools_instructions, build_tools_instructions_stream
    for builder in (build_tools_instructions, build_tools_instructions_stream):
        out = builder("11111111-1111-1111-1111-111111111111")
        assert "NO PUEDES modificar el plan" in out, "redirección explícita"
        assert "Cambiar Plato" in out and "Actualizar platos" in out, \
            "guía al usuario a los botones de la página Plan"
        assert "Usa `modify_single_meal`" not in out
        assert "Usa `regenerate_full_day`" not in out
        assert "Usa `generate_new_plan_from_chat`" not in out
        assert "NUNCA prometas modificar el plan" in out


def test_prompts_restore_with_knob_on(monkeypatch):
    monkeypatch.setenv("MEALFIT_CHAT_PLAN_TOOLS_ENABLED", "true")
    from prompts.chat_agent import build_tools_instructions
    out = build_tools_instructions("11111111-1111-1111-1111-111111111111")
    assert "Usa `modify_single_meal`" in out and "Usa `regenerate_full_day`" in out, \
        "con el knob ON los bullets originales regresan intactos"


def test_doc_moves_retired_tools_out_of_parity_table():
    with open(os.path.join(_BACKEND, "docs", "agent_tools_user_id_table.md"),
              encoding="utf-8") as f:
        doc = f.read()
    assert "Retiradas temporalmente" in doc
    assert "Las 11 tools cubiertas" in doc