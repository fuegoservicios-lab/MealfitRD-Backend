"""[P0-AGENT-1 · 2026-05-11] Force-override de `user_id` en agent tools.

Bug original (audit 2026-05-11):
    `backend/agent.py:execute_tools` invocaba 7 de las 9 tools del agente
    forwardeando `tool_args["user_id"]` que la LLM construyó desde el
    system prompt. Solo `generate_new_plan_from_chat` y `modify_single_meal`
    leían `state["user_id"]` (autenticado) para resolver el caller.

    Vector: prompt injection (mensaje del usuario, contenido importado vía
    vision_agent, recetas externas) → la LLM emite `tool_call` con
    `user_id="<otra-uuid>"` → backend ejecuta sin validar → cross-user
    write/read sobre `user_inventory`, `consumed_meals`, `user_facts`,
    `health_profile`, `meal_plans`.

Cierre:
    En `execute_tools`, ANTES de cualquier branch del if/elif y ANTES
    de `t.invoke(tool_args)`, force-override `tool_args["user_id"] =
    _trusted_uid` donde `_trusted_uid` se resuelve UNA vez del state
    (`state["user_id"]` o fallback a `state["session_id"]`).

Este test enforza:
    1. El bloque de resolución `_trusted_uid` existe ANTES del loop.
    2. El override `tool_args["user_id"] = _trusted_uid` aparece DENTRO
       del loop body, ANTES del primer `if tool_name == "..."` branch.
    3. Funcionalmente: si la LLM emite tool_call con `user_id` ajeno,
       el invoker recibe el trusted_uid (no el inyectado).

Si alguien refactorea `execute_tools` y rompe el invariante, este test
falla con copy explicativo + cross-link a invariante CLAUDE.md.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_AGENT_FP = _REPO_ROOT / "backend" / "agent.py"


@pytest.fixture(scope="module")
def src() -> str:
    return _AGENT_FP.read_text(encoding="utf-8")


def _extract_execute_tools_body(src: str) -> str:
    """Extrae el body de `def execute_tools(state: ChatState):` hasta el
    siguiente `def ` top-level."""
    start = src.find("def execute_tools(state: ChatState):")
    assert start > 0, "`def execute_tools(state: ChatState):` no encontrado en agent.py"
    rest = src[start + len("def execute_tools(state: ChatState):"):]
    next_match = re.search(r"\ndef\s+\w+\(", rest)
    end_offset = next_match.start() if next_match else len(rest)
    return src[start: start + len("def execute_tools(state: ChatState):") + end_offset]


def test_trusted_uid_resolution_present(src: str):
    """El bloque `_trusted_uid = ...` con fallback a session_id debe estar."""
    body = _extract_execute_tools_body(src)
    assert "_trusted_user_id = state.get(\"user_id\")" in body, (
        "P0-AGENT-1 regresión: la resolución de `_trusted_user_id` desapareció. "
        "Restaurar el bloque que lee `state.get('user_id')` y "
        "`state.get('session_id')` ANTES del loop `for tool_call in last_message.tool_calls`. "
        "Sin esto, el override pierde su fuente de verdad."
    )
    assert "_trusted_session_id = state.get(\"session_id\")" in body, (
        "P0-AGENT-1 regresión: el fallback a session_id no se está leyendo. "
        "Guests no tienen user_id real — el override necesita el session_id "
        "como identidad alternativa, mismo patrón que los 2 branches inline."
    )
    assert "_trusted_uid = (" in body or "_trusted_uid =" in body, (
        "P0-AGENT-1 regresión: la variable `_trusted_uid` no se compone. "
        "Debe ser `user_id if user_id and user_id != 'guest' else session_id`."
    )


def test_override_inside_loop_before_branches(src: str):
    """El override `tool_args[\"user_id\"] = _trusted_uid` DEBE aparecer
    DENTRO del loop `for tool_call in ...` y ANTES del primer
    `if tool_name == \"update_form_field\":`."""
    body = _extract_execute_tools_body(src)
    loop_idx = body.find("for tool_call in last_message.tool_calls:")
    assert loop_idx > 0, (
        "P0-AGENT-1 regresión: el loop sobre tool_calls desapareció. "
        "Restaurar el iterador antes de mergear."
    )
    first_branch_idx = body.find("if tool_name == \"update_form_field\":")
    assert first_branch_idx > 0, (
        "P0-AGENT-1 regresión: el primer branch del if/elif desapareció. "
        "Si renombraste las tools, ajustar este test."
    )
    override_idx = body.find('tool_args["user_id"] = _trusted_uid')
    assert override_idx > 0, (
        "P0-AGENT-1 regresión: NO se encontró `tool_args[\"user_id\"] = _trusted_uid` "
        "dentro de `execute_tools`. Esto reabre el IDOR vía LLM-supplied user_id "
        "para 7 de las 9 tools del agente. Restaurar el override antes del primer "
        "`if tool_name == \"...\":` branch. Ver invariante en CLAUDE.md sección "
        "'Anti-patrones de agent tools prohibidos'."
    )
    assert loop_idx < override_idx < first_branch_idx, (
        f"P0-AGENT-1 regresión: el override `tool_args[\"user_id\"] = _trusted_uid` "
        f"existe en la función pero su posición es incorrecta. "
        f"loop@{loop_idx}, override@{override_idx}, primer_branch@{first_branch_idx}. "
        f"Debe cumplirse loop_idx < override_idx < first_branch_idx para que el "
        f"override aplique a TODOS los branches (incluido el `else: t.invoke(tool_args)`)."
    )


def test_warn_log_on_mismatch_present(src: str):
    """Cuando llm_user_id != trusted, debe emitirse `WARN` con tag
    `[P0-AGENT-1]` para telemetría de prompt-injection attempts."""
    body = _extract_execute_tools_body(src)
    assert "[P0-AGENT-1]" in body, (
        "P0-AGENT-1 regresión: el tag de telemetría `[P0-AGENT-1]` desapareció "
        "del log de mismatch. SRE necesita poder grep'ear logs para detectar "
        "bursts de prompt-injection."
    )
    assert "logger.warning" in body, (
        "P0-AGENT-1 regresión: NO se está logueando WARN cuando hay mismatch. "
        "Sin log, los attempts pasan silenciosos — perdemos la señal de "
        "prompt-injection."
    )


def test_override_applies_to_else_branch(src: str):
    """Las 6 tools que caen en `else: t.invoke(tool_args)` (log_consumed_meal,
    modify_pantry_inventory, etc.) DEBEN recibir el override. Esto se
    garantiza estructuralmente porque el override está antes del if/elif/else,
    pero validamos que el `else` siga existiendo y que no haya un
    `tool_args = tool_call["args"]` reasignación entre el override y el else."""
    body = _extract_execute_tools_body(src)
    override_idx = body.find('tool_args["user_id"] = _trusted_uid')
    else_idx = body.find("else:\n", override_idx)
    assert else_idx > override_idx, (
        "P0-AGENT-1 regresión: el branch `else: t.invoke(tool_args)` "
        "desapareció. Sin él, las 6 tools no-pinned pierden su path de "
        "ejecución."
    )
    # No debe haber re-asignación de tool_args entre override y else
    interval = body[override_idx: else_idx]
    assert 'tool_args = tool_call["args"]' not in interval, (
        "P0-AGENT-1 regresión: encontrada re-asignación de `tool_args` entre "
        "el override y el branch `else:`. Esto invalida el override — el "
        "tool_args reseteado al dict del LLM contiene de nuevo el user_id "
        "ajeno potencial. Eliminar la re-asignación."
    )


def test_existing_inline_branches_still_use_state_user_id(src: str):
    """Los 2 branches que ya leían `state.get('user_id')` inline siguen
    haciéndolo (defense-in-depth). El override de tool_args es ortogonal:
    aunque tool_args ya esté pinned, los branches inline pasan user_id por
    parámetro al executor y eso NO depende de tool_args."""
    body = _extract_execute_tools_body(src)
    # generate_new_plan_from_chat branch
    gen_idx = body.find("elif tool_name == \"generate_new_plan_from_chat\":")
    assert gen_idx > 0, "branch `generate_new_plan_from_chat` desaparecido"
    # debe seguir leyendo state.get("user_id") inline
    gen_block = body[gen_idx: gen_idx + 800]
    assert "state.get(\"user_id\")" in gen_block, (
        "P0-AGENT-1 regresión: el branch `generate_new_plan_from_chat` ya no "
        "lee `state.get('user_id')` inline. Defense-in-depth perdida — si el "
        "override del top falla por bug futuro, este branch invocaría con "
        "session_id arbitrario."
    )
    # modify_single_meal branch
    mod_idx = body.find("elif tool_name == \"modify_single_meal\":")
    assert mod_idx > 0, "branch `modify_single_meal` desaparecido"
    mod_block = body[mod_idx: mod_idx + 1200]
    assert "state.get(\"user_id\")" in mod_block, (
        "P0-AGENT-1 regresión: el branch `modify_single_meal` ya no lee "
        "`state.get('user_id')` inline. Defense-in-depth perdida."
    )


# --------------------------------------------------------------------------
# Test funcional: mock de tool_call con user_id ajeno → override aplicado.
# --------------------------------------------------------------------------


@pytest.fixture
def execute_tools_callable(monkeypatch):
    """Importa `execute_tools` con stubs mínimos para evitar tocar DB/LLM
    reales. Los tests parser-based de arriba ya validan estructura; este
    fixture habilita el smoke test funcional.

    IMPORTANTE: usa `monkeypatch.setitem` (no `sys.modules.setdefault`) para
    que los stubs se limpien al terminar el test. El patrón `setdefault`
    deja MagicMocks pegados en sys.modules que rompen tests downstream que
    intentan importar el módulo real (test pollution observada con
    `test_p2_b_post_swap_revalidation`, `test_p1_a_knobs_registry_*`,
    `test_p2_2_post_swap_critical_divergence_alert` cuando este fixture
    corría primero en la sesión).

    Si `agent` ya fue importado antes (por otro test o por collection),
    no añadimos stubs — el módulo real ya está en sys.modules y reusarlo
    es correcto. Solo stubeamos las deps si NO están cargadas todavía."""
    _stub_targets = (
        "db", "db_inventory", "db_profiles", "db_plans", "db_facts",
        "memory_manager", "vision_agent", "fact_extractor", "cpu_tasks",
        "graph_orchestrator", "ai_helpers",
    )
    for mod_name in _stub_targets:
        if mod_name not in sys.modules:
            monkeypatch.setitem(sys.modules, mod_name, MagicMock())

    try:
        from agent import execute_tools
    except Exception as e:
        pytest.skip(f"No se pudo importar agent.execute_tools (deps): {e}")
    return execute_tools


def _make_tool_call(name: str, args: dict, call_id: str = "tc_1") -> dict:
    return {"name": name, "args": dict(args), "id": call_id}


def _make_state(user_id: str, session_id: str, tool_calls: list[dict]) -> dict:
    """Construye un state mínimo compatible con `execute_tools`."""
    last_msg = MagicMock()
    last_msg.tool_calls = tool_calls
    return {
        "messages": [last_msg],
        "user_id": user_id,
        "session_id": session_id,
        "form_data": {},
        "current_plan": {},
        "updated_fields": {},
        "new_plan": None,
        "sys_prompt": "",
    }


def test_functional_else_branch_overrides_llm_user_id(execute_tools_callable, monkeypatch):
    """Smoke test del path crítico: tool en el `else: t.invoke(tool_args)`
    branch (e.g. `log_consumed_meal`) recibe `user_id` autenticado, no el
    que la LLM inyectó."""
    import agent as _agent

    trusted_uid = "AAAAAAAA-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    hostile_uid = "ZZZZZZZZ-zzzz-zzzz-zzzz-zzzzzzzzzzzz"

    captured = {}

    class _FakeTool:
        name = "log_consumed_meal"

        def invoke(self, tool_args):
            # Capturamos el tool_args que el invoker recibió.
            captured["args"] = dict(tool_args)
            return "ok"

    monkeypatch.setattr(_agent, "agent_tools", [_FakeTool()])

    tool_calls = [
        _make_tool_call(
            "log_consumed_meal",
            {
                "user_id": hostile_uid,  # LLM inyecta uid ajeno
                "meal_name": "Mangu",
                "calories": 400,
                "protein": 12,
            },
        )
    ]
    state = _make_state(trusted_uid, "session_x", tool_calls)
    execute_tools_callable(state)

    assert "args" in captured, (
        "P0-AGENT-1 regresión funcional: la tool nunca fue invocada. "
        "El override puede haber excepcionado antes del invoke."
    )
    assert captured["args"]["user_id"] == trusted_uid, (
        f"P0-AGENT-1 regresión funcional CRÍTICA: la tool recibió "
        f"`user_id={captured['args'].get('user_id')!r}` (LLM-supplied) en "
        f"lugar del trusted `{trusted_uid!r}`. Esto reabre el IDOR — el "
        f"override en `execute_tools` no se está aplicando o se está "
        f"aplicando DESPUÉS del invoke."
    )


def test_functional_guest_falls_back_to_session_id(execute_tools_callable, monkeypatch):
    """Cuando el state.user_id es 'guest' o vacío, el trusted debe ser el
    session_id (mismo patrón que los 2 branches inline pre-existentes)."""
    import agent as _agent

    captured = {}

    class _FakeTool:
        name = "check_current_pantry"

        def invoke(self, tool_args):
            captured["args"] = dict(tool_args)
            return "ok"

    monkeypatch.setattr(_agent, "agent_tools", [_FakeTool()])

    session_id = "guest_session_42"
    tool_calls = [
        _make_tool_call(
            "check_current_pantry",
            {"user_id": "ZZZZZZZZ-zzzz-zzzz-zzzz-zzzzzzzzzzzz"},
        )
    ]
    state = _make_state("guest", session_id, tool_calls)
    execute_tools_callable(state)

    assert captured["args"]["user_id"] == session_id, (
        f"P0-AGENT-1 regresión funcional: con user_id='guest', el override "
        f"debió usar session_id={session_id!r}, pero la tool recibió "
        f"{captured['args'].get('user_id')!r}. Mismo bug que en producción "
        f"si el cliente guest manda prompt injection."
    )
