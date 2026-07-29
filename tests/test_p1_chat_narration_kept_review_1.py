"""[P1-CHAT-NARRATION-KEPT-REVIEW-1 · 2026-07-28] Cierra un hallazgo de review
adversarial sobre P1-CHAT-NARRATION-KEPT (2026-07-28):

  "reconcileFinalChatText's 'extend' branch never actually fires for the
  narrate-then-act scenario it was built for, because the backend inserts a
  '\\n\\n' separator between AI passes only in the persisted payload, not in
  the live SSE token stream."

Causa raíz confirmada: `_build_final_content_from_messages` (agent.py) une
las AIMessage de un turno narrate-then-act con `"\\n\\n".join(parts)`, pero
el loop de streaming (`chat_with_agent_stream`) solo emitía `type: 'chunk'`
para el CONTENIDO crudo — nunca un separador. El evento `type: 'progress'`
que sigue a la narración de la 1ra pasada (justo antes de ejecutar la
tool_call) es la ÚNICA señal disponible en el cliente de que una NUEVA
pasada está por comenzar, pero SOLO 6 de ~15 tool_calls posibles emitían un
`progress` (generate_new_plan_from_chat, modify_single_meal,
update_form_field, log_consumed_meal, check_shopping_list,
search_deep_memory) — el resto (check_current_pantry,
modify_pantry_inventory, mark_shopping_list_purchased,
check_hydration_today, log_water_glass, suggest_foods_for_nutrient,
check_clinical_profile, consultar_dia_del_plan, regenerate_full_day, y
cualquier tool futura) no emitía NINGÚN evento en el boundary — el cliente
no tenía forma de saber que una segunda pasada empezaba.

Fix: fallback `else` genérico que garantiza un `type: 'progress'` para
CUALQUIER tool_call, sea cual sea su nombre. El frontend
(AgentPage.jsx/ChatWidget.jsx) usa ese evento para insertar el mismo
separador '\\n\\n' en `fullText` acumulado en vivo — así el texto que el
usuario YA ve coincide byte a byte con `done.response`, y la rama 'extend'
de `reconcileFinalChatText` (chatStreamReconcile.js) realmente se activa
para tráfico real, en vez de caer siempre a 'replace' (reflow visible).

tooltip-anchor: P1-CHAT-NARRATION-KEPT-REVIEW-1
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_AGENT_PY = _BACKEND_ROOT / "agent.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ===========================================================================
# 1. El fallback genérico existe y vive en el sitio correcto: el bloque
#    `for idx, tool_call in enumerate(msg_chunk.tool_calls):` de
#    `chat_with_agent_stream`.
# ===========================================================================

def test_generic_progress_fallback_present_after_named_branches():
    src = _read(_AGENT_PY)
    idx = src.index("for idx, tool_call in enumerate(msg_chunk.tool_calls):")
    # Región que cubre las 6 ramas nombradas + el fallback nuevo (holgada
    # para acomodar el comentario explicativo del fallback).
    region = src[idx: idx + 5000]

    named_branches = [
        'tool_name == "generate_new_plan_from_chat"',
        'tool_name == "modify_single_meal"',
        'tool_name == "update_form_field"',
        'tool_name == "log_consumed_meal"',
        'tool_name == "check_shopping_list"',
        'tool_name == "search_deep_memory"',
    ]
    for branch in named_branches:
        assert branch in region, f"rama nombrada esperada ausente: {branch}"

    # El fallback debe venir DESPUÉS de la última rama nombrada, como
    # `else:` (no otro `elif tool_name == ...`, que seguiría dejando
    # tools sin cubrir).
    last_named_idx = region.rindex('tool_name == "search_deep_memory"')
    tail = region[last_named_idx:]
    assert re.search(r"\belse\s*:\s*\n", tail), (
        "esperaba un `else:` catch-all después de la última rama nombrada "
        "— sin él, tool_calls no-nombradas (check_current_pantry, "
        "modify_pantry_inventory, log_water_glass, etc.) no emiten ningún "
        "`progress`, y el frontend no puede detectar el boundary "
        "narrate-then-act para insertar el separador."
    )
    # El fallback debe emitir un evento `progress` real (no un no-op / pass).
    else_region = tail[tail.index("else"):]
    assert "'type': 'progress'" in else_region, (
        "el `else:` catch-all debe yieldear un evento SSE `type: 'progress'` "
        "— sin emitir nada, el boundary sigue siendo invisible para el "
        "cliente."
    )


# ===========================================================================
# 2. Cobertura de las ~15 tools reales: cada una está o bien en la lista de
#    ramas nombradas, o cae al `else` — nunca queda un tool_call SIN NINGÚN
#    evento en el boundary (el modo de fallo original).
# ===========================================================================

def test_every_known_tool_name_reaches_some_progress_branch():
    src = _read(_AGENT_PY)
    idx = src.index("for idx, tool_call in enumerate(msg_chunk.tool_calls):")
    region = src[idx: idx + 3500]

    named_tool_names = set(re.findall(r'tool_name == "([^"]+)"', region))

    # Tools de `agent_tools` (tools.py) + `_PLAN_MUTATION_TOOLS`. Lista
    # replicada acá a propósito (no importa `tools.py` para no acoplar este
    # test a imports pesados/side-effects de ese módulo) — si `agent_tools`
    # gana una tool nueva, este test NO se rompe silenciosamente: como el
    # fallback es genérico (`else:`), CUALQUIER nombre no listado en
    # `named_tool_names` de todos modos cae al catch-all y sigue cubierto.
    known_tools = {
        "update_form_field", "log_consumed_meal", "search_deep_memory",
        "check_shopping_list", "check_current_pantry",
        "modify_pantry_inventory", "mark_shopping_list_purchased",
        "check_hydration_today", "log_water_glass",
        "suggest_foods_for_nutrient", "check_clinical_profile",
        "consultar_dia_del_plan", "generate_new_plan_from_chat",
        "modify_single_meal", "regenerate_full_day",
    }
    unnamed = known_tools - named_tool_names
    # El punto central del fix: la MAYORÍA de las tools NO tienen bucket
    # dedicado — deben depender del catch-all. Si esta lista quedara vacía
    # (todas nombradas), el `else:` seguiría siendo defensivo pero el test
    # documentaría mal el problema original — falla loud para que alguien
    # revise el fixture si algún día se nombran todas explícitamente.
    assert unnamed, (
        "esperaba al menos una tool SIN bucket de progress dedicado "
        "(dependiendo del catch-all) — si esto está vacío, actualiza el "
        "docstring/fixture de este test."
    )
    assert "else" in region, "sin `else:` catch-all, las tools en " \
        f"{sorted(unnamed)} no emitirían ningún progress event."
