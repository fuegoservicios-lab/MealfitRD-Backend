"""[P1-CHAT-UI-ACTION-INVENTORY · 2026-05-20] Test anti-regresión del tag
silente `[UI_ACTION: REFRESH_INVENTORY]` en el chat.

Bug observado en runtime:
    El system prompt del agente (prompts/chat_agent.py:128) instruye al LLM
    a emitir `[UI_ACTION: REFRESH_INVENTORY]` tras `log_consumed_meal`,
    `modify_pantry_inventory` o `mark_shopping_list_purchased`. El tag
    debe ser **silente** — el frontend lo detecta, ejecuta el refresh,
    y lo remueve del texto antes de renderizar.

    Pre-fix: `AgentPage.jsx` solo manejaba `REFRESH_PLAN` y `REFRESH_HYDRATION`
    — NO `REFRESH_INVENTORY`. Resultado: el tag se renderizaba tal cual al
    user en el chat. Reportado 2026-05-20 con el mensaje "me acabo de comer
    una taza de avena con nueces" → respuesta del agente contenía
    `[UI_ACTION: REFRESH_INVENTORY]` visible.

Fix:
    - `AgentPage.jsx`: 2 callsites (streaming + post-stream done) ahora
      detectan, strip, y dispatch `mealfit:refresh-inventory` custom event.
    - `TrackingProgress.jsx`: listener del custom event que llama a
      `fetchConsumed()` (refresca card "Progreso en Tiempo Real" sin
      esperar al polling de 15s).
    - `Dashboard.jsx`: listener del custom event que llama a
      `refreshInventoryOnFocus()` (refresca el `liveInventory` para la
      Nevera live).

Tests parser-based — anchor literal en cada archivo. NOTA: estos tests
escanean frontend desde tests/ del backend (mismo patrón que
`test_p1_new_a_frontend_no_direct_meal_plans_write.py`).
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_AGENT_PAGE_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "AgentPage.jsx"
_TRACKING_PROGRESS_JSX = _REPO_ROOT / "frontend" / "src" / "components" / "dashboard" / "TrackingProgress.jsx"
_DASHBOARD_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx"
_AGENT_PY = _REPO_ROOT / "backend" / "agent.py"
_CHAT_ROUTER_PY = _REPO_ROOT / "backend" / "routers" / "chat.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ============================================================
# AgentPage.jsx: strip + dispatch en streaming + post-stream
# ============================================================

def test_agent_page_strips_refresh_inventory_in_streaming():
    """[P1-CHAT-UI-ACTION-INVENTORY] El callsite del streaming chunk debe
    detectar `[UI_ACTION: REFRESH_INVENTORY]`, removerlo del texto, y
    dispatch `mealfit:refresh-inventory` custom event."""
    src = _read(_AGENT_PAGE_JSX)
    # El handler de REFRESH_INVENTORY DEBE aparecer >=2 veces (streaming + done).
    inventory_strip = re.findall(
        r"\[UI_ACTION:\s*REFRESH_INVENTORY\]",
        src,
    )
    # 2 ocurrencias mínimas: el `includes('[UI_ACTION: REFRESH_INVENTORY]')` x2
    # + el `replace(/\[UI_ACTION:\s*REFRESH_INVENTORY\]/g, '')` x2 = 4.
    # Si solo aparece 1 vez, falta uno de los callsites.
    assert len(inventory_strip) >= 2, (
        f"Solo {len(inventory_strip)} ocurrencias de `[UI_ACTION: REFRESH_INVENTORY]` "
        f"en AgentPage.jsx. Esperaba >=2 (streaming chunk + post-stream done). "
        f"Sin esto, el tag se renderiza tal cual al user. Ver "
        f"P1-CHAT-UI-ACTION-INVENTORY · 2026-05-20."
    )


def test_agent_page_dispatches_refresh_inventory_event():
    """[P1-CHAT-UI-ACTION-INVENTORY] El dispatch del custom event
    `mealfit:refresh-inventory` debe aparecer en los 2 callsites
    (streaming + post-stream)."""
    src = _read(_AGENT_PAGE_JSX)
    dispatches = re.findall(
        r"CustomEvent\(\s*['\"]mealfit:refresh-inventory['\"]",
        src,
    )
    assert len(dispatches) >= 2, (
        f"Solo {len(dispatches)} dispatch(es) de `mealfit:refresh-inventory` "
        f"encontrados — esperaba >=2 (streaming + post-stream). Sin el dispatch, "
        f"el strip funciona pero el card Progreso no se refresca instantáneo "
        f"(tarda hasta 15s del poll). Ver P1-CHAT-UI-ACTION-INVENTORY."
    )


# ============================================================
# TrackingProgress.jsx: listener del custom event
# ============================================================

def test_tracking_progress_listens_refresh_inventory():
    """[P1-CHAT-UI-ACTION-INVENTORY] El card `TrackingProgress` (Progreso
    en Tiempo Real) debe escuchar `mealfit:refresh-inventory` y refetchear
    `consumed_meals` instantáneamente."""
    src = _read(_TRACKING_PROGRESS_JSX)
    assert "mealfit:refresh-inventory" in src, (
        "TrackingProgress.jsx NO escucha `mealfit:refresh-inventory`. Sin "
        "esto, el card tarda hasta 15s (next poll de setInterval) en "
        "reflejar la comida recién registrada. Ver P1-CHAT-UI-ACTION-INVENTORY."
    )
    # Sanity: addEventListener + removeEventListener pareados.
    add = "addEventListener('mealfit:refresh-inventory'" in src or \
          'addEventListener("mealfit:refresh-inventory"' in src
    remove = "removeEventListener('mealfit:refresh-inventory'" in src or \
             'removeEventListener("mealfit:refresh-inventory"' in src
    assert add, "addEventListener para mealfit:refresh-inventory ausente."
    assert remove, "removeEventListener para mealfit:refresh-inventory ausente — memory leak."


# ============================================================
# Dashboard.jsx: listener del custom event
# ============================================================

def test_dashboard_listens_refresh_inventory():
    """[P1-CHAT-UI-ACTION-INVENTORY] El Dashboard debe escuchar
    `mealfit:refresh-inventory` y refetchear `liveInventory`."""
    src = _read(_DASHBOARD_JSX)
    assert "mealfit:refresh-inventory" in src, (
        "Dashboard.jsx NO escucha `mealfit:refresh-inventory`. Sin esto, "
        "el `liveInventory` que alimenta la Nevera live queda stale tras "
        "log_consumed_meal/modify_pantry_inventory. Ver P1-CHAT-UI-ACTION-INVENTORY."
    )
    add = "addEventListener('mealfit:refresh-inventory'" in src or \
          'addEventListener("mealfit:refresh-inventory"' in src
    remove = "removeEventListener('mealfit:refresh-inventory'" in src or \
             'removeEventListener("mealfit:refresh-inventory"' in src
    assert add and remove, (
        "addEventListener + removeEventListener para mealfit:refresh-inventory "
        "deben estar pareados en Dashboard.jsx."
    )


# ============================================================
# Server-side strip (cierre del bug "desapareció y volvió a aparecer")
# ============================================================

def test_strip_helper_exists_in_agent():
    """[P1-CHAT-UI-ACTION-INVENTORY] Helper `strip_ui_action_tags_for_persist`
    debe existir en `agent.py` como SSOT del strip server-side. Importable
    desde `routers/chat.py` y cualquier otro callsite futuro."""
    src = _read(_AGENT_PY)
    assert "def strip_ui_action_tags_for_persist(" in src, (
        "Helper `strip_ui_action_tags_for_persist` ausente de agent.py. Sin "
        "él, el tag persiste en agent_messages.content y reaparece al "
        "refetch del frontend. Ver P1-CHAT-UI-ACTION-INVENTORY · 2026-05-20."
    )
    # Sanity: la regex que matchea los 3+ UI_ACTION conocidos.
    assert "_UI_ACTION_TAG_RE" in src, (
        "Pattern `_UI_ACTION_TAG_RE` ausente — el helper no opera con regex "
        "consistente."
    )
    # Sanity: regex cubre los 3 actions documentados.
    helper_match = re.search(
        r"_UI_ACTION_TAG_RE\s*=\s*re\.compile\(([^)]+)\)",
        src,
    )
    assert helper_match
    pattern = helper_match.group(1)
    assert "UI_ACTION" in pattern, "Pattern no menciona UI_ACTION literal"


def test_chat_router_strips_ui_actions_before_save():
    """[P1-CHAT-UI-ACTION-INVENTORY] Los 2 callsites de `save_message(...,
    'model', response_text, ...)` en `routers/chat.py` (stream + non-stream)
    DEBEN aplicar `strip_ui_action_tags_for_persist(response_text)` ANTES.
    Sin esto, el tag queda en agent_messages.content y reaparece al
    refetch del frontend."""
    src = _read(_CHAT_ROUTER_PY)
    # Import del helper.
    assert "strip_ui_action_tags_for_persist" in src, (
        "Import de `strip_ui_action_tags_for_persist` ausente de routers/chat.py. "
        "Sin él, los callsites no pueden aplicar el strip."
    )
    # Conteo: el helper debe llamarse >=2 veces (1 por endpoint /stream + 1
    # por /). El import no termina en `(`, así que NO se cuenta — la regex
    # `\s*\(` solo matchea callsites reales.
    calls = re.findall(r"strip_ui_action_tags_for_persist\s*\(", src)
    assert len(calls) >= 2, (
        f"`strip_ui_action_tags_for_persist(...)` invocado solo "
        f"{len(calls)} vez(ces) en routers/chat.py — esperaba >=2 (uno por "
        f"endpoint /stream y otro por /). Ver "
        f"P1-CHAT-UI-ACTION-INVENTORY · 2026-05-20."
    )
