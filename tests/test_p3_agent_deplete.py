"""[P3-AGENT-DEPLETE · 2026-05-22] Tests del flujo "se me acabó X" en el
chat agent: en lugar de eliminar definitivamente del inventario, marca como
agotado (visible en sección "Agotados" de Pantry, listo para re-comprar).

Flujo:
  1. `tools.modify_pantry_inventory(items_to_deplete=[...])`: snapshot la
     cantidad actual del inventario, DELETE la fila, retorna info en marker
     inline `<<PANTRY_DEPLETED_JSON: [...]>>`.
  2. `agent.py::execute_tools`: extrae el marker, propaga al state field
     `pantry_depleted_items`, strip-ea del ToolMessage para que la LLM NO
     vea JSON raw.
  3. SSE `done` event incluye `pantry_depleted_items: list | null`.
  4. `AgentPage.jsx`: merge a `localStorage.mealfit_depleted_items` con
     dedupe por master_ingredient_id (fallback ingredient_name normalizado).
  5. `Pantry.jsx`: storage event listener + lazy useState init ya hidratan
     `depletedItems` desde la key — el merge propaga automáticamente.

Distinción semántica:
  - `items_to_deplete` ("se me acabó", "ya no tengo X") → flujo nuevo.
  - `items_to_remove` ("se me dañó", "bota X") → legacy DELETE sin agotar.

Cross-link convention (P2-HIST-AUDIT-14): slug `p3_agent_deplete` matchea
este archivo.

Tooltip-anchor: P3-AGENT-DEPLETE.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_ROOT.parent
_TOOLS_PY = _BACKEND_ROOT / "tools.py"
_AGENT_PY = _BACKEND_ROOT / "agent.py"
_AGENT_PAGE_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "AgentPage.jsx"


@pytest.fixture(scope="module")
def tools_src() -> str:
    return _TOOLS_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def agent_src() -> str:
    return _AGENT_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def agent_page_src() -> str:
    return _AGENT_PAGE_JSX.read_text(encoding="utf-8")


# ===========================================================================
# Sección 1 — tools.modify_pantry_inventory acepta items_to_deplete
# ===========================================================================

def test_tool_signature_accepts_items_to_deplete(tools_src: str):
    """La signature de `modify_pantry_inventory` debe incluir parámetro
    `items_to_deplete`. Sin este param la LLM no puede invocar el flujo
    de agotado."""
    sig_re = re.compile(
        r"def\s+modify_pantry_inventory\s*\([^)]*items_to_deplete",
        re.DOTALL,
    )
    assert sig_re.search(tools_src), (
        "P3-AGENT-DEPLETE regresión: signature de `modify_pantry_inventory` "
        "no acepta `items_to_deplete`. Sin él la LLM no tiene cómo señalar "
        "el flujo de agotado (vs delete definitivo)."
    )


def test_docstring_explains_three_paths(tools_src: str):
    """La docstring debe explicar los 3 paths (add/deplete/remove) para que
    la LLM aprenda a usar el correcto según el lenguaje del usuario."""
    fn_re = re.compile(
        r"def\s+modify_pantry_inventory\s*\(.*?(?=\n@\w|\ndef\s|\Z)",
        re.DOTALL,
    )
    m = fn_re.search(tools_src)
    assert m is not None
    body = m.group(0)
    for keyword in ("items_to_add", "items_to_deplete", "items_to_remove"):
        assert keyword in body, (
            f"P3-AGENT-DEPLETE regresión: docstring no menciona `{keyword}` "
            f"— la LLM no sabrá cuándo usarlo."
        )
    # Heurística de los 3 cues semánticos.
    assert "se me acab" in body.lower() or "agot" in body.lower(), (
        "P3-AGENT-DEPLETE regresión: docstring no entrena a la LLM en el "
        "cue 'se me acabó' / 'agotado' para distinguir deplete de remove."
    )
    assert "se me dañ" in body.lower() or "bot" in body.lower(), (
        "P3-AGENT-DEPLETE regresión: docstring no entrena a la LLM en el "
        "cue 'se dañó' / 'bota' para distinguir remove de deplete."
    )


def test_tool_emits_pantry_depleted_json_marker(tools_src: str):
    """El tool debe inyectar marker inline `<<PANTRY_DEPLETED_JSON: ...>>`
    en el tool_result cuando hay items agotados. Es el canal por el que
    `execute_tools` extrae la info estructurada."""
    fn_re = re.compile(
        r"def\s+modify_pantry_inventory\s*\(.*?(?=\n@tool|\ndef\s|\Z)",
        re.DOTALL,
    )
    m = fn_re.search(tools_src)
    assert m is not None
    body = m.group(0)
    assert "PANTRY_DEPLETED_JSON" in body, (
        "P3-AGENT-DEPLETE regresión: marker `PANTRY_DEPLETED_JSON` removido. "
        "Sin él, execute_tools no puede extraer la info estructurada del "
        "tool_result."
    )


def test_tool_snapshots_quantity_before_delete(tools_src: str):
    """El flujo de deplete debe leer la cantidad actual ANTES de DELETE para
    poder retornarla en el payload (Pantry.jsx la usa como snapshotQty para
    el restock)."""
    fn_re = re.compile(
        r"def\s+modify_pantry_inventory\s*\(.*?(?=\n@tool|\ndef\s|\Z)",
        re.DOTALL,
    )
    m = fn_re.search(tools_src)
    assert m is not None
    body = m.group(0)
    assert "get_raw_user_inventory" in body, (
        "P3-AGENT-DEPLETE regresión: tool no invoca `get_raw_user_inventory` "
        "para snapshot la qty antes del DELETE. Sin snapshot el frontend "
        "queda con `quantity: undefined` y el restock pierde la cantidad "
        "original."
    )


# ===========================================================================
# Sección 2 — agent.py extrae marker + propaga al state
# ===========================================================================

def test_chatstate_has_pantry_depleted_items_field(agent_src: str):
    """`ChatState` debe declarar `pantry_depleted_items` — sin este field el
    LangGraph reducer descarta la propagación entre nodos."""
    state_re = re.compile(
        r"class\s+ChatState\s*\([^)]*\)\s*:\s*(.*?)(?=^def\s|^class\s)",
        re.DOTALL | re.MULTILINE,
    )
    m = state_re.search(agent_src)
    assert m is not None
    body = m.group(1)
    assert "pantry_depleted_items" in body, (
        "P3-AGENT-DEPLETE regresión: field `pantry_depleted_items` removido "
        "de `ChatState`. El LangGraph reducer descarta la mutación de "
        "`execute_tools` entre nodos."
    )


def test_execute_tools_extracts_marker(agent_src: str):
    """`execute_tools` debe extraer el marker `PANTRY_DEPLETED_JSON` del
    tool_result + parsearlo + propagar al state."""
    fn_re = re.compile(
        r"def\s+execute_tools\s*\(.*?(?=\ndef\s|\nclass\s)",
        re.DOTALL,
    )
    m = fn_re.search(agent_src)
    assert m is not None
    body = m.group(0)
    assert "PANTRY_DEPLETED_JSON" in body, (
        "P3-AGENT-DEPLETE regresión: `execute_tools` no busca el marker "
        "`PANTRY_DEPLETED_JSON`. La info estructurada queda atrapada en el "
        "tool_result string."
    )
    assert "pantry_depleted_items.extend" in body or "pantry_depleted_items.append" in body, (
        "P3-AGENT-DEPLETE regresión: `execute_tools` no acumula los items "
        "parseados al state. El field termina vacío aunque la tool los emita."
    )


def test_execute_tools_strips_marker_from_tool_result(agent_src: str):
    """Tras extraer la info, el marker debe strip-earse del tool_result
    para que la LLM NO vea el JSON raw en su contexto (sería ruido)."""
    fn_re = re.compile(
        r"def\s+execute_tools\s*\(.*?(?=\ndef\s|\nclass\s)",
        re.DOTALL,
    )
    m = fn_re.search(agent_src)
    assert m is not None
    body = m.group(0)
    # Buscar pattern de strip — re.sub() o replace().
    assert re.search(r"_marker_re\.sub\(|\.replace\(.*PANTRY_DEPLETED_JSON", body), (
        "P3-AGENT-DEPLETE regresión: marker NO se strip-ea del tool_result. "
        "La LLM verá el JSON raw en el ToolMessage — contamina su contexto."
    )


def test_sse_done_emits_pantry_depleted_items(agent_src: str):
    """El SSE `done` debe incluir `pantry_depleted_items` en el JSON."""
    done_yield = re.search(
        r"yield\s+f?\"data:\s*\{json\.dumps\(\{[^}]*'type':\s*'done'[^}]*\}\)",
        agent_src,
    )
    assert done_yield is not None
    assert "pantry_depleted_items" in done_yield.group(0), (
        "P3-AGENT-DEPLETE regresión: SSE event `done` no propaga "
        "`pantry_depleted_items`. El frontend no recibe la info."
    )


# ===========================================================================
# Sección 3 — frontend AgentPage merge a localStorage
# ===========================================================================

def test_agent_page_consumes_pantry_depleted_items(agent_page_src: str):
    """`AgentPage.jsx` debe consumir `dataObj.pantry_depleted_items` en el
    handler del SSE `done`."""
    assert "pantry_depleted_items" in agent_page_src, (
        "P3-AGENT-DEPLETE regresión: `AgentPage.jsx` no consume el field "
        "del SSE — los items agotados no llegan al localStorage."
    )


def test_agent_page_merges_into_depleted_localstorage(agent_page_src: str):
    """`AgentPage.jsx` debe hacer setItem a `mealfit_depleted_items` (no
    pisar, debe mergear). Sin esto el feature no aparece visualmente."""
    assert "'mealfit_depleted_items'" in agent_page_src or '"mealfit_depleted_items"' in agent_page_src, (
        "P3-AGENT-DEPLETE regresión: key `mealfit_depleted_items` no se "
        "escribe en `AgentPage.jsx`. Pantry.jsx no lo verá."
    )


def test_agent_page_deduplicates_by_key(agent_page_src: str):
    """El merge debe deduplicar por `master_ingredient_id` o `ingredient_name`
    para no duplicar entries si el user agota el mismo item dos veces."""
    # Buscar dedupe pattern — set/filter por keyOf.
    idx = agent_page_src.find("pantry_depleted_items")
    if idx >= 0:
        window = agent_page_src[idx:idx + 2000]
        assert "master_ingredient_id" in window or "keyOf" in window or "Set(" in window, (
            "P3-AGENT-DEPLETE regresión: merge no deduplica por "
            "master_ingredient_id. Restocks repetidos del mismo item "
            "crean entries duplicadas."
        )
