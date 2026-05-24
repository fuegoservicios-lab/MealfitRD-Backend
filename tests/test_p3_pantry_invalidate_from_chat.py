"""[P3-PANTRY-INVALIDATE-FROM-CHAT · 2026-05-22] Tests del flag de
invalidación de Pantry que el chat agent emite tras ejecutar tools que
mutan `user_inventory` (`modify_pantry_inventory` o `log_consumed_meal`
con ingredients).

Contexto del bug verificado 2026-05-22 04:25-04:26: el chat agent eliminó
"Leche" de `user_inventory` correctamente (verificado en BD prod), pero
el frontend mostraba "Leche · Cartón · 1" en `/dashboard/pantry` porque
(a) cache localStorage TTL=10min pisó el primer paint al navegar de
`/agent` a `/pantry`, (b) Realtime channel de Supabase no entrega
mientras Pantry NO está montado.

Flujo del fix:
  1. `agent.py::execute_tools` set `state["pantry_modified_at"]=epoch_ms`
     cuando tool_name ∈ {modify_pantry_inventory,
     log_consumed_meal(con ingredients)}.
  2. `agent.py::chat_with_agent_stream` propaga al evento SSE `done`.
  3. `AgentPage.jsx` lee `dataObj.pantry_modified_at` y escribe la key
     `localStorage.mealfit_pantry_dirty_at`.
  4. `Pantry.jsx` consume la key al mount + storage event listener →
     `invalidateInventoryCache()` + `fetchData()`.

Cross-link convention (P2-HIST-AUDIT-14): slug
`p3_pantry_invalidate_from_chat` matchea este archivo.

Tooltip-anchor: P3-PANTRY-INVALIDATE-FROM-CHAT (vive en agent.py +
AgentPage.jsx + Pantry.jsx).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_ROOT.parent
_AGENT_PY = _BACKEND_ROOT / "agent.py"
_AGENT_PAGE_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "AgentPage.jsx"
_PANTRY_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Pantry.jsx"


@pytest.fixture(scope="module")
def agent_src() -> str:
    return _AGENT_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def agent_page_src() -> str:
    return _AGENT_PAGE_JSX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def pantry_src() -> str:
    return _PANTRY_JSX.read_text(encoding="utf-8")


# ===========================================================================
# Sección 1 — backend: ChatState field + execute_tools + SSE emit
# ===========================================================================

def test_chatstate_has_pantry_modified_at_field(agent_src: str):
    """`ChatState` debe declarar `pantry_modified_at`. Sin el field el
    LangGraph reducer descarta la mutación entre nodos."""
    state_re = re.compile(
        r"class\s+ChatState\s*\([^)]*\)\s*:\s*(.*?)(?=^def\s|^class\s)",
        re.DOTALL | re.MULTILINE,
    )
    m = state_re.search(agent_src)
    assert m is not None, "Clase ChatState no encontrada — refactor inesperado."
    body = m.group(1)
    assert "pantry_modified_at" in body, (
        "P3-PANTRY-INVALIDATE-FROM-CHAT regresión: field "
        "`pantry_modified_at` removido de `ChatState`. Sin él, el LangGraph "
        "reducer descarta la mutación de `execute_tools` entre nodos."
    )


def test_execute_tools_sets_timestamp_on_modify_pantry(agent_src: str):
    """`execute_tools` debe set `pantry_modified_at` cuando tool_name es
    `modify_pantry_inventory` O `log_consumed_meal` con ingredients."""
    fn_re = re.compile(
        r"def\s+execute_tools\s*\(.*?(?=\ndef\s|\nclass\s)",
        re.DOTALL,
    )
    m = fn_re.search(agent_src)
    assert m is not None
    body = m.group(0)
    assert '"modify_pantry_inventory"' in body, (
        "P3-PANTRY-INVALIDATE-FROM-CHAT regresión: literal "
        "`modify_pantry_inventory` removido del check de mutación pantry."
    )
    assert '"log_consumed_meal"' in body, (
        "P3-PANTRY-INVALIDATE-FROM-CHAT regresión: literal "
        "`log_consumed_meal` removido del check de mutación pantry."
    )
    # El branch debe gate `log_consumed_meal` por presencia de ingredients —
    # sin ingredients la tool no llama deduct_consumed_meal_from_inventory.
    assert "ingredients" in body, (
        "P3-PANTRY-INVALIDATE-FROM-CHAT regresión: `log_consumed_meal` debe "
        "gatear el flag por presencia de `ingredients` — sin esa lista la "
        "tool no toca pantry."
    )
    # Algún cómputo de timestamp (time.time() o equivalente).
    assert re.search(r"time\.time\(\)|datetime\.now", body), (
        "P3-PANTRY-INVALIDATE-FROM-CHAT regresión: no se computa timestamp. "
        "El field debe ser epoch ms para que el frontend pueda comparar."
    )


def test_execute_tools_returns_pantry_modified_at(agent_src: str):
    """El return dict de `execute_tools` debe incluir el field."""
    fn_re = re.compile(
        r"def\s+execute_tools\s*\(.*?(?=\ndef\s|\nclass\s)",
        re.DOTALL,
    )
    m = fn_re.search(agent_src)
    assert m is not None
    body = m.group(0)
    # Patron `"pantry_modified_at": pantry_modified_at` o variant.
    assert re.search(
        r'"pantry_modified_at"\s*:\s*pantry_modified_at',
        body,
    ), (
        "P3-PANTRY-INVALIDATE-FROM-CHAT regresión: return dict de "
        "`execute_tools` no incluye `pantry_modified_at`. LangGraph reducer "
        "no propaga al state si no está en el return."
    )


def test_sse_done_event_emits_pantry_modified_at(agent_src: str):
    """El SSE event `done` debe incluir `pantry_modified_at` en el JSON.
    Sin esto el frontend no recibe la señal."""
    done_yield = re.search(
        r"yield\s+f?\"data:\s*\{json\.dumps\(\{[^}]*'type':\s*'done'[^}]*\}\)",
        agent_src,
    )
    assert done_yield is not None, "Yield del evento `done` no encontrado."
    snippet = done_yield.group(0)
    assert "pantry_modified_at" in snippet, (
        "P3-PANTRY-INVALIDATE-FROM-CHAT regresión: SSE event `done` no "
        "incluye `pantry_modified_at`. El frontend no recibirá la señal."
    )


# ===========================================================================
# Sección 2 — frontend AgentPage: SSE handler escribe la key localStorage
# ===========================================================================

def test_agent_page_reads_pantry_modified_at(agent_page_src: str):
    """`AgentPage.jsx` debe leer `dataObj.pantry_modified_at` en el handler
    del SSE `done`."""
    assert "pantry_modified_at" in agent_page_src, (
        "P3-PANTRY-INVALIDATE-FROM-CHAT regresión: `AgentPage.jsx` no "
        "consume el field del SSE — Pantry nunca recibe la señal."
    )


def test_agent_page_writes_localstorage_key(agent_page_src: str):
    """`AgentPage.jsx` debe `setItem('mealfit_pantry_dirty_at', ...)` cuando
    el field viene populated."""
    assert (
        "'mealfit_pantry_dirty_at'" in agent_page_src
        or '"mealfit_pantry_dirty_at"' in agent_page_src
    ), (
        "P3-PANTRY-INVALIDATE-FROM-CHAT regresión: key "
        "`mealfit_pantry_dirty_at` no se escribe en `AgentPage.jsx`. "
        "Pantry.jsx no podrá invalidar su cache."
    )
    assert ".setItem(" in agent_page_src, (
        "P3-PANTRY-INVALIDATE-FROM-CHAT regresión: no hay `setItem` en "
        "AgentPage.jsx para persistir el dirty mark."
    )


def test_agent_page_setItem_call_is_guarded_against_quota(agent_page_src: str):
    """El setItem debe estar en un try/catch para no fallar bajo iOS Private
    Mode / quota exceeded — el chat NO debe morir por una key auxiliar."""
    # Encontrar la región alrededor de 'mealfit_pantry_dirty_at'.
    idx = agent_page_src.find("mealfit_pantry_dirty_at")
    assert idx >= 0
    # Mirar 600 chars hacia adelante/atrás buscando try/catch wrap.
    window = agent_page_src[max(0, idx - 400):idx + 400]
    assert "try" in window and "catch" in window, (
        "P3-PANTRY-INVALIDATE-FROM-CHAT regresión: setItem de "
        "`mealfit_pantry_dirty_at` no está en try/catch. iOS Private "
        "Mode / quota exceeded matarían el handler del SSE — sólo era "
        "una key auxiliar."
    )


# ===========================================================================
# Sección 3 — frontend Pantry.jsx consume el flag
# ===========================================================================

def test_pantry_imports_invalidate_cache(pantry_src: str):
    """`Pantry.jsx` debe importar `invalidateInventoryCache` desde pantryCache.
    Sin esto el flag no puede limpiar el cache."""
    assert "invalidateInventoryCache" in pantry_src, (
        "P3-PANTRY-INVALIDATE-FROM-CHAT regresión: import "
        "`invalidateInventoryCache` removido de `Pantry.jsx`. Sin él, leer "
        "la key del dirty no logra invalidar el snapshot stale."
    )


def test_pantry_reads_dirty_key_at_mount(pantry_src: str):
    """Pantry.jsx debe consumir la key `mealfit_pantry_dirty_at` al mount."""
    assert "mealfit_pantry_dirty_at" in pantry_src, (
        "P3-PANTRY-INVALIDATE-FROM-CHAT regresión: `Pantry.jsx` no lee la "
        "key `mealfit_pantry_dirty_at`. El flag escrito por AgentPage "
        "queda huérfano."
    )
    # Una llamada explícita a invalidateInventoryCache() o equivalente.
    assert "invalidateInventoryCache()" in pantry_src, (
        "P3-PANTRY-INVALIDATE-FROM-CHAT regresión: `Pantry.jsx` no invoca "
        "`invalidateInventoryCache()`. La detección del flag sin invalidar "
        "no fuerza el refetch — UX queda stale."
    )


def test_pantry_consumes_dirty_one_shot(pantry_src: str):
    """Tras leer el dirty flag, `Pantry.jsx` debe `removeItem` la key — sin
    esto el cache se invalidaría perpetuamente en cada mount, anulando
    el cache TTL=10min y golpeando la BD innecesariamente."""
    # removeItem('mealfit_pantry_dirty_at') o similar.
    assert re.search(
        r"removeItem\(\s*['\"]mealfit_pantry_dirty_at['\"]\s*\)",
        pantry_src,
    ), (
        "P3-PANTRY-INVALIDATE-FROM-CHAT regresión: `Pantry.jsx` no hace "
        "`removeItem('mealfit_pantry_dirty_at')` tras consumir el flag. "
        "Sin one-shot, cada mount invalidaría el cache aunque ya hayamos "
        "leído fresh data."
    )


def test_pantry_listens_storage_event_for_dirty_flag(pantry_src: str):
    """Pantry.jsx debe agregar `storage` listener para detectar cross-tab
    writes del flag. Sin esto, un usuario con Pantry abierto en un tab
    y chateando en otro NO ve los cambios hasta hacer F5."""
    # Heurística: hay alguna addEventListener('storage', ...) Y el handler
    # menciona 'mealfit_pantry_dirty_at' cerca.
    storage_re = re.compile(
        r"addEventListener\(\s*['\"]storage['\"]",
    )
    assert storage_re.search(pantry_src), (
        "P3-PANTRY-INVALIDATE-FROM-CHAT regresión: `Pantry.jsx` no agrega "
        "listener `storage`. Multi-tab use case (Pantry abierta en tab A "
        "+ chat agente en tab B) queda stale hasta F5."
    )
