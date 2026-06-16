"""[P2-AUDIT-NEW-1 · 2026-05-12] Frontend consume `_coherence_warnings`
del backend en 4 surfaces + backend agent propaga al SSE `done`.

Bug original (audit comprehensivo 2026-05-12):
    Backend emitía `_coherence_warnings` en 2 endpoints (top-5 divergencias
    summarized vía `summarize_divergences_for_ui`):

      1. POST /api/plans/recalculate-shopping-list (P2-COHERENCE-1):
         response.success && response.plan_data → response._coherence_warnings.

      2. agent tool `modify_single_meal` (en tools.py): retorna JSON
         con key `_coherence_warnings` que LangGraph propaga vía
         ToolMessage → LLM → final state. PERO el evento SSE `done`
         no incluía el campo, así que el frontend jamás lo recibía.

    `grep _coherence_warnings frontend/src` pre-fix → 0 matches. La
    telemetría útil del guard quedaba en backend sin renderear al usuario.

Fix:
    A) Helper compartido `frontend/src/utils/renderCoherenceWarnings.js`
       con `buildCoherenceToast(warnings)` y `emitCoherenceToast(toast, warnings)`.

    B) 3 consumidores frontend de `response._coherence_warnings`:
       - Pantry.jsx (handleRecalc helper que pega a /recalculate-shopping-list).
       - Dashboard.jsx (groceryDuration option click → recalc).
       - AssessmentContext.jsx (post-swap recalc del swap-meal flow).

    C) Backend agent propaga warnings al SSE `done`:
       - `ChatState.coherence_warnings: list` añadido.
       - `execute_tools` extrae `_coherence_warnings` del tool_result JSON
         de `modify_single_meal` y acumula en el state.
       - `chat_with_agent_stream` yield `done` ahora incluye
         `coherence_warnings` (read del final state snapshot).

    D) AgentPage.jsx consume `dataObj.coherence_warnings` en el branch
       `type=="done"` y emite toast vía `emitCoherenceToast(toast, ...)`.

Lo que este test enforza:
    1. Helper utils existe (file present) y exporta `buildCoherenceToast` +
       `emitCoherenceToast`.
    2. Cada uno de los 4 archivos consumidores (3 frontend + AgentPage.jsx):
       a) Importa `emitCoherenceToast` desde el helper.
       b) Invoca la función con `_coherence_warnings` o `coherence_warnings`.
    3. Backend agent.py:
       a) ChatState declara `coherence_warnings: list`.
       b) execute_tools extrae `_coherence_warnings` del tool_result.
       c) execute_tools retorna `coherence_warnings` en su dict.
       d) chat_with_agent_stream lee `coherence_warnings` del state y lo
          incluye en el yield del evento `done`.

Tooltip-anchor: P2-AUDIT-NEW-1-COHERENCE-CONSUMERS
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_ROOT.parent
_FRONTEND_SRC = _REPO_ROOT / "frontend" / "src"

_HELPER = _FRONTEND_SRC / "utils" / "renderCoherenceWarnings.js"
_PANTRY = _FRONTEND_SRC / "pages" / "Pantry.jsx"
_DASHBOARD = _FRONTEND_SRC / "pages" / "Dashboard.jsx"
_AGENT_PAGE = _FRONTEND_SRC / "pages" / "AgentPage.jsx"
_ASSESSMENT_CTX = _FRONTEND_SRC / "context" / "AssessmentContext.jsx"

_AGENT_PY = _BACKEND_ROOT / "agent.py"


# ---------------------------------------------------------------------------
# 1. Helper utils existe + exports correctos
# ---------------------------------------------------------------------------
def test_helper_file_exists() -> None:
    assert _HELPER.exists(), (
        f"P2-AUDIT-NEW-1 violation: helper {_HELPER} no existe. "
        f"Debe vivir bajo frontend/src/utils/ para que todos los "
        f"consumidores (3 frontend + AgentPage) lo importen del mismo lugar."
    )


def test_helper_exports_buildCoherenceToast() -> None:
    src = _HELPER.read_text(encoding="utf-8")
    assert re.search(
        r"export\s+const\s+buildCoherenceToast\s*=",
        src,
    ), (
        "P2-AUDIT-NEW-1 violation: helper no exporta "
        "`buildCoherenceToast`. Esta es la función pura (sin side "
        "effects) — útil para tests unitarios y para reutilizar la "
        "lógica de severidad/summary sin disparar toasts."
    )


def test_helper_exports_emitCoherenceToast() -> None:
    src = _HELPER.read_text(encoding="utf-8")
    assert re.search(
        r"export\s+const\s+emitCoherenceToast\s*=",
        src,
    ), (
        "P2-AUDIT-NEW-1 violation: helper no exporta "
        "`emitCoherenceToast`. Esta es la función con side effect (llama "
        "toast.warning/toast.info) — la que invocan los 4 consumidores."
    )


def test_helper_uses_existing_label_map() -> None:
    """El helper debe reutilizar `getCoherenceHypothesisLabel` del
    catálogo P1-3 — NO duplicar labels en línea (drift backend↔frontend).
    """
    src = _HELPER.read_text(encoding="utf-8")
    assert "getCoherenceHypothesisLabel" in src, (
        "P2-AUDIT-NEW-1 violation: helper no usa "
        "`getCoherenceHypothesisLabel` de `coherenceLabels.js`. "
        "Duplicar labels inline crea drift con el catálogo SSOT del "
        "Historial (P1-3 enforza paridad backend↔frontend)."
    )


# ---------------------------------------------------------------------------
# 2. Consumidores frontend: 4 archivos
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path,recalc_key", [
    (_PANTRY, "_coherence_warnings"),
    (_DASHBOARD, "_coherence_warnings"),
    (_ASSESSMENT_CTX, "_coherence_warnings"),
    # AgentPage consume `coherence_warnings` (SIN underscore) porque
    # viene del SSE event `done`, donde el backend ya pierde el prefix
    # `_` al serializar como key pública de la response.
    (_AGENT_PAGE, "coherence_warnings"),
])
def test_frontend_file_imports_and_invokes_helper(path: Path, recalc_key: str) -> None:
    """Cada consumidor:
      A) Importa `emitCoherenceToast` desde el helper.
      B) Lo invoca con la key esperada (`response._coherence_warnings`
         para recalc; `dataObj.coherence_warnings` para SSE done).
    """
    assert path.exists(), f"Consumer file {path} no existe."
    src = path.read_text(encoding="utf-8")

    # A) Import del helper
    import_pattern = re.compile(
        r"import\s+\{\s*[^}]*emitCoherenceToast[^}]*\}\s+from\s+['\"][^'\"]*renderCoherenceWarnings['\"]",
    )
    assert import_pattern.search(src), (
        f"P2-AUDIT-NEW-1 violation: {path.name} no importa "
        f"`emitCoherenceToast` desde `renderCoherenceWarnings`. Sin "
        f"este import, el toast no se emite y el usuario nunca ve "
        f"la telemetría del guard."
    )

    # B) Invocación con la key esperada
    # Pattern flexible: `emitCoherenceToast(toast, <obj>.<key>)` o
    # `emitCoherenceToast(toast, <obj>?.<key>)` (optional chaining).
    invocation_pattern = re.compile(
        rf"emitCoherenceToast\s*\(\s*toast\s*,\s*[\w.?\[\]\"']*\b{re.escape(recalc_key)}\b",
    )
    assert invocation_pattern.search(src), (
        f"P2-AUDIT-NEW-1 violation: {path.name} importa "
        f"`emitCoherenceToast` pero no lo invoca con `{recalc_key}`. "
        f"Buscar `emitCoherenceToast(toast, X.{recalc_key})` o "
        f"`emitCoherenceToast(toast, X?.{recalc_key})`."
    )


# ---------------------------------------------------------------------------
# 3. Backend agent.py propaga warnings al SSE done
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def agent_src() -> str:
    return _AGENT_PY.read_text(encoding="utf-8")


def test_chat_state_declares_coherence_warnings(agent_src: str) -> None:
    """A) `ChatState` declara `coherence_warnings: list` para que
    LangGraph propague la key entre nodos.
    """
    # Localizar la class ChatState
    class_match = re.search(r"class\s+ChatState\s*\([^)]+\)\s*:", agent_src)
    assert class_match, "No se encontró `class ChatState(...)` en agent.py."
    class_start = class_match.end()
    # Body: hasta la próxima def top-level o class.
    next_top = re.search(r"\n(?:def|class)\s+\w+", agent_src[class_start:])
    body_end = class_start + (next_top.start() if next_top else len(agent_src) - class_start)
    body = agent_src[class_start:body_end]

    assert re.search(r"coherence_warnings\s*:\s*list", body), (
        "P2-AUDIT-NEW-1 violation: `ChatState` no declara "
        "`coherence_warnings: list`. Sin esto, LangGraph no propaga "
        "la key entre nodos `execute_tools` → final state, y el "
        "stream wrapper no puede leerla."
    )


def test_execute_tools_extracts_coherence_warnings(agent_src: str) -> None:
    """B) `execute_tools` extrae `_coherence_warnings` del tool_result
    JSON (path `parsed_mod.get("_coherence_warnings")`) y los acumula
    en el accumulator local antes de retornar.
    """
    # Localizar def execute_tools
    fn_match = re.search(r"def\s+execute_tools\s*\(", agent_src)
    assert fn_match, "No se encontró `def execute_tools(` en agent.py."
    body_start = fn_match.end()
    # Cuerpo hasta el próximo `def `/`class ` top-level
    next_top = re.search(r"\n(?:def|class|async def)\s+\w+", agent_src[body_start:])
    body_end = body_start + (next_top.start() if next_top else len(agent_src) - body_start)
    body = agent_src[body_start:body_end]

    # Accumulator
    assert re.search(
        r"coherence_warnings\s*=\s*list\s*\(\s*state\.get\(\s*[\"']coherence_warnings[\"']",
        body,
    ), (
        "P2-AUDIT-NEW-1 violation: `execute_tools` no inicializa el "
        "accumulator `coherence_warnings = list(state.get(\"coherence_warnings\") or [])`. "
        "Sin esto, los warnings de tools previas en el mismo turn se "
        "perderían (rare en práctica pero defensive)."
    )

    # Extracción del tool_result
    assert re.search(
        r"parsed_mod\.get\s*\(\s*[\"']_coherence_warnings[\"']\s*\)",
        body,
    ), (
        "P2-AUDIT-NEW-1 violation: `execute_tools` no extrae "
        "`parsed_mod.get(\"_coherence_warnings\")` del tool_result de "
        "`modify_single_meal`. Sin extracción, el friendly string que "
        "se sobrescribe en `tool_result` pisaría la key y se perdería."
    )

    # Retorno incluye coherence_warnings
    return_match = re.search(
        r"return\s+\{[^}]*[\"']coherence_warnings[\"']\s*:\s*coherence_warnings",
        body,
        re.DOTALL,
    )
    assert return_match, (
        "P2-AUDIT-NEW-1 violation: `execute_tools` no retorna "
        "`coherence_warnings` en su dict de salida. LangGraph mergea el "
        "dict retornado contra el state — sin esta key, el state queda "
        "con la versión vieja."
    )


def test_stream_done_includes_coherence_warnings(agent_src: str) -> None:
    """C) El yield del SSE `done` en `chat_with_agent_stream` incluye
    `coherence_warnings` leído del `final_state_snapshot`.
    """
    # Buscar el yield del done.
    # [P2-AUDIT-NEW-1 drift] El yield real serializa via
    # `json.dumps({'type': 'done', ...})`, así que `type` va entre comillas
    # (`'type'`) y precedido por `{json.dumps({`. El patrón anterior asumía
    # `type` sin comilla de apertura y rompía sobre la `{` literal del dict.
    # Buscamos directamente el par clave/valor `"type": "done"` (cualquier
    # estilo de comilla) en una línea `yield`.
    done_yield = re.search(
        r"yield\s+f?[\"'][^\n]*[\"']type[\"']\s*:\s*[\"']done[\"']",
        agent_src,
    )
    assert done_yield, "No se encontró yield del evento `done` en agent.py."

    # Ventana: ~500 chars alrededor del yield para chequear que
    # `coherence_warnings` está en el f-string.
    yield_window = agent_src[done_yield.start(): done_yield.start() + 500]
    assert "coherence_warnings" in yield_window, (
        "P2-AUDIT-NEW-1 violation: el yield del evento SSE `done` no "
        "incluye `coherence_warnings`. El frontend AgentPage lee "
        "`dataObj.coherence_warnings` — si la key no está en el JSON "
        "serializado, el toast nunca se emite."
    )

    # Verificar que se lee del final_state_snapshot
    # (busca hacia ATRÁS del yield, donde se prepara el dict)
    preamble_start = max(0, done_yield.start() - 1500)
    preamble = agent_src[preamble_start: done_yield.start()]
    assert re.search(
        r"final_state_snapshot\.values\.get\s*\(\s*[\"']coherence_warnings[\"']",
        preamble,
    ), (
        "P2-AUDIT-NEW-1 violation: el código que prepara el `done` "
        "yield no lee `final_state_snapshot.values.get(\"coherence_warnings\")`. "
        "Sin lectura del state, `coherence_warnings` queda en el "
        "default [] y el toast siempre silencia, perdiendo el path "
        "real donde el agente generó warnings."
    )


# ---------------------------------------------------------------------------
# 4. Defensa-en-profundidad: no duplicar labels inline
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", [_PANTRY, _DASHBOARD, _ASSESSMENT_CTX, _AGENT_PAGE])
def test_consumers_do_not_inline_hypothesis_labels(path: Path) -> None:
    """Cada consumidor NO debe duplicar el catálogo de labels
    `cap_swallowed_modifier → "Falta en la lista"` etc. Si lo hace,
    está bypaseando el SSOT del helper + `coherenceLabels.js` y
    introduce drift potencial.

    Heuristic: ninguno de los hypothesis codes raw del backend
    aparece como string literal en estos archivos (deben pasar por
    el helper).
    """
    assert path.exists()
    src = path.read_text(encoding="utf-8")
    # Excluir comentarios para reducir falsos positivos
    # (JS line comments + block comments).
    src_no_comments = re.sub(r"//[^\n]*", "", src)
    src_no_comments = re.sub(r"/\*.*?\*/", "", src_no_comments, flags=re.DOTALL)

    forbidden_strings = [
        "cap_swallowed_modifier",
        "yield_uncovered",
        "pantry_overdeduct",
        # `unknown` y `unit_mismatch` son demasiado comunes — no chequear.
    ]
    for code in forbidden_strings:
        # Patrón: el código como string literal `"<code>"` o `'<code>'`
        pattern = re.compile(rf"['\"]{re.escape(code)}['\"]")
        assert not pattern.search(src_no_comments), (
            f"P2-AUDIT-NEW-1 violation: {path.name} contiene el "
            f"código de hipótesis crudo `{code}` como string literal. "
            f"Si necesitas reaccionar a tipos específicos de divergencia, "
            f"importa el catálogo desde `coherenceLabels.js` y compara "
            f"keys del map — no hardcodees el string. Sin esto, drift "
            f"backend→frontend cuando el backend agregue/renombre un código."
        )
