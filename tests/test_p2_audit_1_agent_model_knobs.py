"""[P2-AUDIT-1 · 2026-05-15] Test parser-based: los 5 callsites de
`ChatGoogleGenerativeAI(...)` en `backend/agent.py` leen el modelo
Gemini desde knobs `MEALFIT_*_MODEL`, NO desde literales hardcoded.

Por qué este test:
    Convención del repo `P3-PREVIEW-MODEL-KNOB · 2026-05-12` (CLAUDE.md):
    callsites productivos de modelos LLM `*-preview` de Google leen el ID
    desde env vars con default seguro. Sin knob, swap del modelo requiere
    redeploy — un incidente real (CB rows stale por `gemini-3.1-pro-preview`
    4.4 días seguidos en 2026-05-11) confirmó la fragilidad.

Helpers esperados (ya definidos en agent.py):
    - `_chat_agent_model_name()`        → knob `MEALFIT_CHAT_AGENT_MODEL`
    - `_chat_agent_swap_model_name()`   → knob `MEALFIT_CHAT_AGENT_SWAP_MODEL`
    - `_chat_title_model_name()`        → knob `MEALFIT_CHAT_TITLE_MODEL`
    - `_chat_router_model_name()`       → knob `MEALFIT_CHAT_ROUTER_MODEL`

Callsites cubiertos (los 5):
    1. `llm` (módulo-level, llamado desde swap_meal default)
    2. `swap_llm` dentro de `swap_meal`
    3. `chat_llm` dentro de `call_model` (LangGraph node)
    4. `title_llm` dentro de `generate_session_title`
    5. `router_llm` dentro de `rag_query_router`

Drift detection:
    - Cero `model="gemini-..."` literal en agent.py (todos via helper).
    - Los 4 helpers existen como `def _<name>() -> str:` con
      `os.environ.get("MEALFIT_<KNOB>", "<default>")`.
    - Cada uno de los 5 callsites `ChatGoogleGenerativeAI(...)` usa
      `model=_chat_*_model_name()`.

Cross-link convention (P2-HIST-AUDIT-14): el slug `p2_audit_1` matchea
este archivo `test_p2_audit_1_agent_model_knobs.py`.

Tooltip-anchor: P2-AUDIT-1-START | gap audit 2026-05-15
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_AGENT_PY = _BACKEND_ROOT / "agent.py"


@pytest.fixture(scope="module")
def agent_src() -> str:
    return _AGENT_PY.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Cero literales `model="gemini-..."` activos en agent.py
# ---------------------------------------------------------------------------
def test_no_inline_gemini_model_literal(agent_src: str):
    """Patrón `model="gemini-..."` no debe aparecer como argumento de
    `ChatGoogleGenerativeAI(...)`. Strip comentarios primero — bloques
    narrativos referencian modelos por nombre."""
    no_comments = re.sub(r"#[^\n]*", "", agent_src)
    pattern = re.compile(r'model\s*=\s*["\']gemini-[a-z0-9.-]+["\']')
    matches = pattern.findall(no_comments)
    assert not matches, (
        f"P2-AUDIT-1 regresión: {len(matches)} literales `model=\"gemini-...\"` "
        f"activos en agent.py: {matches}. La convención P3-PREVIEW-MODEL-KNOB "
        f"obliga a leer el modelo via helper `_chat_*_model_name()` que lea "
        f"`MEALFIT_*_MODEL` con default. Reemplazar el literal por "
        f"`model=_chat_<feature>_model_name()`."
    )


# ---------------------------------------------------------------------------
# 2. Los 4 helpers están definidos
# ---------------------------------------------------------------------------
_EXPECTED_HELPERS = [
    ("_chat_agent_model_name", "MEALFIT_CHAT_AGENT_MODEL"),
    ("_chat_agent_swap_model_name", "MEALFIT_CHAT_AGENT_SWAP_MODEL"),
    ("_chat_title_model_name", "MEALFIT_CHAT_TITLE_MODEL"),
    ("_chat_router_model_name", "MEALFIT_CHAT_ROUTER_MODEL"),
]


@pytest.mark.parametrize("helper_name, knob_name", _EXPECTED_HELPERS)
def test_helper_defined_with_env_knob(agent_src: str, helper_name: str, knob_name: str):
    """Cada helper `def _chat_*_model_name(...) -> str:` lee el knob
    `MEALFIT_<KNOB>`. [P0-LLM-PROVIDER-MIGRATION · 2026-06-12] chat/swap aceptan
    `user_id` opcional para tier-routing — la firma admite parámetros."""
    # Definición del helper (con o sin parámetros — tier-routing).
    def_re = re.compile(
        rf"def\s+{re.escape(helper_name)}\s*\([^)]*\)\s*->\s*str\s*:",
    )
    m = def_re.search(agent_src)
    assert m is not None, (
        f"P2-AUDIT-1 regresión: helper `def {helper_name}() -> str:` no "
        f"definido en agent.py. Sin él, el callsite no puede leer el knob "
        f"`{knob_name}` y queda hardcoded."
    )
    # Cuerpo del helper hasta la siguiente def/llm/módulo top-level.
    body_start = m.end()
    next_def = re.search(r"\n(?:def\s|llm\s*=)", agent_src[body_start:])
    body_end = body_start + (next_def.start() if next_def else min(500, len(agent_src) - body_start))
    body = agent_src[body_start:body_end]
    assert knob_name in body, (
        f"P2-AUDIT-1 regresión: helper `{helper_name}` no lee el knob "
        f"`{knob_name}`. El body debe contener `os.environ.get(\"{knob_name}\", "
        f"\"<default>\")`."
    )


# ---------------------------------------------------------------------------
# 3. Cada uno de los 5 callsites usa un helper
# ---------------------------------------------------------------------------
def test_all_5_callsites_use_helper(agent_src: str):
    """Cuenta callsites `ChatGLM(...)` y verifica que cada uno tenga
    `model=_chat_*_model_name(...)` en su lista de args (no un literal).
    [P0-LLM-PROVIDER-MIGRATION] El constructor es ChatGLM y los helpers de
    chat/swap reciben el user_id para tier-routing — el regex acepta args."""
    no_comments = re.sub(r"#[^\n]*", "", agent_src)
    # [P1-SWAP-LUNA · 2026-08-05] El regex cubre AMBOS constructores. `swap_meal` pasó a
    # `build_chat_llm` (fábrica por proveedor) porque su modelo es de OpenAI. Bajar el
    # conteo de 5 a 4 habría puesto el test en verde dejando ese callsite SIN VIGILAR
    # para siempre — que es justo lo contrario de para lo que existe este tripwire.
    callsite_re = re.compile(
        r"(?:ChatGLM|build_chat_llm)\s*\(",
    )
    callsites = list(callsite_re.finditer(no_comments))
    # Cada `ChatGLM(...)` puede cerrar en distintas posiciones.
    # Extraemos un window de ~400 chars tras el paréntesis abierto para
    # capturar `model=...` argument.
    helper_re = re.compile(r"model\s*=\s*_chat_\w+_model_name\s*\([^)]*\)")

    # [P1-SWAP-LUNA · 2026-08-05] También vale `model=<var>` cuando esa variable se asigna
    # desde un helper. El callsite del swap resuelve el modelo UNA vez a una variable porque
    # lo necesitan tres consumidores (detector de proveedor, constructor y gate del circuit
    # breaker); obligarlo a repetir la llamada inline sería peor código.
    #
    # La indirección NO se acepta a ciegas: la variable tiene que estar asignada desde uno de
    # los helpers en este mismo fichero. Si no, `_x = "gpt-4-hardcoded"` + `model=_x` colaría
    # justo lo que este test existe para impedir.
    var_re = re.compile(r"model\s*=\s*([A-Za-z_]\w*)\s*[,)]")
    vars_desde_helper = set(
        re.findall(r"^\s*([A-Za-z_]\w*)\s*=\s*_chat_\w+_model_name\s*\(", no_comments, re.M)
    )

    offenders = []
    for m in callsites:
        window = no_comments[m.end():m.end() + 400]
        if helper_re.search(window):
            continue
        mv = var_re.search(window)
        if mv and mv.group(1) in vars_desde_helper:
            continue
        line_no = no_comments.count("\n", 0, m.start()) + 1
        offenders.append(f"line {line_no}")
    assert not offenders, (
        f"P2-AUDIT-1 regresión: {len(offenders)} callsites de "
        f"`ChatGLM(...)` no usan `model=_chat_*_model_name(...)`: "
        f"{offenders}. Cada callsite debe leer el modelo via uno de los 4 "
        f"helpers `_chat_agent_model_name`, `_chat_agent_swap_model_name`, "
        f"`_chat_title_model_name`, `_chat_router_model_name`."
    )
    # Esperamos exactamente 5 callsites (los 5 documentados en el audit).
    # Si cambia el conteo, sirve como tripwire para auditar el nuevo callsite.
    assert len(callsites) == 5, (
        f"P2-AUDIT-1 advertencia: detectados {len(callsites)} callsites de "
        f"`ChatGLM(...)` en agent.py (esperados 5). Si añadiste "
        f"un callsite nuevo, asegúrate de que usa un helper `_chat_*_model_name()` "
        f"y actualizar `_EXPECTED_HELPERS` + este conteo. Si removiste uno, "
        f"reducir el conteo aquí."
    )


# ---------------------------------------------------------------------------
# 4. Anchor textual P2-AUDIT-1 presente
# ---------------------------------------------------------------------------
def test_anchor_present(agent_src: str):
    """Comment inline `[P2-AUDIT-1 · ...]` cerca de los helpers para
    `grep -r P2-AUDIT-1` localizar el fix sin abrir el archivo."""
    assert "P2-AUDIT-1" in agent_src, (
        "P2-AUDIT-1 regresión: anchor textual `P2-AUDIT-1` perdido en "
        "agent.py. Restaurar para grep cross-incidente."
    )
