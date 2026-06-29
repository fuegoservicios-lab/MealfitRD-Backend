"""[P3-CHAT-MODEL-KNOBS-REGISTRY · 2026-05-15] Test parser-based: los 4
helpers `_chat_*_model_name()` en `backend/agent.py` leen su knob via
`_env_str(...)` (auto-registry en `_KNOBS_REGISTRY`), NO via
`os.environ.get(...)` directo.

Por qué este test
-----------------
Convención del repo `P3-NEW-D` (CLAUDE.md): TODOS los knobs `MEALFIT_*` se
auto-registran en `_KNOBS_REGISTRY` via `_env_int/_env_float/_env_bool/_env_str`.
`get_knobs_registry_snapshot()` expone el set actual al endpoint admin
`GET /api/system/admin/knobs`.

Pre-fix (cerrado P2-AUDIT-1 · 2026-05-15) los 4 helpers usaban
`os.environ.get("MEALFIT_<KNOB>", "<default>")` directo → no aparecían en
el registry. Un SRE que seteaba `MEALFIT_CHAT_AGENT_MODEL=gemini-3.1-flash`
en el VPS Oracle no podía verificar el cambio sin releer el source.

P3-CHAT-MODEL-KNOBS-REGISTRY · 2026-05-15 migra los 4 helpers a `_env_str(...)`,
cerrando el último gap de la convención P3-NEW-D para modelos de chat.

Esto NO supersede `test_p2_audit_1_agent_model_knobs.py` — los dos tests son
complementarios:
- P2-AUDIT-1: existe el helper + el knob name aparece en el body.
- P3-CHAT-MODEL-KNOBS-REGISTRY: el body usa `_env_str(...)` (no
  `os.environ.get(...)`) para que aparezca en el registry snapshot.

Cross-link convention (P2-HIST-AUDIT-14): el slug `p3_chat_model_knobs_registry`
matchea este archivo `test_p3_chat_model_knobs_registry.py`.

Tooltip-anchor: P3-CHAT-MODEL-KNOBS-REGISTRY-START | bundle audit 2026-05-15 noche.
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


_EXPECTED_HELPERS = [
    ("_chat_agent_model_name", "MEALFIT_CHAT_AGENT_MODEL"),
    ("_chat_agent_swap_model_name", "MEALFIT_CHAT_AGENT_SWAP_MODEL"),
    ("_chat_title_model_name", "MEALFIT_CHAT_TITLE_MODEL"),
    ("_chat_router_model_name", "MEALFIT_CHAT_ROUTER_MODEL"),
]


def _extract_helper_body(src: str, helper_name: str) -> str:
    """Extrae el cuerpo de `def _chat_*_model_name(...) -> str: ...` hasta la
    siguiente función o asignación módulo-level. [P0-DEEPSEEK-MIGRATION]
    chat/swap aceptan `user_id` opcional (tier-routing) — el regex admite
    parámetros en la firma."""
    def_re = re.compile(
        rf"def\s+{re.escape(helper_name)}\s*\([^)]*\)\s*->\s*str\s*:",
    )
    m = def_re.search(src)
    assert m is not None, (
        f"Helper `def {helper_name}() -> str:` no encontrado en agent.py. "
        f"P2-AUDIT-1 debería garantizar su presencia — si fue removido, "
        f"actualizar también este test."
    )
    body_start = m.end()
    next_def = re.search(r"\n(?:def\s|llm\s*=)", src[body_start:])
    body_end = body_start + (next_def.start() if next_def else min(500, len(src) - body_start))
    return src[body_start:body_end]


# ---------------------------------------------------------------------------
# 1. Cada helper usa `_env_str(...)` en su body
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("helper_name, knob_name", _EXPECTED_HELPERS)
def test_helper_uses_env_str(agent_src: str, helper_name: str, knob_name: str):
    """Cada helper debe llamar `_env_str("MEALFIT_<KNOB>", "<default>")`.

    Si alguien revierte a `os.environ.get(...)` este test falla con copy
    explicativo apuntando a P3-NEW-D + el endpoint admin que depende del
    registry.
    """
    body = _extract_helper_body(agent_src, helper_name)
    # Patrón flexible para multi-línea:
    # _env_str(
    #     "MEALFIT_CHAT_AGENT_MODEL",
    #     "gemini-3.1-pro-preview",
    # )
    pattern = re.compile(
        rf'_env_str\s*\(\s*["\']?{re.escape(knob_name)}["\']?',
        re.DOTALL,
    )
    assert pattern.search(body), (
        f"P3-CHAT-MODEL-KNOBS-REGISTRY regresión: helper `{helper_name}` no "
        f"llama `_env_str(\"{knob_name}\", ...)`. El body contiene:\n"
        f"---\n{body[:300]}\n---\n"
        f"Convención P3-NEW-D (CLAUDE.md): todos los knobs `MEALFIT_*` "
        f"deben auto-registrarse en `_KNOBS_REGISTRY` via `_env_str/_int/"
        f"_float/_bool`. Sin esto, el knob NO aparece en `GET /api/system/"
        f"admin/knobs` y SRE no puede verificar cambios sin releer source."
    )


# ---------------------------------------------------------------------------
# 2. Ningún helper usa `os.environ.get(...)` directo
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("helper_name, knob_name", _EXPECTED_HELPERS)
def test_helper_does_not_use_os_environ_get(agent_src: str, helper_name: str, knob_name: str):
    """Defensa simétrica: si alguien copia-pegó el pattern viejo,
    bloquear. `os.environ.get(...)` bypasea `_KNOBS_REGISTRY`."""
    body = _extract_helper_body(agent_src, helper_name)
    # Strip comentarios primero — narrativa puede mencionar el pattern viejo.
    body_no_comments = re.sub(r"#[^\n]*", "", body)
    bad_pattern = re.compile(
        rf'os\.environ\.get\s*\(\s*["\']?{re.escape(knob_name)}["\']?',
        re.DOTALL,
    )
    assert not bad_pattern.search(body_no_comments), (
        f"P3-CHAT-MODEL-KNOBS-REGISTRY regresión: helper `{helper_name}` "
        f"usa `os.environ.get(\"{knob_name}\", ...)` — pattern viejo "
        f"(pre-P3-CHAT-MODEL-KNOBS-REGISTRY · 2026-05-15). Migrar a "
        f"`_env_str(\"{knob_name}\", \"<default>\")` para auto-registro "
        f"en `_KNOBS_REGISTRY` (convención P3-NEW-D)."
    )


# ---------------------------------------------------------------------------
# 3. Import `from knobs import _env_str` presente
# ---------------------------------------------------------------------------
def test_env_str_imported(agent_src: str):
    """El módulo debe importar `_env_str` desde `knobs`. Sin import, los
    helpers fallarían con NameError al primer call."""
    # Strip comentarios (narrativa puede mencionar el name).
    no_comments = re.sub(r"#[^\n]*", "", agent_src)
    import_re = re.compile(
        r"from\s+knobs\s+import\s+[^\n]*\b_env_str\b",
    )
    assert import_re.search(no_comments), (
        "P3-CHAT-MODEL-KNOBS-REGISTRY regresión: `from knobs import _env_str` "
        "no encontrado en agent.py. Sin él los 4 helpers `_chat_*_model_name` "
        "fallan con `NameError: _env_str`."
    )


# ---------------------------------------------------------------------------
# 4. Anchor textual P3-CHAT-MODEL-KNOBS-REGISTRY presente
# ---------------------------------------------------------------------------
def test_anchor_present(agent_src: str):
    """`grep -r P3-CHAT-MODEL-KNOBS-REGISTRY backend/` debe localizar el
    fix sin abrir archivos. Anchor textual en agent.py + este test
    convergen al SOP."""
    assert "P3-CHAT-MODEL-KNOBS-REGISTRY" in agent_src, (
        "P3-CHAT-MODEL-KNOBS-REGISTRY regresión: anchor textual "
        "`P3-CHAT-MODEL-KNOBS-REGISTRY` perdido en agent.py. Restaurar "
        "para grep cross-incidente."
    )


# ---------------------------------------------------------------------------
# 5. Funcional: los 4 knobs aparecen en `_KNOBS_REGISTRY` tras invocar
#    los helpers (defensa runtime, no solo parser-based).
# ---------------------------------------------------------------------------
def test_knobs_registered_in_runtime_registry():
    """Smoke test funcional: importa los 4 helpers, llámalos, y verifica
    que las 4 keys aparecen en `get_knobs_registry_snapshot()`.

    Si alguien rompiera `_env_str` (e.g. bypaseando `_register_knob`), este
    test atrapa el regreso silencioso.

    NOTA: este test tolera ImportError de agent.py (CI sin GEMINI_API_KEY)
    porque la importación module-level puede fallar antes de exponer los
    helpers. En ese caso se saltea con xfail — el parser-based test #1
    sigue cubriendo el contrato textual.
    """
    try:
        from agent import (
            _chat_agent_model_name,
            _chat_agent_swap_model_name,
            _chat_title_model_name,
            _chat_router_model_name,
        )
    except Exception as e:
        pytest.xfail(
            f"agent.py no importable en este entorno ({type(e).__name__}: {e}). "
            f"El parser-based test cubre el contrato textual. Re-run con "
            f"GEMINI_API_KEY + DB env vars en staging."
        )
    # Invocar los 4 para gatillar el `_register_knob`.
    _chat_agent_model_name()
    _chat_agent_swap_model_name()
    _chat_title_model_name()
    _chat_router_model_name()
    from knobs import get_knobs_registry_snapshot
    snap = get_knobs_registry_snapshot()
    expected_keys = {
        "MEALFIT_CHAT_AGENT_MODEL",
        "MEALFIT_CHAT_AGENT_SWAP_MODEL",
        "MEALFIT_CHAT_TITLE_MODEL",
        "MEALFIT_CHAT_ROUTER_MODEL",
    }
    missing = expected_keys - set(snap.keys())
    assert not missing, (
        f"P3-CHAT-MODEL-KNOBS-REGISTRY funcional: tras invocar los 4 "
        f"helpers `_chat_*_model_name()`, faltan keys en "
        f"`get_knobs_registry_snapshot()`: {sorted(missing)}. "
        f"Snapshot actual contiene {len(snap)} keys. El bug puede estar "
        f"en `_env_str` o en `_register_knob` — investigar antes de "
        f"relajar este test."
    )
