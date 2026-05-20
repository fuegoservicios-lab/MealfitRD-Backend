"""[P2-CHAT-CLEANUP · 2026-05-20] Bundle de tests para los gaps #5 + #6
del audit production-readiness del módulo agente (2026-05-20):

  #5: `generate_chat_title_background` escribía a `title_debug.log` en disco
      (append-mode sin rotación). Convención repo (P2-LOGGER-MIGRATION)
      prohíbe escritura directa a archivo desde código productivo —
      migrado a `logger.debug`. Este test bloquea regresión: el archivo
      legacy NUNCA debe re-aparecer en disk + el helper `dlog()` NUNCA
      debe re-aparecer en agent.py.

  #6: Tabla "Las 9 tools cubiertas" en CLAUDE.md decía 9 cuando
      `tools.py::agent_tools` ya exportaba 11 (`check_hydration_today` +
      `log_water_glass`). El override genérico SÍ las cubre (al tope del
      loop) pero la doc mentía. Tabla movida a doc canónico
      `backend/docs/agent_tools_user_id_table.md`. Este test enforza
      paridad bidireccional: cada tool en `agent_tools` tiene entry en
      el doc y viceversa.

Cross-link convention (P2-HIST-AUDIT-14): el slug `p2_chat_cleanup` matchea
este archivo `test_p2_chat_cleanup.py`.

Tooltip-anchor: P2-CHAT-CLEANUP.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_ROOT.parent
_AGENT_PY = _BACKEND_ROOT / "agent.py"
_TOOLS_PY = _BACKEND_ROOT / "tools.py"
_TOOLS_DOC = _BACKEND_ROOT / "docs" / "agent_tools_user_id_table.md"
_TITLE_DEBUG_LOG = _BACKEND_ROOT / "title_debug.log"


@pytest.fixture(scope="module")
def agent_src() -> str:
    return _AGENT_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tools_src() -> str:
    return _TOOLS_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tools_doc_src() -> str:
    return _TOOLS_DOC.read_text(encoding="utf-8")


# ===========================================================================
# Sección 1 — gap #5: title_debug.log eliminado
# ===========================================================================

def test_no_title_debug_log_writes_in_agent(agent_src: str):
    """`agent.py` NO debe contener `open("title_debug.log"` ni el helper
    `def dlog(msg):`. Migración a `logger.debug` (P2-CHAT-CLEANUP).
    Regresión: si alguien re-introduce el debug-to-file, este test falla."""
    assert 'open("title_debug.log"' not in agent_src and "open('title_debug.log'" not in agent_src, (
        "P2-CHAT-CLEANUP regresión: `agent.py` re-introdujo escritura a "
        "`title_debug.log`. Convención repo (P2-LOGGER-MIGRATION) prohíbe "
        "escritura directa a archivo desde productivo — usar "
        "`logger.debug(...)` en su lugar."
    )

    # El helper `dlog(msg)` definido inside generate_chat_title_background.
    # Buscamos la definición exacta (no comments que mencionen el nombre).
    dlog_def_re = re.compile(r"^\s*def\s+dlog\s*\(", re.MULTILINE)
    assert not dlog_def_re.search(agent_src), (
        "P2-CHAT-CLEANUP regresión: helper `def dlog(msg):` reapareció en "
        "agent.py. Si quieres debug per-line, usar `logger.debug(...)` "
        "directamente — `dlog()` escribía a `title_debug.log` sin rotación."
    )


def test_title_debug_log_file_not_in_repo():
    """El archivo legacy `backend/title_debug.log` NO debe existir en disk.
    Si reaparece, alguien volvió a escribir desde código productivo."""
    assert not _TITLE_DEBUG_LOG.exists(), (
        f"P2-CHAT-CLEANUP regresión: `{_TITLE_DEBUG_LOG}` existe en disk. "
        f"Eliminar manualmente + buscar el callsite que lo creó. La "
        f"convención post-fix es `logger.debug(...)`, NO escritura a file."
    )


def test_title_background_uses_logger_debug(agent_src: str):
    """Sanity: `generate_chat_title_background` ahora usa `logger.debug(...)`
    para los puntos de debug que antes usaban `dlog()`. Verificamos que el
    patrón aparezca al menos 4 veces en la función (originalmente había
    ~7 dlog calls; ≥4 después de la migración es señal de que el migrate
    se aplicó, no se simplificó al borrar todo el debug)."""
    fn_re = re.compile(r"def\s+generate_chat_title_background\s*\(")
    m = fn_re.search(agent_src)
    assert m is not None
    body_start = m.end()
    next_def = re.search(r"\ndef\s|\nclass\s", agent_src[body_start:])
    body_end = body_start + (
        next_def.start() if next_def else min(12000, len(agent_src) - body_start)
    )
    body = agent_src[body_start:body_end]

    count = body.count("logger.debug(")
    assert count >= 4, (
        f"P2-CHAT-CLEANUP regresión: `generate_chat_title_background` tiene "
        f"{count} `logger.debug(...)` calls (esperado ≥4). Si bajaste el "
        f"debug verbosity intencionalmente, actualizar el threshold; si fue "
        f"accidente del migrate, restaurar los puntos de debug clave."
    )


# ===========================================================================
# Sección 2 — gap #6: paridad bidireccional tools.py ↔ doc
# ===========================================================================

def _extract_agent_tools_list(tools_src: str) -> list[str]:
    """Parsea `agent_tools = [tool1, tool2, ...]` del source de tools.py.
    Retorna la lista de nombres de tool en orden de aparición.

    Asume formato canonical: una sola línea o multilínea con `agent_tools = [`
    seguido de identificadores separados por comas, terminando en `]`."""
    m = re.search(r"agent_tools\s*=\s*\[(.*?)\]", tools_src, re.DOTALL)
    assert m is not None, (
        "tools.py no contiene la asignación `agent_tools = [...]`. ¿Refactor "
        "que movió el export? Actualizar el regex en este test."
    )
    raw_list = m.group(1)
    # Strip comentarios + whitespace; split por coma; filtrar vacíos.
    items = []
    for line in raw_list.split(","):
        # Strip inline comments after `#`
        line = re.sub(r"#.*$", "", line).strip()
        if line:
            items.append(line)
    return items


def _extract_doc_tool_names(doc_src: str) -> list[str]:
    """Parsea las rows de la tabla 'Las 11 tools cubiertas' y extrae los
    nombres de tool de la columna 2 (con backticks alrededor).

    Asume formato markdown: | n | tool_name | descripcion |."""
    # Buscar todas las rows que tienen formato `| <num> | \`<name>\` | ... |`
    row_re = re.compile(r"^\|\s*\d+\s*\|\s*`([a-zA-Z_][a-zA-Z0-9_]*)`\s*\|", re.MULTILINE)
    matches = row_re.findall(doc_src)
    return matches


def test_doc_lists_all_agent_tools(tools_src: str, tools_doc_src: str):
    """Cada tool en `tools.py::agent_tools` debe tener entry en el doc.
    Si añades tool nueva sin documentarla, este test falla con la lista
    de tools faltantes."""
    tools_in_code = set(_extract_agent_tools_list(tools_src))
    tools_in_doc = set(_extract_doc_tool_names(tools_doc_src))

    missing_in_doc = tools_in_code - tools_in_doc
    assert not missing_in_doc, (
        f"P2-CHAT-CLEANUP regresión: tools en `agent_tools` SIN entry en "
        f"`backend/docs/agent_tools_user_id_table.md`: {sorted(missing_in_doc)}. "
        f"Añadir row al doc + bumpear el contador 'Las N tools cubiertas' "
        f"en CLAUDE.md (root + backend) si N cambió."
    )


def test_no_orphan_tools_in_doc(tools_src: str, tools_doc_src: str):
    """Cada tool en el doc debe existir en `agent_tools`. Si eliminaste
    una tool de tools.py pero olvidaste podar el doc, este test falla."""
    tools_in_code = set(_extract_agent_tools_list(tools_src))
    tools_in_doc = set(_extract_doc_tool_names(tools_doc_src))

    orphans_in_doc = tools_in_doc - tools_in_code
    assert not orphans_in_doc, (
        f"P2-CHAT-CLEANUP regresión: tools en el doc PERO NO en "
        f"`agent_tools`: {sorted(orphans_in_doc)}. Si eliminaste la tool "
        f"intencionalmente, también eliminar su row del doc."
    )


def test_doc_header_count_matches_actual(tools_src: str, tools_doc_src: str):
    """El header `## Las N tools cubiertas` en el doc debe coincidir con
    `len(agent_tools)`. Sanity check de la narrativa del doc."""
    tools_count = len(_extract_agent_tools_list(tools_src))
    header_re = re.compile(r"##\s+Las\s+(\d+)\s+tools\s+cubiertas")
    m = header_re.search(tools_doc_src)
    assert m is not None, (
        "Doc no tiene header `## Las N tools cubiertas`. Si renombraste la "
        "sección, actualizar este test."
    )
    n_in_header = int(m.group(1))
    assert n_in_header == tools_count, (
        f"P2-CHAT-CLEANUP regresión: header del doc dice "
        f"`Las {n_in_header} tools cubiertas` pero `agent_tools` tiene "
        f"{tools_count}. Sincronizar el número en el header del doc."
    )


def test_claudemd_references_correct_count(tools_src: str):
    """CLAUDE.md (root + backend) deben referir el conteo correcto en
    `### Las N tools cubiertas`. Si bumpeas `agent_tools`, bumpear ambos
    CLAUDE.md también — el operador que abre CLAUDE.md NO debería leer
    'Las 9' cuando son 11."""
    tools_count = len(_extract_agent_tools_list(tools_src))
    expected_header = f"Las {tools_count} tools cubiertas"

    for path in [_REPO_ROOT / "CLAUDE.md", _BACKEND_ROOT / "CLAUDE.md"]:
        src = path.read_text(encoding="utf-8")
        assert expected_header in src, (
            f"P2-CHAT-CLEANUP regresión: `{path.name}` no contiene "
            f"`{expected_header}` (esperado para `len(agent_tools)={tools_count}`). "
            f"Sincronizar el header de la sección 'Anti-patrones de agent "
            f"tools prohibidos'."
        )


# ===========================================================================
# Sección 3 — tooltip-anchor + cross-link sanity
# ===========================================================================

def test_tooltip_anchor_present(agent_src: str):
    """El marker `P2-CHAT-CLEANUP` aparece ≥1 vez en agent.py (al menos
    en el comment de cierre del fix de title_debug.log)."""
    assert "P2-CHAT-CLEANUP" in agent_src, (
        "P2-CHAT-CLEANUP regresión: tooltip-anchor desaparecido de "
        "agent.py. Si un rename del slug ocurrió, restaurar el marker "
        "en el comment del fix `generate_chat_title_background`."
    )


def test_doc_exists_and_has_tooltip_anchor(tools_doc_src: str):
    """El doc canónico contiene el marker para que un rename del slug
    rompa el cross-link antes de causar daño en producción."""
    assert "P2-CHAT-CLEANUP" in tools_doc_src, (
        "P2-CHAT-CLEANUP regresión: tooltip-anchor desaparecido del doc "
        "`backend/docs/agent_tools_user_id_table.md`."
    )
