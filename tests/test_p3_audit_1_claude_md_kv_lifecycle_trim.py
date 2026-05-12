"""[P3-AUDIT-1 · 2026-05-12] El detalle verbose de la subsección
"Ciclo de vida del KV `llm_circuit_breaker:*`" en CLAUDE.md se movió a un
runbook externo para reducir el contexto cargado en cada turn.

Contexto:
    El audit production-readiness 2026-05-12 identificó CLAUDE.md como
    el principal consumidor de tokens por turn (60+ KB cargados siempre).
    La subsección de KV lifecycle ocupaba ~3.7 KB de detalle verbose
    (diagrama de transiciones, SOPs, SQL de verificación) que pertenecen
    a un runbook para SREs, no al contexto base del agente.

Diseño:
    - Subsección en CLAUDE.md: trimmed a ~2.7 KB con el contrato mínimo
      (header, marker P3-NEW-E, los 5 knob names, las 3 vías de reset
      como bullets, referencia al runbook).
    - Runbook nuevo: `~/.claude/projects/.../memory/runbook_llm_circuit_breaker_kv_lifecycle_2026_05_12.md`
      con el detalle completo + SOPs adicionales.
    - Test `test_p3_new_e_cb_kv_lifecycle_doc.py` (legacy) sigue pasando
      porque los anchors críticos están preservados.

Lo que este test enforza:
  A) CLAUDE.md SÍ menciona explícitamente la migración al runbook
     (marker P3-AUDIT-1 en el cuerpo de la subsección).
  B) La subsección de CLAUDE.md está efectivamente más corta que ~3.5 KB
     (sino el trim no ocurrió o fue revertido).
  C) Test legacy `test_p3_new_e_cb_kv_lifecycle_doc.py` sigue passing
     (anchors preservados) — verificado indirectamente al chequear que
     los 5 knob names todavía aparecen.

Tooltip-anchor: P3-AUDIT-1-CLAUDE-TRIM.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLAUDE_MD = _REPO_ROOT / "CLAUDE.md"


@pytest.fixture(scope="module")
def claude_md() -> str:
    return _CLAUDE_MD.read_text(encoding="utf-8")


def _extract_kv_subsection(text: str) -> str:
    start = text.find("### Ciclo de vida del KV `llm_circuit_breaker:*`")
    assert start >= 0, "Subsección KV lifecycle no encontrada en CLAUDE.md."
    # Busca el siguiente header (### o ##) tras el contenido.
    pattern = re.compile(r"^(?:### |## )", re.MULTILINE)
    next_match = pattern.search(text, start + 10)
    end = next_match.start() if next_match else len(text)
    return text[start:end]


# ---------------------------------------------------------------------------
# A) Marker P3-AUDIT-1 presente en la subsección (señala el trim).
# ---------------------------------------------------------------------------

def test_a_p3_audit_1_marker_in_kv_subsection(claude_md: str):
    section = _extract_kv_subsection(claude_md)
    assert "P3-AUDIT-1" in section, (
        "P3-AUDIT-1 regresión: la subsección KV no menciona P3-AUDIT-1 "
        "como el P-fix que movió el detalle al runbook. Sin este marker, "
        "un futuro mantenedor no entiende dónde vivía el contenido antes."
    )


# ---------------------------------------------------------------------------
# B) La subsección está trimmed (post-trim < 3.5 KB).
# ---------------------------------------------------------------------------

def test_b_kv_subsection_is_compact(claude_md: str):
    section = _extract_kv_subsection(claude_md)
    # Pre-trim: ~3700 chars. Post-trim objetivo: ~2700 chars. Floor 3500
    # cubre el caso 'el trim ocurrió' sin ser estricto sobre el target exacto.
    assert len(section) < 3500, (
        f"P3-AUDIT-1 regresión: subsección KV lifecycle tiene "
        f"{len(section)} chars (esperado <3500 post-trim). El detalle "
        f"verbose debe vivir en el runbook "
        f"`runbook_llm_circuit_breaker_kv_lifecycle_2026_05_12.md`, "
        f"no en CLAUDE.md."
    )


# ---------------------------------------------------------------------------
# C) Anchors legacy (test_p3_new_e) preservados.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("knob", [
    "MEALFIT_CB_FAILURE_THRESHOLD",
    "MEALFIT_CB_RESET_TIMEOUT_S",
    "MEALFIT_CB_LOCAL_HEALTH_TTL_S",
    "MEALFIT_CB_KV_STALENESS_HOURS",
    "MEALFIT_CB_KV_STALENESS_SWEEP_INTERVAL_MIN",
])
def test_c_knob_preserved_in_trimmed_subsection(claude_md: str, knob: str):
    """Los 5 knob names que test_p3_new_e enforza siguen presentes."""
    section = _extract_kv_subsection(claude_md)
    assert knob in section, (
        f"P3-AUDIT-1 violation: trim de la subsección eliminó el knob "
        f"`{knob}` que test_p3_new_e_cb_kv_lifecycle_doc enforza. "
        f"Restaurar el knob en la tabla resumen de CLAUDE.md (también "
        f"mantener sincronizado el runbook)."
    )


# ---------------------------------------------------------------------------
# D) Referencia al runbook presente.
# ---------------------------------------------------------------------------

def test_d_runbook_referenced(claude_md: str):
    section = _extract_kv_subsection(claude_md)
    assert "runbook_llm_circuit_breaker_kv_lifecycle" in section, (
        "P3-AUDIT-1: subsección KV no referencia el runbook donde vive "
        "el detalle. Sin el link, un SRE no sabe dónde buscar el "
        "diagrama de transiciones / SOPs / SQL de verificación live."
    )
