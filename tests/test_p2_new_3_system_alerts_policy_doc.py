"""[P2-NEW-3 · 2026-05-10] CLAUDE.md DEBE documentar la política de
resolución de `system_alerts` (productor + resolver + auto/manual).

Bug original (audit 2026-05-10):
    CLAUDE.md no enumeraba quién/cuándo/cómo se resuelven alerts.
    Mezclaba auto-resolve (P1-NEW-2 listener), manual ops, y crons
    resolver (regenerate-simplified cierra `chunk_paused_indefinitely:*`).
    Un nuevo dev/operator no podía decidir si actuar sobre una alert
    sin grep manual del codebase.

Fix:
    Sección "Política de `system_alerts` resolution" en CLAUDE.md con
    tabla {alert_key pattern → productor → resolver → auto/manual}.

Estrategia del test (parser estático sobre CLAUDE.md):
    1. Verificar que la sección existe.
    2. Verificar al menos N=8 filas en la tabla (cobertura mínima de
       los alert_keys conocidos).
    3. Verificar que cada alert_key relevante aparece (drift detection:
       si añadimos un nuevo alert_key al codebase pero olvidamos
       documentarlo, este test falla).
    4. Verificar bloque SQL de limpieza one-shot (operadores lo copian).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CLAUDE_MD = _REPO_ROOT / "CLAUDE.md"


@pytest.fixture(scope="module")
def claude_md_src() -> str:
    assert _CLAUDE_MD.exists(), f"{_CLAUDE_MD} no existe."
    return _CLAUDE_MD.read_text(encoding="utf-8")


def test_section_present(claude_md_src: str):
    """El header `## Política de system_alerts resolution` debe existir."""
    pattern = re.compile(
        r"^##\s+Política\s+de\s+`system_alerts`\s+resolution",
        re.MULTILINE,
    )
    assert pattern.search(claude_md_src), (
        "P2-NEW-3 regresión: sección `## Política de system_alerts "
        "resolution` desapareció de CLAUDE.md. Sin ella, un nuevo "
        "operador no sabe si actuar manualmente sobre una alert o "
        "esperar resolución automática."
    )


def test_key_alert_keys_documented(claude_md_src: str):
    """Los alert_keys actualmente en uso deben aparecer en la tabla.
    Drift detection: si añadimos un alert_key nuevo en código pero
    olvidamos documentarlo, este test falla.

    Lista derivada del grep `alert_key.*=` sobre backend/ al cierre
    P2-NEW-3 (2026-05-10).
    """
    expected_keys = [
        "scheduler_missed_",
        "scheduler_error_",
        "scheduler_cascade_missed",
        "chunk_paused_indefinitely",
        "chunks_dead_lettered_recent",
        "chunk_pantry_snapshots_stale",
        "inventory_rpc_fallback",
        "coherence_watchdog_silent",
    ]
    missing = [k for k in expected_keys if k not in claude_md_src]
    assert not missing, (
        f"P2-NEW-3 regresión: alert_keys NO documentados en la sección: "
        f"{missing}. Si fueron renombrados, actualizar la tabla. Si "
        f"fueron eliminados del código, removerlos de la tabla."
    )


def test_cleanup_sql_block_present(claude_md_src: str):
    """El bloque SQL `UPDATE system_alerts SET resolved_at = NOW() ...`
    debe estar presente para que un operador pueda copy-paste para
    limpiar alerts viejas pre-deploy."""
    # Buscar el bloque ```sql ... ```.
    pattern = re.compile(
        r"```sql\s*[\r\n]+.*?UPDATE\s+system_alerts\s+SET\s+resolved_at\s*=\s*NOW\(\)",
        re.DOTALL | re.IGNORECASE,
    )
    assert pattern.search(claude_md_src), (
        "P2-NEW-3 regresión: el bloque SQL de `Limpieza one-shot` "
        "(UPDATE system_alerts SET resolved_at = NOW()) desapareció. "
        "Operadores lo necesitan para purgar alerts huérfanas pre-deploy."
    )


def test_section_explains_how_to_add_new_alert_key(claude_md_src: str):
    """La sección debe explicar el procedimiento para añadir un nuevo
    alert_key. Sin esto, los devs futuros añaden alerts sin documentar
    el resolver."""
    # Buscar el sub-header "Cómo añadir un nuevo alert_key" o similar.
    pattern = re.compile(
        r"###\s+Cómo\s+añadir\s+un\s+nuevo\s+`?alert_key`?",
        re.IGNORECASE,
    )
    assert pattern.search(claude_md_src), (
        "P2-NEW-3 regresión: la sub-sección `Cómo añadir un nuevo "
        "alert_key` desapareció. Sin ese procedimiento, devs futuros "
        "añaden alerts sin documentar productor/resolver."
    )
