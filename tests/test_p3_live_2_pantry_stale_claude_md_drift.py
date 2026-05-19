"""[P3-LIVE-2 . 2026-05-11] Drift detection cross-language:
CLAUDE.md fila del alert `chunk_pantry_snapshots_stale` debe
documentar el filtro P1-LIVE-1 (`_pantry_captured_at IS NOT NULL`),
y el codigo (cron_tasks.py) debe contener el predicate matching.

Por que existe:
    P1-LIVE-1 (2026-05-11) cambio la semantica del alert: ahora solo
    cuenta chunks con `_pantry_captured_at IS NOT NULL` (no usa
    COALESCE a `created_at`). CLAUDE.md fila correspondiente se
    actualizo, pero sin un test que enforze el cross-link, un
    refactor futuro podria:
      (a) Restaurar el COALESCE en la query sin actualizar CLAUDE.md
          → operador se confia del doc cuando la realidad es otra.
      (b) Editar CLAUDE.md (e.g., simplificar la fila) sin sincronizar
          el codigo → operador asume el filtro ya no aplica.

    Este test ancla ambos lados: si uno cambia sin el otro, falla en CI.

Estrategia (parser estatico):
    1. CLAUDE.md contiene una fila para `chunk_pantry_snapshots_stale`
       en la tabla de "Politica system_alerts".
    2. Esa fila menciona EXPLICITAMENTE:
        - El predicate `_pantry_captured_at IS NOT NULL`.
        - El marker `P1-LIVE-1`.
    3. El codigo (cron_tasks.py, funcion productora) contiene el
       predicate `_pantry_captured_at` IS NOT NULL en la WHERE clause.
       (Es el contract anchor de P1-LIVE-1, replicado aqui como cross-link.)

    Si CLAUDE.md o el codigo divergen, el test falla con copy
    explicativo sobre cual lado se rompio.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
# [P3-CLAUDEMD-CAP] La tabla "Política system_alerts resolution" fue movida
# de CLAUDE.md a `backend/docs/system_alerts_resolution_table.md` por el cap
# de CLAUDE.md (P3-CLAUDEMD-CAP · 2026-05-14). El test ahora valida la
# nueva ubicación canónica.
_CLAUDE_MD = _REPO_ROOT / "backend" / "docs" / "system_alerts_resolution_table.md"
_CRON_PY = _REPO_ROOT / "backend" / "cron_tasks.py"


@pytest.fixture(scope="module")
def claude_md_src() -> str:
    return _CLAUDE_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def cron_src() -> str:
    return _CRON_PY.read_text(encoding="utf-8")


def _extract_alert_row(claude_md_src: str, alert_key: str) -> str:
    """Extrae la fila de la tabla CLAUDE.md que corresponde a `alert_key`.
    Las filas son `| \\`alert_key\\` | ... | ... | ... |` en una sola linea.
    """
    # Markdown table row: comienza con `|` y termina antes del proximo `\n`.
    pattern = re.compile(
        rf"^\|\s*`{re.escape(alert_key)}`\s*\|.*$",
        re.MULTILINE,
    )
    m = pattern.search(claude_md_src)
    assert m, (
        f"P3-LIVE-2: no se encontro la fila de `{alert_key}` en CLAUDE.md. "
        f"Posible regresion: alguien borro la fila de la tabla "
        f"'Politica system_alerts resolution'."
    )
    return m.group(0)


def test_claude_md_row_exists(claude_md_src: str):
    """Sanity: la fila para `chunk_pantry_snapshots_stale` existe."""
    _extract_alert_row(claude_md_src, "chunk_pantry_snapshots_stale")


def test_claude_md_row_mentions_captured_at_filter(claude_md_src: str):
    """La fila debe mencionar el predicate `_pantry_captured_at IS NOT NULL`
    (semantica P1-LIVE-1)."""
    row = _extract_alert_row(claude_md_src, "chunk_pantry_snapshots_stale")
    assert "_pantry_captured_at IS NOT NULL" in row, (
        "P3-LIVE-2 regresion: la fila de CLAUDE.md para "
        "`chunk_pantry_snapshots_stale` no menciona el predicate "
        "`_pantry_captured_at IS NOT NULL`. Si el codigo aplica "
        "ese filtro pero el doc no lo dice, un operador asume "
        "que chunks sin snapshot tambien cuentan."
    )


def test_claude_md_row_references_p1_live_1_marker(claude_md_src: str):
    """La fila debe linkear el marker `P1-LIVE-1` para trazabilidad
    al P-fix que introdujo la semantica."""
    row = _extract_alert_row(claude_md_src, "chunk_pantry_snapshots_stale")
    assert "P1-LIVE-1" in row, (
        "P3-LIVE-2 regresion: la fila no menciona el marker "
        "`P1-LIVE-1`. Sin esa cross-reference, un futuro auditor "
        "no encuentra el P-fix que introdujo el cambio de "
        "semantica del alert."
    )


def test_code_contains_captured_at_not_null_predicate(cron_src: str):
    """El codigo (funcion productora del alert) debe contener el
    predicate `_pantry_captured_at IS NOT NULL` en la query. Si
    CLAUDE.md lo dice pero el codigo no, doc != realidad."""
    # Localizar el body de la funcion productora.
    func_match = re.search(
        r"^def\s+_alert_chunk_pantry_snapshots_stale\s*\([^)]*\)[^:]*:(.*?)(?=^def\s)",
        cron_src,
        re.DOTALL | re.MULTILINE,
    )
    assert func_match, (
        "P3-LIVE-2 setup: funcion `_alert_chunk_pantry_snapshots_stale` "
        "no encontrada en cron_tasks.py."
    )
    body = func_match.group(1)
    pattern = re.compile(
        r"pipeline_snapshot\s*->\s*'form_data'\s*->>\s*'_pantry_captured_at'\s*\)\s*IS\s+NOT\s+NULL",
        re.IGNORECASE,
    )
    assert pattern.search(body), (
        "P3-LIVE-2 regresion: el codigo NO contiene el predicate "
        "`_pantry_captured_at IS NOT NULL` que CLAUDE.md "
        "documenta. Doc != codigo. Si quisieras restaurar el "
        "COALESCE legacy, actualiza CLAUDE.md primero y "
        "documenta el motivo del rollback."
    )
