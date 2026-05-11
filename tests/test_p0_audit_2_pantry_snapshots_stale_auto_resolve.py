"""[P0-AUDIT-2 · 2026-05-10] Auto-resolve de `chunk_pantry_snapshots_stale`
cuando el counter cae bajo umbral.

Bug original (audit 2026-05-10):
    CLAUDE.md tabla "Política `system_alerts`" declara este alert como
    "Auto — cron resetea si counter cae a 0". Pero el código en
    `_alert_chunk_pantry_snapshots_stale` (cron_tasks.py) hacía
    `return` temprano cuando `len(rows) < CHUNK_PANTRY_STALENESS_ALERT_MIN_COUNT`
    sin tocar `system_alerts`. Producción acumulaba alerts stuck por
    horas tras recuperación efectiva (1 alert observada con 3h+ stuck).

    Discrepancia entre doc y código: el doc decía "Auto" pero el
    código solo emitía.

Fix:
    En el early-return path (counter bajo umbral), UPDATE
    `system_alerts SET resolved_at = NOW() WHERE
    alert_key = 'chunk_pantry_snapshots_stale' AND resolved_at IS NULL`.
    Best-effort, idempotente (0 rows si no hay alert abierta).

Estrategia del test (parser estático sobre cron_tasks.py):
    1. Función `_alert_chunk_pantry_snapshots_stale` define el
       `alert_key` ANTES del early-return.
    2. El early-return path contiene UPDATE `resolved_at = NOW()`.
    3. UPDATE filtra por `resolved_at IS NULL` (idempotente — no
       pisa cerradas).
    4. UPDATE filtra por `alert_key = 'chunk_pantry_snapshots_stale'`
       específicamente (no namespace genérico).
    5. Best-effort: except envuelve el UPDATE.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_PY = _BACKEND_ROOT / "cron_tasks.py"


@pytest.fixture(scope="module")
def cron_src() -> str:
    return _CRON_PY.read_text(encoding="utf-8")


def _extract_function_body(src: str, name: str) -> str:
    """Cuerpo de una función top-level hasta el siguiente `def`."""
    m = re.search(
        rf"^def\s+{re.escape(name)}\s*\([^)]*\)[^:]*:(.*?)(?=^def\s)",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert m, f"No se encontró función top-level `{name}`."
    return m.group(1)


def test_function_exists(cron_src: str):
    """Sanity: la función productora sigue existiendo."""
    assert re.search(
        r"^def\s+_alert_chunk_pantry_snapshots_stale\s*\(",
        cron_src,
        re.MULTILINE,
    ), "P0-AUDIT-2: función productora desapareció."


def test_alert_key_defined_before_early_return(cron_src: str):
    """`alert_key = 'chunk_pantry_snapshots_stale'` debe definirse
    ANTES del early-return (`if len(rows) < min_count: ...`).

    Sin esto, el early-return no puede referenciar el alert_key
    para resolverlo — bug original.
    """
    body = _extract_function_body(cron_src, "_alert_chunk_pantry_snapshots_stale")

    # Localizar la asignación del alert_key.
    alert_key_match = re.search(
        r'alert_key\s*=\s*["\']chunk_pantry_snapshots_stale["\']',
        body,
    )
    assert alert_key_match, (
        "P0-AUDIT-2 regresión: `alert_key = 'chunk_pantry_snapshots_stale'` "
        "no se encuentra en `_alert_chunk_pantry_snapshots_stale`."
    )

    # Localizar el early-return (`if len(rows) < ... return`).
    early_return_match = re.search(
        r"if\s+len\(rows\)\s*<\s*int\(CHUNK_PANTRY_STALENESS_ALERT_MIN_COUNT\)",
        body,
    )
    assert early_return_match, (
        "P0-AUDIT-2 regresión: early-return path "
        "(`if len(rows) < CHUNK_PANTRY_STALENESS_ALERT_MIN_COUNT`) "
        "ya no existe — ¿se refactorizó la guard?"
    )

    assert alert_key_match.start() < early_return_match.start(), (
        "P0-AUDIT-2 regresión: `alert_key` se define DESPUÉS del "
        "early-return. El bloque de auto-resolve no puede referenciarlo. "
        "Mover la asignación de `alert_key` antes del early-return."
    )


def test_early_return_resolves_alert(cron_src: str):
    """Después del early-return debe haber UPDATE `resolved_at = NOW()`
    para `alert_key = 'chunk_pantry_snapshots_stale'`."""
    body = _extract_function_body(cron_src, "_alert_chunk_pantry_snapshots_stale")

    # El bloque del early-return: desde `if len(rows) < ...` hasta el
    # primer `return` que sigue.
    early_return_block = re.search(
        r"if\s+len\(rows\)\s*<\s*int\(CHUNK_PANTRY_STALENESS_ALERT_MIN_COUNT\)\s*:"
        r"(.*?)return",
        body,
        re.DOTALL,
    )
    assert early_return_block, (
        "P0-AUDIT-2 regresión: no se encontró el bloque del early-return."
    )

    block = early_return_block.group(1)
    update_pattern = re.compile(
        r"UPDATE\s+system_alerts\s+SET\s+resolved_at\s*=\s*NOW\(\)",
        re.IGNORECASE,
    )
    assert update_pattern.search(block), (
        "P0-AUDIT-2 regresión: el early-return NO ejecuta "
        "`UPDATE system_alerts SET resolved_at = NOW()`. Las alerts "
        "previas quedan stuck hasta resolución manual."
    )


def test_auto_resolve_filters_unresolved_only(cron_src: str):
    """UPDATE debe incluir `resolved_at IS NULL` — sin esto se pisan
    timestamps de alerts cerradas manualmente."""
    body = _extract_function_body(cron_src, "_alert_chunk_pantry_snapshots_stale")
    early_return_block = re.search(
        r"if\s+len\(rows\)\s*<\s*int\(CHUNK_PANTRY_STALENESS_ALERT_MIN_COUNT\)\s*:"
        r"(.*?)return",
        body,
        re.DOTALL,
    )
    assert early_return_block
    block = early_return_block.group(1)

    pattern = re.compile(r"resolved_at\s+IS\s+NULL", re.IGNORECASE)
    assert pattern.search(block), (
        "P0-AUDIT-2 regresión: UPDATE no filtra por `resolved_at IS NULL`. "
        "Cada run pisa timestamps de alerts ya resueltas — perdemos "
        "historial real de resolución."
    )


def test_auto_resolve_filters_specific_alert_key(cron_src: str):
    """UPDATE debe filtrar por `alert_key = %s` y bind del valor
    específico. Un wildcard sería peligroso (mataría otros alert_keys)."""
    body = _extract_function_body(cron_src, "_alert_chunk_pantry_snapshots_stale")
    early_return_block = re.search(
        r"if\s+len\(rows\)\s*<\s*int\(CHUNK_PANTRY_STALENESS_ALERT_MIN_COUNT\)\s*:"
        r"(.*?)return",
        body,
        re.DOTALL,
    )
    assert early_return_block
    block = early_return_block.group(1)

    # WHERE alert_key = %s (placeholder bound), no wildcard.
    assert re.search(r"alert_key\s*=\s*%s", block, re.IGNORECASE), (
        "P0-AUDIT-2 regresión: UPDATE no usa `alert_key = %s` con bind. "
        "Hardcodear el alert_key o usar LIKE es riesgoso."
    )
    # Tampoco debe haber LIKE en este bloque (que indicaría wildcard).
    assert "LIKE" not in block.upper().split("RETURN")[0], (
        "P0-AUDIT-2 regresión: UPDATE usa `LIKE` (wildcard) en el "
        "early-return. Solo `alert_key = 'chunk_pantry_snapshots_stale'` "
        "debe resolverse acá."
    )


def test_auto_resolve_is_best_effort(cron_src: str):
    """El UPDATE debe estar dentro de try/except — un fallo de DB
    no debe pausar el cron ni propagar al scheduler."""
    body = _extract_function_body(cron_src, "_alert_chunk_pantry_snapshots_stale")
    early_return_block = re.search(
        r"if\s+len\(rows\)\s*<\s*int\(CHUNK_PANTRY_STALENESS_ALERT_MIN_COUNT\)\s*:"
        r"(.*?)return",
        body,
        re.DOTALL,
    )
    assert early_return_block
    block = early_return_block.group(1)

    assert "try:" in block and "except" in block, (
        "P0-AUDIT-2 regresión: auto-resolve NO está envuelto en "
        "try/except. Un blip de Supabase haría que el cron entero "
        "propague la excepción al scheduler → genera "
        "`scheduler_error_alert_chunk_pantry_snapshots_stale`, "
        "ironía total."
    )
