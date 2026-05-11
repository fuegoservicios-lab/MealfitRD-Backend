"""[P3-NEW-5 · 2026-05-11] Docstring del comportamiento `min_count`
para auto-resolve del alert `chunk_pantry_snapshots_stale`.

Gap original (audit 2026-05-11):
    La función `_alert_chunk_pantry_snapshots_stale` auto-resuelve la
    alert cuando `len(rows) < CHUNK_PANTRY_STALENESS_ALERT_MIN_COUNT`
    (P0-AUDIT-2), pero NO documentaba qué pasa con casos parciales:
      - 1 chunk stale + min_count=3 → NO mantiene alert viva.
      - 5 chunks stale + min_count=3 → alert viva hasta que <3.
      - 10 chunks stale + min_count=1 → todos deben refrescarse.

    Sin docstring, un operador que ajusta el umbral via knob no sabe
    qué trade-off está haciendo.

Fix: docstring extendido en `_alert_chunk_pantry_snapshots_stale`
documentando:
    - Semántica del umbral (counter ≥ min_count → alert viva).
    - Trade-offs de subir/bajar.
    - Casos parciales con counter < min_count NO mantienen alert.
    - Chunks individuales stale los captura otro alert (dead_lettered).

Este test parser-based enforza la presencia del docstring.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CRON_FP = _REPO_ROOT / "backend" / "cron_tasks.py"


@pytest.fixture(scope="module")
def src() -> str:
    return _CRON_FP.read_text(encoding="utf-8")


def _extract_function_body(src: str, name: str) -> str:
    m = re.search(
        rf"^def\s+{re.escape(name)}\s*\([^)]*\)[^:]*:(.*?)(?=^def\s)",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert m, f"función `{name}` no encontrada"
    return m.group(1)


def test_function_exists(src: str):
    """Sanity: la función debe existir."""
    assert re.search(
        r"^def\s+_alert_chunk_pantry_snapshots_stale\s*\(",
        src,
        re.MULTILINE,
    ), "P3-NEW-5: función ausente — gap mayor que docstring."


def test_docstring_includes_p3_new_5_marker(src: str):
    """El docstring debe identificar el P-fix `P3-NEW-5` para
    trazabilidad post-mortem."""
    body = _extract_function_body(src, "_alert_chunk_pantry_snapshots_stale")
    assert "P3-NEW-5" in body, (
        "P3-NEW-5 regresión: marker P3-NEW-5 ausente en docstring. "
        "Sin marker, un futuro auditor no sabe si la sección de "
        "documentación es la añadida en este P-fix."
    )


def test_docstring_documents_min_count_semantics(src: str):
    """El docstring debe explicar la semántica del umbral
    `min_count` — específicamente que casos parciales (counter <
    min_count) NO mantienen alert viva."""
    body = _extract_function_body(src, "_alert_chunk_pantry_snapshots_stale")
    # Debe mencionar `min_count` Y el comportamiento de auto-resolve.
    assert "min_count" in body.lower(), (
        "P3-NEW-5 regresión: docstring no menciona `min_count`. "
        "Sin esa referencia explícita, el comportamiento del umbral "
        "queda implícito."
    )
    # Debe mencionar el trade-off de ajustar el umbral.
    assert "trade-off" in body.lower() or "umbral" in body.lower(), (
        "P3-NEW-5 regresión: docstring no documenta trade-offs de "
        "ajustar el umbral. Un operador que cambia el knob necesita "
        "saber qué se gana/pierde."
    )


def test_docstring_references_auto_resolve_path(src: str):
    """El docstring debe referenciar el auto-resolve via P0-AUDIT-2
    (UPDATE resolved_at en early-return) — el comportamiento de
    auto-cierre cuando counter cae bajo umbral."""
    body = _extract_function_body(src, "_alert_chunk_pantry_snapshots_stale")
    # auto-resolve explicit reference
    has_auto = ("auto-resuelve" in body.lower()
                or "auto_resolve" in body.lower()
                or "P0-AUDIT-2" in body)
    assert has_auto, (
        "P3-NEW-5 regresión: docstring no menciona el auto-resolve "
        "(P0-AUDIT-2). Sin esa referencia, el operador no entiende "
        "que la alert SE CIERRA sola si counter cae bajo umbral."
    )
