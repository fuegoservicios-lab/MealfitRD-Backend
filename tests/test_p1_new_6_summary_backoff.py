"""[P1-NEW-6 · 2026-05-11] Backoff exponencial en
`memory_manager.summarize_and_prune` para evitar quemar API calls
de Gemini cuando hay racha consecutiva de fallos.

Bug original (audit 2026-05-11):
    Pre-fix: si Gemini estaba down (cuota / API key rotada / modelo
    eliminado), el cron invocaba `summarize_and_prune` cada N min ×
    M sesiones activas. Cada llamada consumía API tokens (cost),
    saturaba el pool de threads (otros crons MISSED), y llenaba
    logs sin valor.

    El threshold P1-19 (>=5 fallos) emite alert pero no detiene
    futuras invocaciones.

Fix:
    `_summary_backoff_should_skip()` retorna True cuando el contador
    cruza umbrales de `_SUMMARY_BACKOFF_TABLE` y el tiempo desde el
    último intento es menor que el wait calculado.

Tabla de backoff:
    count 1-4:   no skip
    count 5-9:   skip si < 10min desde last_attempt
    count 10-19: skip si < 30min
    count 20-39: skip si < 1h
    count >=40:  skip si < 4h

Estrategia del test (mix parser + behavior):
    1. Función `_summary_backoff_should_skip` definida.
    2. Tabla `_SUMMARY_BACKOFF_TABLE` con los 4 thresholds.
    3. Lee knob kill-switch `MEALFIT_SUMMARY_BACKOFF_ENABLED`.
    4. `summarize_and_prune` invoca el guard al inicio.
    5. Behavior: con count=0 → False; con count=5 + ts reciente → True;
       con count=5 + ts > 10min → False.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

import memory_manager


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MM_FP = _REPO_ROOT / "backend" / "memory_manager.py"


@pytest.fixture(scope="module")
def src() -> str:
    return _MM_FP.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Parser-based: estructura
# ---------------------------------------------------------------------------
def test_should_skip_function_defined(src: str):
    """`_summary_backoff_should_skip` debe existir top-level."""
    assert re.search(
        r"^def\s+_summary_backoff_should_skip\s*\(",
        src,
        re.MULTILINE,
    ), "P1-NEW-6 regresión: `_summary_backoff_should_skip` no existe."


def test_backoff_table_defined(src: str):
    """`_SUMMARY_BACKOFF_TABLE` debe estar declarada con al menos 4
    umbrales (5/10/20/40 fallos)."""
    assert "_SUMMARY_BACKOFF_TABLE" in src, (
        "P1-NEW-6 regresión: `_SUMMARY_BACKOFF_TABLE` no existe. "
        "Sin tabla, el wait time es fijo o ausente."
    )
    # Esperar tuplas (threshold, wait_seconds).
    pattern = re.compile(
        r"_SUMMARY_BACKOFF_TABLE[^=]*=\s*\[(.*?)\]",
        re.DOTALL,
    )
    m = pattern.search(src)
    assert m, "Tabla no parseable."
    body = m.group(1)
    # Esperar 4 tuplas (al menos 5, 10, 20, 40).
    threshold_count = len(re.findall(r"\(\s*\d+\s*,", body))
    assert threshold_count >= 4, (
        f"P1-NEW-6 regresión: solo {threshold_count} thresholds en la "
        "tabla (esperado ≥4 para curva 5/10/20/40)."
    )


def test_kill_switch_knob(src: str):
    """Knob `MEALFIT_SUMMARY_BACKOFF_ENABLED` debe leerse — kill
    switch sin redeploy."""
    assert "MEALFIT_SUMMARY_BACKOFF_ENABLED" in src, (
        "P1-NEW-6 regresión: knob kill-switch removido. Sin él, no "
        "se puede desactivar el backoff en producción para debug."
    )


def test_summarize_and_prune_invokes_guard(src: str):
    """`summarize_and_prune` debe llamar `_summary_backoff_should_skip()`
    al inicio (antes de adquirir lock o invocar Gemini)."""
    m = re.search(
        r"^def\s+summarize_and_prune\s*\([^)]*\)[^:]*:(.*?)(?=^def\s)",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert m, "summarize_and_prune no encontrada"
    body = m.group(1)
    skip_idx = body.find("_summary_backoff_should_skip()")
    acquire_idx = body.find("acquire_summarizing_lock")
    assert skip_idx >= 0, (
        "P1-NEW-6 regresión: summarize_and_prune no invoca el guard."
    )
    assert skip_idx < acquire_idx, (
        "P1-NEW-6 regresión: guard se invoca DESPUÉS de adquirir el "
        "lock. Debe ser ANTES — un skip exitoso debe ser cheap."
    )


# ---------------------------------------------------------------------------
# Behavior tests
# ---------------------------------------------------------------------------
def test_no_skip_when_zero_failures(monkeypatch):
    """count=0 → no skip (path feliz)."""
    monkeypatch.setattr(
        memory_manager, "_summarize_failures",
        {"count": 0, "last_error": None, "last_attempt_at": 0.0},
    )
    assert memory_manager._summary_backoff_should_skip() is False


def test_no_skip_below_first_threshold(monkeypatch):
    """count=4 (< 5) → no skip (todavía no entra el backoff)."""
    monkeypatch.setattr(
        memory_manager, "_summarize_failures",
        {"count": 4, "last_error": "err", "last_attempt_at": time.time()},
    )
    assert memory_manager._summary_backoff_should_skip() is False


def test_skip_when_count_5_and_recent(monkeypatch):
    """count=5 + last_attempt hace 1 minuto → skip (wait=10min)."""
    monkeypatch.setattr(
        memory_manager, "_summarize_failures",
        {"count": 5, "last_error": "err", "last_attempt_at": time.time() - 60},
    )
    assert memory_manager._summary_backoff_should_skip() is True


def test_no_skip_when_wait_elapsed(monkeypatch):
    """count=5 + last_attempt hace 11 minutos → no skip (>10min)."""
    monkeypatch.setattr(
        memory_manager, "_summarize_failures",
        {"count": 5, "last_error": "err", "last_attempt_at": time.time() - 11 * 60},
    )
    assert memory_manager._summary_backoff_should_skip() is False


def test_high_count_longer_wait(monkeypatch):
    """count=40 + last_attempt hace 30 min → skip (wait=4h)."""
    monkeypatch.setattr(
        memory_manager, "_summarize_failures",
        {"count": 40, "last_error": "err", "last_attempt_at": time.time() - 30 * 60},
    )
    assert memory_manager._summary_backoff_should_skip() is True


def test_kill_switch_off_forces_no_skip(monkeypatch):
    """Knob `MEALFIT_SUMMARY_BACKOFF_ENABLED=false` → no skip
    independiente del contador."""
    monkeypatch.setenv("MEALFIT_SUMMARY_BACKOFF_ENABLED", "false")
    monkeypatch.setattr(
        memory_manager, "_summarize_failures",
        {"count": 100, "last_error": "err", "last_attempt_at": time.time()},
    )
    assert memory_manager._summary_backoff_should_skip() is False


def test_failures_dict_has_last_attempt_at_field():
    """El dict `_summarize_failures` debe incluir `last_attempt_at`
    como key — sin él, el guard no puede calcular `elapsed`."""
    assert "last_attempt_at" in memory_manager._summarize_failures, (
        "P1-NEW-6 regresión: `_summarize_failures` no tiene "
        "`last_attempt_at`. Sin ese timestamp, el backoff no funciona."
    )
