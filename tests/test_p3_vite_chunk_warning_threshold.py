"""[P3-VITE-CHUNK-WARNING-THRESHOLD · 2026-05-15] Anchor + regression guard.

Pre-fix: `chunkSizeWarningLimit: 500` (KB) en `frontend/vite.config.js`.
Es el default de Vite — un chunk de 499 KB pasa silencioso. El bundle ya
tiene chunks lazy intencionalmente grandes (html2pdf-*.js ~976 KB, P2-LAZY-PDF)
y eso está OK, pero un import accidentado puede colar 800 KB en un entry
chunk sin warning.

Fix: bajar el cap a 300 KB. Captura regresiones de entry chunks que crecen
accidentalmente. Los chunks intencionalmente lazy seguirán emitiendo
warning (ruido conocido) — la señal útil es cuando aparece un NUEVO chunk
> 300 KB que NO se esperaba.

Defensas que este test enforza:
  1. Anchor `P3-VITE-CHUNK-WARNING-THRESHOLD` presente en vite.config.js.
  2. `chunkSizeWarningLimit` <= 300 KB.
"""
from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_VITE_CONFIG = _REPO_ROOT / "frontend" / "vite.config.js"


def _read() -> str:
    return _VITE_CONFIG.read_text(encoding="utf-8")


def test_anchor_present_in_vite_config():
    src = _read()
    assert "P3-VITE-CHUNK-WARNING-THRESHOLD" in src, (
        "Falta anchor `P3-VITE-CHUNK-WARNING-THRESHOLD` en vite.config.js."
    )


def test_chunk_size_warning_limit_at_or_below_300kb():
    """El cap DEBE ser <= 300 KB para capturar regresiones. Si futuras
    optimizaciones bajan más el threshold (200 / 150), también pasa.
    Si alguien lo sube de vuelta a 500, falla."""
    src = _read()
    m = re.search(r"chunkSizeWarningLimit\s*:\s*(\d+)", src)
    assert m is not None, "No se encontró `chunkSizeWarningLimit: <N>` en vite.config.js."
    value = int(m.group(1))
    assert value <= 300, (
        f"`chunkSizeWarningLimit={value}` excede el cap del fix "
        f"(P3-VITE-CHUNK-WARNING-THRESHOLD reduce a <= 300). "
        f"Si necesitás más, justificá en comentario inline + actualizá este test."
    )


def test_chunk_size_warning_limit_not_disabled():
    """Si alguien pone `chunkSizeWarningLimit: 0` o `Infinity`, el cap
    queda efectivamente desactivado. Bloquear esos casos: el cap debe
    estar entre [1, 300] KB (1 KB es absurdo pero formalmente válido;
    el valor canónico actual es 300)."""
    src = _read()
    m = re.search(r"chunkSizeWarningLimit\s*:\s*(\d+)", src)
    assert m is not None
    value = int(m.group(1))
    assert value >= 1, (
        f"`chunkSizeWarningLimit={value}` desactiva el warning (debe ser >= 1)."
    )
