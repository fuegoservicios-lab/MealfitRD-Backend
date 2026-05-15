"""[P3-MARKER-MEMORY-XLINK · 2026-05-15] Cross-link entre `_LAST_KNOWN_PFIX`
en `app.py` y la presencia del marker en el índice `MEMORY.md` externo.

Modo de fallo que este test cierra (audit 2026-05-15):
    Dos P-fixes (`P3-FACT-EXTRACTOR-SHADOW-AB`, `P1-18 + UNIFICATION`) se
    cerraron con: (a) commit en `backend/`, (b) test de regresión en
    `backend/tests/`, (c) marker bumpeado en `_LAST_KNOWN_PFIX` — pero
    SIN entrada narrativa en `MEMORY.md`. Resultado: futuros audits no
    pueden reconstruir la decisión sin leer el código línea por línea.

    El test `P2-HIST-AUDIT-14` enforza (a)+(b)+(c) → existencia de
    `tests/test_<slug>*.py`. NO enforza (d) → entrada en `MEMORY.md`.
    Este test cierra ese gap específico.

Limitaciones (intencionales):
    - `MEMORY.md` vive fuera del repo (`~/.claude/projects/.../memory/`)
      y NO es accesible desde un runner CI estándar. El test SKIPea
      cuando no encuentra el archivo — es defense para el developer
      local que corre `pytest backend/tests/` antes de commit, no para CI.
    - El default path es Windows-specific (matches la máquina del
      developer principal). Override via env var `CLAUDE_MEMORY_INDEX_PATH`.
    - NO enforza la calidad ni longitud de la entrada — solo presencia
      del prefijo del marker como substring.

Cómo activar en CI (futuro):
    Setear `CLAUDE_MEMORY_INDEX_PATH` apuntando a un checkout/copia del
    `MEMORY.md` accesible al runner. Sin esto, el test queda en modo
    skip-friendly y no rompe builds.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_APP_PY = _BACKEND_ROOT / "app.py"
_DEFAULT_MEMORY_INDEX = Path(
    os.path.expanduser(
        "~/.claude/projects/C--Users-angel-OneDrive-Escritorio-MealfitRD-IA/memory/MEMORY.md"
    )
)


def _read_marker_value() -> str:
    text = _APP_PY.read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*[\'"]([^\'"]+)[\'"]', text)
    assert m is not None, "_LAST_KNOWN_PFIX no encontrado en app.py"
    return m.group(1)


def _marker_prefix(marker: str) -> str:
    """`P3-FACT-EXTRACTOR-SHADOW-AB · 2026-05-14` → `P3-FACT-EXTRACTOR-SHADOW-AB`."""
    return marker.split("·", 1)[0].strip()


def _resolve_memory_path() -> Path | None:
    raw = os.environ.get("CLAUDE_MEMORY_INDEX_PATH")
    if raw:
        p = Path(os.path.expanduser(raw))
        return p if p.exists() else None
    return _DEFAULT_MEMORY_INDEX if _DEFAULT_MEMORY_INDEX.exists() else None


def test_marker_appears_in_memory_index():
    """El prefijo del marker `_LAST_KNOWN_PFIX` debe aparecer en `MEMORY.md`.

    Skipea si `MEMORY.md` no es accesible (CI estándar, contributor sin
    acceso al directorio personal del developer principal). Falla local
    del developer cuando bumpean el marker sin actualizar la narrativa.
    """
    memory_path = _resolve_memory_path()
    if memory_path is None:
        pytest.skip(
            "MEMORY.md no accesible. Setea CLAUDE_MEMORY_INDEX_PATH si querés "
            "activar el guard en este entorno."
        )

    marker = _read_marker_value()
    prefix = _marker_prefix(marker)

    content = memory_path.read_text(encoding="utf-8")
    assert prefix in content, (
        f"Marker {prefix!r} (extraído de _LAST_KNOWN_PFIX en app.py) NO aparece "
        f"en {memory_path}.\n"
        f"Antes de commit:\n"
        f"  1. Escribe archivo de detalle en "
        f"`~/.claude/projects/.../memory/project_{prefix.lower().replace('-', '_')}_*.md`\n"
        f"  2. Añade una línea al índice MEMORY.md referenciándolo.\n"
        f"Este test cierra el gap 'marker bumpeado sin doc-pass' detectado en "
        f"audit 2026-05-15 (P3-FACT-EXTRACTOR-SHADOW-AB y P1-18 + UNIFICATION)."
    )


def test_marker_prefix_extraction_regression():
    """Sanity check: el helper `_marker_prefix` no se rompe con el formato
    canónico `Pn-X-Y · YYYY-MM-DD`. Si la convención cambia, este test y
    `test_p3_1_last_known_pfix_freshness` deben actualizarse a la vez."""
    assert _marker_prefix("P3-FACT-EXTRACTOR-SHADOW-AB · 2026-05-14") == "P3-FACT-EXTRACTOR-SHADOW-AB"
    assert _marker_prefix("P1-18 + UNIFICATION · 2026-05-14") == "P1-18 + UNIFICATION"
    assert _marker_prefix("P0-1 · 2026-01-01") == "P0-1"
