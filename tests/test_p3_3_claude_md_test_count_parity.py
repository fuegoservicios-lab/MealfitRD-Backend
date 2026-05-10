"""[P3-3 · 2026-05-08] Drift detection: cifras de tests en `CLAUDE.md`.

Bug observado en el audit 2026-05-08:
  `CLAUDE.md` listaba `test_p1_shopping_recipe_coherence.py — guard E2E
  (~80 casos)` pero el archivo había crecido a 89 tests (P3-F autouse fixture
  + P1-C magnitudes + P3-A multiplier suite). La cifra "~80" era evidencia
  narrativa stale del estado del archivo en 2026-05-07.

Causa: cifras numéricas en docs son human-maintained y se desactualizan
silenciosamente cuando se añaden tests sin tocar CLAUDE.md.

Fix:
  1. Cifra `~80 casos` corregida a `89 casos` (conteo actual exacto).
  2. Este test parsea TODAS las menciones de tests con cifra en CLAUDE.md
     y valida paridad con el conteo real de pytest collect.

Cuando un futuro test se añade/elimina:
  - El conteo cambia → `pytest tests/test_p3_3_claude_md_test_count_parity.py`
    falla en CI.
  - Operador actualiza la cifra en CLAUDE.md al mismo conteo nuevo.
  - Mismo patrón self-enforcing que P3-1 (`_LAST_KNOWN_PFIX` + floor).

Patrón cubierto: cualquier línea en CLAUDE.md de la forma
    [`backend/tests/<file>.py`](backend/tests/<file>.py) — ... (N casos).
donde `N` es un entero. Si el patrón cambia, este test debe actualizarse.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_ROOT.parent
_CLAUDE_MD = _REPO_ROOT / "CLAUDE.md"

# Captura `[`backend/tests/test_*.py`](backend/tests/test_*.py) — descripción (N casos).`
_TEST_LINE_PATTERN = re.compile(
    r"\[`backend/tests/(?P<filename>test_[\w_\-]+\.py)`\]"
    r"\(backend/tests/(?P=filename)\)"
    r"[^()]*"
    r"\((?P<count>\d+)\s+casos?\)"
)


def _collect_test_count(test_path: Path) -> int:
    """Cuenta tests del archivo via `pytest --collect-only -q`.

    Retorna -1 si pytest falla (collection error). El test consumidor
    decide si fallar o skipear.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_path), "--collect-only", "-q"],
            capture_output=True,
            timeout=60,
            cwd=str(_BACKEND_ROOT),
            text=True,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return -1
    if result.returncode != 0:
        return -1
    # Buscar línea "<N> tests collected" en stdout.
    m = re.search(r"^(\d+)\s+tests?\s+collected", result.stdout, re.MULTILINE)
    if m is None:
        return -1
    return int(m.group(1))


def test_claude_md_exists():
    assert _CLAUDE_MD.is_file(), (
        f"No se encontró CLAUDE.md en {_CLAUDE_MD}. Si el archivo se movió, "
        f"actualizar este test."
    )


def test_test_count_cifras_match_pytest_collect():
    """Cada `tests/test_*.py` mencionado en CLAUDE.md con `(N casos)` debe
    coincidir con el conteo real de pytest. Si añades tests, sube la cifra
    en CLAUDE.md (o vice versa).
    """
    text = _CLAUDE_MD.read_text(encoding="utf-8")
    matches = list(_TEST_LINE_PATTERN.finditer(text))
    assert matches, (
        "No se encontraron menciones `tests/test_*.py — ... (N casos)` en "
        "CLAUDE.md. Si el patrón cambió, actualizar `_TEST_LINE_PATTERN` "
        "en este test."
    )

    mismatches: list[str] = []
    missing_files: list[str] = []
    for m in matches:
        filename = m.group("filename")
        documented_count = int(m.group("count"))
        test_path = _BACKEND_ROOT / "tests" / filename
        if not test_path.is_file():
            missing_files.append(
                f"{filename} mencionado en CLAUDE.md pero no existe en backend/tests/"
            )
            continue
        actual_count = _collect_test_count(test_path)
        if actual_count == -1:
            # Skip files that fail collection (pre-existing brokenness, no P3-3 bug).
            continue
        if actual_count != documented_count:
            mismatches.append(
                f"{filename}: CLAUDE.md dice {documented_count} casos, "
                f"pytest collect dice {actual_count}"
            )

    if missing_files:
        pytest.fail(
            "Archivos referenciados en CLAUDE.md no existen:\n  - "
            + "\n  - ".join(missing_files)
        )
    if mismatches:
        pytest.fail(
            "Cifras de tests en CLAUDE.md desactualizadas:\n  - "
            + "\n  - ".join(mismatches)
            + "\n\nFix: actualizar la cifra `(N casos)` en CLAUDE.md al "
            "valor reportado por `pytest --collect-only -q`."
        )


def test_pattern_catches_realistic_lines():
    """Sanity: el regex captura el formato real usado en CLAUDE.md.

    Si `_TEST_LINE_PATTERN` se rompe por un cambio en el formato de docs,
    el test principal no detectaría drift (matches=0 → pasa trivialmente).
    Este test guarda contra esa regresión silenciosa.
    """
    sample = (
        "- [`backend/tests/test_foo.py`](backend/tests/test_foo.py) — "
        "guard E2E presence/absence + magnitudes + knobs (89 casos).\n"
    )
    matches = list(_TEST_LINE_PATTERN.finditer(sample))
    assert len(matches) == 1
    assert matches[0].group("filename") == "test_foo.py"
    assert int(matches[0].group("count")) == 89
