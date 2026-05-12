"""[P2-AUDIT-1 · 2026-05-12] La raíz del repo NO debe contener archivos
scratch/junk que ensucien la navegación.

Contexto:
    Pre-audit, la raíz del repo MealfitRD.IA contenía:
      - 8 helpers one-off: `fix.py`, `patch.py`, `update.py`, `test_chuleta.py`,
        `test_db_pool.py`, `test_humanize.py`, `test_parse.py`,
        `test_pre_consolidate.py`
      - 2 outputs stale: `test_output.txt` (39 KB), `test_output_1.txt`
      - 1 doc legacy: `analysis_nevera_rotacion.md`
      - 1 test JS escapado: `test_shopping_regex.js`

    P2-AUDIT-1 los movió a `scratch/legacy_root_*/` o eliminó cuando eran
    regenerables. Este test bloquea regresión: nadie debe re-introducir
    helpers o tests al root sin justificarlo.

Lo que este test enforza:
  A) Patrones prohibidos en raíz (los moveidos al scratch).
  B) `.gitignore` raíz existe y bloquea los patrones a futuro.
  C) `scratch/README.md` documenta la convención.
  D) `backend/scripts/README.md` documenta los scripts operacionales.

Si necesitas un helper en raíz justificado (e.g., para tooling de CI), añadir
una excepción explícita en `.gitignore` y documentar acá por qué.

Tooltip-anchor: P2-AUDIT-1-REPO-ROOT-CLEAN.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRATCH_README = _REPO_ROOT / "scratch" / "README.md"
_BACKEND_SCRIPTS_README = _REPO_ROOT / "backend" / "scripts" / "README.md"
_GITIGNORE = _REPO_ROOT / ".gitignore"


# Patrones que el cleanup eliminó de la raíz. Si alguno reaparece, falla.
_PROHIBITED_ROOT_PATTERNS = [
    "fix.py",
    "patch.py",
    "update.py",
    "test_chuleta.py",
    "test_db_pool.py",
    "test_humanize.py",
    "test_parse.py",
    "test_pre_consolidate.py",
    "test_shopping_regex.js",
    "test_output.txt",
    "test_output_1.txt",
    "analysis_nevera_rotacion.md",
]


def test_a_root_does_not_contain_legacy_junk():
    """Patrones movidos a scratch/ NO deben existir en raíz."""
    violations = [
        name for name in _PROHIBITED_ROOT_PATTERNS
        if (_REPO_ROOT / name).exists()
    ]
    assert not violations, (
        "P2-AUDIT-1 regresión: archivos legacy/scratch reaparecieron en "
        "raíz del repo:\n  " + "\n  ".join(violations) + "\n\n"
        "Mover a `scratch/legacy_root_helpers/` o `scratch/legacy_root_tests/` "
        "según corresponda. Si necesitas el archivo en raíz, justificarlo "
        "en `scratch/README.md` y añadir excepción en `.gitignore`."
    )


def test_b_gitignore_exists_and_blocks_pattern():
    """`.gitignore` raíz debe existir + contener globs que prevengan
    regresión (no es backend/ ni frontend/ — la raíz no tenía .gitignore
    pre-audit)."""
    assert _GITIGNORE.exists(), (
        "P2-AUDIT-1: falta `.gitignore` en la raíz del repo. Sin él, "
        "git track-ea cualquier helper one-off colocado en root."
    )
    text = _GITIGNORE.read_text(encoding="utf-8")
    # Patrones críticos que el cleanup añadió.
    must_have = [
        ("test_output*.txt", "outputs stale de test"),
        ("lint_results.json", "output stale de lint"),
        ("/fix.py", "helper one-off"),
        ("/patch.py", "helper one-off"),
        ("/update.py", "helper one-off"),
        ("/test_*.py", "tests ad-hoc en raíz"),
        ("scratch/*", "scratch dir excepto README"),
        ("!scratch/README.md", "excepción para preservar el README"),
    ]
    missing = [(p, reason) for p, reason in must_have if p not in text]
    assert not missing, (
        "P2-AUDIT-1: `.gitignore` raíz no contiene los patrones requeridos:\n"
        + "\n".join(f"  - `{p}` ({r})" for p, r in missing)
    )


def test_c_scratch_readme_exists():
    assert _SCRATCH_README.exists(), (
        "P2-AUDIT-1: falta `scratch/README.md`. Sin él, futuros mantenedores "
        "no saben qué hay en `scratch/` y por qué (los archivos están "
        "gitignored excepto el README, que es la única señal trackeada)."
    )
    text = _SCRATCH_README.read_text(encoding="utf-8")
    assert "P2-AUDIT-1" in text, (
        "P2-AUDIT-1: `scratch/README.md` perdió el anchor — refactor "
        "perdió el contexto del cleanup."
    )


def test_d_backend_scripts_readme_exists():
    assert _BACKEND_SCRIPTS_README.exists(), (
        "P2-AUDIT-1: falta `backend/scripts/README.md`. Sin él, los scripts "
        "operacionales (check_scaling.py, etc.) no tienen documentación de "
        "uso ni convenciones (no DELETE/UPDATE sin user_id filtro, etc.)."
    )
    text = _BACKEND_SCRIPTS_README.read_text(encoding="utf-8")
    # Debe mencionar al menos uno de los scripts movidos en P2-AUDIT-1.
    assert "check_scaling.py" in text, (
        "P2-AUDIT-1: `backend/scripts/README.md` no menciona `check_scaling.py` "
        "(el reemplazo del endpoint debug eliminado en P1-AUDIT-NEW-1). Sin "
        "esta doc, un nuevo SRE no sabe que existe el script."
    )


def test_e_no_bak_files_in_backend_root():
    """`backend/_deprecated_*.py.bak` debían eliminarse — confirmar que
    ninguno quedó atrás."""
    backend_root = _REPO_ROOT / "backend"
    bak_files = list(backend_root.glob("_deprecated_*.py.bak"))
    bak_files.extend(backend_root.glob("*.py.orphan_review_*.bak"))
    assert not bak_files, (
        f"P2-AUDIT-1: archivos `.bak` legacy reaparecieron en backend/:\n"
        f"  " + "\n  ".join(str(f.relative_to(_REPO_ROOT)) for f in bak_files)
        + "\n\nDeben eliminarse — están gitignored (`*.bak`) pero ensucian "
        "el workdir. Si necesitas preservar el contenido, mover a "
        "`scratch/legacy_backend_orphans/`."
    )


def test_f_misplaced_backend_test_moved():
    """`backend/test_flexible_mode_pantry_safety.py` era un duplicado
    desactualizado (96 vs 1133 lineas) del que vive en `backend/tests/`.
    Confirmar que la versión orfana NO esté en `backend/` (debe estar en
    `scratch/legacy_backend_orphans/`)."""
    orphan = _REPO_ROOT / "backend" / "test_flexible_mode_pantry_safety.py"
    proper = _REPO_ROOT / "backend" / "tests" / "test_flexible_mode_pantry_safety.py"
    assert not orphan.exists(), (
        "P2-AUDIT-1: orfano `backend/test_flexible_mode_pantry_safety.py` "
        "reapareció — es duplicado desactualizado (96 lineas vs 1133 del "
        "de tests/). Mover a `scratch/legacy_backend_orphans/`."
    )
    assert proper.exists(), (
        "P2-AUDIT-1 sanity: `backend/tests/test_flexible_mode_pantry_safety.py` "
        "desapareció. Este es el test REAL — recuperarlo del historial git "
        "o del backup `scratch/legacy_backend_orphans/`."
    )
