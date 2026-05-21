"""[P3-TRACEBACK-PRINT-EXC · 2026-05-15] Blanket regression guard.

Pre-fix: 8 callsites de `traceback.print_exc()` en código productivo
(agent.py, graph_orchestrator.py, routers/chat.py x3, routers/plans.py x3).
`traceback.print_exc()` emite el stack a stdout/stderr directamente — no
respeta `LOG_LEVEL`, no respeta Sentry sampling, compite con `logger.basicConfig`
format. El test `test_p3_pdf_polish_4.py:211` banea el patrón en un único
handler PDF, pero el resto del backend quedó sin enforcement.

Fix: reemplazo cada callsite por `logger.exception(...)` (o
`logger.warning(..., exc_info=True)` cuando se quiere preservar nivel WARNING).

Defensas que este test enforza:
  1. Anchor `P3-TRACEBACK-PRINT-EXC` presente en al menos uno de los archivos
     migrados (sirve para grep cross-incidente).
  2. CERO `traceback.print_exc()` en código productivo. Whitelist explícita
     para scripts one-shot, tests, refactor_plans.py (scratch).
  3. CERO `import traceback` "raw" en los archivos migrados — el módulo
     `traceback` solo debe importarse en utilidades que sí lo usan
     legítimamente (formatters internos, fallback handlers).
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent

# Archivos en los que se ejecutó la migración. Cada uno DEBE tener cero
# `traceback.print_exc()` post-fix.
_MIGRATED_PROD_FILES = [
    _BACKEND_ROOT / "agent.py",
    _BACKEND_ROOT / "graph_orchestrator.py",
    _BACKEND_ROOT / "routers" / "chat.py",
    _BACKEND_ROOT / "routers" / "plans.py",
]

# Whitelist: archivos donde `traceback.print_exc()` se acepta intencionalmente.
# Scripts one-shot, tests, refactor scratch.
#
# [P3-DEBUG-TIME-CLEANUP · 2026-05-20] `refactor.py` y `refactor_plans.py`
# movidos a `backend/scratch/legacy_root_helpers/` (audit gaps-audit-2026-05
# A1). El whitelist queda vacío pero preservamos el set como hook por si
# reaparece un caso legítimo — preferir marker inline en lugar de re-añadir.
_WHITELIST_PATHS: set = set()


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Anchor presente en al menos uno de los archivos migrados.
# ---------------------------------------------------------------------------
def test_anchor_present_in_at_least_one_migrated_file():
    found = False
    for p in _MIGRATED_PROD_FILES:
        if "P3-TRACEBACK-PRINT-EXC" in _read(p):
            found = True
            break
    assert found, (
        "Falta anchor `P3-TRACEBACK-PRINT-EXC` en alguno de los archivos "
        f"migrados ({[str(p.relative_to(_BACKEND_ROOT)) for p in _MIGRATED_PROD_FILES]}). "
        "Sin anchor, un futuro reader que vea `logger.exception` no sabrá "
        "el modo de fallo que cierra (stack a stdout sin sink configurado)."
    )


# ---------------------------------------------------------------------------
# 2. Cada archivo migrado tiene CERO `traceback.print_exc()`.
# ---------------------------------------------------------------------------
def test_migrated_files_have_no_traceback_print_exc():
    offenders = []
    for p in _MIGRATED_PROD_FILES:
        src = _read(p)
        # Buscar el patrón en LÍNEAS NO comentadas. Las menciones en
        # comentarios narrativos (que documentan el patrón legacy) están OK.
        for i, line in enumerate(src.splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if "traceback.print_exc()" in line:
                offenders.append(f"{p.relative_to(_BACKEND_ROOT)}:{i}: {line.strip()}")
    assert not offenders, (
        "P3-TRACEBACK-PRINT-EXC regresión: callsite(s) sin migrar:\n  "
        + "\n  ".join(offenders)
        + "\n\nReemplazar por `logger.exception(...)` (incluye stack + "
        "respeta LOG_LEVEL + Sentry sampling) o "
        "`logger.warning(..., exc_info=True)` si querés mantener nivel WARNING."
    )


# ---------------------------------------------------------------------------
# 3. Blanket scan: TODO archivo en backend/ (no test, no whitelist) debe
#    estar libre de `traceback.print_exc()`.
# ---------------------------------------------------------------------------
def test_no_traceback_print_exc_in_production_paths():
    offenders = []
    # Carpetas excluidas (sustring match — `test_venv`, `.venv-pkg`, etc.
    # caen bajo `venv` o `_venv`).
    _EXCLUDED_DIR_SUBSTRINGS = (
        "tests", "scripts", "venv", "_venv", ".venv", "__pycache__",
        "site-packages", "node_modules",
    )
    for py in _BACKEND_ROOT.rglob("*.py"):
        rel = py.relative_to(_BACKEND_ROOT)
        # Excluir cualquier subdir cuya parts incluya una de las substrings.
        if any(any(s in part for s in _EXCLUDED_DIR_SUBSTRINGS) for part in rel.parts):
            continue
        if py in _WHITELIST_PATHS:
            continue
        src = _read(py)
        for i, line in enumerate(src.splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if "traceback.print_exc()" in line:
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, (
        "Producción aún tiene `traceback.print_exc()`:\n  "
        + "\n  ".join(offenders)
        + "\n\nMigrar a `logger.exception(...)` o añadir a `_WHITELIST_PATHS` "
        "si es script one-shot."
    )
