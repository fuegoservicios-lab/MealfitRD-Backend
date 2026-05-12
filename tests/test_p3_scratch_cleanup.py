"""[P3-SCRATCH-CLEANUP · 2026-05-13] Regression guard: scratch dev-scripts
NO deben reaparecer en `backend/` root.

Contexto del audit production-readiness 2026-05-12:
    `backend/scratch_create_user.py` (14 LOC, insert ad-hoc en auth.users
    + user_profiles para testing manual) y `backend/scratch_get_tables.py`
    (11 LOC, lista tablas via information_schema) vivían en el root del
    backend desde hacía meses. Cero callers en código de producción. La
    convención del repo (test_p2_logger_migration KNOWN_PRINT_EXEMPT_PATHS)
    los whitelisteaba para permitir `print(...)`, pero su mera presencia
    polucionaba `ls backend/` y confundía a un nuevo lector sobre qué es
    código productivo vs dev-script.

Fix (este commit):
    `git rm backend/scratch_create_user.py backend/scratch_get_tables.py`
    + remover whitelist entries en test_p2_logger_migration.

Si necesitas un script ad-hoc en el futuro, ponlo en una de estas dos
ubicaciones (ambas EXCLUIDAS de los paths de producción del logger test):
    - `backend/scratch/`  → trabajo descartable, no merger
    - `backend/scripts/`  → utilities con valor permanente (publish_pfix_marker, etc.)

Tooltip-anchor: P3-SCRATCH-CLEANUP.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"

# Lista de scratch files que fueron eliminados en P3-SCRATCH-CLEANUP. Si
# alguno reaparece (e.g. alguien copia el viejo del git history sin leer
# la convención), este test falla y obliga a reubicar.
_FORBIDDEN_AT_BACKEND_ROOT = (
    "scratch_create_user.py",
    "scratch_get_tables.py",
)


def test_no_scratch_files_at_backend_root():
    """Bloquea re-introducción de scratch_*.py en backend/ root."""
    reappeared = [
        name for name in _FORBIDDEN_AT_BACKEND_ROOT
        if (BACKEND_ROOT / name).exists()
    ]
    assert not reappeared, (
        f"[P3-SCRATCH-CLEANUP] scratch files reaparecieron en backend/ root: "
        f"{reappeared}.\n"
        "Si necesitas un dev-script ad-hoc, ubícalo en:\n"
        "  - backend/scratch/  (trabajo descartable)\n"
        "  - backend/scripts/  (utility permanente — añadir a path_validators si necesario)\n"
        "Ambas ubicaciones están excluidas del logger production check.\n"
        "NO añadir whitelist en test_p2_logger_migration.KNOWN_PRINT_EXEMPT_PATHS "
        "para re-permitir scratch en root — el cleanup fue deliberado."
    )


def test_no_other_scratch_pattern_at_backend_root():
    """Patrón general: NINGÚN `scratch*.py` o `scratch_*.py` debe estar
    en backend/ root. Cierra el gap si alguien crea `scratch_new_thing.py`
    pensando que solo los 2 históricos están prohibidos.
    """
    scratch_files = sorted(
        p.name for p in BACKEND_ROOT.glob("scratch*.py")
        if p.is_file()
    )
    assert not scratch_files, (
        f"[P3-SCRATCH-CLEANUP] Archivos con prefix `scratch` detectados en "
        f"backend/ root: {scratch_files}. Mover a backend/scratch/ o "
        f"backend/scripts/ según convención."
    )
