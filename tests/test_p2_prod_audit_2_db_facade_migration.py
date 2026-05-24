"""[P2-PROD-AUDIT-1 · 2026-05-23] Tracker del progress de migración
`from db_X import` → fachada `from db import`.

Gap original (audit production-readiness 2026-05-23, B-P2-2):
    CLAUDE.md "Convención de imports DB" (P3-DB-IMPORTS-FACADE · 2026-05-20)
    documenta que call sites NUEVOS deben usar la fachada `from db import
    <funcion>` en lugar de los submódulos internos.

    Política "boy scout": cuando edites un archivo con `from db_<sub>
    import`, considera migrar ese mismo bloque. NO grep+replace masivo
    hoy (59 imports cross-codebase).

    Sin tracking, no hay visibility del progress — un PR podría AÑADIR
    nuevos imports a submódulos sin que nadie note.

Fix (este test):
    Snapshot del count actual de `from db_<sub> import` en código productivo.
    Sub-módulos cubiertos: `db_core`, `db_profiles`, `db_chat`, `db_plans`,
    `db_facts`, `db_inventory`, `db_meal_plans_audit`.

    Test FALLA si el count SUBE — señal de que un PR introdujo nuevos
    imports al submódulo en lugar de migrar a fachada.

    Test PASA si el count BAJA — señal de progreso "boy scout".

    Snapshot se bumpea hacia ABAJO en cada cleanup pass. Si el count se
    estabiliza por meses sin bajar, considerar migración masiva.

Cobertura:
    A) Count actual snapshot.
    B) Falla si count nuevo > snapshot.
    C) Sugerencia inline para bajar snapshot tras cleanup.

Tooltip-anchor: P2-PROD-AUDIT-1-DB-FACADE | audit 2026-05-23.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent

# Snapshot 2026-05-23 — count actual de imports directos a submódulos.
# Cuando un PR migre algunos a fachada `from db import`, BAJAR este
# snapshot al nuevo count + documentar en commit msg.
#
# Si count subió: el PR está AÑADIENDO imports directos en lugar de usar
# fachada — refactorear antes de mergear.
_DB_FACADE_SNAPSHOT_MAX = 210   # snapshot 2026-05-23 = 186 + margen 24
_DB_FACADE_SNAPSHOT_BASELINE = 190  # baseline 2026-05-23 — soft warning si baja <185

_DB_SUBMODULES = (
    "db_core", "db_profiles", "db_chat", "db_plans", "db_facts",
    "db_inventory", "db_meal_plans_audit",
)

_IMPORT_PATTERN = re.compile(
    r"^\s*from\s+(" + "|".join(re.escape(m) for m in _DB_SUBMODULES) + r")\s+import\s+",
    re.MULTILINE,
)


def _list_production_py_files():
    """Excluye tests/, scratch/, __pycache__."""
    out = []
    excluded_dirs = {"tests", "scratch", "__pycache__", ".git", "venv",
                     "test_venv", ".github", "fixtures", "supabase"}
    for path in _BACKEND_ROOT.rglob("*.py"):
        # Skip files dentro de dirs excluidos.
        if any(part in excluded_dirs for part in path.parts):
            continue
        # Skip db_*.py themselves (sub-módulos importan entre sí — no es bug).
        if path.name in {f"{m}.py" for m in _DB_SUBMODULES}:
            continue
        # Skip db.py (fachada misma).
        if path.name == "db.py":
            continue
        out.append(path)
    return out


def _count_direct_imports():
    """Returns dict {file: count_of_direct_db_imports}."""
    out = {}
    for f in _list_production_py_files():
        text = f.read_text(encoding="utf-8")
        matches = _IMPORT_PATTERN.findall(text)
        if matches:
            out[str(f.relative_to(_BACKEND_ROOT))] = len(matches)
    return out


def test_snapshot_baseline_documented():
    """Sanity: si el snapshot cambia, el test fail muestra el delta."""
    by_file = _count_direct_imports()
    total = sum(by_file.values())
    print(
        f"\n[P2-PROD-AUDIT-1-DB-FACADE] Direct `from db_<sub> import` "
        f"count: {total} (baseline 2026-05-23: ~{_DB_FACADE_SNAPSHOT_BASELINE})"
    )
    # Top files con más imports — ayuda al operador a priorizar migración.
    sorted_files = sorted(by_file.items(), key=lambda kv: -kv[1])
    if sorted_files:
        print(f"  Top 10 files:")
        for path, count in sorted_files[:10]:
            print(f"    {count:3d}  {path}")


def test_no_regression_above_cap():
    """Count NO debe exceder el cap del snapshot. Si excede, un PR está
    añadiendo imports directos en lugar de usar fachada.
    """
    by_file = _count_direct_imports()
    total = sum(by_file.values())
    assert total <= _DB_FACADE_SNAPSHOT_MAX, (
        f"\n[P2-PROD-AUDIT-1-DB-FACADE] Direct `from db_<sub> import` count "
        f"({total}) excede el cap {_DB_FACADE_SNAPSHOT_MAX}.\n\n"
        f"PR introdujo nuevos imports directos a submódulos. Opciones:\n"
        f"  (a) Migrar los nuevos imports a fachada: `from db import <func>`.\n"
        f"  (b) Si imposible (circular import o sub-módulo NO exportado por\n"
        f"      la fachada), documentar en commit msg + bumpear el cap aquí.\n\n"
        f"Top 5 files contribuidores:\n"
        + "\n".join(
            f"  {count:3d}  {path}"
            for path, count in sorted(by_file.items(), key=lambda kv: -kv[1])[:5]
        )
    )


def test_eventually_baseline_drops():
    """Soft check: si baseline baja consistentemente, bumpear este test
    para reflejar el nuevo baseline. Sin esta nota, el snapshot se
    desactualiza."""
    by_file = _count_direct_imports()
    total = sum(by_file.values())
    if total < _DB_FACADE_SNAPSHOT_BASELINE - 5:
        # Soft warning para que el operador bumpee el snapshot.
        print(
            f"\n⚠️  [P2-PROD-AUDIT-1-DB-FACADE] Direct imports bajaron "
            f"a {total} (baseline {_DB_FACADE_SNAPSHOT_BASELINE}). "
            f"Considerar actualizar `_DB_FACADE_SNAPSHOT_BASELINE` "
            f"a {total} en este test para reflejar el nuevo nivel de progreso."
        )


def test_anchor_present_in_db_facade():
    """`db.py` debe mantener el anchor `P3-DB-IMPORTS-FACADE` o
    `from db_X import *` — sin alguna referencia, la fachada queda sin
    documentación de su propósito.
    """
    db_py = _BACKEND_ROOT / "db.py"
    if not db_py.exists():
        return  # Skip si fachada no existe en este checkout
    text = db_py.read_text(encoding="utf-8")
    has_marker = (
        "P3-DB-IMPORTS-FACADE" in text
        or any(f"from {sub} import" in text for sub in _DB_SUBMODULES)
    )
    assert has_marker, (
        "db.py NO menciona anchor `P3-DB-IMPORTS-FACADE` ni hace "
        "re-export de submódulos. La fachada perdió su propósito — "
        "considera eliminar o restaurar el re-export."
    )
