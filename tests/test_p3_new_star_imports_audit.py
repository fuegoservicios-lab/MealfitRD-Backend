"""[P3-NEW-STAR-IMPORTS-AUDIT · 2026-05-15] Guard contra name conflicts en
los 6 `from db_* import *` de `backend/db.py`.

`db.py` es un facade gateway:

    from db_core import *
    from db_profiles import *
    from db_chat import *
    from db_plans import *
    from db_facts import *
    from db_inventory import *

El `import *` colapsa el namespace; si dos módulos exportan un símbolo top-
level con el mismo nombre (función o clase pública), el último import gana
silenciosamente. Modo de fallo: refactor en `db_plans.py` que mueve un
helper a `db_core.py` con el mismo nombre — por orden de import el `db_core`
gana, pero el call site en otro módulo que esperaba el `db_plans` ahora
ejecuta el `db_core` (semántica diferente, sin error). Difícil de
diagnosticar.

Refactor a imports explícitos es invasivo (enumerar 100+ exports). Como
defense-in-depth, este guard parsea cada `db_*.py` con AST, extrae los
top-level names PÚBLICOS (función/clase, no `_` prefijo), y falla si
algún nombre aparece en >1 módulo.

Si un conflicto legítimo se introduce (ej. símbolo re-exportado a
propósito), añadir entry a `_KNOWN_DUPLICATES_WHITELIST` con razón.

Defensas que el test enforza:
  1. Anchor `P3-NEW-STAR-IMPORTS-AUDIT` presente en `db.py`.
  2. Cero name conflicts entre los 6 módulos `db_*` (excepto whitelist).
"""

from __future__ import annotations

import ast
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _REPO_ROOT / "backend"
_DB_PY = _BACKEND / "db.py"

_DB_MODULES = (
    "db_core",
    "db_profiles",
    "db_chat",
    "db_plans",
    "db_facts",
    "db_inventory",
)

# Símbolos legítimamente compartidos entre módulos (re-exports intencionales,
# names genéricos triviales). Añadir aquí cuando un conflicto sea by-design.
_KNOWN_DUPLICATES_WHITELIST: set[str] = set()


def _public_top_level_names(path: Path) -> set[str]:
    """Extrae nombres de funciones/clases top-level no privadas (sin `_` prefijo).

    Excluye:
      - Nombres con `_` prefijo (private convention).
      - Variables de módulo (constants).
      - Imports (no son del módulo).
    """
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
    return names


def test_anchor_present_in_db_py():
    src = _DB_PY.read_text(encoding="utf-8")
    assert "P3-NEW-STAR-IMPORTS-AUDIT" in src, (
        "Falta anchor `P3-NEW-STAR-IMPORTS-AUDIT` en backend/db.py. "
        "Sin él, futuros readers no entienden por qué este test guard existe."
    )


def test_no_name_conflicts_between_db_modules():
    """Cero conflictos de nombres top-level entre los 6 módulos `db_*`.

    Si este test falla, el último `from db_X import *` en db.py gana — un
    bug silencioso esperando a manifestarse. Resolver:
      a) Renombrar el símbolo en uno de los módulos para hacerlo único.
      b) Si el conflicto es by-design (re-export a propósito), añadir
         a `_KNOWN_DUPLICATES_WHITELIST` con comment de razón.
    """
    module_names: dict[str, set[str]] = {}
    for mod in _DB_MODULES:
        path = _BACKEND / f"{mod}.py"
        if not path.exists():
            # Defensivo: si un módulo desapareció, fallar con mensaje claro.
            raise AssertionError(
                f"Módulo `{mod}.py` no existe en backend/. "
                f"Si fue removido, actualizar `_DB_MODULES` y `db.py` import."
            )
        module_names[mod] = _public_top_level_names(path)

    # Construir mapa nombre → módulos que lo definen.
    name_to_modules: dict[str, list[str]] = {}
    for mod, names in module_names.items():
        for name in names:
            name_to_modules.setdefault(name, []).append(mod)

    conflicts = {
        name: mods
        for name, mods in name_to_modules.items()
        if len(mods) > 1 and name not in _KNOWN_DUPLICATES_WHITELIST
    }

    assert not conflicts, (
        f"Name conflicts entre módulos `db_*` (cada nombre definido en >1 "
        f"módulo, último import en db.py gana silenciosamente):\n  "
        + "\n  ".join(f"`{n}` → {mods}" for n, mods in sorted(conflicts.items()))
        + f"\n\nResolución: renombrar símbolo o whitelist explícito."
    )


def test_db_py_imports_all_listed_modules():
    """Sanity: db.py debe importar TODOS los módulos listados en
    `_DB_MODULES`. Si alguien remueve un import, este test falla loud
    (y el name conflict check arriba pierde cobertura)."""
    src = _DB_PY.read_text(encoding="utf-8")
    missing = []
    for mod in _DB_MODULES:
        if f"from {mod} import *" not in src:
            missing.append(mod)
    assert not missing, (
        f"db.py no importa con `from <mod> import *` los módulos: {missing}. "
        f"Si fue intencional (refactor a imports explícitos), actualizar "
        f"`_DB_MODULES` arriba."
    )


def test_anchor_present_in_test_file():
    src = Path(__file__).read_text(encoding="utf-8")
    assert "P3-NEW-STAR-IMPORTS-AUDIT" in src
