"""[P2-NEW-2 · 2026-05-10] `MEALFIT_INVENTORY_RPC_STRICT` debe estar
pre-registrado en `_KNOBS_REGISTRY` al import-time de `db_inventory`
(no solo cuando el path patológico ejecuta).

Bug original (audit 2026-05-10):
    P1-NEW-1 introdujo el knob pero lo leía solo dentro del except block
    del fallback RPC. Eso significa que el knob NUNCA aparecía en
    `/admin/knobs` ni en `/health/version` hasta que el except path
    ejecutara — exactamente lo opuesto de lo que necesita un operador
    en triage ("¿strict está on en prod?").

Fix:
    `db_inventory.py` ejecuta `_env_bool("MEALFIT_INVENTORY_RPC_STRICT",
    False)` al module top-level — el valor se persiste en
    `_KNOBS_REGISTRY` (vía side-effect del helper) desde el primer
    import del módulo. La lectura dentro del except sigue funcionando
    (idempotente, retorna del registry tras la primera vez).

Estrategia del test:
    1. Verificar parser-based que el módulo tiene el call al import-time
       (no solo dentro del except).
    2. Behavior test: importar `get_knobs_registry_snapshot` y
       verificar que la entry está presente.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DB_INVENTORY_PY = _REPO_ROOT / "backend" / "db_inventory.py"


@pytest.fixture(scope="module")
def db_inventory_src() -> str:
    return _DB_INVENTORY_PY.read_text(encoding="utf-8")


def test_strict_knob_read_at_module_load(db_inventory_src: str):
    """El call `_env_bool("MEALFIT_INVENTORY_RPC_STRICT", ...)` debe
    aparecer al module-top-level (no dentro de ningún `def`), para que
    el side-effect del registro ocurra al import-time.

    Estrategia (AST): parsear el módulo y buscar el call dentro de los
    statements top-level que NO sean def/class. Tolera anidamiento en
    `try:`/`if:` de top-level (que sí ejecutan en import-time) y alias
    del helper (`from knobs import _env_bool as _knob_env_bool`).

    [P1-NEON-DB-MIGRATION · 2026-06-12] La heurística regex previa
    ("def-a-def en columna 0") clasificaba mal el bloque top-level cuando un
    `def` nuevo (e.g. `_db_available`) aparece ANTES del call en el archivo —
    el AST resuelve la pertenencia real al module body."""
    matches = list(
        re.finditer(
            r'_env_bool\(\s*["\']MEALFIT_INVENTORY_RPC_STRICT["\']',
            db_inventory_src,
        )
    )
    assert matches, (
        "P2-NEW-2 regresión: ya no hay ningún call a "
        "`_env_bool('MEALFIT_INVENTORY_RPC_STRICT', ...)`. Si el knob "
        "fue removido del módulo, `/admin/knobs` no lo expone."
    )

    import ast

    tree = ast.parse(db_inventory_src)

    def _contains_knob_call(node: ast.AST) -> bool:
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            func = sub.func
            fname = getattr(func, "id", None) or getattr(func, "attr", None) or ""
            if not fname.endswith("_env_bool") or not sub.args:
                continue
            first = sub.args[0]
            if isinstance(first, ast.Constant) and first.value == "MEALFIT_INVENTORY_RPC_STRICT":
                return True
        return False

    top_level_match = any(
        _contains_knob_call(stmt)
        for stmt in tree.body
        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )

    assert top_level_match, (
        "P2-NEW-2 regresión: `_env_bool('MEALFIT_INVENTORY_RPC_STRICT', "
        "...)` solo aparece DENTRO de funciones (no se ejecuta al "
        "import-time). El knob no se auto-registra hasta que el path "
        "patológico ejecute. Restaurar una lectura al module-top-level "
        "cerca de los imports."
    )


def test_strict_knob_appears_in_registry_after_import():
    """Behavior test: tras importar `db_inventory`, el knob DEBE estar
    en `get_knobs_registry_snapshot()`. Sin esto el `/admin/knobs`
    endpoint queda blind al knob."""
    # Importar primero db_inventory para forzar el side-effect del registro.
    try:
        import db_inventory  # noqa: F401
    except Exception as e:
        pytest.skip(
            f"db_inventory no se puede importar en este entorno "
            f"(probable falta de env Supabase): {e}"
        )

    try:
        from graph_orchestrator import get_knobs_registry_snapshot
    except Exception as e:
        pytest.skip(
            f"get_knobs_registry_snapshot no importable: {e}"
        )

    snap = get_knobs_registry_snapshot()
    assert isinstance(snap, dict), (
        "P2-NEW-2: `get_knobs_registry_snapshot()` no retornó dict. "
        f"Type: {type(snap).__name__}"
    )

    assert "MEALFIT_INVENTORY_RPC_STRICT" in snap, (
        "P2-NEW-2 regresión: `MEALFIT_INVENTORY_RPC_STRICT` no está en "
        "`_KNOBS_REGISTRY` tras importar db_inventory. Si la lectura "
        "top-level desapareció, el knob queda invisible a "
        "`/admin/knobs` y `/health/version` hasta que el except path "
        "ejecute — exactamente cuando NO quieres descubrirlo."
    )

    entry = snap["MEALFIT_INVENTORY_RPC_STRICT"]
    assert isinstance(entry, dict), (
        f"P2-NEW-2: entry para el knob no es dict: {type(entry).__name__}"
    )
    assert entry.get("type") == "bool", (
        f"P2-NEW-2: type del knob no es 'bool' (es {entry.get('type')!r}). "
        "Si pasó a int/str el contrato de strict-mode rompe."
    )


def test_p2_new_2_anchor_present(db_inventory_src: str):
    """Anchor `P2-NEW-2` cerca del call top-level."""
    assert "P2-NEW-2" in db_inventory_src, (
        "P2-NEW-2 regresión: anchor textual desapareció. Restaurar para "
        "que un `grep` rápido localice la lectura top-level."
    )
