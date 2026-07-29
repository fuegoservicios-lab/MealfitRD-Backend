"""[P3-FREQ-TRACKING-CATALOG-DB · 2026-07-29] El helper de tracking de frecuencias debe ser
AUTOSUFICIENTE.

Bug real, vivo ~4 h en producción y cazado por el LOG, no por la suite:
    ⚠️ [BACKGROUND ERROR] Error en freq-tracking del plan: name '_catalog_db' is not defined

Al extraer `_track_ingredient_frequencies` de `_save_plan_and_track_background`
(P2-FREQ-TRACKING-CHUNKED, ese mismo día) el bloque movido se quedó SIN dos dependencias que vivían
en la función original, fuera del trozo: `_catalog_db` y `_eb`. Cada llamada lanzaba `NameError`,
que el `except Exception` de "best-effort" convertía en un ERROR de log. Resultado: el tracking
quedó roto para TODOS — peor que el bug que el helper venía a arreglar (antes al menos funcionaba
en la rama no-chunked).

Por qué la suite no lo vio: los tests del batch P2 eran parser-based (verificaban que el helper
EXISTIERA y que la rama chunked lo LLAMARA), nunca lo EJECUTARON. Un helper recién extraído necesita
al menos una invocación real.

Este archivo hace las dos cosas:
  1. lo INVOCA de verdad (con dobles, sin tocar Neon) y exige que no lance;
  2. comprueba estáticamente por AST que no le queden nombres LIBRES sin resolver — la comprobación
     que habría encontrado ambas dependencias de una sola pasada.
"""
from __future__ import annotations

import ast
import builtins
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
_SERVICES = os.path.join(_BACKEND, "services.py")


def _free_names_of(func_name: str, path: str) -> list:
    """Nombres que la función LEE y que no asigna, ni recibe como argumento, ni importa dentro, ni
    existen a nivel de módulo o en builtins. Los bindings de `except X as e` cuentan como asignados."""
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == func_name)

    assigned, loaded = set(), set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Name):
            (assigned if isinstance(n.ctx, ast.Store) else loaded).add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                assigned.add(a.asname or a.name.split(".")[0])
        elif isinstance(n, ast.arg):
            assigned.add(n.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            assigned.add(n.name)          # `except X as _cat_e`
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not fn:
            assigned.add(n.name)          # helpers anidados

    module_level = set()
    for n in tree.body:                    # SOLO el nivel superior del módulo
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_level.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                module_level.add(a.asname or a.name.split(".")[0])
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            for t in ([n.target] if isinstance(n, ast.AnnAssign) else n.targets):
                if isinstance(t, ast.Name):
                    module_level.add(t.id)
        elif isinstance(n, ast.Try):       # imports en try/except a nivel de módulo
            for sub in ast.walk(n):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for a in sub.names:
                        module_level.add(a.asname or a.name.split(".")[0])

    return sorted(x for x in (loaded - assigned - module_level) if not hasattr(builtins, x))


def test_helper_has_no_unresolved_free_names():
    """La comprobación que habría cazado `_catalog_db` Y `_eb` de una sola pasada."""
    missing = _free_names_of("_track_ingredient_frequencies", _SERVICES)
    assert not missing, (
        f"`_track_ingredient_frequencies` lee nombres que no resuelve: {missing}. "
        f"Son dependencias que se quedaron en la función de origen al extraer el helper — "
        f"cada llamada lanzará NameError, y el `except` best-effort lo convertirá en un ERROR de "
        f"log en vez de un fallo visible.")


def test_helper_actually_runs_without_raising(monkeypatch):
    """Invocación REAL. El batch P2 solo comprobaba que el helper existiera y que lo llamaran —
    nunca lo ejecutó, y por eso el NameError llegó a producción."""
    import services

    calls = {}
    monkeypatch.setattr(services, "increment_ingredient_frequencies",
                        lambda uid, items: calls.setdefault("inc", []).append((uid, list(items))))
    monkeypatch.setattr(services, "log_unknown_ingredients",
                        lambda *a, **kw: calls.setdefault("unknown", []).append(a))

    # No debe lanzar NADA, ni siquiera si el catálogo no está disponible.
    services._track_ingredient_frequencies(
        "00000000-0000-0000-0000-000000000000",
        ["150 g de pollo", "1 taza de arroz blanco", "100 g de habichuelas rojas"],
    )
    assert "inc" in calls, "con ingredientes canónicos reconocibles debe incrementar frecuencias"
    uid, items = calls["inc"][0]
    assert items, f"no trackeó ningún ingrediente canónico: {calls}"


def test_helper_is_noop_on_empty_input():
    import services
    services._track_ingredient_frequencies("u", [])      # no debe lanzar
    services._track_ingredient_frequencies("u", None)    # ni con None


def test_extract_raw_ingredients_matches_the_inline_shape():
    """El extractor comparte criterio con el aplanado inline del path no-chunked."""
    import services
    plan = {"days": [{"meals": [{"ingredients": ["a", "b"]}, {"ingredients": None}]},
                     {"meals": [{"ingredients": ["c"]}]}, "no-dict"]}
    assert services.extract_raw_ingredients_from_plan(plan) == ["a", "b", "c"]
    assert services.extract_raw_ingredients_from_plan({}) == []
    assert services.extract_raw_ingredients_from_plan(None) == []
