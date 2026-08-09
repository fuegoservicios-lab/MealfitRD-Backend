# [P1-SWAP-TELEMETRY-NAMEERROR · 2026-08-08] `change_swap` = 0 filas DESDE SIEMPRE (vs
# change_regen_day que sí emitía) con 74 swaps reales en 36h: los 4 call sites del emit en
# `api_swap_meal` referenciaban `body.mealType` — pero la función recibe `data: dict`. El
# NameError se evaluaba DENTRO del try best-effort y `except Exception: pass` se lo tragaba
# en cada llamada. Un best-effort que traga NameError convierte un typo en telemetría muerta
# de nacimiento (clase P1-G: código inerte que parece sano). Este test la mata por AST:
# reintroducir CUALQUIER referencia a un nombre no definido tipo `body` en la función falla.
import ast
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_SRC_PATH = os.path.join(os.path.dirname(__file__), "..", "routers", "plans.py")


def _function_node(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def test_api_swap_meal_no_referencia_body():
    tree = ast.parse(open(_SRC_PATH, encoding="utf-8").read())
    fn = _function_node(tree, "api_swap_meal")
    assert fn is not None, "api_swap_meal desapareció de routers/plans.py"
    # nombres definidos en la función: params + asignaciones + imports + for/with targets
    defined = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                defined.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
    assert "body" not in defined, (
        "si defines `body` de verdad, actualiza este test — nació porque `body` NO existía")
    loads = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    assert "body" not in loads, (
        "api_swap_meal referencia `body`, que no está definido: el NameError muere dentro del "
        "try best-effort de telemetría y la vuelve inerte (0 filas change_swap). Usa `data`.")


def test_emit_swap_usa_data_mealtype():
    src = open(_SRC_PATH, encoding="utf-8").read()
    assert src.count('meal_type=data.get("mealType")') >= 4, (
        "los 4 call sites del emit de change_swap deben leer data.get('mealType')")
