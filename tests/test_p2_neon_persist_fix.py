"""[P2-NEON-PERSIST-FIX · 2026-06-13] El path de persistencia no-chunked / fallback
(`services._save_plan_and_track_background`) NO debe gatear el guardado en `if supabase:`.

Bug destapado por un test en vivo autenticado: en modo Neon `db.supabase` es None → el guard
`if supabase:` saltaba el INSERT ENTERO y luego crasheaba con UnboundLocalError en
`raw_ingredients` (definido dentro del bloque saltado). El plan no-chunked (totalDays ≤ 3) o
el fallback cuando el SSE generator muere pre-postprocess (conexión del cliente cae a mitad de
generación) se PERDÍA silenciosamente + emitía alerta plan_persist_failed. El path chunked
(`save_partial_plan_get_id`, default 7 días) nunca tuvo el guard → por eso pasó desapercibido.
"""
import ast
import os

_SVC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "services.py")


def _func_node(name: str):
    src = open(_SVC, encoding="utf-8").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node, src
    raise AssertionError(f"función {name} no encontrada en services.py")


def _func_source(name: str) -> str:
    node, src = _func_node(name)
    return ast.get_source_segment(src, node)


def _gates_on_name(node, var: str) -> bool:
    """True si hay un `if <var>:` real (AST), ignorando menciones en comentarios/strings."""
    for n in ast.walk(node):
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == var:
            return True
    return False


def test_save_background_no_gatea_en_if_supabase():
    node, _ = _func_node("_save_plan_and_track_background")
    assert not _gates_on_name(node, "supabase"), (
        "_save_plan_and_track_background NO debe gatear el guardado en `if supabase:` "
        "(None en Neon → se salta el INSERT + UnboundLocalError)"
    )


def test_save_background_llama_atomic_incondicional():
    body = _func_source("_save_plan_and_track_background")
    assert "save_new_meal_plan_atomic(" in body, "debe invocar el save Neon-native"

    # [P1-UPDATE-PROTAGONIST-FLOOR · 2026-07-29 · reanclado] El par de `index()` de antes
    # (`"raw_ingredients = []"` antes de `"if raw_ingredients:"`) era un anclaje POSICIONAL por
    # texto y caducó cuando `_track_ingredient_frequencies` se extrajo a su propio helper: el
    # consumo (`if raw_ingredients:`) se mudó fuera de esta función, así que el segundo `index()`
    # lanzaba ValueError. La invariante NO cambió — el bug original era un `UnboundLocalError`
    # porque la variable se definía DENTRO de un guard que en modo Neon se saltaba entero, y con
    # ella se perdía el plan en silencio. Lo que hay que exigir es eso, no un orden de dos
    # literales: la asignación tiene que estar en un camino que SIEMPRE se ejecuta.
    _fn, _ = _func_node("_save_plan_and_track_background")

    # líneas de asignación que cuelgan de algún `if` → esas NO garantizan nada
    _bajo_if = set()
    for _node in ast.walk(_fn):
        if isinstance(_node, ast.If):
            for _d in ast.walk(_node):
                if (isinstance(_d, ast.Name) and _d.id == "raw_ingredients"
                        and isinstance(_d.ctx, ast.Store)):
                    _bajo_if.add(_d.lineno)
    _asigna_libre = any(
        isinstance(_t, ast.Name) and _t.id == "raw_ingredients" and _t.lineno not in _bajo_if
        for _node in ast.walk(_fn) if isinstance(_node, ast.Assign)
        for _t in _node.targets)

    assert _asigna_libre, (
        "`raw_ingredients` ya no se asigna en un camino incondicional de "
        "_save_plan_and_track_background — vuelve el UnboundLocalError que perdía el plan "
        "no-chunked entero en modo Neon (y emitía plan_persist_failed sin decir por qué)")
    assert "_track_ingredient_frequencies(" in body, (
        "el tracking salió a su helper; esta función debe seguir invocándolo")


def test_partial_chunked_path_tampoco_gatea():
    # El path chunked ya estaba correcto; lo anclamos para que no reintroduzcan el guard.
    body = _func_source("save_partial_plan_get_id")
    assert "if supabase:" not in body
