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
    # raw_ingredients debe definirse antes de su uso (no dentro de un guard saltable)
    assert "raw_ingredients = []" in body
    assert body.index("raw_ingredients = []") < body.index("if raw_ingredients:")


def test_partial_chunked_path_tampoco_gatea():
    # El path chunked ya estaba correcto; lo anclamos para que no reintroduzcan el guard.
    body = _func_source("save_partial_plan_get_id")
    assert "if supabase:" not in body
