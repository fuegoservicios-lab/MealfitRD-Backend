"""[P2-4 · re-anclado P1-NEON-DB-MIGRATION · 2026-06-12] Regression test que
protege la frontera de seguridad del incremento atómico de inventario.

Historia:
  P2-4 (2026-05-07) anclaba la RPC `public.increment_inventory_quantity`
  (SECURITY DEFINER + `WHERE user_id = auth.uid()`), invocada por el frontend
  vía PostgREST. Con la migración de datos a Neon (P1-NEON-DB-MIGRATION ·
  2026-06-12) esa RPC quedó FUERA del serving path: en Neon no hay contexto
  JWT (`auth.uid()` = NULL) y el frontend (Pantry.jsx velocímetro) llama ahora
  a `POST /api/inventory/increment` ([backend/routers/user_data.py]) que
  implementa la MISMA frontera con filtro explícito server-side
  `WHERE id = %s AND user_id = %s`, usando el uid verificado por
  `get_verified_user_id` (invariante I2).

  Si un dev futuro elimina el filtro `AND user_id = %s`, acepta `user_id` del
  body, o interpola el SQL sin parametrizar, la decisión P2-4 deja de ser
  válida y se convierte en un IDOR P0 silencioso. Este test bloquea ese drift
  parseando CÓDIGO EJECUTABLE (AST: el literal SQL + la tupla de params), no
  comentarios.

Mapeo de invariantes (RPC vieja → endpoint nuevo):

  | RPC P2-4                                | Endpoint Neon                                       |
  |-----------------------------------------|-----------------------------------------------------|
  | function exists in public schema        | route POST /inventory/increment definida            |
  | SECURITY DEFINER + EXECUTE authenticated| identidad inyectada por `Depends(get_verified_user_id)` |
  | body enforces `user_id = auth.uid()`    | SQL `WHERE id = %s AND user_id = %s` + param `uid`   |
  | body uses `user_inventory`              | SQL `UPDATE user_inventory`                          |
  | search_path locked (anti-shadowing)     | SQL 100% parametrizado (sin f-string/format/concat)  |
  | GREATEST(0, ...) clamp                  | clamp ≥0 en el SQL del endpoint                      |
  | COMMENT referencia P2-4 / auth.uid()    | docstring referencia la RPC reemplazada + scoping    |

Ejecutar:
    cd backend && python -m pytest tests/test_p2_4_increment_inventory_user_scoping.py -v
"""
import ast
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_USER_DATA_FP = _BACKEND_ROOT / "routers" / "user_data.py"
_FRONTEND_SRC = _BACKEND_ROOT.parent / "frontend" / "src"


@pytest.fixture(scope="module")
def user_data_src() -> str:
    return _USER_DATA_FP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def user_data_tree(user_data_src: str) -> ast.Module:
    return ast.parse(user_data_src)


def _find_function(tree: ast.Module, name: str):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


@pytest.fixture(scope="module")
def increment_fn(user_data_tree: ast.Module):
    fn = _find_function(user_data_tree, "api_increment_inventory")
    assert fn is not None, (
        "`api_increment_inventory` no encontrado en routers/user_data.py. "
        "Si renombraste el handler, re-ancla este test ANTES de cambiar producción."
    )
    return fn


def _find_execute_sql_write_call(fn_node) -> ast.Call:
    """Localiza la llamada `execute_sql_write(sql, params, ...)` dentro del
    handler (vive en el closure `_inc`). Parser sobre el AST — código
    ejecutable, no comentarios."""
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name == "execute_sql_write":
                return node
    pytest.fail(
        "No hay llamada a `execute_sql_write` dentro de `api_increment_inventory`. "
        "Si el write migró a otro helper, re-ancla este test verificando que el "
        "filtro user_id sobreviva en el nuevo path."
    )


def _extract_sql_literal(call: ast.Call) -> str:
    """El primer argumento DEBE ser un literal estático (ast.Constant str).
    Si es un JoinedStr (f-string) u otra expresión, la parametrización se
    rompió — eso es parte de la frontera (ver test_sql_is_parameterized)."""
    assert call.args, "execute_sql_write llamado sin argumentos posicionales."
    sql_node = call.args[0]
    assert isinstance(sql_node, ast.Constant) and isinstance(sql_node.value, str), (
        "El SQL de api_increment_inventory ya NO es un literal estático "
        f"(encontrado nodo {type(sql_node).__name__}). Un f-string/concatenación "
        "abre vector de inyección que puede falsificar la frontera user_id."
    )
    return sql_node.value


# ---------------------------------------------------------------------------
# 1. El endpoint existe y está ruteado
# ---------------------------------------------------------------------------
def test_endpoint_exists_in_user_data_router(user_data_src: str, increment_fn):
    assert '"/inventory/increment"' in user_data_src, (
        "Route `/inventory/increment` no encontrada en routers/user_data.py. "
        "El velocímetro de Pantry.jsx depende de este endpoint post-Neon."
    )
    # El decorator @router.post debe colgar exactamente de este handler.
    decorated = False
    for dec in increment_fn.decorator_list:
        if isinstance(dec, ast.Call):
            for arg in dec.args:
                if isinstance(arg, ast.Constant) and arg.value == "/inventory/increment":
                    decorated = True
    assert decorated, (
        "`api_increment_inventory` no está decorado con la route "
        "`/inventory/increment` — handler huérfano o route movida."
    )


# ---------------------------------------------------------------------------
# 2. La identidad la inyecta el backend (≈ SECURITY DEFINER + auth.uid())
# ---------------------------------------------------------------------------
def test_identity_comes_from_verified_dependency_only(increment_fn):
    """ESTA ES LA MITAD AUTENTICACIÓN DE LA FRONTERA. El uid usado en el SQL
    debe nacer de `Depends(get_verified_user_id)` (token Supabase verificado
    server-side), nunca del payload del cliente."""
    found_dependency = False
    for default in increment_fn.args.defaults:
        if isinstance(default, ast.Call) and getattr(default.func, "id", "") == "Depends":
            dep_arg = default.args[0] if default.args else None
            if getattr(dep_arg, "id", "") == "get_verified_user_id":
                found_dependency = True
    assert found_dependency, (
        "FRONTERA VIOLADA: `api_increment_inventory` ya no inyecta la identidad "
        "via `Depends(get_verified_user_id)`. Sin la dependency, el endpoint "
        "no tiene uid confiable que filtrar — IDOR universal."
    )
    # `uid = _require_user(verified_user_id)` — gate 401 para anónimos.
    requires_user = False
    for node in ast.walk(increment_fn):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_require_user":
            requires_user = True
    assert requires_user, (
        "`_require_user(verified_user_id)` ausente — un request sin token "
        "llegaría al SQL con uid=None en vez de 401 fail-secure."
    )


def test_body_model_does_not_accept_user_id(user_data_tree: ast.Module):
    """El cliente NO puede inyectar identidad: `InventoryIncrementBody` solo
    declara `item_id` y `delta`. Si alguien añade `user_id` al model, el
    endpoint podría preferirlo sobre el verificado (anti-patrón P0-AGENT-1)."""
    body_cls = None
    for node in ast.walk(user_data_tree):
        if isinstance(node, ast.ClassDef) and node.name == "InventoryIncrementBody":
            body_cls = node
            break
    assert body_cls is not None, "`InventoryIncrementBody` no encontrado."
    fields = [
        stmt.target.id
        for stmt in body_cls.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    ]
    assert "item_id" in fields and "delta" in fields, (
        f"Campos del body cambiaron (encontrados: {fields}). Re-ancla el test "
        f"y verifica que la identidad siga fuera del payload."
    )
    assert "user_id" not in fields, (
        "FRONTERA VIOLADA: `InventoryIncrementBody` acepta `user_id` del "
        "cliente. La identidad DEBE venir solo de get_verified_user_id."
    )


# ---------------------------------------------------------------------------
# 3. El SQL enforza el scoping (≈ WHERE user_id = auth.uid())
# ---------------------------------------------------------------------------
def test_sql_enforces_user_id_filter(increment_fn):
    """ESTA ES LA FRONTERA DE SEGURIDAD (invariante I2). Si esta cláusula
    desaparece, cualquier usuario autenticado puede mutar el inventario de
    otro adivinando `item_id` (ids enteros — enumerables)."""
    call = _find_execute_sql_write_call(increment_fn)
    sql = _extract_sql_literal(call)
    assert re.search(r"WHERE\s+id\s*=\s*%s\s+AND\s+user_id\s*=\s*%s", sql, re.IGNORECASE), (
        "FRONTERA DE SEGURIDAD VIOLADA: el UPDATE de api_increment_inventory "
        "ya NO filtra `WHERE id = %s AND user_id = %s`. Equivale a borrar el "
        f"`user_id = auth.uid()` de la RPC P2-4 original. SQL actual: {sql!r}"
    )
    # La tupla de params debe cerrar el binding: el ÚLTIMO placeholder
    # (user_id) recibe el uid verificado, no un valor del cliente.
    assert len(call.args) >= 2, "execute_sql_write sin tupla de params."
    params = call.args[1]
    assert isinstance(params, ast.Tuple) and params.elts, (
        "Params de execute_sql_write no son una tupla literal — imposible "
        "verificar estáticamente el binding del uid."
    )
    last_param = params.elts[-1]
    assert isinstance(last_param, ast.Name) and last_param.id == "uid", (
        "El último param del UPDATE (binding de `user_id = %s`) ya no es la "
        "variable `uid` (derivada de get_verified_user_id). Si reordenaste "
        "los placeholders, re-ancla este test verificando el binding correcto."
    )
    # Sanity: ningún param viene de un campo `user_id` del body.
    for p in params.elts:
        if isinstance(p, ast.Attribute):
            assert p.attr != "user_id", (
                "Un param del UPDATE lee `*.user_id` (¿body.user_id?) — la "
                "identidad debe ser el `uid` verificado."
            )


def test_sql_uses_user_inventory_table(increment_fn):
    sql = _extract_sql_literal(_find_execute_sql_write_call(increment_fn))
    assert re.search(r"UPDATE\s+user_inventory\b", sql, re.IGNORECASE), (
        "El UPDATE ya no toca `user_inventory`. Si la tabla fue renombrada, "
        "actualizar memoria P2-4 y este test."
    )


def test_sql_is_parameterized_no_interpolation(increment_fn):
    """≈ `SET search_path` lockdown de la RPC: sin vector que permita
    falsificar la frontera. El SQL es literal estático (verificado en
    `_extract_sql_literal`) y todos los valores viajan como placeholders."""
    call = _find_execute_sql_write_call(increment_fn)
    sql = _extract_sql_literal(call)  # falla si es JoinedStr/BinOp
    n_placeholders = sql.count("%s")
    params = call.args[1]
    assert isinstance(params, ast.Tuple) and len(params.elts) == n_placeholders, (
        f"Mismatch placeholders ({n_placeholders}) vs params "
        f"({len(getattr(params, 'elts', []))}) — binding roto o valor "
        f"interpolado fuera de params."
    )
    assert n_placeholders >= 3, (
        f"Esperaba ≥3 placeholders (delta, item_id, user_id); hay {n_placeholders}. "
        f"¿Algún valor quedó hardcoded/interpolado?"
    )


# ---------------------------------------------------------------------------
# 4. Paridad de semántica con la RPC: clamp a >= 0
# ---------------------------------------------------------------------------
def test_sql_clamps_to_zero_via_greatest(increment_fn):
    """La RPC P2-4 hacía `GREATEST(0, quantity + p_delta)`: ráfagas de clicks
    negativos (o dos tabs decrementando concurrentemente — el frontend computa
    delta contra un baseline local) nunca dejaban quantity < 0. El endpoint
    DEBE preservar ese clamp server-side: el floor qty=1 del velocímetro es
    solo client-side y no protege bajo concurrencia."""
    sql = _extract_sql_literal(_find_execute_sql_write_call(increment_fn))
    assert re.search(r"GREATEST\s*\(\s*0", sql, re.IGNORECASE), (
        "Clamp `GREATEST(0, ...)` ELIMINADO del UPDATE — semántica de la RPC "
        "P2-4 perdida en la migración Neon. Dos tabs decrementando "
        "concurrentemente pueden dejar quantity negativa (row invisible para "
        "GET /api/inventory pero bloqueante para el INSERT 409-dedup). "
        "Fix: `SET quantity = GREATEST(0, quantity + %s::numeric)`."
    )


# ---------------------------------------------------------------------------
# 5. Lineage documentado (≈ COMMENT ON FUNCTION de P2-4)
# ---------------------------------------------------------------------------
def test_docstring_references_replaced_rpc(increment_fn):
    """El docstring documenta la frontera y el lineage para que cualquier dev
    entienda por qué este endpoint reemplaza a la RPC SECURITY DEFINER."""
    doc = ast.get_docstring(increment_fn) or ""
    assert "increment_inventory_quantity" in doc, (
        "Docstring ya no referencia la RPC `increment_inventory_quantity` "
        "reemplazada — el lineage P2-4 → endpoint se pierde."
    )
    assert "user_id" in doc, (
        "Docstring no menciona el scoping por user_id — documenta la frontera "
        "igual que el COMMENT ON FUNCTION de P2-4."
    )


# ---------------------------------------------------------------------------
# 6. El frontend ya no llama a la RPC muerta (anti split-brain)
# ---------------------------------------------------------------------------
def test_frontend_does_not_call_legacy_rpc():
    """Post-cutover, PostgREST apunta al Postgres de Supabase (datos stale).
    Un callsite `supabase.rpc('increment_inventory_quantity', ...)` que
    regrese al frontend escribiría en la DB equivocada (split-brain). Escaneo
    de callsites ejecutables (`.rpc(`), no comentarios."""
    offenders = []
    for ext in ("*.js", "*.jsx"):
        for fp in _FRONTEND_SRC.rglob(ext):
            try:
                src = fp.read_text(encoding="utf-8")
            except Exception:
                continue
            if re.search(r"\.rpc\(\s*['\"]increment_inventory_quantity['\"]", src):
                offenders.append(str(fp))
    assert not offenders, (
        f"Callsites de la RPC legacy `increment_inventory_quantity` "
        f"reintroducidos en el frontend: {offenders}. Usa POST "
        f"/api/inventory/increment (backend, filtro user_id explícito)."
    )
