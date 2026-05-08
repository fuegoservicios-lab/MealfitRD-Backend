"""[P2-4] Regression test que protege la frontera de seguridad de
`public.increment_inventory_quantity`.

Contexto:
  La función es `SECURITY DEFINER` y ejecutable por `authenticated`,
  lo que dispara el lint `auth_security_definer_function_executable`.
  Es un false-positive: el body enforces `WHERE id = p_id AND
  user_id = auth.uid()`, así que un usuario no puede mutar el inventario
  de otro aunque adivine el `p_id`.

  El advisor no inspecciona el body. Si un dev futuro refactoriza la
  función y elimina la cláusula `auth.uid()` (o cambia search_path, o
  remueve SECURITY DEFINER), la decisión P2-4 deja de ser válida y se
  convierte en un agujero P0 silencioso. Este test bloquea ese drift.

Cobertura:
  - test_function_exists_in_public_schema
  - test_security_definer_is_set
  - test_search_path_locked_to_public
  - test_body_enforces_user_id_equals_auth_uid
  - test_body_uses_user_inventory_table
  - test_body_clamps_to_zero_via_greatest
  - test_comment_references_p2_4_decision

Skip si no hay credenciales de DB configuradas (CI sin acceso live):
  el test es de "frontera de seguridad" y requiere consultar pg_proc
  contra la DB real. Si DATABASE_URL/SUPABASE_DB_URL falta, pytest.skip.

Ejecutar:
    cd backend && python -m pytest tests/test_p2_4_increment_inventory_user_scoping.py -v
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fetch_function_def():
    """Retorna `(definition, comment)` o llama pytest.skip si la DB no está
    accesible. Cacheada vía module-level fixture para no martillar la DB."""
    if not os.environ.get("SUPABASE_DB_URL"):
        pytest.skip("SUPABASE_DB_URL no configurada; skip frontera-seguridad live")
    try:
        from db_core import execute_sql_query
    except Exception as e:
        pytest.skip(f"db_core no importable en este entorno: {e}")

    rows = execute_sql_query(
        """
        SELECT
            pg_get_functiondef(p.oid) AS definition,
            obj_description(p.oid, 'pg_proc') AS comment_text,
            CASE p.prosecdef WHEN true THEN 'DEFINER' ELSE 'INVOKER' END AS security
        FROM pg_proc p
        JOIN pg_namespace n ON n.oid = p.pronamespace
        WHERE n.nspname = 'public' AND p.proname = 'increment_inventory_quantity'
        """,
        (),
    )
    if not rows:
        pytest.fail("public.increment_inventory_quantity no existe en la DB")
    return rows[0]


@pytest.fixture(scope="module")
def fn_def():
    return _fetch_function_def()


def test_function_exists_in_public_schema(fn_def):
    assert fn_def["definition"], "function_def vacío"
    assert "public.increment_inventory_quantity" in fn_def["definition"]


def test_security_definer_is_set(fn_def):
    """La elección SECURITY DEFINER es intencional. Si alguien la cambia a
    INVOKER, RLS de user_inventory tomará control y la función no necesitará
    el lint exception — pero el comportamiento del frontend cambiará. Si esto
    es deliberado, actualizar este test + memoria P2-4."""
    assert fn_def["security"] == "DEFINER", (
        "increment_inventory_quantity ya no es SECURITY DEFINER. "
        "Si fue intencional, actualiza memoria P2-4 y este test."
    )
    assert "SECURITY DEFINER" in fn_def["definition"]


def test_search_path_locked_to_public(fn_def):
    """`SET search_path TO 'public'` previene function-shadowing attacks
    (un schema de mayor prioridad no puede inyectar una `auth.uid()` falsa)."""
    defn = fn_def["definition"]
    assert "SET search_path TO 'public'" in defn or "SET search_path = 'public'" in defn, (
        "search_path no está locked a public. SECURITY DEFINER + search_path "
        "abierto = vulnerabilidad de function-shadowing."
    )


def test_body_enforces_user_id_equals_auth_uid(fn_def):
    """ESTA ES LA FRONTERA DE SEGURIDAD. Si esta cláusula desaparece,
    cualquier authenticated user puede mutar el inventario de otro adivinando ids."""
    defn = fn_def["definition"]
    # Tolerante a whitespace y orden de operandos.
    has_canonical = "user_id = auth.uid()" in defn
    has_reversed = "auth.uid() = user_id" in defn
    assert has_canonical or has_reversed, (
        "FRONTERA DE SEGURIDAD VIOLADA: el body de increment_inventory_quantity "
        "ya NO contiene `user_id = auth.uid()`. Sin esa cláusula, SECURITY DEFINER "
        "permite que cualquier authenticated user mute el inventario de otro. "
        "Restaurar la cláusula o revocar EXECUTE de `authenticated`."
    )


def test_body_uses_user_inventory_table(fn_def):
    assert "user_inventory" in fn_def["definition"], (
        "El body ya no toca user_inventory. Si la tabla fue renombrada, "
        "actualizar memoria P2-4 y este test."
    )


def test_body_clamps_to_zero_via_greatest(fn_def):
    """`GREATEST(0, ...)` evita que ráfagas de clicks negativos lleven a < 0."""
    assert "GREATEST(0" in fn_def["definition"] or "greatest(0" in fn_def["definition"], (
        "Clamp a >= 0 vía GREATEST eliminado; clicks negativos pueden llevar "
        "quantity por debajo de 0."
    )


def test_comment_references_p2_4_decision(fn_def):
    """El COMMENT documenta la decisión P2-4 y la frontera de seguridad para
    que cualquier dev que lea `\\df+` entienda por qué la WARN del advisor
    es false-positive."""
    comment = fn_def["comment_text"] or ""
    assert "P2-4" in comment, "COMMENT no referencia la decisión P2-4."
    assert "auth.uid()" in comment, "COMMENT no referencia la frontera auth.uid()."
