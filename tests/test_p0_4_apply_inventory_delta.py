"""[P0-4 · 2026-05-10] Regression guard: deducción de inventario es
atómica vía RPC `apply_inventory_delta`, no SELECT-MODIFY-WRITE
en app-layer.

Bug original (audit 2026-05-10):
    [`backend/db_inventory.py:1085-1146`] `add_or_update_inventory_item`
    hacía:
        1. SELECT id, quantity, unit FROM user_inventory WHERE user_id+name.
        2. for row: convert_amount(...); new_qty = current + converted_qty.
        3. UPDATE SET quantity = new_qty (o DELETE si < 0.01).

    Bajo concurrencia (dos chunks deduciendo en paralelo el mismo
    ingrediente, o `log_consumed_meal` 2× duplicado), dos threads
    podían leer la misma `current_qty`, computar deltas distintos
    desde ese snapshot stale, y ambos hacer UPDATE — el segundo pisa
    al primero perdiendo unidades.

    `test_pantry_auto_sync_between_chunks.py` existe pero mockea —
    no ejerce DB real → la race quedaba invisible.

Fix (`supabase/migrations/p0_4_apply_inventory_delta_rpc.sql`):
    RPC `apply_inventory_delta(p_user_id, p_row_id, p_delta, ...)`:
      - SECURITY DEFINER (backend service_role no setea auth.uid()).
      - `SELECT … FOR UPDATE` lockea la fila → serializa concurrent calls.
      - `WHERE id = p_row_id AND user_id = p_user_id` — ownership check
        explícito (defense-in-depth contra IDs adivinados).
      - DELETE si new_qty < 0.01, else UPDATE — mismo threshold legacy.
      - GRANT solo a `service_role` (frontend usa
        `increment_inventory_quantity` con `auth.uid()`).

Cobertura de este test (parser-based, no DB):
    1. La migración SSOT existe y declara los 4 invariantes.
    2. El callsite `add_or_update_inventory_item` llama a `.rpc("apply_inventory_delta", ...)`.
    3. Fallback al UPDATE legacy SI la RPC falla (deploy lag) está
       presente y loguea loud (race condition no se enmascara).
    4. El INSERT path (fila nueva) sigue en app-layer (no necesita
       RPC porque no hay race posible — la fila no existe aún).

Out of scope (siguientes P-fixes):
    - Test runtime que ejecute 2 threads concurrentes contra DB real
      y valide ausencia de lost-update. Requiere DB de staging
      separada; queda como P3.
    - `failed_inventory_deductions` recibe escrituras cuando todas
      las RPCs fallan — esto es P0-5 (separado), no P0-4.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MIGRATION_PATH = _REPO_ROOT / "supabase" / "migrations" / "p0_4_apply_inventory_delta_rpc.sql"
_DB_INVENTORY_PATH = _REPO_ROOT / "backend" / "db_inventory.py"


def _read(path: Path) -> str:
    assert path.exists(), f"Archivo requerido no encontrado: {path}"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Migración SSOT declara la RPC + invariantes
# ---------------------------------------------------------------------------
def test_migration_declares_rpc():
    sql = _read(_MIGRATION_PATH)
    assert re.search(
        r"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+public\.apply_inventory_delta",
        sql, re.IGNORECASE,
    ), "Migración debe declarar `public.apply_inventory_delta`."


def test_migration_uses_security_definer():
    """SECURITY DEFINER es necesario porque backend service_role no setea
    `auth.uid()`. Si alguien cambia a INVOKER, las llamadas desde backend
    fallarían silenciosamente con RLS."""
    sql = _read(_MIGRATION_PATH)
    assert re.search(r"SECURITY\s+DEFINER", sql, re.IGNORECASE), (
        "RPC debe ser SECURITY DEFINER (backend service_role no tiene "
        "auth.uid() context — un INVOKER fallaría con RLS)."
    )


def test_migration_locks_row_for_update():
    """El lock FOR UPDATE es el corazón del fix — sin esto, concurrent
    calls vuelven a tener lost-update race."""
    sql = _read(_MIGRATION_PATH)
    assert re.search(r"FOR\s+UPDATE", sql, re.IGNORECASE), (
        "RPC debe usar `SELECT … FOR UPDATE` para lockear la fila "
        "contra concurrent calls. Sin esto, la atomicidad se pierde."
    )


def test_migration_enforces_ownership_check():
    """Ownership check explícito en el SELECT — sin esto un SECURITY DEFINER
    puede ser abusado pasando IDs ajenos."""
    sql = _read(_MIGRATION_PATH)
    # Esperamos `AND user_id = p_user_id` en el SELECT.
    assert re.search(
        r"user_id\s*=\s*p_user_id", sql, re.IGNORECASE,
    ), (
        "RPC debe verificar `user_id = p_user_id` en el SELECT — defense "
        "in depth porque SECURITY DEFINER bypassea RLS."
    )


def test_migration_grants_only_service_role():
    """Permisos restrictivos: solo backend (service_role). Frontend usa
    `increment_inventory_quantity` (auth.uid() interno) para sus paths
    autenticados."""
    sql = _read(_MIGRATION_PATH)
    # REVOKE de PUBLIC, anon, authenticated. GRANT solo a service_role.
    assert re.search(
        r"REVOKE\s+ALL\s+ON\s+FUNCTION\s+public\.apply_inventory_delta.*FROM\s+PUBLIC",
        sql, re.IGNORECASE | re.DOTALL,
    ), "Migración debe REVOKE de PUBLIC explícitamente."
    assert re.search(
        r"REVOKE\s+ALL\s+ON\s+FUNCTION\s+public\.apply_inventory_delta.*FROM\s+authenticated",
        sql, re.IGNORECASE | re.DOTALL,
    ), "Migración debe REVOKE de authenticated (frontend usa otra RPC)."
    assert re.search(
        r"GRANT\s+EXECUTE\s+ON\s+FUNCTION\s+public\.apply_inventory_delta.*TO\s+service_role",
        sql, re.IGNORECASE | re.DOTALL,
    ), "Migración debe GRANT EXECUTE solo a service_role."


def test_migration_deletes_on_threshold():
    """DELETE cuando new_qty < 0.01 — mismo threshold legacy (db_inventory.py:1120),
    crítico para que el comportamiento sea idéntico al pre-fix."""
    sql = _read(_MIGRATION_PATH)
    assert re.search(r"v_new_qty\s*<\s*0\.01", sql, re.IGNORECASE), (
        "RPC debe aplicar threshold `< 0.01` (mismo que app-layer legacy)."
    )
    assert re.search(r"DELETE\s+FROM\s+public\.user_inventory", sql, re.IGNORECASE), (
        "RPC debe ejecutar DELETE cuando la qty cae bajo threshold."
    )


# ---------------------------------------------------------------------------
# 2. db_inventory.py usa la RPC
# ---------------------------------------------------------------------------
def test_add_or_update_calls_rpc():
    src = _read(_DB_INVENTORY_PATH)
    # Buscar el bloque de `add_or_update_inventory_item` y confirmar que
    # contiene la llamada a la RPC.
    func_match = re.search(
        r"def\s+add_or_update_inventory_item\(.*?(?=\ndef\s+|\Z)",
        src, re.DOTALL,
    )
    assert func_match is not None, (
        "Función `add_or_update_inventory_item` no encontrada en db_inventory.py."
    )
    body = func_match.group(0)
    assert "apply_inventory_delta" in body, (
        "`add_or_update_inventory_item` NO llama a la RPC `apply_inventory_delta`. "
        "El path UPDATE/DELETE quedó en app-layer — la lost-update race vuelve."
    )
    # [P1-NEON-DB-MIGRATION · 2026-06-12] La RPC se invoca via SQL directo
    # `SELECT public.apply_inventory_delta(...)` (antes `supabase.rpc(...)`).
    # La atomicidad FOR UPDATE de la función se preserva idéntica.
    assert re.search(r"SELECT\s+public\.apply_inventory_delta\(", body), (
        "Llamada a la RPC debe ser `SELECT public.apply_inventory_delta(...)` "
        "via execute_sql_query."
    )


def test_rpc_call_passes_required_params():
    """La llamada SQL a la RPC debe pasar los 5 params canónicos con sus
    casts explícitos. [P1-NEON-DB-MIGRATION · 2026-06-12] Equivalencia con el
    payload PostgREST legacy `{p_user_id, p_row_id, p_delta, p_mutation_type,
    p_master_id}`: ahora viajan posicionales — `%s::uuid` (user), `%s::bigint`
    (row), `%s::numeric` (delta — float8→numeric NO es cast implícito en
    resolución de funciones), `%s` (mutation_type), `%s::uuid` (master)."""
    src = _read(_DB_INVENTORY_PATH)
    func_match = re.search(
        r"def\s+add_or_update_inventory_item\(.*?(?=\ndef\s+|\Z)",
        src, re.DOTALL,
    )
    body = func_match.group(0)
    assert '"%s::uuid, %s::bigint, %s::numeric, %s, %s::uuid"' in body, (
        "Llamada a apply_inventory_delta debe pasar los 5 params posicionales "
        "con casts explícitos (%s::uuid, %s::bigint, %s::numeric, %s, %s::uuid). "
        "Sin el ::numeric explícito, Postgres no resuelve la función para "
        "float8; sin los 5 params, la RPC devuelve error o aplica defaults "
        "silenciosamente."
    )
    # Y la tupla de argumentos debe enviar los 5 valores canónicos en orden.
    call_region_match = re.search(
        r"SELECT\s+public\.apply_inventory_delta\([\s\S]{0,800}?fetch_one=True",
        body,
    )
    assert call_region_match is not None, (
        "No se encontró la región de la llamada RPC (hasta fetch_one=True). "
        "Si cambiaste el shape de la llamada, actualiza este test."
    )
    call_region = call_region_match.group(0)
    for ident in ("user_id", "row_id", "round(converted_qty, 4)", "mutation_type", "master_id"):
        assert ident in call_region, (
            f"Llamada a apply_inventory_delta omite `{ident}` en los params. "
            f"Sin todos los params, la RPC devuelve error o aplica defaults "
            f"silenciosamente."
        )


def test_rpc_fallback_logs_loud_on_failure():
    """Fallback al path legacy si la RPC no está disponible (deploy lag).
    Debe loguear con logger.error (visible en alertas), no swallow."""
    src = _read(_DB_INVENTORY_PATH)
    func_match = re.search(
        r"def\s+add_or_update_inventory_item\(.*?(?=\ndef\s+|\Z)",
        src, re.DOTALL,
    )
    body = func_match.group(0)
    # Debe haber un except que capture errores de la RPC y loguee error
    # (no warning ni debug — son too quiet).
    assert re.search(
        r"logger\.error\(\s*[fr]?[\"'].*apply_inventory_delta",
        body, re.IGNORECASE,
    ), (
        "Fallback de la RPC debe loguear con `logger.error` (no warning/debug). "
        "Sin esto, una RPC missing post-deploy queda invisible."
    )
