"""[P0-3 / P0.2] Backfill de plan anchors — migración SQL canónica.

Contexto del gap (cerrado por P0.2):
    Antes existían DOS implementaciones del backfill de
    `_plan_start_date` / `grocery_start_date`:

      1. `supabase/p0_3_backfill_plan_anchors.sql` (aplicada manualmente).
      2. `cron_tasks._backfill_plan_anchors_oneshot()` — runtime, invocada
         al startup desde `register_plan_chunk_scheduler` y gobernada por
         la env var `BACKFILL_PLAN_ANCHORS_DONE` para evitar doble-corrida.

    El path runtime era frágil: si la env var no se seteaba en deploy, los
    anchors se reescribían en cada startup (idempotente, pero con WAL traffic
    creciente y log noise); si se seteaba antes de aplicar la migración,
    los planes legacy quedaban sin anchor y el cron worker caía al fallback
    hardcoded "8am UTC" desalineando hasta 24h en TZ no-UTC.

    P0.2 elimina el helper Python y consolida el backfill en
    `supabase/migrations/p0_3_backfill_plan_anchors.sql` como fuente única
    de verdad. Operador aplica vía `supabase db push` o SQL editor.

Este test verifica:
    Group A — Migración SQL existe y vive en la carpeta canónica.
    Group B — Idempotencia (WHERE IS NULL en ambos UPDATEs).
    Group C — Formatos correctos (ISO+TZ vs date-only) para los dos anchors.
    Group D — Estructura segura (jsonb_set + COALESCE para no pisar plan_data).
    Group E — Regresión: el helper runtime y la env var ya no existen en
              código activo (solo se permiten en comentarios documentales).
"""
import os
import re

import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MIGRATION_PATH = os.path.join(
    REPO_ROOT, "supabase", "migrations", "p0_3_backfill_plan_anchors.sql"
)
LEGACY_SQL_PATH = os.path.join(
    REPO_ROOT, "supabase", "p0_3_backfill_plan_anchors.sql"
)
CRON_TASKS_PATH = os.path.join(
    REPO_ROOT, "backend", "cron_tasks.py"
)


@pytest.fixture(scope="module")
def migration_sql() -> str:
    assert os.path.exists(MIGRATION_PATH), (
        f"Migración canon no encontrada en {MIGRATION_PATH}. "
        f"P0.2 movió el SQL desde supabase/ raíz a supabase/migrations/. "
        f"Si fue renombrada, actualiza este test."
    )
    with open(MIGRATION_PATH, "r", encoding="utf-8") as fh:
        return fh.read()


@pytest.fixture(scope="module")
def cron_tasks_source() -> str:
    with open(CRON_TASKS_PATH, "r", encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Group A — La migración SQL existe en la ubicación canon
# ---------------------------------------------------------------------------
def test_migration_lives_in_canonical_migrations_folder(migration_sql):
    """El SQL del backfill debe vivir en `supabase/migrations/`, junto al
    resto de migraciones formales del proyecto (p1_2, p1_3, etc.).
    Si alguien lo mueve fuera, este test falla y los flujos de aplicación
    de migración (`supabase db push`) lo dejarían atrás."""
    assert migration_sql.strip(), (
        f"La migración existe pero está vacía en {MIGRATION_PATH}"
    )


def test_legacy_sql_in_supabase_root_was_removed():
    """P0.2 elimina la copia vieja en `supabase/` raíz para evitar drift de
    contenido entre el archivo aplicado (operador) y el "documental"."""
    assert not os.path.exists(LEGACY_SQL_PATH), (
        f"El SQL legacy en {LEGACY_SQL_PATH} sigue presente. P0.2 requiere "
        f"que solo exista la copia canon en supabase/migrations/. Si lo "
        f"reintrodujiste, los dos archivos pueden divergir."
    )


# ---------------------------------------------------------------------------
# Group B — Idempotencia: ambos UPDATEs filtran solo NULLs en ventana 60d
# ---------------------------------------------------------------------------
def test_migration_filters_only_null_rows_for_idempotency(migration_sql):
    """Cada UPDATE debe filtrar `(plan_data->>'<anchor>') IS NULL` para que
    re-aplicar la migración sea no-op. Sin esto, un segundo `supabase db push`
    sobrescribiría valores ya rellenados (daño bajo pero rompe el contrato)."""
    assert re.search(
        r"plan_data->>'_plan_start_date'\)\s+IS\s+NULL",
        migration_sql,
        re.IGNORECASE,
    ), "UPDATE de _plan_start_date no filtra por IS NULL — perdería idempotencia."

    assert re.search(
        r"plan_data->>'grocery_start_date'\)\s+IS\s+NULL",
        migration_sql,
        re.IGNORECASE,
    ), "UPDATE de grocery_start_date no filtra por IS NULL — perdería idempotencia."


def test_migration_constrains_to_60_day_window(migration_sql):
    """`created_at > NOW() - INTERVAL '60 days'` evita escanear/tocar planes
    antiguos. Sin la ventana, la migración escala con el tamaño total de
    `meal_plans` y el costo crece sin tope con cada deploy."""
    update_blocks = re.findall(
        r"UPDATE\s+meal_plans.*?(?:;|\Z)", migration_sql, re.IGNORECASE | re.DOTALL
    )
    assert len(update_blocks) >= 2, (
        f"Esperaba al menos 2 UPDATEs (uno por anchor). Got: {len(update_blocks)}"
    )
    for block in update_blocks:
        assert "INTERVAL '60 days'" in block, (
            f"UPDATE sin ventana 60d:\n{block!r}"
        )
        assert "created_at IS NOT NULL" in block, (
            f"UPDATE sin guard de created_at NOT NULL:\n{block!r}"
        )


# ---------------------------------------------------------------------------
# Group C — Formatos correctos por anchor
# ---------------------------------------------------------------------------
def _split_update_blocks(sql: str) -> list[str]:
    """Devuelve cada UPDATE statement aislado (`UPDATE meal_plans ... ;`),
    SIN incluir el comentario del statement siguiente. Necesario porque un
    split simple por lookahead deja los comentarios `-- 2. Backfill ...` del
    bloque 2 pegados al final del bloque 1, contaminando los filtros por
    anchor.
    """
    return re.findall(
        r"UPDATE\s+meal_plans[^;]*;",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )


def test_plan_start_date_uses_iso_with_timezone(migration_sql):
    """`_plan_start_date` debe escribirse en formato ISO completo con TZ
    (`YYYY-MM-DD"T"HH24:MI:SS"+00:00"`), idéntico al producido por el
    pipeline en `routers/plans.py` para planes nuevos. Drift entre legacy
    backfilleado y nuevo causaría parsing inconsistente downstream."""
    blocks = _split_update_blocks(migration_sql)
    psd_blocks = [b for b in blocks if "_plan_start_date" in b and "grocery_start_date" not in b]
    assert psd_blocks, (
        f"No se encontró UPDATE aislado para _plan_start_date. "
        f"Bloques detectados: {len(blocks)}"
    )
    block = psd_blocks[0]
    assert 'YYYY-MM-DD"T"HH24:MI:SS"+00:00"' in block, (
        f"_plan_start_date debe usar formato ISO con TZ. Block:\n{block!r}"
    )


def test_grocery_start_date_uses_date_only_format(migration_sql):
    """`grocery_start_date` debe escribirse en formato date-only `YYYY-MM-DD`,
    idéntico al producido por `db_plans._ensure_grocery_start_date`. Si esta
    migración usa accidentalmente el formato ISO, la deduplicación shopping-list
    rompe (compara strings exactos)."""
    blocks = _split_update_blocks(migration_sql)
    gsd_blocks = [b for b in blocks if "grocery_start_date" in b and "_plan_start_date" not in b]
    assert gsd_blocks, (
        f"No se encontró UPDATE aislado para grocery_start_date. "
        f"Bloques detectados: {len(blocks)}"
    )
    block = gsd_blocks[0]
    assert "'YYYY-MM-DD'" in block, (
        f"grocery_start_date debe usar formato date-only. Block:\n{block!r}"
    )
    assert 'HH24:MI:SS"+00:00"' not in block, (
        f"grocery_start_date NO debe usar formato ISO con TZ. Block:\n{block!r}"
    )


# ---------------------------------------------------------------------------
# Group D — Estructura segura del UPDATE
# ---------------------------------------------------------------------------
def test_migration_uses_jsonb_set_to_preserve_other_fields(migration_sql):
    """`jsonb_set(COALESCE(plan_data, '{}'::jsonb), ...)` preserva el resto de
    plan_data. Sin esto, un UPDATE simple con `plan_data = '{...}'::jsonb`
    pisaría `days`, `total_days_requested`, `learning`, etc."""
    update_blocks = re.findall(
        r"UPDATE\s+meal_plans.*?(?:;|\Z)", migration_sql, re.IGNORECASE | re.DOTALL
    )
    for block in update_blocks:
        assert "jsonb_set(" in block, (
            f"UPDATE no usa jsonb_set — pisaría el resto de plan_data:\n{block!r}"
        )
        assert "COALESCE(plan_data, '{}'::jsonb)" in block, (
            f"UPDATE no usa COALESCE para tolerar plan_data NULL:\n{block!r}"
        )


# ---------------------------------------------------------------------------
# Group E — Regresión: el helper runtime ya no existe en código activo
# ---------------------------------------------------------------------------
def test_runtime_helper_definition_was_removed(cron_tasks_source):
    """`def _backfill_plan_anchors_oneshot` (definición de la función) NO debe
    existir en cron_tasks.py. Si reaparece, P0.2 fue revertido y la lógica
    duplicada con la migración SQL volverá a divergir."""
    assert not re.search(
        r"^\s*def\s+_backfill_plan_anchors_oneshot\s*\(",
        cron_tasks_source,
        re.MULTILINE,
    ), (
        "El helper `_backfill_plan_anchors_oneshot` fue redefinido en "
        "cron_tasks.py. P0.2 lo eliminó intencionalmente — el backfill vive "
        "exclusivamente en supabase/migrations/p0_3_backfill_plan_anchors.sql."
    )


def test_runtime_helper_invocation_was_removed(cron_tasks_source):
    """`_backfill_plan_anchors_oneshot()` no debe ser invocado desde ningún
    sitio (startup u otros). Distinguir invocación de comentario buscando
    paréntesis explícitos."""
    # Busca la función llamada (con paréntesis), excluyendo backticks dentro
    # de comentarios. La forma más simple es contar matches que NO están
    # precedidas inmediatamente por backtick.
    pattern = r"(?<!`)_backfill_plan_anchors_oneshot\s*\("
    matches = re.findall(pattern, cron_tasks_source)
    assert not matches, (
        f"Se encontró invocación al helper eliminado: {matches}. P0.2 lo "
        f"removió de register_plan_chunk_scheduler. Si necesitas backfill "
        f"runtime, replantea el diseño antes de reintroducirlo."
    )


def test_backfill_env_var_no_longer_read(cron_tasks_source):
    """`os.environ.get("BACKFILL_PLAN_ANCHORS_DONE", ...)` NO debe leerse en
    código activo. La env var era el toggle del helper eliminado y ya no
    tiene sentido. Comentarios documentales que la mencionan están bien."""
    pattern = r"os\.environ\.get\(\s*['\"]BACKFILL_PLAN_ANCHORS_DONE['\"]"
    matches = re.findall(pattern, cron_tasks_source)
    assert not matches, (
        f"La env var BACKFILL_PLAN_ANCHORS_DONE sigue siendo leída en "
        f"cron_tasks.py: {matches}. P0.2 la deprecó."
    )
