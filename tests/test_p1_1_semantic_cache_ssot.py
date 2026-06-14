"""[P1-1 · 2026-05-08] Drift detection: SSOT del semantic cache.

Bug observado en el audit 2026-05-08:
  El script `backend/scripts/add_semantic_cache.py` declaraba 3 objetos del
  semantic cache (column `profile_embedding`, índice HNSW, RPC
  `match_similar_plan`) en runtime DDL, pero el repo NO conservaba el
  archivo `.sql` correspondiente en `migrations/`. El historial
  remoto de Supabase sí los tenía (vía `semantic_cache_migration` 2026-04-17
  y `fix_match_similar_plan_search_path_extensions` 2026-05-06), pero un
  greenfield clon del repo dependía del script Python para reproducir el
  schema — mismo anti-patrón que cerró P1-A para los scripts price_per_*/
  paypal_*.

Fix:
  - Migración SSOT `p1_1_consolidate_semantic_cache_ddl.sql` (idempotente).
  - Aplicada al remoto (versión 20260508192515) como NOOP funcional.
  - Este test es la red de seguridad: verifica que la migración SSOT
    permanece presente y declara los 3 objetos canónicos, y que el script
    Python legacy y los call sites del backend no derivan en nombres.

Patrón de drift cubierto: si un futuro refactor renombra la column / el
índice / la función en la migración, este test falla antes de que el
caller (db_plans.search_similar_plan) explote en runtime con
"function does not exist" o "column does not exist".
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_ROOT.parent
_MIGRATION_PATH = _REPO_ROOT / "migrations" / "p1_1_consolidate_semantic_cache_ddl.sql"
# [P3-2 · 2026-05-08] El script original `backend/scripts/add_semantic_cache.py`
# fue archivado a `_deprecated_*.py.bak` con exec guard. Buscamos el archivado
# primero; si no existe (regresión: alguien lo restauró sin renombrar), miramos
# en la ubicación original como fallback.
_LEGACY_SCRIPT_BAK = _BACKEND_ROOT / "_deprecated_add_semantic_cache.py.bak"
_LEGACY_SCRIPT_ORIG = _BACKEND_ROOT / "scripts" / "add_semantic_cache.py"
_LEGACY_SCRIPT_PATH = _LEGACY_SCRIPT_BAK if _LEGACY_SCRIPT_BAK.is_file() else _LEGACY_SCRIPT_ORIG
_DB_PLANS_PATH = _BACKEND_ROOT / "db_plans.py"


def test_migration_file_exists() -> None:
    """SSOT: la migración debe existir como archivo en migrations/.

    Sin este archivo, un greenfield clone del repo no puede reproducir el
    schema del semantic cache desde el filesystem y dependería del script
    Python deprecated o de un dump del remoto.
    """
    assert _MIGRATION_PATH.exists(), (
        f"Migración SSOT del semantic cache no encontrada en {_MIGRATION_PATH}. "
        f"Si fue movida/renombrada, actualizar este test y la referencia en "
        f"`backend/scripts/add_semantic_cache.py`."
    )


def test_migration_declares_canonical_objects() -> None:
    """La migración debe declarar los 3 objetos canónicos del semantic cache.

    Verificación textual (no AST SQL) — busca patrones idempotentes que el
    código de producción ya consume.
    """
    text = _MIGRATION_PATH.read_text(encoding="utf-8")

    # 1. Column profile_embedding como vector(768).
    assert re.search(
        r"ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\s+profile_embedding\s+vector\(768\)",
        text, re.IGNORECASE,
    ), "Migración no declara `profile_embedding vector(768)` con IF NOT EXISTS."

    # 2. Índice HNSW vector_cosine_ops.
    assert re.search(
        r"CREATE\s+INDEX\s+IF\s+NOT\s+EXISTS\s+meal_plans_profile_emb_idx",
        text, re.IGNORECASE,
    ), "Migración no declara índice `meal_plans_profile_emb_idx` con IF NOT EXISTS."
    assert re.search(
        r"USING\s+hnsw\s*\(\s*profile_embedding\s+vector_cosine_ops\s*\)",
        text, re.IGNORECASE,
    ), "Índice no usa HNSW con vector_cosine_ops (drift de operator class)."

    # 3. RPC match_similar_plan con CREATE OR REPLACE (idempotente).
    assert re.search(
        r"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+public\.match_similar_plan",
        text, re.IGNORECASE,
    ), "Migración no declara `CREATE OR REPLACE FUNCTION public.match_similar_plan`."

    # 4. SET search_path hardening — sin esto la función queda vulnerable a
    # search_path attacks (advisor Supabase function_search_path_mutable).
    assert re.search(
        r"SET\s+search_path\s+TO\s+'public'\s*,\s*'pg_catalog'\s*,\s*'extensions'",
        text, re.IGNORECASE,
    ), (
        "RPC `match_similar_plan` no fija search_path. Restaurar el SET "
        "search_path = 'public', 'pg_catalog', 'extensions' (ver migración "
        "`fix_match_similar_plan_search_path_extensions` 2026-05-06)."
    )


def test_legacy_script_objects_match_migration() -> None:
    """Drift detection script↔migración: los 3 objetos del script legacy
    deben coincidir con los nombres declarados en la migración.

    [P3-2 · 2026-05-08] Tras archivar a `.bak`, el script preserva el
    cuerpo histórico como comentarios (no alcanzable, ver exec guard).
    Los nombres canónicos (`profile_embedding`, `meal_plans_profile_emb_idx`,
    `match_similar_plan`) siguen apareciendo en esas líneas comentadas,
    así que el test sigue validando paridad. Si el `.bak` se borra
    completamente, skipeamos (no hay drift posible sin script).

    Si alguien edita el script/.bak (revivirlo con nombres distintos)
    o renombra en la migración, este test falla y obliga a reconciliar
    antes de mergear.
    """
    if not _LEGACY_SCRIPT_PATH.exists():
        pytest.skip(
            "Script legacy borrado completamente (no `.bak` ni original); "
            "drift detection no aplica."
        )

    script_text = _LEGACY_SCRIPT_PATH.read_text(encoding="utf-8")

    # Los nombres que el script declara DEBEN ser los mismos de la migración.
    canonical_names = (
        "profile_embedding",
        "meal_plans_profile_emb_idx",
        "match_similar_plan",
    )
    for name in canonical_names:
        assert name in script_text, (
            f"Script legacy `add_semantic_cache.py` (o su `.bak`) no "
            f"menciona `{name}`. Si fue renombrado, actualizar también "
            f"la migración SSOT y los call sites en backend/db_plans.py."
        )


def test_callers_match_migration_object_names() -> None:
    """Drift detection caller↔migración: db_plans.py invoca los nombres
    canónicos declarados en la migración SSOT.

    Cubre los 2 call sites conocidos:
      - search_similar_plan() invoca la RPC `match_similar_plan`.
      - _build_meal_plan_insert_sql() referencia la column `profile_embedding`.

    Sin este check, un rename en la migración rompería ambos en runtime y
    solo se vería en producción cuando el cache hit rate cae a 0.
    """
    db_plans_text = _DB_PLANS_PATH.read_text(encoding="utf-8")

    # [P1-NEON-DB-MIGRATION · 2026-06-12] El callsite migró de PostgREST
    # (`supabase.rpc("match_similar_plan", ...)`) a SQL directo
    # (`SELECT ... FROM public.match_similar_plan(...)`). Ambas formas son
    # callsites válidos del MISMO objeto canónico — lo que este test ancla
    # es la paridad de NOMBRE con la migración SSOT, no el transporte.
    rpc_callsite = re.compile(
        r'rpc\(\s*"match_similar_plan"'                 # forma PostgREST legacy
        r"|FROM\s+public\.match_similar_plan\s*\(",     # forma SQL directa (psycopg)
        re.IGNORECASE,
    )
    assert rpc_callsite.search(db_plans_text), (
        "`db_plans.search_similar_plan()` no invoca la función "
        "`match_similar_plan` (ni via rpc() ni via `FROM "
        "public.match_similar_plan(...)`). Si fue renombrada en la "
        "migración SSOT, actualizar el caller — un mismatch silencioso "
        "degrada el cache hit rate a 0% sin error visible."
    )

    assert "profile_embedding" in db_plans_text, (
        "db_plans.py no menciona la column `profile_embedding`. Si fue "
        "renombrada en la migración SSOT, actualizar el INSERT helper "
        "(_build_meal_plan_insert_sql) y los lectores."
    )
