"""[P2-NEW-1 · 2026-05-10] La migración `p2_new_1_set_meal_plans_updated_at_search_path.sql`
DEBE estar presente y aplicar `SET search_path = ''` a la trigger function
`set_meal_plans_updated_at`.

Bug original (audit 2026-05-10):
    La migración P0-2 (`p0_2_meal_plans_updated_at.sql`) creó la trigger
    function sin `SET search_path`. El advisor Supabase
    `function_search_path_mutable` la flag como WARN. Para una trigger
    function que solo usa `NOW()` el riesgo práctico es bajo, pero
    Supabase convención es lockear search_path en TODAS las funciones
    (defense-in-depth).

Fix:
    Migración nueva `p2_new_1_set_meal_plans_updated_at_search_path.sql`
    con `CREATE OR REPLACE FUNCTION ... SET search_path = ''`. Aplicada
    a producción 2026-05-10; advisor desaparecido del linter.

Estrategia del test (parser estático sobre la migración):
    1. Verificar que la migración existe en `migrations/`.
    2. Verificar `CREATE OR REPLACE FUNCTION public.set_meal_plans_updated_at`.
    3. Verificar `SET search_path = ''` (cadena vacía exactamente, no
       'public' — la cadena vacía es el lock más estricto).
    4. Verificar `LANGUAGE plpgsql` preservado.
    5. Verificar que la signature (RETURNS TRIGGER) no cambió.

Sin behavior test directo contra Postgres porque:
  - Requeriría conexión Supabase en CI (no garantizada).
  - El audit post-aplicación (`get_advisors security`) ya confirmó que
    el advisor desapareció — el contrato producción↔código está en la
    SQL del archivo de migración (SSOT).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MIGRATION_PATH = (
    _REPO_ROOT
    / "migrations"
    / "p2_new_1_set_meal_plans_updated_at_search_path.sql"
)


@pytest.fixture(scope="module")
def migration_src() -> str:
    if not _MIGRATION_PATH.exists():
        pytest.fail(
            f"P2-NEW-1 regresión: la migración {_MIGRATION_PATH.name} "
            f"no existe en `migrations/`. Si fue renombrada o "
            f"eliminada, restaurarla o actualizar este test. La migración "
            f"es el SSOT que aplica `SET search_path = ''` a "
            f"`set_meal_plans_updated_at`."
        )
    return _MIGRATION_PATH.read_text(encoding="utf-8")


def test_create_or_replace_function_present(migration_src: str):
    """La migración debe usar `CREATE OR REPLACE FUNCTION` (idempotente
    y preserva el trigger existente sin DROP/RECREATE)."""
    pattern = re.compile(
        r"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+public\.set_meal_plans_updated_at\s*\(",
        re.IGNORECASE,
    )
    assert pattern.search(migration_src), (
        "P2-NEW-1 regresión: la migración no usa "
        "`CREATE OR REPLACE FUNCTION public.set_meal_plans_updated_at`. "
        "Si pasó a `DROP FUNCTION + CREATE FUNCTION`, verifica que el "
        "trigger `trg_meal_plans_set_updated_at` siga conectado tras "
        "el DROP (Postgres lo desconectaría)."
    )


def test_search_path_locked_to_empty_string(migration_src: str):
    """El lock debe ser `SET search_path = ''` (cadena vacía) — el lock
    más estricto. `'public'` es aceptable funcionalmente pero deja
    vulnerable a shadowing via temp tables; cadena vacía fuerza schema
    qualifier explícito en cualquier referencia futura.
    """
    pattern = re.compile(
        r"SET\s+search_path\s*=\s*''",
        re.IGNORECASE,
    )
    assert pattern.search(migration_src), (
        "P2-NEW-1 regresión: la migración NO setea `SET search_path = ''`. "
        "Si pasó a `SET search_path = 'public'` o se removió, el advisor "
        "`function_search_path_mutable` puede reaparecer y los futuros "
        "callers de la función quedan vulnerables a shadowing por temp "
        "tables."
    )


def test_returns_trigger_preserved(migration_src: str):
    """La signature `RETURNS TRIGGER` no debe cambiar — sino el trigger
    `trg_meal_plans_set_updated_at` se rompe."""
    pattern = re.compile(
        r"RETURNS\s+TRIGGER",
        re.IGNORECASE,
    )
    assert pattern.search(migration_src), (
        "P2-NEW-1 regresión: `RETURNS TRIGGER` desapareció de la "
        "definición. Trigger functions DEBEN retornar TRIGGER; sin esto "
        "el `CREATE TRIGGER ... EXECUTE FUNCTION` falla al aplicar."
    )


def test_language_plpgsql_preserved(migration_src: str):
    """`LANGUAGE plpgsql` debe estar presente (la función usa `NEW.*` y
    `RETURN NEW` — pl/pgsql syntax)."""
    pattern = re.compile(
        r"LANGUAGE\s+plpgsql",
        re.IGNORECASE,
    )
    assert pattern.search(migration_src), (
        "P2-NEW-1 regresión: `LANGUAGE plpgsql` desapareció. Sin él, "
        "Postgres no parsea `NEW.updated_at = NOW()` (sintaxis pl/pgsql)."
    )


def test_anchor_present(migration_src: str):
    """Anchor textual `P2-NEW-1` en comentarios — `grep` debe poder
    localizar el fix rápidamente."""
    assert "P2-NEW-1" in migration_src, (
        "P2-NEW-1 regresión: anchor textual `P2-NEW-1` desapareció. "
        "Restaurar para `grep -r P2-NEW-1` en `migrations/`."
    )
