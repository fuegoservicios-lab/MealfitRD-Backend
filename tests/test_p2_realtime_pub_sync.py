"""[P2-REALTIME-PUB-SYNC · 2026-06-01] Regression guard: la publicación
`supabase_realtime` debe estar alineada con las suscripciones REALES del
frontend.

Contexto (auditoría de velocidad 2026-06-01, Supabase MCP):
    El lector WAL de Realtime era ~86% del tiempo de ejecución de la DB y
    descartaba el 99.94% de los eventos procesados. Causa raíz: la publicación
    estaba desalineada del consumo real en dos puntos opuestos.

    (1) `meal_plans` — el canal `meal-plan-chunk-updates`
        (AssessmentContext.jsx, `table: 'meal_plans'`) se suscribía pero la
        tabla NO estaba publicada → `postgres_changes` nunca entregaba
        eventos; las semanas de chunking solo llegaban por el fallback pesado
        (refetch REST completo). Mismo bug que P3-DEPLETED-BD-REALTIME-FIX.

    (2) `custom_shopping_items` — publicada CON `REPLICA IDENTITY FULL` pero
        SIN ningún consumidor Realtime (solo REST backend). Procesamiento WAL
        + logueo de fila-vieja completa para cero clientes.

Fix: migración SSOT `p2_realtime_pub_sync_2026_06_01.sql` (en AMBOS dirs)
    publica meal_plans, despublica custom_shopping_items y le pone REPLICA
    IDENTITY DEFAULT. Idempotente + sanity fail-loud.

Este test es parser-based (NO toca la DB) y ancla:
    - La migración existe en los DOS dirs SSOT y es byte-idéntica.
    - Contiene los 3 cambios canónicos + guards de idempotencia + sanity.
    - El consumidor frontend (suscripción a meal_plans) sigue presente — si
      alguien remueve la publicación sin la suscripción (o viceversa), el
      cross-link lo documenta.
    - NINGÚN consumidor Realtime de custom_shopping_items en el frontend
      (justifica su despublicación).

Limitaciones: NO valida el estado vivo de la DB (eso lo confirma el sanity
DO $$ de la propia migración al aplicarse). Solo ancla el SSOT en el repo.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_WORKSPACE_ROOT = _BACKEND_ROOT.parent

_MIGRATION_NAME = "p2_realtime_pub_sync_2026_06_01.sql"
_ROOT_MIGRATION = _WORKSPACE_ROOT / "supabase" / "migrations" / _MIGRATION_NAME
_BACKEND_MIGRATION = _BACKEND_ROOT / "supabase" / "migrations" / _MIGRATION_NAME

_ASSESSMENT_CTX = (
    _WORKSPACE_ROOT / "frontend" / "src" / "context" / "AssessmentContext.jsx"
)


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. La migración existe en AMBOS dirs SSOT (P3-MIGRATIONS-SSOT) e idéntica.
# ---------------------------------------------------------------------------
def test_migration_exists_in_both_ssot_dirs():
    assert _ROOT_MIGRATION.exists(), (
        f"Falta la migración en workspace-root: {_ROOT_MIGRATION}. "
        "P3-MIGRATIONS-SSOT exige el archivo en AMBOS dirs."
    )
    assert _BACKEND_MIGRATION.exists(), (
        f"Falta la migración en backend repo: {_BACKEND_MIGRATION}. "
        "P3-MIGRATIONS-SSOT exige el archivo en AMBOS dirs."
    )


def test_migration_ssot_dirs_byte_identical():
    assert _read(_ROOT_MIGRATION) == _read(_BACKEND_MIGRATION), (
        "Las dos copias SSOT de la migración divergen. Deben mantenerse "
        "byte-idénticas (P3-MIGRATIONS-SSOT)."
    )


# ---------------------------------------------------------------------------
# 2. La migración contiene los 3 cambios canónicos.
# ---------------------------------------------------------------------------
def test_migration_publishes_meal_plans():
    sql = _read(_ROOT_MIGRATION)
    assert "ALTER PUBLICATION supabase_realtime ADD TABLE public.meal_plans" in sql, (
        "La migración debe AÑADIR meal_plans a supabase_realtime (activa el "
        "push-merge de chunks)."
    )


def test_migration_unpublishes_custom_shopping_items():
    sql = _read(_ROOT_MIGRATION)
    assert (
        "ALTER PUBLICATION supabase_realtime DROP TABLE public.custom_shopping_items"
        in sql
    ), "La migración debe REMOVER custom_shopping_items de supabase_realtime."


def test_migration_sets_replica_identity_default():
    sql = _read(_ROOT_MIGRATION)
    assert (
        "ALTER TABLE public.custom_shopping_items REPLICA IDENTITY DEFAULT" in sql
    ), "La migración debe poner custom_shopping_items en REPLICA IDENTITY DEFAULT."


# ---------------------------------------------------------------------------
# 3. Idempotencia + sanity fail-loud (P3-MIGRATION-IDEMPOTENCE-DOC).
# ---------------------------------------------------------------------------
def test_migration_is_idempotent_guarded():
    sql = _read(_ROOT_MIGRATION)
    assert "DO $$" in sql, "Guards de idempotencia DO $$ ausentes."
    # ADD gateado por NOT EXISTS; DROP gateado por EXISTS.
    assert "IF NOT EXISTS (" in sql, "Guard NOT EXISTS (para el ADD) ausente."
    assert "IF EXISTS (" in sql, "Guard EXISTS (para el DROP) ausente."


def test_migration_has_failloud_sanity_checks():
    sql = _read(_ROOT_MIGRATION)
    # Las 3 invariantes post-apply deben fallar loud.
    assert sql.count("RAISE EXCEPTION") >= 3, (
        "Faltan sanity checks fail-loud: se esperan ≥3 RAISE EXCEPTION "
        "(meal_plans publicada, custom_shopping_items despublicada, "
        "replica identity default)."
    )
    assert "relreplident" in sql, (
        "El sanity de REPLICA IDENTITY (relreplident = 'd') debe estar presente."
    )


# ---------------------------------------------------------------------------
# 4. Cross-link con el consumidor frontend.
# ---------------------------------------------------------------------------
def test_frontend_meal_plans_subscription_still_present():
    """La suscripción a meal_plans (que la publicación habilita) debe seguir
    existiendo. Si se remueve sin despublicar la tabla, la publicación queda
    como desperdicio (el caso opuesto que cerramos para custom_shopping_items).
    """
    if not _ASSESSMENT_CTX.exists():
        pytest.skip("Frontend no presente en este checkout.")
    src = _read(_ASSESSMENT_CTX)
    assert "table: 'meal_plans'" in src, (
        "La suscripción Realtime a meal_plans desapareció de "
        "AssessmentContext.jsx. Si fue intencional, despublica también la "
        "tabla (migración) para no dejar WAL de Realtime sin consumidor."
    )
    assert "P2-REALTIME-PUB-SYNC" in src, (
        "El anchor P2-REALTIME-PUB-SYNC (que documenta la dependencia de la "
        "publicación) desapareció del frontend."
    )


def test_frontend_has_no_custom_shopping_items_realtime_consumer():
    """Justificación de la despublicación: ningún canal Realtime del frontend
    consume custom_shopping_items. Si alguien añade uno, debe re-publicar la
    tabla (y este test obliga a reconsiderar la decisión)."""
    if not _ASSESSMENT_CTX.exists():
        pytest.skip("Frontend no presente en este checkout.")
    # Escanear todo frontend/src por una suscripción postgres_changes a la tabla.
    frontend_src = _WORKSPACE_ROOT / "frontend" / "src"
    if not frontend_src.exists():
        pytest.skip("frontend/src no presente.")
    offenders = []
    for jsx in frontend_src.rglob("*.jsx"):
        text = _read(jsx)
        if "table: 'custom_shopping_items'" in text or 'table: "custom_shopping_items"' in text:
            offenders.append(str(jsx.relative_to(_WORKSPACE_ROOT)))
    assert not offenders, (
        "Se encontró una suscripción Realtime a custom_shopping_items en "
        f"{offenders}, pero la migración P2-REALTIME-PUB-SYNC la despublicó. "
        "Re-publica la tabla o elimina la suscripción."
    )
