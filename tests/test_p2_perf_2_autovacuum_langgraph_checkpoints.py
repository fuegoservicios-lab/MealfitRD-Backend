"""[P2-PERF-2 · 2026-05-12] Migración `p2_perf_2_autovacuum_langgraph_checkpoints.sql`
tunea autovacuum en las 3 tablas LangGraph (`checkpoints`, `checkpoint_blobs`,
`checkpoint_writes`).

Bug observado (audit 2026-05-11):
    Las 3 tablas tenían último autovacuum 2026-03-23 (50 días). Defaults
    globales (`scale_factor=0.2 + threshold=50`) requieren ~50+0.2*live
    dead rows para disparar. Con flujos INSERT-heavy + UPDATE ocasional
    de LangGraph, el VACUUM corría 1×/50d. Estadísticas del planner stale
    + bloat creciente bajo escalado del chat agent.

Fix:
    Migración SSOT con `ALTER TABLE ... SET (autovacuum_*)` × 3 tablas
    (`scale_factor=0.1`, `threshold=50`). Defaults intencionalmente más
    conservadores que P1-B (esas eran UPDATE-heavy puras sobre <50 filas;
    estas son INSERT-heavy con history grande). `checkpoint_migrations`
    queda con defaults (estática).

Lo que este test enforza:
    A) Migración SSOT existe.
    B) Migración contiene los 4 `autovacuum_*` parameters para cada una
       de las 3 tablas LangGraph.
    C) Migración escribe `COMMENT ON TABLE` con anchor P2-PERF-2 para
       cada tabla — descubrible vía `\\d+ <tabla>` post-deploy.
    D) Defaults usan `scale_factor=0.1` (más conservador que P1-B's 0.05
       porque estos son INSERT-heavy, no UPDATE-heavy).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MIGRATION = (
    _REPO_ROOT / "migrations"
    / "p2_perf_2_autovacuum_langgraph_checkpoints.sql"
)

_TABLES = ("checkpoints", "checkpoint_blobs", "checkpoint_writes")


@pytest.fixture(scope="module")
def migration_src() -> str:
    assert _MIGRATION.exists(), (
        "P2-PERF-2: migración SSOT "
        "`migrations/p2_perf_2_autovacuum_langgraph_checkpoints.sql` "
        "no encontrada. Restaurarla — sin SSOT, el tuning vive solo en "
        "memoria del operador y se pierde tras un dump/restore."
    )
    return _MIGRATION.read_text(encoding="utf-8")


def test_a_migration_alters_three_langgraph_tables(migration_src: str):
    """Cada una de las 3 tablas debe aparecer en un `ALTER TABLE`."""
    for table in _TABLES:
        pattern = re.compile(
            rf"ALTER\s+TABLE\s+public\.{re.escape(table)}\s+SET\s*\(",
            re.IGNORECASE,
        )
        assert pattern.search(migration_src), (
            f"P2-PERF-2: ALTER TABLE para `public.{table}` ausente en "
            f"la migración. Sin ese ALTER el autovacuum tuning no aplica."
        )


def test_b_each_table_sets_four_autovacuum_params(migration_src: str):
    """Cada `ALTER TABLE` declara los 4 parameters: vacuum_scale_factor,
    vacuum_threshold, analyze_scale_factor, analyze_threshold."""
    expected_params = (
        "autovacuum_vacuum_scale_factor",
        "autovacuum_vacuum_threshold",
        "autovacuum_analyze_scale_factor",
        "autovacuum_analyze_threshold",
    )
    for table in _TABLES:
        alter_match = re.search(
            rf"ALTER\s+TABLE\s+public\.{re.escape(table)}\s+SET\s*\((.*?)\);",
            migration_src,
            re.DOTALL | re.IGNORECASE,
        )
        assert alter_match, f"ALTER TABLE bloque para {table} no aislable."
        block = alter_match.group(1)
        for param in expected_params:
            assert param in block, (
                f"P2-PERF-2: parameter `{param}` ausente en ALTER TABLE "
                f"`public.{table}`. Todos los 4 son necesarios — vacuum y "
                f"analyze tienen umbrales independientes."
            )


def test_c_comment_anchors_present(migration_src: str):
    """Cada tabla debe tener `COMMENT ON TABLE ... '[P2-PERF-2 ...]'`."""
    for table in _TABLES:
        pattern = re.compile(
            rf"COMMENT\s+ON\s+TABLE\s+public\.{re.escape(table)}\s+IS\s+'\[P2-PERF-2",
            re.IGNORECASE,
        )
        assert pattern.search(migration_src), (
            f"P2-PERF-2: COMMENT con anchor [P2-PERF-2 ausente para "
            f"`public.{table}`. Sin COMMENT, un audit futuro vía "
            f"\\d+ no encuentra justificación del tuning."
        )


def test_d_scale_factor_uses_conservative_0_1(migration_src: str):
    """`scale_factor=0.1` (no 0.05 de P1-B). Estos son INSERT-heavy, no
    UPDATE-heavy — 0.05 sería agresivo de más."""
    assert "scale_factor = 0.1" in migration_src or "scale_factor=0.1" in migration_src, (
        "P2-PERF-2: scale_factor no es 0.1 como define el plan. Si "
        "experimentación pide cambiar, actualizar este test y la "
        "memoria de cierre juntos."
    )


def test_e_anchor_present(migration_src: str):
    assert "P2-PERF-2" in migration_src, (
        "P2-PERF-2: anchor desapareció del header de la migración."
    )
