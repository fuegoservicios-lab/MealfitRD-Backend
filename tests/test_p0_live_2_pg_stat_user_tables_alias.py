"""[P0-LIVE-2 · 2026-05-12] Tests parser-based: prohíbe `FROM
pg_stat_user_tables` sin alias en backend de prod.

Contexto del bug:
    En `_emit_hot_table_bloat_tick` (cron_tasks.py, P2-AUDIT-2) la query
    leía `pg_stat_user_tables` sin alias y proyectaba columnas como
    `relname`, `n_live_tup`, `last_autovacuum` sin prefix. En Postgres
    17.6 (DB live `17.6.1.063`), el planner expande internamente la vista
    (que JOINs `pg_class` + `pg_namespace` + `pg_stat_all_tables`) y
    expone `relname` desde múltiples lados → "column reference 'relname'
    is ambiguous" en runtime.

    Consecuencia: el watchdog P2-AUDIT-2 instalado el 2026-05-12 estaba
    *muerto* desde su primer tick — cero filas `_hot_table_bloat_tick` en
    `pipeline_metrics`, cero alerts `hot_table_bloat:*` posibles, y CLAUDE.md
    documentando una alert que el código jamás emite.

Fix canónico:
    `FROM pg_stat_user_tables t` + prefix `t.` consistente en TODAS las
    referencias (SELECT, WHERE, expresiones internas).

Test estrategia:
    Escanea backend/*.py (excluido tests/). Para cada bloque que use
    `FROM pg_stat_user_tables`:
      - Si está sin alias y la query referencia columnas potencialmente
        ambiguas (relname, schemaname, oid, reltuples), falla.
      - Aceptamos `FROM pg_stat_user_tables AS x` y `FROM pg_stat_user_tables x`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

P0_LIVE_2_ANCHOR = "P0-LIVE-2"

BACKEND_DIR = Path(__file__).resolve().parent.parent

# Columnas que cuelgan también de pg_class y por tanto son ambiguas si
# pg_stat_user_tables se usa sin alias en Postgres 17+.
_AMBIGUOUS_COLS = {"relname", "schemaname", "oid", "reltuples"}

_FROM_PG_STAT_RE = re.compile(
    r"FROM\s+pg_stat_user_tables(?P<after>[\s\S]{0,40})",
    re.IGNORECASE,
)


def _has_alias(after: str) -> bool:
    """Detecta si el FROM tiene alias (con o sin AS, identifier corto)."""
    m = re.match(r"\s+(?:AS\s+)?([a-zA-Z_]\w*)\b", after, re.IGNORECASE)
    if not m:
        return False
    candidate = m.group(1).lower()
    # 'where', 'join', etc no son alias.
    return candidate not in {"where", "join", "left", "right", "inner",
                             "on", "limit", "order", "group", "having",
                             "select", "and", "or"}


def _iter_prod_py_files() -> list[Path]:
    files: list[Path] = []
    for f in BACKEND_DIR.glob("*.py"):
        if f.is_file():
            files.append(f)
    return files


def test_a_pg_stat_user_tables_has_alias_when_used():
    """Toda `FROM pg_stat_user_tables` en backend de prod DEBE tener alias."""
    errors: list[str] = []
    for f in _iter_prod_py_files():
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue
        for m in _FROM_PG_STAT_RE.finditer(text):
            after = m.group("after")
            if _has_alias(after):
                continue
            line_no = text.count("\n", 0, m.start()) + 1
            errors.append(
                f"  {f.name}:{line_no} — `FROM pg_stat_user_tables` sin "
                f"alias. En Postgres 17+ el planner expande la vista contra "
                f"pg_class y columnas como `relname` quedan ambiguas. Usar "
                f"`FROM pg_stat_user_tables t` + prefix `t.` en todas las "
                f"columnas (relname, schemaname, n_live_tup, n_dead_tup, "
                f"last_autovacuum)."
            )
    if errors:
        pytest.fail(
            "[P0-LIVE-2] `FROM pg_stat_user_tables` sin alias encontrado:\n"
            + "\n".join(errors)
        )


def test_b_hot_table_bloat_tick_uses_alias_prefix():
    """Regresión específica del cron `_emit_hot_table_bloat_tick`: la query
    DEBE usar `FROM pg_stat_user_tables t` y prefix `t.relname` (no `relname`
    suelto en SELECT/WHERE)."""
    f = BACKEND_DIR / "cron_tasks.py"
    text = f.read_text(encoding="utf-8")
    fn_marker = "def _emit_hot_table_bloat_tick"
    idx = text.find(fn_marker)
    assert idx != -1, (
        "[P0-LIVE-2] Función `_emit_hot_table_bloat_tick` no encontrada en "
        "cron_tasks.py — el watchdog P2-AUDIT-2 desapareció o fue renombrado."
    )
    body = text[idx : idx + 6000]
    assert "FROM pg_stat_user_tables t" in body, (
        "[P0-LIVE-2] Esperado `FROM pg_stat_user_tables t` (con alias `t`) "
        "en `_emit_hot_table_bloat_tick`. Sin alias, Postgres 17 lanza "
        "'column reference \"relname\" is ambiguous' en cada tick — el "
        "watchdog P2-AUDIT-2 NO emite telemetría ni alerts."
    )
    # Las columnas críticas en SELECT/WHERE deben ir prefixed.
    assert "t.relname" in body, (
        "[P0-LIVE-2] Esperado `t.relname` (prefix explícito) en la query — "
        "sin prefix Postgres no resuelve cuál `relname` usar."
    )
    assert "t.schemaname" in body or "WHERE t.schemaname" in body, (
        "[P0-LIVE-2] Esperado `t.schemaname` en el WHERE — sin prefix queda "
        "ambiguo igual que relname."
    )
    # Sanity: NO debe quedar el patrón legacy sin alias.
    body_lines = body.splitlines()
    for i, line in enumerate(body_lines):
        stripped = line.strip()
        # Permitir comments (header explicativo del fix).
        if stripped.startswith("#"):
            continue
        if "FROM pg_stat_user_tables" in stripped:
            assert "FROM pg_stat_user_tables t" in stripped, (
                f"[P0-LIVE-2] Línea no-comentario con `FROM "
                f"pg_stat_user_tables` sin alias `t`: {stripped!r}"
            )


def test_c_anchor_marker_present():
    """Anchor `P0-LIVE-2` debe seguir cerca del fix."""
    f = BACKEND_DIR / "cron_tasks.py"
    text = f.read_text(encoding="utf-8")
    assert "[P0-LIVE-2" in text, (
        "[P0-LIVE-2] Anchor `[P0-LIVE-2 · ...]` removido de cron_tasks.py. "
        "Restaurar para que un futuro refactor entienda por qué el alias "
        "es load-bearing en Postgres 17."
    )
