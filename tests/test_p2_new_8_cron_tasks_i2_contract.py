"""[P2-NEW-8 · 2026-05-11] Lock-the-contract I2 sobre cron_tasks.py.

Convención del repo (CLAUDE.md "Lifecycle de plan_id", invariante I2):
    "Toda mutación de meal_plans filtra AND user_id = %s."

P2-NEXT-1 + P3-NEXT-1 (2026-05-11) cerraron el invariante para
routers/plans.py. cron_tasks.py quedó sin test análogo. Los 4 UPDATE
meal_plans del chunk worker viven en cron-trusted contexts (meal_plan_id
ya claimeado de plan_chunk_queue + SELECT FOR UPDATE + advisory lock) →
defense-in-depth intencionalmente exenta.

Este test escanea TODOS los `UPDATE meal_plans SET` de cron_tasks.py y
exige UNO de:
    - El SQL contiene `AND user_id = %s` (filter explícito a DB-level).
    - O un comentario `# I2-EXEMPT: <razón ≥ 12 chars>` en las ~12
      líneas previas (justifica explícitamente el contexto trusted).

Sin este test, un refactor futuro que añada un UPDATE meal_plans en
cron_tasks.py sin filtrar por user_id (e.g., un nuevo backfill cron o
un sweep que toque plan_data) pasaría CI silente. P2-NEW-8 cierra el
gap análogo a P3-NEXT-1 sobre routers/plans.py.

Drift detection:
    - Nuevo UPDATE sin filter ni I2-EXEMPT → falla principal.
    - Borrar I2-EXEMPT de un sitio trusted existente → falla principal.
    - Marker con razón < 12 chars (decorativo, sin explicación) → no
      cuenta como exempt (regex `.{11,}`).

Tooltip-anchor: P2-NEW-8-I2-CRON | lock-the-contract paralelo a P3-NEXT-1
"""
from __future__ import annotations

import re
from pathlib import Path


_CRON_TASKS_PY = Path(__file__).resolve().parent.parent / "cron_tasks.py"

_UPDATE_MEAL_PLANS_RE = re.compile(r"UPDATE\s+meal_plans\s+SET", re.IGNORECASE)
_USER_ID_FILTER_RE = re.compile(r"AND\s+user_id\s*=\s*%s", re.IGNORECASE)
# Razón ≥ 12 chars (1 \S + .{11,}). Marker decorativo (corto) NO califica.
_I2_EXEMPT_RE = re.compile(r"#\s*I2-EXEMPT\s*:\s*(\S.{11,})", re.IGNORECASE)

# Ventanas:
#  - SQL siguiente al match: 5 líneas (cubre WHERE multilínea defensivo
#    aunque los 4 sitios actuales sean single-line).
#  - I2-EXEMPT precedente: 12 líneas (los 4 sitios actuales tienen el
#    marker en las 4-6 líneas anteriores; margen razonable para futuros).
_SQL_WINDOW_LINES = 5
_EXEMPT_LOOKBACK_LINES = 12


def _read_cron_tasks_lines() -> list[str]:
    return _CRON_TASKS_PY.read_text(encoding="utf-8").splitlines()


def _find_violation_sites(lines: list[str]) -> list[tuple[int, str]]:
    """Retorna lista de `(line_no, line_text)` para cada UPDATE
    meal_plans SET sin filter user_id y sin I2-EXEMPT precedente."""
    violations: list[tuple[int, str]] = []
    for line_no, line in enumerate(lines, start=1):
        # Saltar líneas que SON comentarios (heurística: strip empieza con #).
        # No saltar líneas de código que contienen `# comment` inline —
        # `UPDATE meal_plans` no va inline en comentarios típicamente.
        if line.strip().startswith("#"):
            continue
        if not _UPDATE_MEAL_PLANS_RE.search(line):
            continue
        # Ventana SQL: línea actual + N siguientes (capta WHERE multilínea).
        sql_window = "\n".join(
            lines[line_no - 1 : min(len(lines), line_no - 1 + _SQL_WINDOW_LINES)]
        )
        if _USER_ID_FILTER_RE.search(sql_window):
            continue
        # Sin filter — buscar I2-EXEMPT en lookback.
        lookback_start = max(0, line_no - 1 - _EXEMPT_LOOKBACK_LINES)
        lookback = "\n".join(lines[lookback_start : line_no - 1])
        if _I2_EXEMPT_RE.search(lookback):
            continue
        violations.append((line_no, line.strip()))
    return violations


# ---------------------------------------------------------------------------
# 1. Contrato principal: cada UPDATE filtra o tiene exempt
# ---------------------------------------------------------------------------
def test_every_update_meal_plans_filters_or_exempts():
    """Cada `UPDATE meal_plans SET` en cron_tasks.py debe tener
    `AND user_id = %s` en el SQL o un comentario
    `# I2-EXEMPT: <razón ≥ 12 chars>` en las 12 líneas previas.
    """
    lines = _read_cron_tasks_lines()
    violations = _find_violation_sites(lines)
    assert not violations, (
        f"P2-NEW-8 regresión: {len(violations)} `UPDATE meal_plans` en "
        f"cron_tasks.py SIN `AND user_id = %s` y SIN comentario "
        f"`# I2-EXEMPT: <razón ≥ 12 chars>` en las {_EXEMPT_LOOKBACK_LINES} "
        f"líneas anteriores. Convención del repo (CLAUDE.md invariante I2): "
        f"toda mutación de meal_plans filtra por user_id, EXCEPTO en "
        f"contextos cron-trusted explícitamente justificados.\n\n"
        f"Violaciones:\n"
        + "\n".join(f"  línea {ln}: {txt}" for ln, txt in violations)
        + "\n\nFix: añadir `AND user_id = %s` al UPDATE (si el caller "
          "tiene user_id disponible) o `# I2-EXEMPT: <razón>` justificando "
          "el contexto trusted (FOR UPDATE, advisory lock, plan_chunk_queue "
          "claim previo, etc)."
    )


# ---------------------------------------------------------------------------
# 2. Sanity: los sitios sin filter tienen marker (más diagnóstico que #1)
# ---------------------------------------------------------------------------
def test_count_of_unfiltered_sites_matches_marker_count():
    """Sanity: count de UPDATE sin filter == count de markers I2-EXEMPT
    detectados. Si no calzan, hay un sitio que perdió el marker o uno
    extra (marker huérfano).
    """
    lines = _read_cron_tasks_lines()
    sites_without_filter = 0
    sites_with_marker = 0
    for line_no, line in enumerate(lines, start=1):
        if line.strip().startswith("#"):
            continue
        if not _UPDATE_MEAL_PLANS_RE.search(line):
            continue
        sql_window = "\n".join(
            lines[line_no - 1 : min(len(lines), line_no - 1 + _SQL_WINDOW_LINES)]
        )
        if _USER_ID_FILTER_RE.search(sql_window):
            continue
        sites_without_filter += 1
        lookback_start = max(0, line_no - 1 - _EXEMPT_LOOKBACK_LINES)
        lookback = "\n".join(lines[lookback_start : line_no - 1])
        if _I2_EXEMPT_RE.search(lookback):
            sites_with_marker += 1

    assert sites_without_filter > 0, (
        "P2-NEW-8: no se detectaron UPDATE meal_plans SIN filter en "
        "cron_tasks.py. ¿Refactor masivo movió todo a routers? "
        "Reconsiderar si este test sigue siendo necesario."
    )
    assert sites_with_marker == sites_without_filter, (
        f"P2-NEW-8: {sites_without_filter} UPDATE meal_plans sin filter "
        f"detectados pero solo {sites_with_marker} tienen marker "
        f"`# I2-EXEMPT`. Falta marker en "
        f"{sites_without_filter - sites_with_marker} sitio(s) — ver "
        f"test_every_update_meal_plans_filters_or_exempts para localizar."
    )


# ---------------------------------------------------------------------------
# 3. Regex del marker enforza razón ≥ 12 chars
# ---------------------------------------------------------------------------
def test_marker_regex_rejects_short_reasons():
    """El marker `# I2-EXEMPT: <razón>` debe tener razón ≥ 12 chars
    (no es marker decorativo — debe explicar por qué el filter no aplica).
    Este test verifica que el regex `.{11,}` está vigente y no debilitado.
    """
    # Sanity check sobre el regex compilado.
    assert _I2_EXEMPT_RE.search("# I2-EXEMPT: trusted cron context"), (
        "Regex `_I2_EXEMPT_RE` no acepta razón válida (≥ 12 chars). "
        "¿Patrón roto?"
    )
    assert not _I2_EXEMPT_RE.search("# I2-EXEMPT: short"), (
        "Regex `_I2_EXEMPT_RE` acepta razón muy corta — debilitamiento "
        "accidental. Mantener `.{11,}` después del `:` para forzar "
        "razones substantivas (mín 12 chars total tras `:` + space)."
    )
    assert not _I2_EXEMPT_RE.search("# I2-EXEMPT:"), (
        "Regex `_I2_EXEMPT_RE` acepta marker vacío."
    )


# ---------------------------------------------------------------------------
# 4. Anchor cross-link slug
# ---------------------------------------------------------------------------
def test_marker_anchor_present():
    """Slug del filename matchea el marker `P2-NEW-8` para
    `test_p2_hist_audit_14_marker_test_link` (cross-link enforcer)."""
    expected_slug = "p2_new_8"
    assert expected_slug in __file__.replace("\\", "/").lower(), (
        "El nombre de este archivo debe contener el slug del P-fix "
        "(`p2_new_8`) para que el cross-link `test_p2_hist_audit_14_"
        "marker_test_link` lo matchee con el marker `_LAST_KNOWN_PFIX "
        "= \"P2-NEW-8 · 2026-05-11\"`."
    )
