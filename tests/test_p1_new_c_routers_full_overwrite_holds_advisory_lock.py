"""[P1-NEW-C · 2026-05-11] Lock-the-contract: cada `UPDATE meal_plans SET
plan_data = %s::jsonb` (full-overwrite) en `routers/plans.py` DEBE
tener `acquire_meal_plan_advisory_lock(...purpose="general"...)` en
las líneas previas del mismo flujo.

Espejo de P1-NEW-B (que cubre cron_tasks.py), pero para los handlers
HTTP. Hoy hay un solo sitio aplicable:
    - `api_shift_plan` (routers/plans.py ~L2078): UPDATE full-overwrite
      tras computar `shifted_data` con días renumerados.

`api_shift_plan` toma `acquire_meal_plan_advisory_lock(cursor, plan_id,
purpose="general")` ~475 líneas antes (L1603) dentro del mismo
`with conn.cursor()`. El mismo `purpose="general"` que `_chunk_worker`
T1/T2 — garantiza serialización inter-worker.

Motivación:
    P1-NEW-B cubre cron-trusted. Este test ancla el mismo contrato en
    el lado HTTP. Si un futuro endpoint añade un full-overwrite de
    `plan_data` sin tomar el lock, dos handlers concurrentes (HTTP
    + cron + HTTP) producirían lost-update silente. Los endpoints que
    hoy hacen mutaciones quirúrgicas vía `jsonb_set` (P0-NEW-A, P0-NEW-B,
    retry-chunk, recipe/expand, etc.) NO están sujetos a este test —
    `jsonb_set` opera atómicamente a nivel de key, no full-overwrite.

Drift detection:
    - Nuevo endpoint con full-overwrite sin lock → falla con archivo +
      línea + snippet.
    - Refactor que mueve el lock fuera del scope del UPDATE → falla.

Whitelist:
    No prevista. Si un futuro endpoint NECESITA full-overwrite sin
    lock (e.g. admin endpoint que opera sobre planes huérfanos),
    documentar la excepción inline con un comentario
    `# [P1-NEW-C WHITELIST: <razón>]` en las 12 líneas previas, igual
    que P3-NEXT-1 / P2-NEW-8 (no soportado en este test todavía —
    añadir cuando aparezca el primer caso real).

Tooltip-anchor: P1-NEW-C-START | gap P1 audit 2026-05-11
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_PLANS_PY = Path(__file__).resolve().parent.parent.parent / "backend" / "routers" / "plans.py"


# [P2-FULLOVERWRITE-FSTRING-ANCHOR · 2026-06-18] (audit fresco P2) Además de la forma literal, capturamos la
# forma f-string dinámica `UPDATE meal_plans SET {', '.join(set_clauses)}` (api_restore_plan_local, P1-OPEN-1),
# donde set_clauses incluye `plan_data = %s::jsonb`. Antes el blanket era ciego a ella → el lock de
# /restore-local solo lo anclaba su test dedicado; ahora el contrato I7 blanket también lo cubre.
_FULL_OVERWRITE_RE = re.compile(
    r"UPDATE\s+meal_plans\s+SET\s+plan_data\s*=\s*%s::jsonb"   # forma literal (api_shift_plan)
    r"|UPDATE\s+meal_plans\s+SET\s+\{",                         # forma f-string (api_restore_plan_local)
    re.IGNORECASE,
)
_LOCK_CALL_RE = re.compile(
    r"acquire_meal_plan_advisory_lock\s*\([^)]*purpose\s*=\s*['\"]general['\"]",
    re.DOTALL,
)
_LOCK_ALIAS_IMPORT_RE = re.compile(
    r"from\s+db_plans\s+import\s+acquire_meal_plan_advisory_lock\s+as\s+(\w+)"
)

# Ventana lookback. El sitio actual está ~475 líneas del lock; 1000
# da margen razonable para inserciones futuras.
_LOOKBACK_LINES = 1000


def _find_full_overwrite_sites(lines: list[str]) -> list[tuple[int, str]]:
    sites: list[tuple[int, str]] = []
    for idx, line in enumerate(lines, start=1):
        if line.strip().startswith("#"):
            continue
        if _FULL_OVERWRITE_RE.search(line):
            sites.append((idx, line.strip()))
    return sites


def _has_lock_in_lookback(lines: list[str], target_line: int) -> bool:
    start = max(0, target_line - 1 - _LOOKBACK_LINES)
    end = target_line - 1
    window = "\n".join(lines[start:end])
    if _LOCK_CALL_RE.search(window):
        return True
    for m in _LOCK_ALIAS_IMPORT_RE.finditer(window):
        alias = m.group(1)
        alias_call_re = re.compile(
            rf"{re.escape(alias)}\s*\([^)]*purpose\s*=\s*['\"]general['\"]",
            re.DOTALL,
        )
        if alias_call_re.search(window):
            return True
    return False


@pytest.fixture(scope="module")
def plans_lines() -> list[str]:
    return _PLANS_PY.read_text(encoding="utf-8").splitlines()


# ---------------------------------------------------------------------------
# 1. Contrato principal
# ---------------------------------------------------------------------------
def test_every_full_overwrite_has_advisory_lock_in_lookback(plans_lines: list[str]):
    """Cada `UPDATE meal_plans SET plan_data = %s::jsonb` en routers/plans.py
    DEBE tener `acquire_meal_plan_advisory_lock(...purpose='general'...)`
    en las 1000 líneas previas.
    """
    sites = _find_full_overwrite_sites(plans_lines)
    # Sanity: no fallar si todos los handlers ya migraron a jsonb_set.
    # En ese caso el test queda como "permitido" y se podría considerar
    # eliminarlo manualmente. Por ahora documentamos el escenario.
    if not sites:
        pytest.skip(
            "Sin full-overwrites en routers/plans.py — todos los handlers "
            "están en jsonb_set quirúrgico. Test no aplica."
        )

    offenders: list[str] = []
    for line_no, line_text in sites:
        if _has_lock_in_lookback(plans_lines, line_no):
            continue
        snippet = line_text[:160]
        offenders.append(f"  routers/plans.py:{line_no} → {snippet}")

    assert not offenders, (
        "P1-NEW-C violation: uno o más `UPDATE meal_plans SET plan_data "
        "= %s::jsonb` (full-overwrite) en routers/plans.py NO tienen un "
        "`acquire_meal_plan_advisory_lock(...purpose='general'...)` en "
        "las 1000 líneas previas. Sin ese lock, dos handlers concurrentes "
        "que ambos hacen full-overwrite producen lost-update silente "
        "(handler HTTP vs `_chunk_worker` T1/T2 cron, dos handlers HTTP "
        "paralelos del mismo usuario, etc.).\n\n"
        "Offenders:\n"
        + "\n".join(offenders)
        + "\n\nFix: añadir antes del UPDATE:\n"
        "    from db_plans import acquire_meal_plan_advisory_lock\n"
        "    acquire_meal_plan_advisory_lock(cursor, plan_id, purpose=\"general\")\n"
        "\nO migrar el endpoint a `jsonb_set` quirúrgico (mejor opción "
        "si el handler solo muta claves específicas — espejo de P0-NEW-A "
        "/swap-meal/persist o P0-NEW-B /grocery-start-date)."
    )


# ---------------------------------------------------------------------------
# 2. Cross-link slug del marker
# ---------------------------------------------------------------------------
def test_marker_anchor_present():
    """Slug del filename matchea el marker `P1-NEW-C`."""
    expected_slug = "p1_new_c"
    assert expected_slug in __file__.replace("\\", "/").lower(), (
        "El nombre de este archivo debe contener el slug del P-fix "
        "(`p1_new_c`) para que el cross-link "
        "`test_p2_hist_audit_14_marker_test_link` lo matchee cuando "
        "el marker se bumpee a `P1-NEW-C · 2026-05-11`."
    )
