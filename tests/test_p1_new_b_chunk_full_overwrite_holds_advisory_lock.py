"""[P1-NEW-B · 2026-05-11] Lock-the-contract: cada `UPDATE meal_plans SET
plan_data = %s::jsonb` (full-overwrite, NO jsonb_set quirúrgico) en
cron_tasks.py DEBE tener un `acquire_meal_plan_advisory_lock(...
purpose="general"...)` en las líneas previas del mismo flujo.

Motivación:
    Tras P0-NEW-A/B (2026-05-11), los escritores externos de `plan_data`
    desde el frontend están eliminados (`supabase.from('meal_plans').update(...)`
    sustituido por endpoints backend con jsonb_set quirúrgico). El único
    riesgo restante de lost-update es **inter-worker backend**:

      - `_chunk_worker` (cron_tasks.py): hace 3 full-overwrites de
        plan_data en T1 (chunk_already_merged y persist principal) y T2
        (incremental persist).
      - `_background_shift_plan_for_user` (routers/plans.py): hace 2
        full-overwrites cuando el plan necesita rolling refresh.
      - Otros workers backend que en el futuro toquen plan_data full.

    Si dos full-overwriters concurren sin coordinarse, lost-update:
    cada uno lee plan_data, muta su slice, y reescribe sobreescribiendo
    lo que el otro persistió.

    Los sitios actuales coordinan vía
    `acquire_meal_plan_advisory_lock(cursor, meal_plan_id, purpose="general")`.
    Mientras TODOS los escritores tomen el mismo lock, están serializados
    (Postgres pg_advisory_xact_lock). El riesgo es un refactor futuro
    que añada un full-overwrite SIN tomar el lock → lost-update silente.

    Este test cierra ese riesgo: enforce que cada UPDATE full-overwrite
    de plan_data tiene una llamada `acquire_meal_plan_advisory_lock`
    con `purpose="general"` en las ~2000 líneas previas (cap defensivo
    para el T1 persist principal, que vive ~1170 líneas dentro del
    mismo `with` block que el lock advisory inicial del worker).

Sitios actuales cubiertos (cron_tasks.py):
    - T1 chunk_already_merged (~L21729): re-acquire explícito a 6 líneas.
    - T1 persist principal (~L22350): lock inicial del worker ~L21177.
    - T2 incremental persist (~L22735): lock T2 propio ~L22708.

Drift detection:
    - Nuevo UPDATE full-overwrite sin lock → falla con archivo + línea.
    - Refactor que mueve el lock fuera del scope del UPDATE → falla.

Tooltip-anchor: P1-NEW-B-START | gap P1 audit 2026-05-11
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_CRON_TASKS_PY = Path(__file__).resolve().parent.parent / "cron_tasks.py"


# Patrón: UPDATE meal_plans SET plan_data = %s::jsonb (full overwrite).
# NO matchea jsonb_set quirúrgico (ese va contra una key específica).
_FULL_OVERWRITE_RE = re.compile(
    r"UPDATE\s+meal_plans\s+SET\s+plan_data\s*=\s*%s::jsonb",
    re.IGNORECASE,
)

# El lock canónico: `acquire_meal_plan_advisory_lock(cursor, ..., purpose="general")`.
# Tolerante a:
#   - Import aliased: `from db_plans import acquire_meal_plan_advisory_lock as _foo`
#     y llamada `_foo(...)`. Detectamos la llamada al alias mediante el
#     import line precedente.
#   - Llamada multilínea: regex con re.DOTALL si necesario.
_LOCK_CALL_RE = re.compile(
    r"acquire_meal_plan_advisory_lock\s*\([^)]*purpose\s*=\s*['\"]general['\"]",
    re.DOTALL,
)
# Detección de alias aliased: `as _xxx_lock` en un from-import.
_LOCK_ALIAS_IMPORT_RE = re.compile(
    r"from\s+db_plans\s+import\s+acquire_meal_plan_advisory_lock\s+as\s+(\w+)"
)

# Ventana en líneas hacia atrás (cap defensivo). El sitio más distante
# hoy es ~1170 líneas; 2000 deja margen 70% para refactors antes que el
# test deje de catchear el lock.
_LOOKBACK_LINES = 2000


def _find_full_overwrite_sites(lines: list[str]) -> list[tuple[int, str]]:
    sites: list[tuple[int, str]] = []
    for idx, line in enumerate(lines, start=1):
        # Skip puro comentario.
        if line.strip().startswith("#"):
            continue
        if _FULL_OVERWRITE_RE.search(line):
            sites.append((idx, line.strip()))
    return sites


def _has_lock_in_lookback(lines: list[str], target_line: int) -> bool:
    """True si existe una llamada `acquire_meal_plan_advisory_lock(...
    purpose="general"...)` (o su alias importado) en las
    `_LOOKBACK_LINES` líneas previas a `target_line` (1-indexed).
    """
    start = max(0, target_line - 1 - _LOOKBACK_LINES)
    end = target_line - 1
    window = "\n".join(lines[start:end])

    # 1. Llamada directa.
    if _LOCK_CALL_RE.search(window):
        return True

    # 2. Alias import detectado + llamada al alias con purpose="general".
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
def cron_lines() -> list[str]:
    return _CRON_TASKS_PY.read_text(encoding="utf-8").splitlines()


# ---------------------------------------------------------------------------
# 1. Contrato principal: cada full-overwrite tiene lock advisory previo
# ---------------------------------------------------------------------------
def test_every_full_overwrite_has_advisory_lock_in_lookback(cron_lines: list[str]):
    """Cada `UPDATE meal_plans SET plan_data = %s::jsonb` DEBE tener una
    llamada `acquire_meal_plan_advisory_lock(...purpose="general"...)`
    (o su alias importado) en las 2000 líneas previas.

    Sin esta defensa, un escritor concurrente que también haga
    full-overwrite (otro worker, futuro endpoint backend) podría leer
    el snapshot stale + reescribir → lost-update silente.
    """
    sites = _find_full_overwrite_sites(cron_lines)
    assert sites, (
        "P1-NEW-B sanity: no se encontró ningún `UPDATE meal_plans SET "
        "plan_data = %s::jsonb` en cron_tasks.py. ¿Refactor que migró "
        "todos a jsonb_set? Si sí, considerar eliminar este test y "
        "documentar el cambio. Si no, el regex se rompió."
    )

    offenders: list[str] = []
    for line_no, line_text in sites:
        if _has_lock_in_lookback(cron_lines, line_no):
            continue
        snippet = line_text[:160]
        offenders.append(f"  cron_tasks.py:{line_no} → {snippet}")

    assert not offenders, (
        "P1-NEW-B violation: uno o más `UPDATE meal_plans SET plan_data "
        "= %s::jsonb` (full-overwrite) en cron_tasks.py NO tienen un "
        "`acquire_meal_plan_advisory_lock(...purpose='general'...)` en "
        "las 2000 líneas previas. Sin ese lock, dos escritores backend "
        "concurrentes que ambos hacen full-overwrite producen "
        "lost-update silente.\n\n"
        "Offenders:\n"
        + "\n".join(offenders)
        + "\n\nFix: añadir antes del UPDATE:\n"
        "    from db_plans import acquire_meal_plan_advisory_lock\n"
        "    acquire_meal_plan_advisory_lock(cursor, meal_plan_id, purpose=\"general\")\n"
        "\n(Idempotente — los callers existentes ya usan el mismo lock; "
        "tomar el mismo purpose='general' garantiza serialización.)"
    )


# ---------------------------------------------------------------------------
# 2. Cross-link slug del marker
# ---------------------------------------------------------------------------
def test_marker_anchor_present():
    """Slug del filename matchea el marker `P1-NEW-B`."""
    expected_slug = "p1_new_b"
    assert expected_slug in __file__.replace("\\", "/").lower(), (
        "El nombre de este archivo debe contener el slug del P-fix "
        "(`p1_new_b`) para que el cross-link "
        "`test_p2_hist_audit_14_marker_test_link` lo matchee cuando "
        "el marker se bumpee a `P1-NEW-B · 2026-05-11`."
    )
