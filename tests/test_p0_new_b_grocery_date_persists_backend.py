"""[P0-NEW-B · 2026-05-11] Test parser-based: la persistencia del
`grocery_start_date` y `cycle_start_date` DEBE pasar por el endpoint
backend `/api/plans/{plan_id}/grocery-start-date`, NO por un direct
write desde el frontend a `meal_plans`.

Bug original (gap P0 detectado audit 2026-05-11):
    `frontend/src/context/AssessmentContext.jsx:560` hacía
    ``supabase.from('meal_plans').update({plan_data: latestPlan}).eq('id', planId)``
    cuando inyectaba `grocery_start_date` y/o `cycle_start_date` en un
    plan recién cargado. Mismo modo de fallo que P0-NEW-A (swap):

      - Frontend leía `latestPlan` (snapshot inicial del cargador).
      - El cron `_resolve_grocery_start_date` (cron_tasks.py:15327)
        podía estar rellenando `grocery_start_date` en paralelo via
        jsonb_set quirúrgico con idempotencia
        `(plan_data->>'grocery_start_date') IS NULL`.
      - `_chunk_worker` podía estar persistiendo `days[*]` o
        `_chunk_lessons` simultáneamente.
      - El frontend pisaba TODO con `latestPlan` (snapshot del state
        local) cuando solo quería actualizar 2 keys, perdiendo lo que
        cron/worker habían persistido.

    El riesgo agregado vs swap (P0-NEW-A) es menor porque el path solo
    se ejecuta una vez al cargar el plan (no en hot loop), pero el modo
    de fallo es idéntico y bypassea la idempotencia del cron.

Cierre P0-NEW-B:
    1. Backend gana endpoint `/{plan_id}/grocery-start-date` que hace
       `jsonb_set(plan_data, '{grocery_start_date}', %s, true)` + UPDATE
       separado para `cycle_start_date`. Cada UPDATE incluye:
         - `AND user_id = %s` defense-in-depth.
         - `AND (plan_data->>'<key>') IS NULL` idempotencia (mismo
           patrón que el cron upstream).
    2. Frontend reemplaza el `supabase.from('meal_plans').update(...)`
       por `fetchWithAuth(POST /grocery-start-date)`.

Tests (parser-based, sin DB ni red):
    1. `test_handler_exists_with_jsonb_set_idempotency_and_user_id` —
       `api_set_grocery_start_date` existe; su SQL usa `jsonb_set`,
       `(plan_data->>'grocery_start_date') IS NULL` idempotencia, y
       filtra `AND user_id = %s`.
    2. `test_frontend_grocery_uses_backend_endpoint` — AssessmentContext.jsx
       referencia `/api/plans/${planId}/grocery-start-date`.
    3. `test_frontend_grocery_no_longer_direct_writes_meal_plans` — el
       bloque condicional `if (didInjectGroceryDate && ...)` ya NO
       contiene `supabase.from('meal_plans').update({plan_data`.
    4. `test_marker_anchor_present` — slug `p0_new_b` aparece en el
       filename.

Tooltip-anchor: P0-NEW-B-START | gap P0 audit 2026-05-11
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PLANS_PY = _REPO_ROOT / "backend" / "routers" / "plans.py"
_ASSESSMENT_JSX = _REPO_ROOT / "frontend" / "src" / "context" / "AssessmentContext.jsx"


def _extract_function_body(src: str, fn_name: str) -> str:
    pattern = re.compile(rf"def\s+{re.escape(fn_name)}\s*\(")
    m = pattern.search(src)
    if not m:
        raise AssertionError(
            f"No se encontró `def {fn_name}(` en plans.py — endpoint "
            f"renombrado/eliminado."
        )
    start = m.start()
    next_def = re.search(r"\n(?:@router\.|@app\.|def\s)", src[start + 1:])
    end = (start + 1 + next_def.start()) if next_def else len(src)
    return src[start:end]


@pytest.fixture(scope="module")
def plans_src() -> str:
    return _PLANS_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def assessment_src() -> str:
    return _ASSESSMENT_JSX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def grocery_body(plans_src: str) -> str:
    return _extract_function_body(plans_src, "api_set_grocery_start_date")


# ---------------------------------------------------------------------------
# 1. Backend handler: jsonb_set + idempotencia + user_id filter
# ---------------------------------------------------------------------------
def test_handler_exists_with_jsonb_set_idempotency_and_user_id(grocery_body: str):
    """El handler `api_set_grocery_start_date` debe:
      - Usar `jsonb_set(...)` (NO full overwrite del plan_data).
      - Incluir `(plan_data->>'grocery_start_date') IS NULL` para
        idempotencia espejo del cron upstream (cron_tasks.py:15327).
      - Filtrar `AND user_id = %s` defense-in-depth.

    Sin idempotencia, dos calls concurrentes (frontend + cron) podrían
    racear y pisarse — la primera fila a llegar gana, la segunda se
    convierte en no-op silente, lo cual ya cumple el contrato pero
    `IS NULL` lo hace explícito y enforced a DB-level.
    """
    assert "jsonb_set" in grocery_body, (
        "P0-NEW-B regresión: `api_set_grocery_start_date` no usa "
        "`jsonb_set`. Si degradó a full overwrite, reabriría el "
        "lost-update que P0-NEW-B cerró."
    )

    grocery_idempotency = re.search(
        r"\(\s*plan_data\s*->>\s*'grocery_start_date'\s*\)\s+IS\s+NULL",
        grocery_body,
        re.IGNORECASE,
    )
    assert grocery_idempotency, (
        "P0-NEW-B regresión: el UPDATE de grocery_start_date no incluye "
        "`(plan_data->>'grocery_start_date') IS NULL`. Sin esta condición, "
        "el endpoint podría pisar un valor recién persistido por el cron "
        "`_resolve_grocery_start_date` (espejo de la idempotencia del "
        "cron upstream)."
    )

    cycle_idempotency = re.search(
        r"\(\s*plan_data\s*->>\s*'cycle_start_date'\s*\)\s+IS\s+NULL",
        grocery_body,
        re.IGNORECASE,
    )
    assert cycle_idempotency, (
        "P0-NEW-B regresión: el UPDATE de cycle_start_date no incluye "
        "`(plan_data->>'cycle_start_date') IS NULL`. Misma razón que el "
        "grocery: idempotencia explícita a DB-level."
    )

    # AND user_id = %s en al menos un WHERE del handler (cubre ambos
    # UPDATE separados).
    user_id_filter = re.findall(
        r"AND\s+user_id\s*=\s*%s",
        grocery_body,
        re.IGNORECASE,
    )
    assert len(user_id_filter) >= 2, (
        "P0-NEW-B regresión: el handler no filtra `AND user_id = %s` en "
        "ambos UPDATE (grocery + cycle). Sin esto, un futuro refactor del "
        "ownership check upstream podría abrir IDOR a DB-level. Restaurar "
        "el doble candado (espejo de retry-chunk P0-HIST-IDOR-1)."
    )


# ---------------------------------------------------------------------------
# 2. Frontend invoca el nuevo endpoint
# ---------------------------------------------------------------------------
def test_frontend_grocery_uses_backend_endpoint(assessment_src: str):
    """AssessmentContext.jsx debe invocar
    `/api/plans/${planId}/grocery-start-date` cuando `didInjectGroceryDate`
    sea true. Sin esta llamada, las fechas inyectadas solo viven en state
    local y se pierden en el próximo refresh o se pisan con un snapshot
    obsoleto desde la nube.
    """
    pattern = re.compile(
        r"/api/plans/\$\{[^}]*\}/grocery-start-date",
    )
    assert pattern.search(assessment_src), (
        "P0-NEW-B regresión: AssessmentContext.jsx NO referencia "
        "`/api/plans/${planId}/grocery-start-date`. La rama "
        "`didInjectGroceryDate` debe POST-ear al endpoint backend en "
        "lugar de hacer `supabase.from('meal_plans').update({plan_data})` "
        "directo (que produce lost-update con el cron y _chunk_worker)."
    )


# ---------------------------------------------------------------------------
# 3. Frontend ya no escribe direct a meal_plans en el bloque grocery
# ---------------------------------------------------------------------------
def test_frontend_grocery_no_longer_direct_writes_meal_plans(assessment_src: str):
    """El bloque condicional `if (didInjectGroceryDate && ...)` ya NO
    debe contener `supabase.from('meal_plans').update({plan_data:...})`.

    Escaneamos solo el bloque del grocery (delimitado por la condición
    `didInjectGroceryDate &&`) porque el resto del archivo cae bajo
    P0-NEW-A (swap) o es legítimo (lecturas SELECT, etc.).
    """
    marker_idx = assessment_src.find("didInjectGroceryDate &&")
    assert marker_idx >= 0, (
        "P0-NEW-B regresión: no se encontró el marcador "
        "`didInjectGroceryDate &&` en AssessmentContext.jsx — refactor "
        "del flow. Si la condición cambió de nombre, actualizar el "
        "marker en este test."
    )
    # Escanear los próximos ~1500 chars (suficiente para cubrir el bloque
    # condicional completo incluyendo .then/.catch).
    grocery_block = assessment_src[marker_idx : marker_idx + 1500]
    # Strip líneas que son puro comentario JS para evitar falsos positivos
    # en la documentación inline del fix P0-NEW-B, que cita el patrón
    # legacy en un comentario explicativo.
    no_comments = re.sub(
        r"^\s*//.*$",
        "",
        grocery_block,
        flags=re.MULTILINE,
    )
    legacy_pattern = re.compile(
        r"supabase\s*\.\s*from\s*\(\s*['\"]meal_plans['\"]\s*\)"
        r"\s*\.\s*update\s*\(\s*\{\s*plan_data",
        re.DOTALL,
    )
    matches = legacy_pattern.findall(no_comments)
    assert not matches, (
        "P0-NEW-B regresión: el bloque `if (didInjectGroceryDate && ...)` "
        "todavía contiene `supabase.from('meal_plans').update({plan_data:...})` "
        "como línea ejecutable. Este patrón produce lost-update con el "
        "cron y _chunk_worker. Restaurar el reemplazo por "
        "`fetchWithAuth(POST /grocery-start-date)`."
    )


# ---------------------------------------------------------------------------
# 4. Cross-link slug del marker
# ---------------------------------------------------------------------------
def test_marker_anchor_present():
    """Slug del filename matchea el marker `P0-NEW-B`."""
    expected_slug = "p0_new_b"
    assert expected_slug in __file__.replace("\\", "/").lower(), (
        "El nombre de este archivo debe contener el slug del P-fix "
        "(`p0_new_b`) para que el cross-link "
        "`test_p2_hist_audit_14_marker_test_link` lo matchee cuando "
        "se bumpee `_LAST_KNOWN_PFIX` a `P0-NEW-B · 2026-05-11`."
    )
