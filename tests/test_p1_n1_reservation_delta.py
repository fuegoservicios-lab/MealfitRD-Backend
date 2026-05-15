"""[P1-N1-RESERVATION-DELTA · 2026-05-15] Anchor parser-based:
`reserve_plan_ingredients` debe hacer batch-fetch del inventory del user
UNA VEZ y pasar `prefetched_rows=` a cada llamada a `_apply_reservation_delta`,
evitando 30+ SELECTs por plan.

Contexto del bug original:
    Plan con 30 ingredientes = 30 roundtrips Supabase. Cada
    `_apply_reservation_delta` ejecutaba su propio
    `supabase.table("user_inventory").select(...).execute()` per
    ingrediente. CAS-retry mitigaba lost-update pero NO el N+1.

Fix:
    - `_apply_reservation_delta` acepta `prefetched_rows: Optional[List[Dict]] = None`.
    - Primer attempt usa `prefetched_rows` si != None; retries siempre
      re-SELECT (state fresh post-conflict).
    - `reserve_plan_ingredients` hace batch SELECT upfront indexado por
      `ingredient_name` y pasa subset a cada call.

Tooltip-anchor: P1-N1-RESERVATION-DELTA-START
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent.parent
_DB_INVENTORY = _BACKEND / "db_inventory.py"


@pytest.fixture(scope="module")
def db_inv_src() -> str:
    return _DB_INVENTORY.read_text(encoding="utf-8")


def _extract_fn_body(src: str, fn_name: str) -> str:
    anchor = re.search(rf"def {re.escape(fn_name)}\b[^\n]*\n", src)
    assert anchor is not None, f"function `{fn_name}` not found — ¿renombrada?"
    start = anchor.end()
    rest = src[start:]
    next_decl = re.search(r"\n(?:def |class |# ---)", rest)
    end = start + (next_decl.start() if next_decl else len(rest))
    return src[start:end]


def test_apply_reservation_delta_accepts_prefetched(db_inv_src: str):
    """`_apply_reservation_delta` debe declarar el parámetro `prefetched_rows`."""
    assert re.search(
        r"def _apply_reservation_delta\([\s\S]+?prefetched_rows:\s*Optional",
        db_inv_src,
    ), (
        "P1-N1-RESERVATION-DELTA: `_apply_reservation_delta` no acepta "
        "`prefetched_rows` — el batch-fetch upstream no puede compartir filas."
    )


def test_first_attempt_uses_prefetched(db_inv_src: str):
    """En attempt==0, si prefetched_rows is not None, debe usarse en vez de
    SELECT."""
    body = _extract_fn_body(db_inv_src, "_apply_reservation_delta")
    assert re.search(
        r"attempt\s*==\s*0[\s\S]{0,200}prefetched_rows\s+is\s+not\s+None",
        body,
    ), (
        "P1-N1-RESERVATION-DELTA: la rama `attempt==0 AND prefetched_rows` "
        "no aparece — el primer attempt sigue haciendo SELECT redundante."
    )


def test_retries_still_refetch(db_inv_src: str):
    """En attempts >= 1, debe forzarse SELECT fresh (no usar prefetched stale)."""
    body = _extract_fn_body(db_inv_src, "_apply_reservation_delta")
    # El else-branch del check `attempt == 0 and prefetched_rows is not None`
    # debe contener el SELECT fresh.
    assert re.search(
        r"else:\s*\n[\s\S]{0,200}supabase\.table\(\"user_inventory\"\)\.select",
        body,
    ), (
        "P1-N1-RESERVATION-DELTA: el else-branch (retries) ya no hace SELECT "
        "fresh — usar prefetched stale en retries podría perder mutaciones "
        "concurrentes detectadas via CAS."
    )


def test_reserve_plan_does_batch_fetch(db_inv_src: str):
    """`reserve_plan_ingredients` debe hacer un SELECT al inicio agrupando
    por `ingredient_name`."""
    body = _extract_fn_body(db_inv_src, "reserve_plan_ingredients")
    assert "rows_by_name" in body, (
        "P1-N1-RESERVATION-DELTA: el diccionario `rows_by_name` (batch index "
        "por ingredient_name) no aparece."
    )
    assert re.search(
        r"supabase\.table\(\"user_inventory\"\)\.select",
        body,
    ), (
        "P1-N1-RESERVATION-DELTA: el SELECT batch upfront sobre user_inventory "
        "no aparece dentro de `reserve_plan_ingredients`."
    )


def test_call_passes_prefetched(db_inv_src: str):
    """La llamada a `_apply_reservation_delta` desde reserve_plan_ingredients
    debe pasar `prefetched_rows=`."""
    body = _extract_fn_body(db_inv_src, "reserve_plan_ingredients")
    assert re.search(
        r"_apply_reservation_delta\([\s\S]{0,300}prefetched_rows\s*=",
        body,
    ), (
        "P1-N1-RESERVATION-DELTA: `reserve_plan_ingredients` ya no pasa "
        "`prefetched_rows=` — la optimización quedó dead code."
    )


def test_marker_tooltip_present(db_inv_src: str):
    assert "P1-N1-RESERVATION-DELTA" in db_inv_src, (
        "Marker tooltip ausente — si renombras el bloque, actualiza este test."
    )
