"""[P2-NEW-9 · 2026-05-11] `_alert_high_synthesized_lesson_ratio` excluye
chunks de planes `abandoned`/`cancelled` del cálculo del ratio.

Bug original (audit 2026-05-11):
    El cron contaba `total_chunks` filtrando solo por
    `status IN ('completed', 'failed') AND updated_at > window`. Planes
    marcados `abandoned` por P1-LIVE-3 (sintéticos) o P2-NEXT-3 (orphans)
    tenían sus chunks completados sumados al denominador SIN sumar al
    numerador (sus lessons nunca se observan por usuarios → no inflan
    `synthesized_events`). Resultado: ratio falso-negativo cuando se
    acumulaba stock de planes abandoned → alerta SRE válida suprimida
    cuando learning realmente estaba roto.

Fix:
    AND NOT EXISTS (
        SELECT 1 FROM meal_plans mp
        WHERE mp.id = pcq.meal_plan_id
          AND mp.plan_data->>'generation_status' IN ('abandoned', 'cancelled')
    )

Estrategia del test:
    1. La query DEBE contener la cláusula NOT EXISTS contra meal_plans
       filtrando los dos generation_status.
    2. El marker `P2-NEW-9` aparece en el comentario inline para que un
       futuro refactor sepa por qué la subquery existe.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CRON_FP = _REPO_ROOT / "backend" / "cron_tasks.py"


@pytest.fixture(scope="module")
def src() -> str:
    return _CRON_FP.read_text(encoding="utf-8")


def test_synth_ratio_query_excludes_abandoned(src: str):
    """La query de `_alert_high_synthesized_lesson_ratio` debe excluir
    chunks de planes abandoned/cancelled del denominador."""
    func_start = src.find("def _alert_high_synthesized_lesson_ratio(")
    assert func_start > 0, "_alert_high_synthesized_lesson_ratio no encontrado"
    # Boundary: la siguiente función o ~600 líneas máx (la función es chica).
    body = src[func_start:func_start + 8000]

    # Patrón canónico: NOT EXISTS contra meal_plans con los 2 status.
    pattern = re.compile(
        r"NOT\s+EXISTS\s*\(\s*"
        r"SELECT\s+1\s+FROM\s+meal_plans\s+mp\s+"
        r"WHERE\s+mp\.id\s*=\s*pcq\.meal_plan_id\s+"
        r"AND\s+mp\.plan_data->>'generation_status'\s+IN\s*\(\s*"
        r"'abandoned'\s*,\s*'cancelled'\s*\)",
        re.IGNORECASE | re.MULTILINE,
    )
    assert pattern.search(body), (
        "P2-NEW-9 regresión: la query NO contiene la subquery NOT EXISTS "
        "que excluye chunks de planes abandoned/cancelled. Sin esta exclusión, "
        "el ratio synth:real puede falsearse cuando se acumula stock de "
        "abandoned (sus chunks completed inflan total_chunks pero NO inflan "
        "synthesized_events). Alerta válida suprimida."
    )


def test_pcq_alias_used_in_select_and_subquery(src: str):
    """Sanity check: la subquery usa `pcq.meal_plan_id` y `pcq.status`,
    es decir, el SELECT principal aliasa `plan_chunk_queue` como `pcq`."""
    func_start = src.find("def _alert_high_synthesized_lesson_ratio(")
    body = src[func_start:func_start + 8000]
    assert "FROM plan_chunk_queue pcq" in body, (
        "P2-NEW-9 regresión: el SELECT ya no usa el alias `pcq` para "
        "plan_chunk_queue. La subquery NOT EXISTS depende de ese alias "
        "para correlacionar `pcq.meal_plan_id`."
    )


def test_marker_present_in_inline_comment(src: str):
    """El marker `P2-NEW-9` debe aparecer en el comentario inline para
    documentar por qué la subquery existe."""
    func_start = src.find("def _alert_high_synthesized_lesson_ratio(")
    body = src[func_start:func_start + 8000]
    assert "P2-NEW-9" in body, (
        "P2-NEW-9 regresión: el comentario inline ya no menciona el marker. "
        "Sin él, un revisor futuro puede simplificar la subquery creyendo "
        "que es código muerto."
    )
