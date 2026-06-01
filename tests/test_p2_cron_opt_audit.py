"""[P2-CRON-OPT · 2026-05-29] Anchor + regresión del pass de optimización de
`cron_tasks.py` (audit multi-agente 2026-05-29).

Este test consolida los invariantes estructurales clave que el pass de optimización
introdujo, para que un refactor futuro que los revierta falle loud. Cubre los fixes de
mayor valor de los 6 gaps; los detalles conductuales viven en sus tests dedicados
(`test_gap_1_consumed_meals_fetcher_contract.py`, `test_gap_4_5_inject_helpers.py`).

Es además el test cross-link del marker `_LAST_KNOWN_PFIX = "P2-CRON-OPT · 2026-05-29"`
(slug `p2_cron_opt`, enforzado por `test_p2_hist_audit_14_marker_test_link.py`).

Tooltip-anchor: P2-CRON-OPT.
"""
from __future__ import annotations

import re
from pathlib import Path

import cron_tasks


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CRON_PY = _REPO_ROOT / "backend" / "cron_tasks.py"


def _src() -> str:
    return _CRON_PY.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# GAP-1: el bug estrella (feature silenciosamente roto)
# ---------------------------------------------------------------------------

def test_gap1_fatigue_and_dow_read_consumed_at():
    src = _src()
    # Ninguna de las dos funciones debe volver a leer la clave inexistente created_at.
    for fn in ("calculate_ingredient_fatigue", "calculate_day_of_week_adherence"):
        start = src.find(f"def {fn}(")
        assert start >= 0
        after = src[start:]
        nxt = re.search(r"\ndef \w", after[10:])
        body = after[: nxt.start() + 10] if nxt else after
        assert "meal.get('created_at')" not in body, f"{fn} no debe leer created_at"


# ---------------------------------------------------------------------------
# GAP-3 / GAP-6: SQL pushdown + colapsos
# ---------------------------------------------------------------------------

def test_s15_3_jsonb_compare_not_text_cast():
    src = _src()
    assert "pipeline_snapshot IS DISTINCT FROM '{}'::jsonb" in src
    # `WHERE pipeline_snapshot::text` sería el uso en SQL (no el literal en el comentario
    # explicativo que cita el patrón viejo). El GC ya no castea ~10MB JSONB a texto/fila.
    assert "WHERE pipeline_snapshot::text" not in src, (
        "S15-3: el GC de snapshots no debe castear ~10MB JSONB a texto por fila"
    )


def test_s10_6_no_redundant_counter_imports():
    src = _src()
    # Counter se importa UNA vez a nivel módulo; cero imports dentro de funciones.
    assert src.count("from collections import Counter") == 1


def test_s18_4_no_correlated_subquery_in_rolling_refill():
    src = _src()
    assert "SELECT id FROM meal_plans mp2" not in src, (
        "S18-4: la subquery correlacionada redundante debe estar eliminada"
    )


# ---------------------------------------------------------------------------
# GAP-4: SSOT helpers + divergence bug
# ---------------------------------------------------------------------------

def test_s10_1_s10_2_helpers_exist():
    assert callable(getattr(cron_tasks, "_should_auto_activate_adversarial", None))
    assert callable(getattr(cron_tasks, "_quality_trend_hint", None))
    assert callable(getattr(cron_tasks, "_consumed_for_window", None))
    assert callable(getattr(cron_tasks, "_coerce_consumed_at_to_dt", None))


def test_s12_2_dead_letter_cooldown_has_resolved_at_filter():
    """El cooldown SELECT del dead-letter alert debe filtrar resolved_at IS NULL
    (alineado con sus hermanos) — pre-fix suprimía re-alertas legítimas."""
    src = _src()
    start = src.find("def _alert_new_dead_lettered_chunks(")
    assert start >= 0
    nxt = src.find("\ndef ", start + 10)
    body = src[start:nxt] if nxt > 0 else src[start:]
    # Localizar el cooldown SELECT y verificar que incluye resolved_at IS NULL.
    m = re.search(r"SELECT triggered_at FROM system_alerts.*?LIMIT 1", body, re.S)
    assert m, "no se encontró el cooldown SELECT en _alert_new_dead_lettered_chunks"
    assert "resolved_at IS NULL" in m.group(0), (
        "S12-2: el cooldown del dead-letter debe filtrar resolved_at IS NULL"
    )


# ---------------------------------------------------------------------------
# GAP-6: S14-1 batched RETURNING + I2-EXEMPT
# ---------------------------------------------------------------------------

def test_s14_1_batched_blanket_reactivation():
    src = _src()
    start = src.find("def _reactivate_shopping_list_after_perishable_cycle(")
    assert start >= 0
    nxt = src.find("\ndef ", start + 10)
    body = src[start:nxt] if nxt > 0 else src[start:]
    # Debe usar un UPDATE ... RETURNING id (batched) y tener el marker I2-EXEMPT.
    assert "RETURNING id" in body
    assert "I2-EXEMPT" in body, "S14-1: sweep cross-user requiere marker I2-EXEMPT"
