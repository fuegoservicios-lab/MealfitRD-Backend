"""[P2-CRITIQUE-RESPECTS-ROUTINE · 2026-09-04] La autocrítica respeta la política de recurrencia.

Logs del plan 55cf90bb del dueño (F3 en `enforce`, «Rutina» elegida en el formulario): el
paso SELF-CRITIQUE marcó «staples repetidos: avena en 2 días, yogur griego en 3, lechosa» y
reescribió los 3 días (2,5 min de los 8); el día 3 falló dos veces al reescribirse y quedó con
`_critique_unresolved`, lo que disparó además la regen quirúrgica post-aprobación. Los cuatro
contadores determinísticos de REPETICIÓN (staples cross-día, ingrediente concentrado, plato-base
repetido, monotonía de proteína pesada) entran al evaluador como «no opinable» y fuerzan
needs_correction, contradiciendo lo que el usuario pidió y lo que el revisor ya respeta
(`filter_variety_issues_for_policy`, F3).

Arreglo: `horizon.filter_repetition_counts_for_policy` (misma regla por modo que el revisor)
aplicada a los cuatro contadores en `self_critique_node`. Sin enforce (shadow) nada cambia.
"""
from __future__ import annotations

import re
from pathlib import Path

import horizon as h

_BACKEND = Path(__file__).resolve().parents[1]


def _eff(mode: str, anchors=()):
    return {
        "recurrence": {"global_mode": mode, "slot_modes": {}},
        "food_anchors": [{"name": n, "ingredient_id": n.lower()} for n in anchors],
    }


COUNTS = {"avena": 2, "yogurt griego": 3, "lechosa": 2}


def test_routine_enforced_drops_every_repetition_signal():
    assert h.filter_repetition_counts_for_policy(COUNTS, _eff("routine"), enforced=True) == {}


def test_balanced_enforced_drops_only_anchor_signals():
    kept = h.filter_repetition_counts_for_policy(COUNTS, _eff("balanced", anchors=("Avena",)), enforced=True)
    assert kept == {"yogurt griego": 3, "lechosa": 2}


def test_explore_enforced_keeps_everything():
    assert h.filter_repetition_counts_for_policy(COUNTS, _eff("explore"), enforced=True) == COUNTS


def test_shadow_or_no_policy_keeps_everything():
    assert h.filter_repetition_counts_for_policy(COUNTS, _eff("routine"), enforced=False) == COUNTS
    assert h.filter_repetition_counts_for_policy(COUNTS, None, enforced=True) == COUNTS
    assert h.filter_repetition_counts_for_policy(COUNTS, {}, enforced=True) == COUNTS
    assert h.filter_repetition_counts_for_policy(None, _eff("routine"), enforced=True) == {}


def test_self_critique_node_filters_the_four_counters():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    start = src.index("async def self_critique_node(")
    body = src[start:start + 40_000]
    assert "from horizon import filter_repetition_counts_for_policy as _filter_rep_counts" in body
    assert 'bool(form_data.get("_policy_enforced"))' in body
    for call in (
        '_sc_policy_filter(_count_staple_repetitions(days), "staples cross-día")',
        '_count_cross_day_heavy_protein_repetition(days), "monotonía de proteína pesada")',
        '"ingrediente concentrado")',
        '"plato-base repetido")',
    ):
        assert call in body, call
    # ningún contador de repetición queda asignado SIN pasar por el filtro
    assert not re.search(r"\n\s*staple_repetitions = _count_staple_repetitions\(days\)\n", body)
    assert not re.search(r"\n\s*heavy_protein_monotony = _count_cross_day_heavy_protein_repetition\(days\)\n", body)
    assert not re.search(r"\n\s*cross_day_dish_repeats = build_variety_report\(", body)


def test_marker_present():
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    assert "P2-CRITIQUE-RESPECTS-ROUTINE · 2026-09-04" in app
