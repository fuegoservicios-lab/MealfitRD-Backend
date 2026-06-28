"""[P3-BARIATRIC-PROTEIN-FLOOR-RELAX · 2026-06-28] Bariátrico + gain_muscle + piso 80g es estructuralmente
sobre-restringido: el pouch limita la ingesta, el target ya está capeado a 80g, y exigir 90% (72g) en comidas
diminutas/dulces es inviable — fuerza re-cierres agresivos (+102g) que rompen las bandas de grasa/kcal (corr=10efde32:
Día 2 65/80, band 0.58 tras +102g). Fix: para bariátrica el piso de proteína se relaja a 80% (64g) — clínicamente
razonable post-bariátrica (metas típicas 60-80g/día); la condición bariátrica manda sobre la severidad de gain_muscle.
FASE A también apunta al 80% → re-cierre menos agresivo → grasa/kcal en banda.
"""
from __future__ import annotations

from pathlib import Path

import graph_orchestrator as g

_BACKEND = Path(g.__file__).resolve().parent

_PLAN = {"macros": {"protein": "80"}, "days": [
    {"day": 1, "meals": [{"protein": 75}]},
    {"day": 2, "meals": [{"protein": 65}]},   # 81% del target
    {"day": 3, "meals": [{"protein": 73}]},
]}


def test_non_bariatric_rejects_below_90():
    short = g._protein_floor_shortfall(_PLAN, renal_capped=False, form_data={"medicalConditions": ["Ninguna"]})
    assert 2 in [d[0] for d in short], "no-bariátrico: Día 2 (65/80=81%) debe quedar bajo el piso de 90%"


def test_bariatric_relaxed_to_80():
    short = g._protein_floor_shortfall(_PLAN, renal_capped=False,
                                       form_data={"medicalConditions": ["Cirugía Bariátrica"]})
    assert short == [], f"bariátrico: 65/80=81% y 73/80=91% cumplen el piso relajado de 80%: {short}"


def test_bariatric_still_rejects_real_deficit():
    plan = {"macros": {"protein": "80"}, "days": [{"day": 1, "meals": [{"protein": 50}]}]}  # 62.5% < 80%
    short = g._protein_floor_shortfall(plan, renal_capped=False,
                                       form_data={"medicalConditions": ["Cirugía Bariátrica"]})
    assert 1 in [d[0] for d in short], "bariátrico: un déficit REAL (50/80=62%) sí debe rechazar"


def test_renal_still_exempt():
    short = g._protein_floor_shortfall(_PLAN, renal_capped=True,
                                       form_data={"medicalConditions": ["Cirugía Bariátrica"]})
    assert short == []  # renal exento siempre (techo KDIGO)


def test_knob_and_anchors():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "PROTEIN_FLOOR_HARD_PCT_BARIATRIC" in src and "P3-BARIATRIC-PROTEIN-FLOOR-RELAX" in src
    # FASE A apunta al piso relajado para bariátrica
    i = src.index("def _repair_protein_floor_post_caps")
    body = src[i:src.index("\ndef ", i + 10)]
    assert "PROTEIN_FLOOR_HARD_PCT_BARIATRIC if _is_baria else" in body
