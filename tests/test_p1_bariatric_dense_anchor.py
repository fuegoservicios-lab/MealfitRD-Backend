"""[P1-BARIATRIC-DENSE-ANCHOR · 2026-06-28] El esqueleto bariátrico anclaba la proteína en quesos-relleno como
"Queso de Freír" (corr=3b318e57: "Salteado de Queso de Freír" — relleno + frito) que el swap de densidad NO cazaba
(no estaban en _LOW_DENSITY_AS_MAIN). Los caps los destripaban → FASE A añadía +176g de rescate → band 0.58.

Fix: set bariátrico-only `_BARIATRIC_LOW_DENSITY_AS_MAIN` (quesos pobres-como-ancla) que se une al global SOLO si
_is_bariatric. Calibrado por review CLÍNICA (ASMBS): NO degrada cottage/ricotta/yogurt griego/claras — ésas son anclas
LEGÍTIMAS post-bariátricas (húmedas, densas, bien toleradas).
"""
from __future__ import annotations

import random
from pathlib import Path

import ai_helpers as a
from ai_helpers import _LOW_DENSITY_AS_MAIN, _BARIATRIC_LOW_DENSITY_AS_MAIN

_SRC = (Path(a.__file__).resolve().parent / "ai_helpers.py").read_text(encoding="utf-8")


def test_bad_cheeses_in_bariatric_set():
    for q in ("queso de freir", "queso blanco", "queso mozzarella", "queso de hoja",
              "queso parmesano", "queso cheddar", "queso gouda"):
        assert q in _BARIATRIC_LOW_DENSITY_AS_MAIN, q


def test_clinical_anchors_NOT_degraded():
    # cottage / yogurt griego / claras son anclas legítimas post-bariátricas → NO deben estar en el set bariátrico
    assert "queso cottage" not in _BARIATRIC_LOW_DENSITY_AS_MAIN
    assert "yogurt griego" not in _BARIATRIC_LOW_DENSITY_AS_MAIN
    assert "yogurt griego" not in _LOW_DENSITY_AS_MAIN  # ya estaba permitido (alta densidad)
    assert not any("clara" in x for x in _BARIATRIC_LOW_DENSITY_AS_MAIN)


def test_bariatric_set_not_in_global():
    # no contamina el set global → gain_muscle (no bariátrico) puede usar estos quesos como main
    for q in _BARIATRIC_LOW_DENSITY_AS_MAIN:
        assert q not in _LOW_DENSITY_AS_MAIN, q


def test_branch_present_and_knob_reused():
    assert "_BARIATRIC_LOW_DENSITY_AS_MAIN" in _SRC
    assert "_is_bariatric and _pl in _BARIATRIC_LOW_DENSITY_AS_MAIN" in _SRC
    # reusa el knob existente, NO crea uno nuevo
    assert "MEALFIT_GAINMUSCLE_HIGH_DENSITY_PROTEIN" in _SRC
    assert "MEALFIT_BARIATRIC_DENSE" not in _SRC and "MEALFIT_BARIATRIC_PROTEIN_DENSITY_ANCHOR" not in _SRC


def test_functional_bariatric_no_freir_as_main():
    # bariátrico balanced: "queso de freír" no debe sobrevivir como proteína principal en el esqueleto
    baria = {"mainGoal": "maintain", "medicalConditions": ["Cirugía Bariátrica"],
             "allergies": [], "dislikes": [], "dietType": ""}
    hits = 0
    for seed in range(25):
        random.seed(seed)
        p = a.get_deterministic_variety_prompt("", form_data=baria)
        if "queso de fre" in p.lower():
            hits += 1
    assert hits == 0, f"'queso de freír' apareció como main en {hits}/25 prompts bariátricos"


def test_functional_gainmuscle_nonbariatric_unaffected():
    # control no-regresión: gain_muscle NO bariátrico no usa el set bariátrico (genera sin crash)
    gm = {"mainGoal": "gain_muscle", "medicalConditions": [], "allergies": [], "dislikes": [], "dietType": ""}
    random.seed(1)
    p = a.get_deterministic_variety_prompt("", form_data=gm)
    assert isinstance(p, str) and p
