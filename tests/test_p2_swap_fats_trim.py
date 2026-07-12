"""[P2-SWAP-FATS-TRIM · 2026-07-12] El swap recorta el EXCESO de grasa del candidato
determinísticamente antes del validador — espejo del closer de proteína (déficit).

Caso vivo (regen-día v3, 05:28Z, plan 1bfda745): el validador quemó los 3 reintentos LLM
del moro por fats=8g vs target 5g (delta de 3g!) → SWAP_LLM_RETRIES_EXHAUSTED → slot
conservado ×2 (moro + panqueques) → el día sin libertad para cuadrar → band 0.5 con
carbs/kcal fuera. Un recorte determinista de 3g de grasa habría salvado ambos candidatos.

Cadena del guardrail: TRUTHUP → RESCALE → PROTEIN-CLOSER (déficit) → **FATS-TRIM (exceso)**
→ validador. Reusa `_trim_day_fats_to_target` (SSOT de S1/relevel: shrink de fuentes de
grasa añadida, portadores de micros protegidos — el MISMO helper que acaba de validarse en
vivo vía P2-REGEN-DAY-FATS-RELEVEL). Solo actúa sobre exceso MATERIAL (>115%, el umbral
del validador). Nota local-env: el helper resuelve densidades vía catálogo Neon — la
validación funcional es en prod (el relevel ya lo probó); estos tests anclan el wiring.

tooltip-anchor: P2-SWAP-FATS-TRIM
"""
from __future__ import annotations

from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_AGENT = (_BACKEND / "agent.py").read_text(encoding="utf-8")


def test_wired_after_protein_closer_before_validator():
    i_pc = _AGENT.find("tooltip-anchor: P2-SWAP-PROTEIN-CLOSER")
    i_ft = _AGENT.find("MEALFIT_SWAP_FATS_TRIM")
    i_val = _AGENT.find("[P1-SWAP-MACROS] Drift detectado attempt-pending")
    assert -1 not in (i_pc, i_ft, i_val)
    assert i_pc < i_ft < i_val, (
        "orden load-bearing: closer (déficit de proteína) → trim (exceso de grasa) → el "
        "validador juzga el estado FINAL — si el trim va después, sigue quemando retries"
    )


def test_reuses_ssot_trimmer_and_threshold():
    i = _AGENT.find("tooltip-anchor: P2-SWAP-FATS-TRIM")
    win = _AGENT[i:i + 3000]
    assert "_trim_day_fats_to_target" in win, (
        "reusa el SSOT de S1 (mismo helper del relevel validado en vivo) — cero lógica nueva"
    )
    assert "_ft_cur > _ft_target * 1.15" in win, (
        "solo exceso MATERIAL (>115% = umbral del validador); el déficit es del closer"
    )
    assert "_truth_up_meal_macros_from_strings" in win and "_sync_recipe_step_quantities" in win, (
        "tras el trim: números honestos desde strings + pasos re-sincronizados"
    )


def test_copy_back_and_failsafe():
    i = _AGENT.find("tooltip-anchor: P2-SWAP-FATS-TRIM")
    win = _AGENT[i:i + 3500]
    assert '"ingredients_raw"' in win, "lockstep raw en el copy-back (la lista lee raw)"
    assert "except Exception as _ft_exc" in win, "fail-safe total: jamás aborta el swap"


def test_marker_anchored():
    assert _AGENT.count("P2-SWAP-FATS-TRIM") >= 2
