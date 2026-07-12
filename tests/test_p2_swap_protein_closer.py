"""[P2-SWAP-PROTEIN-CLOSER · 2026-07-12] El swap cierra el déficit de proteína del candidato
DETERMINÍSTICAMENTE antes del validador de macros.

Pedido del owner ("¿no puedes mejorarlo más para que sea más preciso?") tras el toast
"Plato cambiado, pero menos preciso" (moro de camarones 25g vs 38g target). Antes: el
validador (±15%) quemaba retries LLM pidiendo aritmética o entregaba con el toast. Ahora:
si el candidato queda <85% del objetivo de proteína tras el solver, corre el MISMO closer
determinista de la generación (`_close_protein_gap_for_meal`: scale-first → candidato
allergen-safe día-aware, bolt cap 180g, sweet-guard, no-dup-cheese, wording SSOT) y el
validador queda como red para lo irreparable.

Orden en la cadena del guardrail (invoke_with_retry):
  TRUTHUP → DETERMINISTIC-RESCALE → **PROTEIN-CLOSER** → validador de macros → gates.

tooltip-anchor: P2-SWAP-PROTEIN-CLOSER
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))
_AGENT = (_BACKEND / "agent.py").read_text(encoding="utf-8")


def _cands():
    mk = lambda n, p, k, c, f: SimpleNamespace(name=n, protein=p, kcal=k, carbs=c, fats=f)  # noqa: E731
    return [(0.25, "Atún en agua", mk("Atún en agua", 25.0, 100.0, 0.0, 1.0)),
            (0.20, "Pechuga de pollo", mk("Pechuga de pollo", 23.0, 114.0, 0.0, 2.0)),
            (0.18, "Queso cottage", mk("Queso cottage", 11.0, 84.0, 3.0, 2.5))]


def test_functional_deficit_closed_day_aware():
    """El core que el swap invoca: cierra a target exacto, evita labels del día, renombra
    y añade el paso SSOT (caso vivo: candidato 12g vs target 38g con pollo/huevo usados)."""
    import graph_orchestrator as go
    from nutrition_db import IngredientNutritionDB
    cand = {"meal": "Cena", "name": "Vegetales Salteados con Arroz",
            "ingredients": ["0.25 taza de arroz blanco", "1 taza de brocoli"],
            "ingredients_raw": ["0.25 taza de arroz blanco", "1 taza de brocoli"],
            "recipe": ["Saltea.", "MONTAJE: sirve."],
            "protein": 12, "carbs": 45, "fats": 18, "cals": 420}
    g = go._close_protein_gap_for_meal(
        cand, 38.0, IngredientNutritionDB(), _cands(), allergies=[], fill_pct=1.0,
        slot_cal_target=630.0, enforce_min_threshold=False,
        day_used_proteins={"huevo", "pollo"})
    assert g > 0 and cand["protein"] >= 32, "déficit material debe cerrarse (≥85% del target)"
    labels = go._protein_gate_labels_in_meal(cand)
    assert "pollo" not in labels and "huevo" not in labels, (
        "day-aware: jamás reintroduce una proteína ya usada hoy (lección del quemador same-day)"
    )
    assert "atún" in cand["name"].lower() or "atun" in cand["name"].lower(), (
        "la proteína añadida se refleja en el nombre (no esconderla)"
    )
    assert any("💪" in s for s in cand["recipe"]), "paso SSOT del closer presente"


def test_wired_between_rescale_and_validator():
    i_rs = _AGENT.find("P0-SWAP-DETERMINISTIC-RESCALE] porciones re-escaladas")
    i_pc = _AGENT.find("MEALFIT_SWAP_PROTEIN_CLOSER")
    i_val = _AGENT.find("[P1-SWAP-MACROS] Drift detectado attempt-pending")
    assert -1 not in (i_rs, i_pc, i_val)
    assert i_rs < i_pc < i_val, (
        "orden load-bearing: solver corrige ratio → closer cierra déficit residual → el "
        "validador juzga el estado FINAL (si va después del validador, sigue quemando retries)"
    )


def test_day_aware_from_gate_ssot_and_repair_semantics():
    i = _AGENT.find("tooltip-anchor: P2-SWAP-PROTEIN-CLOSER")
    win = _AGENT[i:i + 3500]
    assert "_protein_gate_labels_in_text" in win, "bans del día derivados con el SSOT del gate"
    assert "same_day_other_meal_blobs" in win
    assert "enforce_min_threshold=False" in win, (
        "semántica de REPARACIÓN (FASE A): el cierre pequeño no se bloquea por umbral de tamaño"
    )
    assert '"name"' in win, "el copy-back incluye name (el closer renombra el plato)"


def test_only_deficit_never_surplus():
    i = _AGENT.find("tooltip-anchor: P2-SWAP-PROTEIN-CLOSER")
    win = _AGENT[i:i + 2000]
    assert "_pc_cur < _pc_target * 0.85" in win, (
        "solo actúa bajo el 85% del objetivo — el exceso lo maneja el solver/validador"
    )


def test_marker_anchored():
    assert _AGENT.count("P2-SWAP-PROTEIN-CLOSER") >= 2
