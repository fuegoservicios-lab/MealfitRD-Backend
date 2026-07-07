"""[P1-RECIPE-QTY-SYNC · 2026-07-01] (audit recetas P1-A)

El prompt exige MEDIDAS en los pasos ("Mise en place: … pesa 50 g de arroz") pero el solver/caps/
closers/quantize reescalan `ingredients[]` DESPUÉS de escritos los pasos sin tocar `recipe[]` →
el paso decía 50 g mientras ingredients/lista decían 80 g: el usuario cocinaba mal el plato con
macros "clavados". Afectaba a TODO plan de formulario y a los updates.

Fix: `_sync_recipe_step_quantities` (determinista, conservador, fail-safe) reescribe la mención
"<qty> <unit> de <alimento>" del paso a la cantidad ACTUAL del ingrediente, cableado post-quantize
en assemble + finalizador de updates + persist boundary.
"""
from __future__ import annotations

from pathlib import Path

import graph_orchestrator as g

_GRAPH = (Path(g.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")


def _meal(ings, steps):
    return {"name": "Plato", "ingredients": list(ings), "recipe": list(steps)}


def test_step_qty_resyncs_to_ingredient(monkeypatch):
    monkeypatch.setattr(g, "RECIPE_STEP_QTY_SYNC_ENABLED", True)
    meal = _meal(
        ["80 g de arroz blanco", "2 cdas de aceite de oliva"],
        ["Mise en place: pesa 50 g de arroz y pica la cebolla.",
         "El Toque de Fuego: cocina el arroz 15 min a fuego medio.",
         "Montaje: sirve caliente."],
    )
    n = g._sync_recipe_step_quantities(meal)
    assert n == 1, f"esperada 1 mención reescrita, dio {n}"
    assert "80 g de arroz" in meal["recipe"][0], meal["recipe"][0]
    assert "50 g" not in meal["recipe"][0]
    assert meal["recipe"][1] == "El Toque de Fuego: cocina el arroz 15 min a fuego medio.", \
        "menciones sin cantidad no se tocan"


def test_unit_change_also_syncs(monkeypatch):
    """quantize convierte '1 taza' → gramos: el paso hereda la unidad actual."""
    monkeypatch.setattr(g, "RECIPE_STEP_QTY_SYNC_ENABLED", True)
    meal = _meal(["80 g de avena"], ["Mise en place: mide 1 taza de avena."])
    assert g._sync_recipe_step_quantities(meal) == 1
    assert "80 g de avena" in meal["recipe"][0]


def test_matching_qty_is_noop(monkeypatch):
    monkeypatch.setattr(g, "RECIPE_STEP_QTY_SYNC_ENABLED", True)
    meal = _meal(["80 g de arroz blanco"], ["Mise en place: pesa 80 g de arroz."])
    assert g._sync_recipe_step_quantities(meal) == 0


def test_ambiguous_token_untouched(monkeypatch):
    """Dos ingredientes con el mismo token principal → no reescribir (no se sabe cuál es cuál)."""
    monkeypatch.setattr(g, "RECIPE_STEP_QTY_SYNC_ENABLED", True)
    meal = _meal(["80 g de arroz blanco", "50 g de arroz integral"],
                 ["Mise en place: pesa 60 g de arroz."])
    assert g._sync_recipe_step_quantities(meal) == 0
    assert "60 g" in meal["recipe"][0]


def test_safety_notes_untouched(monkeypatch):
    monkeypatch.setattr(g, "RECIPE_STEP_QTY_SYNC_ENABLED", True)
    note = "⚠️ Seguridad alimentaria: usa 50 g de yuca SIEMPRE BIEN COCIDA."
    meal = _meal(["80 g de yuca"], [note])
    assert g._sync_recipe_step_quantities(meal) == 0
    assert meal["recipe"][0] == note


def test_unknown_food_untouched(monkeypatch):
    monkeypatch.setattr(g, "RECIPE_STEP_QTY_SYNC_ENABLED", True)
    meal = _meal(["80 g de arroz blanco"], ["Añade 10 g de mantequilla clarificada."])
    assert g._sync_recipe_step_quantities(meal) == 0


def test_knob_off_is_noop(monkeypatch):
    monkeypatch.setattr(g, "RECIPE_STEP_QTY_SYNC_ENABLED", False)
    meal = _meal(["80 g de arroz blanco"], ["pesa 50 g de arroz"])
    assert g._sync_recipe_step_quantities(meal) == 0
    assert "50 g" in meal["recipe"][0]


def test_fail_safe_on_bad_input():
    assert g._sync_recipe_step_quantities(None) == 0
    assert g._sync_recipe_step_quantities({"ingredients": "x", "recipe": None}) == 0


# ---------------------------------------------------------------------------
# wiring: assemble (post-quantize) + finalizador de updates + persist boundary
# ---------------------------------------------------------------------------
def test_wired_in_three_surfaces():
    n_calls = _GRAPH.count("_sync_recipe_step_quantities(")
    assert n_calls >= 4, (  # 1 def + assemble + finalizador updates + persist boundary
        f"esperados ≥4 usos de _sync_recipe_step_quantities (def + 3 superficies), hay {n_calls}"
    )
    assert "P1-RECIPE-QTY-SYNC" in _GRAPH


def test_runs_after_final_quantize_in_assemble():
    i_q = _GRAPH.find('logger.info(f"📏 [P1-CLOSER-COHERENCE] quantize final')
    assert i_q != -1
    # [P2-POSTQUANTIZE-RECHECK · 2026-07-02] entre el quantize y el qty-sync ahora vive el pase corrector
    # de drift de redondeo (rebalance+requantize acotado) → ventana ampliada 1500→4500. El ORDEN sigue
    # anclado: quantize → recheck → qty-sync (el sync ve el estado final).
    # [P2-AUDIT-V6-BATCH · 2026-07-03] la ventana quedó corta tras los bloques shrink-floor (GAP-05)
    # + refinador global (P1-NEXT-LEVEL-SOLVER) del 2026-07-02 y el bugfix del recheck (P1-UPDATE-
    # MACRO-PARITY) → 4500→9000. [P1-CHEESE-DUMP-FINAL · 2026-07-07] el cap final de queso + strip de
    # azúcar entre el relevel y el recheck empujó el qty-sync final → 9000→10500. El contrato REAL son
    # los asserts de ORDEN de abajo, no el tamaño.
    seg = _GRAPH[i_q:i_q + 10500]
    assert "_sync_recipe_step_quantities" in seg, \
        "el sync debe correr tras el quantize final de assemble (última mutación de porciones)"
    i_rq = seg.find("P2-POSTQUANTIZE-RECHECK")
    i_sync = seg.find("_sync_recipe_step_quantities")
    assert i_rq != -1 and i_rq < i_sync, "el recheck post-quantize corre ANTES del qty-sync"
