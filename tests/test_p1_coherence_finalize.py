"""[P1-COHERENCE-FINALIZE · 2026-06-28] Los 3 fixes de coherencia de unidad (slice-grams/leaf-cap/quantize) SOLO corrían
en assemble_plan_node. Un plan entregado por un path que salta assemble (partial/rechazado-pero-entregado/SSE-fallback) se
persistía con unidades vagas ("1¼ lonjas/pedazos de queso") — verificado en DB real (generation_status='partial'). Fix:
`finalize_plan_data_coherence` aplicado DEFENSIVAMENTE en los persist boundaries (INSERT en db_plans + chunk T1 en
cron_tasks). Idempotente → no-op donde assemble ya corrió (band 1.00 intacto), fail-safe.
"""
from __future__ import annotations

import re
from pathlib import Path

import graph_orchestrator as g

_BACKEND = Path(g.__file__).resolve().parent


class _DB:
    def grams_from_ingredient_string(self, s):
        m = re.search(r"\((\d+(?:\.\d+)?)\s*g\)", s) or re.search(r"^\s*(\d+(?:\.\d+)?)\s*g\b", s)
        return float(m.group(1)) if m else None

    def macros_from_ingredient_string(self, s):
        return {}


def _wire(monkeypatch):
    # [P1-FINALIZE-TRUTHUP-ALL · 2026-07-07] El truth-up REAL es idempotente: retorna False cuando las
    # macros ya = string-sum (el caso band-1.00 que test_idempotent_band_safe representa). El stub debe
    # modelar eso (antes era `lambda: True` siempre → rompía el conteo del truth-up final incondicional).
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda m, db: False)


def _days():
    return [{"day": 1, "meals": [{"meal": "Merienda", "name": "X", "ingredients": [
        "1¼ lonjas/pedazos de queso", "¾ lonja/pedazo de queso", "5.5 taza de rucula fresca (165g)", "2 huevos enteros"]}]}]


def test_converts_vague_units(monkeypatch):
    _wire(monkeypatch)
    d = _days()
    n, summ = g.finalize_plan_data_coherence(d, _DB())
    assert n >= 3
    joined = " ".join(d[0]["meals"][0]["ingredients"]).lower()
    assert "lonja" not in joined and "¼" not in joined and "¾" not in joined
    # [P2-INGREDIENT-LINE-CONSOLIDATE · 2026-07-01] las DOS lonjas de queso ahora se convierten a gramos
    # (slice) Y se fusionan en UNA línea sumada en la misma pasada — antes quedaban "30 g" + "20 g" separadas.
    _queso_lines = [i for i in d[0]["meals"][0]["ingredients"] if "queso" in str(i).lower()]
    assert len(_queso_lines) == 1, f"debe quedar UNA línea de queso consolidada: {_queso_lines}"
    assert "g de queso" in joined


def test_idempotent_band_safe(monkeypatch):
    _wire(monkeypatch)
    d = _days()
    g.finalize_plan_data_coherence(d, _DB())
    snapshot = [list(m["ingredients"]) for m in d[0]["meals"]]
    n2, _ = g.finalize_plan_data_coherence(d, _DB())  # 2ª pasada sobre plan ya-fijo
    assert n2 == 0, "2ª pasada debe ser no-op (band 1.00 intacto en el caso bueno)"
    assert [list(m["ingredients"]) for m in d[0]["meals"]] == snapshot


def test_failsafe_does_not_raise(monkeypatch):
    _wire(monkeypatch)
    monkeypatch.setattr(g, "_recipe_slice_units_to_grams", lambda days, db=None: (_ for _ in ()).throw(RuntimeError("boom")))
    # no debe propagar — devuelve aunque una fix lance
    n, _ = g.finalize_plan_data_coherence(_days(), _DB())
    assert isinstance(n, int)


def test_empty_days_noop(monkeypatch):
    _wire(monkeypatch)
    assert g.finalize_plan_data_coherence([], _DB()) == (0, "")
    assert g.finalize_plan_data_coherence(None, _DB()) == (0, "")


def test_insert_chokepoint_wired():
    # [P0-PERSIST-TXN-IDLE · 2026-07-10] el bloque de pases se extrajo del builder a
    # _finalize_plan_data_for_insert para poder correrlo FUERA de la transacción del
    # path atomic (idle-in-transaction timeout mataba el INSERT). El chokepoint sigue
    # siendo universal: el builder delega al helper salvo skip explícito del atomic
    # (que lo pre-ejecuta él mismo). Ver test_p0_persist_txn_idle.py.
    src = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
    i = src.index("def _finalize_plan_data_for_insert")
    body = src[i:i + 4000]
    assert "finalize_plan_data_coherence" in body
    assert "P1-COHERENCE-FINALIZE" in body
    # import LAZY (no a nivel módulo — ciclo db_plans↔graph_orchestrator)
    assert "from graph_orchestrator import finalize_plan_data_coherence" in body
    # el builder conserva el escudo central (default skip=False → delega al helper)
    ib = src.index("def _build_meal_plan_insert_sql")
    bbody = src[ib:ib + 2200]
    assert "_finalize_plan_data_for_insert(data)" in bbody


def test_chunk_t1_chokepoint_wired():
    # [P0-BAND-PRE-REVIEW · 2026-07-10] el chunk T1 ya no importa fpc directo: delega en
    # `apply_plan_quality_finalize_chain` (SSOT db_plans), que corre fpc + el resto del
    # shield (banda all-4, step-parity, polish-refire, etc.) para las semanas 2+.
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    assert "P1-COHERENCE-FINALIZE" in src
    assert "apply_plan_quality_finalize_chain" in src


def test_order_slice_before_quantize():
    # el orden interno (slice → leaf → quantize) es load-bearing; ancla que no se reordene.
    # [P1-UPDATE-RECIPE-FINALIZE · 2026-06-29] El veg-guard ahora corre ANTES (al tope de la fn) y el
    # recipe-nonempty backstop DESPUÉS; la ventana se amplía a 4000 chars para abarcar la fn completa,
    # pero la invariante slice→leaf→quantize sigue intacta.
    # [P0-VEG-GUARD-ALLERGEN · 2026-07-01] +comment del filtro de alérgenos → ventana a 5000 chars.
    # [P1-DISH-REALISM-BATCH · 2026-07-01] +carb-ghost/realism-cap/consolidate en el boundary → 9500 chars.
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("def finalize_plan_data_coherence")
    body = src[i:i + 9500]
    p_slice = body.index("_recipe_slice_units_to_grams")
    p_leaf = body.index("_cap_leaf_volume_in_meals")
    p_quant = body.index("_apply_portion_quantization")
    assert p_slice < p_leaf < p_quant, "orden slice→leaf→quantize es load-bearing"
    # El veg-guard corre ANTES del slice (para que el veg añadido entre a slice/quantize).
    # [P0-VEG-GUARD-ALLERGEN · 2026-07-01] el callsite ahora pasa allergies= (filtro de alérgenos).
    assert body.index("_add_missing_recipe_step_vegetables(days, allergies=allergies)") < p_slice
