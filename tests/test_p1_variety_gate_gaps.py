"""[P1-VARIETY-GATE-GAPS · 2026-07-29] Tres huecos de variedad vistos en el plan vivo
73db1e79 (corr=8469264b, revisión de logs del loop):

1. [CONEJO-INVISIBLE] "Conejo Estofado" (almuerzo) + "Conejo Guisado" (cena) el MISMO
   jueves y `same_day_protein_repeats=0`: `_HEAVY_PROTEIN_LABELS` no incluía conejo (ni
   chivo/calamar/langosta/higado) — la lista creció por-incidente (pavo 2026-06, mariscos
   2026-07-10) y cada especie ausente es invisible para el gate, el report Y el
   self-critique a la vez. Se alinea con el universo de especies de `_DM_SPECIES_TOKENS`.
2. [SNACK-PAIR] merienda casi idéntica en días CONSECUTIVOS ("Batido Refrescante de
   Guineo y Mantequilla de Maní" → "Guineo con Mantequilla de Maní y Yogurt"): ningún
   contador la veía (fruit-repeat es mismo-día; dish-gate exige nombre repetido).
   Advisory nuevo `cross_day_snack_pair_repeats`: mismo slot (índice), día i vs i+1,
   ≥2 alimentos featured compartidos en el NOMBRE. NO bloquea (calidad blanda).
3. [GARNISH-BACKFILL] "Espolvorea con perejil picado" sin perejil en ingredientes (mero
   del miércoles; misma clase que el cilantro fantasma del day-gen): backfill determinista
   "1 ramita de {hierba} fresco (para decorar)" — la receta ya la usa, la lista la compra.

tooltip-anchor: P1-VARIETY-GATE-GAPS
"""
from __future__ import annotations

import graph_orchestrator as go


def _day(meals, day=1):
    return {"day": day, "meals": meals}


def _meal(name, ings=None, recipe=None):
    return {"name": name, "ingredients": ings or [], "ingredients_raw": list(ings or []),
            "recipe": recipe or [], "protein": 30, "carbs": 30, "fats": 10, "cals": 400}


# ─────────────────────────── 1. conejo/chivo visibles ───────────────────────────

def test_conejo_same_day_repeat_detected():
    plan = {"days": [_day([
        _meal("Conejo Estofado con Arroz Blanco y Maní Crocante", ["150 g de conejo"]),
        _meal("Yogurt Griego con Granada", ["1 taza de yogurt griego"]),
        _meal("Conejo Guisado con Coliflor Salteada", ["140 g de conejo"]),
    ])]}
    rep = go.build_variety_report(plan)
    assert rep.get("same_day_protein_repeats", 0) >= 1, (
        "conejo ×2 el mismo día debe contar como repetición de proteína principal "
        "(caso vivo jueves plan 73db1e79)"
    )
    assert go._days_with_same_day_protein_repeat(plan), "espejo del self-critique también lo ve"


def test_chivo_and_calamar_visible():
    for esp, line in (("Chivo Guisado", "180 g de chivo"),
                      ("Calamares Salteados", "150 g de calamares")):
        plan = {"days": [_day([
            _meal(f"{esp} al Estilo RD", [line]),
            _meal(f"{esp} con Vegetales", [line]),
        ])]}
        rep = go.build_variety_report(plan)
        assert rep.get("same_day_protein_repeats", 0) >= 1, f"{esp} ×2 mismo día invisible"


def test_different_days_not_flagged():
    plan = {"days": [
        _day([_meal("Conejo Estofado", ["150 g de conejo"])], day=1),
        _day([_meal("Conejo Guisado", ["140 g de conejo"])], day=2),
    ]}
    assert not go._days_with_same_day_protein_repeat(plan), (
        "días DISTINTOS es la preferencia del owner — solo fatiga el mismo día"
    )


# ─────────────────────────── 2. merienda repetida días consecutivos ───────────────────────────

def test_cross_day_snack_pair_advisory():
    plan = {"days": [
        _day([_meal("Tostadas de Pan Integral con Mango"),
              _meal("Batido Refrescante de Guineo y Mantequilla de Maní")], day=1),
        _day([_meal("Smoothie Bowl de Mango con Yogurt"),
              _meal("Guineo con Mantequilla de Maní y Yogurt")], day=2),
    ]}
    rep = go.build_variety_report(plan)
    assert rep.get("cross_day_snack_pair_repeats", 0) >= 1, (
        "mismo slot, días consecutivos, ≥2 featured compartidos (guineo+maní) → advisory"
    )
    assert any("consecutivos" in str(i) for i in rep.get("issues", []))


def test_cross_day_snack_pair_not_flagged_when_distinct():
    plan = {"days": [
        _day([_meal("Batido de Guineo y Mantequilla de Maní")], day=1),
        _day([_meal("Yogurt Griego con Granada y Avena")], day=2),
    ]}
    rep = go.build_variety_report(plan)
    assert rep.get("cross_day_snack_pair_repeats", 0) == 0


# ─────────────────────────── 3. garnish fantasma ───────────────────────────

def test_garnish_herb_backfill():
    days = [_day([
        _meal("Mero Salteado con Espárragos y Puré de Yautía",
              ["200 g de mero", "150 g de yautía"],
              ["Saltea el mero.", "Espolvorea con perejil picado."]),
        _meal("Pimientos Rellenos de Cerdo",
              ["150 g de cerdo"],
              ["Rellena los pimientos y decora con cilantro fresco."]),
        _meal("Pollo con Cilantro",
              ["150 g de pollo", "0.5 cda de cilantro fresco picado"],
              ["Sirve y decora con cilantro."]),
    ])]
    n = go._garnish_herb_mention_backfill(days)
    assert n == 2
    m0, m1, m2 = days[0]["meals"]
    assert any("perejil" in s for s in m0["ingredients"])
    assert any("1 ramita de cilantro fresco" in s for s in m1["ingredients"])
    assert sum("cilantro" in s for s in m2["ingredients"]) == 1, "ya presente → no duplica"
    assert go._garnish_herb_mention_backfill(days) == 0, "idempotente"
