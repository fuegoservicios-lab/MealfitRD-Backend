"""[P1-FRUIT-DEDUP · 2026-06-26] (auditoría gap #7) Calidad blanda: de-dup determinista de fruta repetida
+ BENCHMARK de apetecibilidad/coherencia (la auditoría notó "CERO test/benchmark mide la dimensión").

1. dedup_featured_fruits_in_plan: la 2ª+ aparición de una fruta dulce repetida el mismo día se reescribe a
   una fruta distinta (nombre + ingredientes), ANTES de la lista de compras → coherente. Cierra el cierre que
   el gate de variedad dejaba abierto (en el intento final entregaba la repetición).
2. Benchmark de pareo: un corpus de platos "chocantes" (deben flagearse) vs "apetecibles" (no) mide
   recall/especificidad del detector de coherencia (gap #5) → primera medición objetiva de la dimensión.
"""
from __future__ import annotations

import graph_orchestrator as go


# ── 1. De-dup determinista de fruta repetida ─────────────────────────────────────────────
def test_dedup_rewrites_repeated_fruit():
    plan = {"days": [{"day": 1, "meals": [
        {"name": "Avena con mango", "ingredients": ["1 taza de avena (40g)", "100g de mango (100g)"]},
        {"name": "Yogur con mango", "ingredients": ["1 pote de yogur (150g)", "80g de mango (80g)"]},
    ]}]}
    n = go.dedup_featured_fruits_in_plan(plan)
    assert n == 1, "debe reescribir 1 repetición (mango ×2 el mismo día)"
    names = [m["name"].lower() for m in plan["days"][0]["meals"]]
    assert sum("mango" in nm for nm in names) == 1, "mango debe quedar en 1 sola comida del día"
    # la 2ª comida ya NO menciona mango ni en nombre ni en ingredientes
    m2 = plan["days"][0]["meals"][1]
    assert "mango" not in m2["name"].lower()
    assert not any("mango" in str(i).lower() for i in m2["ingredients"])


def test_dedup_handles_accented_fruit():
    plan = {"days": [{"day": 1, "meals": [
        {"name": "Batido de Piña", "ingredients": ["1 taza de piña (150g)"]},
        {"name": "Ensalada con Piña", "ingredients": ["80g de piña (80g)"]},
    ]}]}
    n = go.dedup_featured_fruits_in_plan(plan)
    assert n == 1, "el matcher accent-flex debe localizar 'Piña' (acentuada) y reescribirla"
    names = " ".join(m["name"].lower() for m in plan["days"][0]["meals"])
    assert names.count("piña") + names.count("pina") == 1


def test_dedup_noop_when_no_repeat():
    plan = {"days": [{"day": 1, "meals": [
        {"name": "Avena con mango", "ingredients": []},
        {"name": "Yogur con lechosa", "ingredients": []},
    ]}]}
    assert go.dedup_featured_fruits_in_plan(plan) == 0


def test_dedup_failsafe_on_garbage():
    assert go.dedup_featured_fruits_in_plan(None) == 0
    assert go.dedup_featured_fruits_in_plan({"days": "no-es-lista"}) == 0


# ── 2. Benchmark de apetecibilidad/coherencia de pareo ───────────────────────────────────
# Corpus curado: platos que un dominicano consideraría CHOCANTES (fruta dulce + base salada) vs APETECIBLES.
_CLASHING = [
    "Arroz blanco con mango",
    "Revoltillo de huevos con piña",
    "Coliflor gratinada con lechosa",
    "Espagueti con melón",
    "Berenjena guisada con mango",
    "Moro con papaya",
]
_APPETIZING = [
    "Yogur griego con mango y granola",
    "Avena con lechosa",
    "Pollo a la plancha con piña asada",   # proteína + fruta tropical = aceptable
    "Cerdo guisado con guayaba",
    "Ensalada verde con manzana",
    "Batido de papaya",
    "Pechuga de pavo con vegetales al vapor",
    "Sancocho dominicano",
    "Tostones con pollo guisado",
]


def _clash_flagged(dish_name: str) -> bool:
    rep = go.build_variety_report({"days": [{"day": 1, "meals": [{"name": dish_name, "ingredients": []}]}]})
    return int(rep.get("sweet_savory_clash", 0)) >= 1


def test_appetibility_benchmark_recall_and_specificity():
    """El detector debe flagear los pareos chocantes (recall alto) sin falsos positivos (especificidad alta)."""
    flagged_bad = [d for d in _CLASHING if _clash_flagged(d)]
    flagged_good = [d for d in _APPETIZING if _clash_flagged(d)]
    recall = len(flagged_bad) / len(_CLASHING)
    specificity = 1 - len(flagged_good) / len(_APPETIZING)
    # Umbrales medibles (primera baseline de la dimensión). Conservador → prioriza CERO falsos positivos.
    assert specificity == 1.0, f"falsos positivos en apetecibles: {flagged_good}"
    assert recall >= 0.8, f"recall bajo en chocantes ({recall:.0%}); no flageados: {[d for d in _CLASHING if d not in flagged_bad]}"


def test_named_owner_case_is_caught():
    """El caso textual del owner ('mango con arroz') siempre debe flagearse."""
    assert _clash_flagged("Arroz con mango")
    assert _clash_flagged("Mango con arroz blanco")
