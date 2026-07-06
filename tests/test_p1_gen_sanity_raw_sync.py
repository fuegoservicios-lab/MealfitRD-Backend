"""[P1-GEN-SANITY-RAW-SYNC + P2-FRUIT-DEDUP-BOUNDED · 2026-07-06]

CIERRE del actor #2 del desalineador crónico (el tracer lo bracketeó en post_humanize→boundary
y el código lo confirmó): `_generation_sanity_autofix` reconstruía SOLO `ingredients` al dropear
glitches/incongruentes — el raw quedaba con el glitch VIVO para siempre y desalineado a mitad de
lista (todo lockstep posicional posterior degradaba). Ahora filtra el raw por índices conservados.

+ 'Eslechosacas' (vivo en guard-blind): el fruit-dedup sustituía "pina"→"lechosa" SIN límite de
palabra — "Es-PINA-cas" → "Es-lechosa-cas". Detección y rewrite bounded (\\b...s?\\b) — cuarta
aparición de la clase 'res'↔'fresas'.

+ whitelist v2 del guard-blind: "Agua fría/tibia/para hervir" tampoco son comprables.
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()
with open(os.path.join(_BACKEND, "shopping_calculator.py"), encoding="utf-8") as f:
    _SC = f.read()


@pytest.fixture()
def go():
    import graph_orchestrator as g
    return g


# ───────────── gen-sanity raw lockstep ─────────────

def test_gen_sanity_drops_from_both_lists(go, monkeypatch):
    monkeypatch.setattr(go, "GEN_SANITY_AUTOFIX_ENABLED", True)
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda m, db: None)

    class _DB:
        def macros_from_ingredient_string(self, s):
            return {} if "eslechosacas" in str(s).lower() else {"grams": 100}

        def lookup(self, w):
            return None

    import shopping_calculator as sc
    monkeypatch.setattr(sc, "_is_verified_for_shopping",
                        lambda s: "eslechosacas" not in str(s).lower())
    plan = {"days": [{"day": 1, "meals": [{
        "meal": "Cena", "name": "Ensalada Fresca",
        "ingredients": ["100 g de pollo", "75g de Eslechosacas frescas", "1 tomate"],
        "ingredients_raw": ["100 g de pollo", "75g de Eslechosacas frescas", "1 tomate"],
        "recipe": ["Mise: x.", "Montaje: y."],
    }]}]}
    assert go._generation_sanity_autofix(plan, db=_DB()) >= 1
    m = plan["days"][0]["meals"][0]
    assert len(m["ingredients"]) == len(m["ingredients_raw"]) == 2, \
        "el drop del gen-sanity DEBE salir de AMBAS listas (actor #2 del desalineador)"
    assert not any("eslechosacas" in str(x).lower() for x in m["ingredients_raw"]), \
        "el glitch no queda vivo en el raw"


def test_gen_sanity_misaligned_lists_display_only(go, monkeypatch):
    """Con listas YA desalineadas, el filtro por índice sería corrupto → solo display (regla-clase)."""
    monkeypatch.setattr(go, "GEN_SANITY_AUTOFIX_ENABLED", True)
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda m, db: None)

    class _DB:
        def macros_from_ingredient_string(self, s):
            return {} if "eslechosacas" in str(s).lower() else {"grams": 100}

        def lookup(self, w):
            return None

    import shopping_calculator as sc
    monkeypatch.setattr(sc, "_is_verified_for_shopping",
                        lambda s: "eslechosacas" not in str(s).lower())
    plan = {"days": [{"day": 1, "meals": [{
        "meal": "Cena", "name": "Ensalada",
        "ingredients": ["100 g de pollo", "75g de Eslechosacas frescas"],
        "ingredients_raw": ["100 g de pollo", "75g de Eslechosacas frescas", "Sal al gusto"],
        "recipe": ["Mise: x."],
    }]}]}
    go._generation_sanity_autofix(plan, db=_DB())
    m = plan["days"][0]["meals"][0]
    assert len(m["ingredients_raw"]) == 3, "desalineado previo → el raw no se filtra por idx (jamás corromper)"


def test_tracer_probe_post_gen_sanity_wired():
    assert '_trace_misalign(result.get("days"), "post_gen_sanity")' in _GO


# ───────────── fruit-dedup bounded ─────────────

def test_espinacas_never_rewritten_by_pina_dedup(go):
    plan = {"days": [{"day": 1, "meals": [
        {"meal": "Desayuno", "name": "Batido de Piña Fresca",
         "ingredients": ["1 taza de piña"], "ingredients_raw": ["1 taza de piña"],
         "recipe": ["Montaje: licúa."]},
        {"meal": "Almuerzo", "name": "Ensalada de Espinacas con Piña",
         "ingredients": ["2 tazas de espinacas", "½ taza de piña"],
         "ingredients_raw": ["2 tazas de espinacas", "½ taza de piña"],
         "recipe": ["Montaje: mezcla."]},
    ]}]}
    go.dedup_featured_fruits_in_plan(plan)
    _alm = plan["days"][0]["meals"][1]
    _blob = _alm["name"] + " " + " ".join(str(x) for x in _alm["ingredients"])
    assert "eslechosacas" not in _blob.lower() and "Eslechos" not in _blob, \
        f"'pina' JAMÁS matchea dentro de 'Espinacas' (rewrite bounded): {_blob}"
    assert "espinacas" in _blob.lower(), "las espinacas sobreviven intactas"


# ───────────── whitelist agua v2 ─────────────

def test_water_variants_whitelisted():
    i = _SC.index("[P3-GUARD-BLIND-WATER-WHITELIST v2")
    win = _SC[i:i + 600]
    assert 'startswith("agua ")' in win, \
        "'Agua fría/tibia/para hervir' no son desobediencia del LLM (y 'aguacate' no matchea)"
