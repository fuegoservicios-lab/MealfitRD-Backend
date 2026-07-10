"""[P1-RECENT-DISHES-FEEDFORWARD · 2026-07-10] Feed-forward de la blocklist de platos recientes
al intento #1 (attempt-1-right).

Fallo vivo corr=45f05b9c: el intento 1 se quemó completo porque el day-gen inventó 3 platos que
YA estaban en los planes recientes del usuario — lista que el pipeline YA había fetcheado antes de
generar (get_recent_meals_from_plans, prep :35407) pero que (a) viajaba solo dentro de
history_context, que el compresor LLM resume perdiendo los nombres literales, y (b) NUNCA llegaba
al day-generator (el nodo que inventa los nombres). El gate anti-repetición (review :29752) queda
INTACTO como backstop; este bloque solo ayuda a pasarlo al primer intento.

Contrato:
1. `build_recent_dishes_blocklist_context(form_data)` → bloque determinista con los nombres, o ""
   (byte-equivalencia → prompt-cache preservado) para guests/primer plan/knob OFF.
2. El bloque se inyecta en AMBOS prompts dinámicos (skeleton Y day-gen) — el bug fue exactamente
   un contexto que solo llegaba al skeleton (parser-based; espeja P1-MED-CONTEXT-DAYGEN).
tooltip-anchor: P1-RECENT-DISHES-FEEDFORWARD
"""
import re
from pathlib import Path

import graph_orchestrator as go

_GO_SRC = Path(go.__file__).read_text(encoding="utf-8")


def test_knob_defaults():
    assert go.RECENT_DISHES_FEEDFORWARD is True      # nudge aditivo, no gate → nace ON
    assert go.RECENT_DISHES_FEEDFORWARD_MAX == 40


def test_builder_empty_for_guests_and_first_plan():
    # sin key / lista vacía → "" exacto (byte-equivalencia, prompt-cache)
    assert go.build_recent_dishes_blocklist_context({}) == ""
    assert go.build_recent_dishes_blocklist_context({"_recent_dishes_blocklist": []}) == ""
    assert go.build_recent_dishes_blocklist_context(None) == ""


def test_builder_contains_names_and_ingredient_clause():
    fd = {"_recent_dishes_blocklist": ["Yogurt Griego con Lechosa y Miel",
                                       "Avena Cremosa con Chinola"]}
    block = go.build_recent_dishes_blocklist_context(fd)
    assert "Yogurt Griego con Lechosa y Miel" in block
    assert "Avena Cremosa con Chinola" in block
    # cláusula anti-sobre-evitación: ingredientes SÍ reutilizables (espeja la REGLA DE ORO)
    assert "INGREDIENTES" in block and "PROHIBIDO" in block


def test_builder_respects_cap(monkeypatch):
    monkeypatch.setattr(go, "RECENT_DISHES_FEEDFORWARD_MAX", 5)
    fd = {"_recent_dishes_blocklist": [f"Plato {i}" for i in range(30)]}
    block = go.build_recent_dishes_blocklist_context(fd)
    assert block.count("⛔ Plato") == 5  # cap aplicado (newest-first)


def test_builder_off_when_knob_disabled(monkeypatch):
    monkeypatch.setattr(go, "RECENT_DISHES_FEEDFORWARD", False)
    fd = {"_recent_dishes_blocklist": ["Mangú de Plátano"]}
    assert go.build_recent_dishes_blocklist_context(fd) == ""


def test_ctx_key_exists_in_shared_context_builder():
    # parser: la key debe construirse en _build_shared_context
    assert '"recent_dishes_blocklist_context"' in _GO_SRC


def test_injected_into_BOTH_dynamic_prompts():
    """El bug raíz fue un contexto que solo llegaba al skeleton. Este parser exige la inyección
    en los DOS f-strings dinámicos (skeleton + day-gen). Si un refactor borra una, CI falla."""
    occurrences = _GO_SRC.count("{ctx['recent_dishes_blocklist_context']}")
    assert occurrences >= 2, (
        f"esperaba el bloque en skeleton Y day-gen (>=2 inyecciones), encontré {occurrences}"
    )


def test_prep_populates_blocklist_from_recent_meals():
    # parser: el prep del pipeline debe poblar form_data['_recent_dishes_blocklist']
    # desde el fetch existente de recent_meals (sin roundtrip extra).
    assert '_recent_dishes_blocklist' in _GO_SRC
    m = re.search(r"_recent_dishes_blocklist.{0,4000}?REGLA DE ORO|recent_meals.{0,4000}?_recent_dishes_blocklist",
                  _GO_SRC, re.S)
    assert m, "el prep no puebla _recent_dishes_blocklist cerca del fetch de recent_meals"
