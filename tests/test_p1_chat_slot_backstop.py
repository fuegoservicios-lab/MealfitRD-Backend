"""[P1-CHAT-SLOT-BACKSTOP · 2026-06-29] (audit objetivo · paridad chat-modify ↔ swap/S1)

El chat-modify (`execute_modify_single_meal` en tools.py) era la ÚNICA superficie de update que
NO revalidaba la apropiación de HORARIO: swap S3 tiene el backstop con ValueError→retry
(agent.py:1311), S1 lo enforza como gate en review_plan_node, pero el chat-modify solo tenía el
prompt advisory. Un "cámbiame la cena" podía colar 'arroz de noche' o comida-de-desayuno
(cereal/panqueque) en la cena sin filtro.

Este fix añade el backstop dentro de `invoke_with_retry` con presión de retry SELECTIVA:
solo fuerza retry cuando el item fuera de horario NO está en el texto del cambio del usuario
(deseo explícito gana), y degrada a advisory en el intento final (nunca convierte una
incoherencia cosmética/fail-open en el abort de "FALLO POR INVENTARIO").

Tests: (1) parser-based del wiring en tools.py; (2) lógica pura "deseo explícito gana" sobre la
SSOT `constants.slot_violations_for_meal_name` (sin DB).
"""
from __future__ import annotations

import re
from pathlib import Path

from constants import canonical_slot_key, slot_violations_for_meal_name

_BACKEND = Path(__file__).resolve().parent.parent
_TOOLS = (_BACKEND / "tools.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Parser-based: el backstop está cableado en tools.py
# ---------------------------------------------------------------------------
def test_marker_present():
    assert "P1-CHAT-SLOT-BACKSTOP" in _TOOLS, "marker P1-CHAT-SLOT-BACKSTOP ausente en tools.py"


def test_imports_slot_ssot():
    """tools.py debe importar el gate maestro de graph_orchestrator y la SSOT de slot de constants."""
    assert "SLOT_APPROPRIATENESS_GATE_ENABLED" in _TOOLS, "falta import del gate maestro de slot"
    assert "canonical_slot_key" in _TOOLS and "slot_violations_for_meal_name" in _TOOLS, \
        "falta import de la SSOT de slot (canonical_slot_key / slot_violations_for_meal_name)"


def test_backstop_inside_retry_with_selective_logic():
    """El backstop debe: (a) gatear por SLOT_APPROPRIATENESS_GATE_ENABLED, (b) comparar las
    violaciones del plato vs las que el usuario pidió en `changes` (deseo explícito gana),
    (c) degradar en el intento final (`_slot_attempt[0] < 3`), (d) raise ValueError para retry."""
    # El bloque vive tras el macro-validator y antes de `return res` (dentro de invoke_with_retry).
    assert re.search(r"_slot_attempt\s*=\s*\[0\]", _TOOLS), "falta el contador de intentos del slot"
    assert "_slot_attempt[0] += 1" in _TOOLS, "el contador de intentos no se incrementa por attempt"
    # Deseo explícito gana: se computan las violaciones del texto del usuario (`changes`).
    assert "slot_violations_for_meal_name(changes" in _TOOLS, \
        "no se evalúa el texto del cambio del usuario (deseo explícito gana)"
    # Degradación en intento final (no raise en attempt 3).
    assert "_slot_attempt[0] < 3" in _TOOLS, "falta el guard de intento final (degradar a advisory)"
    # Presión de retry.
    assert re.search(r"raise ValueError\(f?\"plato fuera de horario", _TOOLS), \
        "el backstop no fuerza retry vía ValueError"


def test_advisory_flag_on_delivery():
    """Cuando el plato se entrega fuera de horario (deseo explícito o intento final), se marca
    `_slot_advisory` para telemetría/frontend (espejo de `_slot_appropriateness_advisory_final` de S1)."""
    assert '_slot_advisory' in _TOOLS, "falta el flag advisory de telemetría en la entrega"


# ---------------------------------------------------------------------------
# 2. Lógica pura "deseo explícito gana" (sin DB) — el corazón del fix
# ---------------------------------------------------------------------------
def _unrequested_labels(meal_name: str, user_changes: str, slot_key: str) -> list:
    """Réplica de la lógica del backstop: las violaciones del plato que el usuario NO pidió."""
    meal_v = slot_violations_for_meal_name(meal_name, slot_key)
    requested = {v["label"] for v in slot_violations_for_meal_name(user_changes or "", slot_key)}
    return [v for v in meal_v if v["label"] not in requested]


def test_llm_introduced_arroz_de_noche_triggers_retry():
    """El LLM mete 'arroz' en la cena ante un 'cámbiame la cena' vago → debe haber violación no-pedida."""
    slot = canonical_slot_key("Cena")
    assert slot == "cena"
    out = _unrequested_labels("Arroz blanco con pollo guisado", "cámbiame la cena", slot)
    assert out, "una cena con arroz no-pedida debería disparar retry"
    assert any("arroz" in v["label"] for v in out)


def test_user_explicitly_asked_arroz_no_retry():
    """El usuario pide 'ponme arroz en la cena' → su deseo gana, NO se reintenta."""
    slot = canonical_slot_key("Cena")
    out = _unrequested_labels("Arroz blanco con pollo guisado", "ponme arroz en la cena", slot)
    assert out == [], "si el usuario pide arroz explícitamente, no debe forzarse retry"


def test_creativity_modifier_not_flagged():
    """'Panqueques de harina de arroz' en desayuno NO se flagea (protección de creatividad G5)."""
    slot = canonical_slot_key("Desayuno")
    assert slot == "desayuno"
    out = _unrequested_labels("Panqueques de harina de arroz", "algo dulce", slot)
    assert out == [], "el modificador 'harina de arroz' no debe disparar la violación de arroz"


def test_breakfast_food_in_dinner_triggers_when_unrequested():
    """Cereal/panqueque en la cena, no pedido → violación soft no-pedida → retry."""
    slot = canonical_slot_key("Cena")
    out = _unrequested_labels("Panqueques con miel", "hazla más ligera", slot)
    assert out, "comida de desayuno en la cena (no pedida) debería disparar retry"
