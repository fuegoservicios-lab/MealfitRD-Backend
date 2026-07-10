"""[P1-COHERENCE-COUNT-MATERIAL · 2026-07-09] La regla count>=MIN_COUNT del severe-only debe
contar solo divergencias MATERIALES, no sobrevivientes del ruido de fondo.

Evidencia (SQL forense plan 3d11d96e + corr=45f05b9c): el intento 2 se quemó COMPLETO porque 2
divergencias de magnitud ('Res', 'Queso fresco') activaron la regla count>=2 — pero el plan
APROBADO del intento 3 carga 33 divergencias de magnitud hypothesis='unknown' equivalentes (el
ruido de agregación es endémico; los filtros cap-aware/pkg-noise dejan pasar 2-3 al azar). Contar
2 drifts de 10-25%% como "sistemático" es confundir ruido con señal → full regen LLM (~5-10x el
costo) para una lista DERIVADA que igual queda con ~33 drifts marginales.

Fix: la regla de CONTEO cuenta una divergencia solo si es material — presence/no-magnitude SIEMPRE
cuentan (alimento ausente = material por definición); magnitude cuenta solo si |delta| >=
MEALFIT_REVIEW_COHERENCE_COUNT_MIN_DELTA (default 0.25). La regla de delta egregio (|d|>=0.50
individual) queda INTACTA. Floor=0.0 => comportamiento previo exacto (rollback sin redeploy).

tooltip-anchor: P1-COHERENCE-COUNT-MATERIAL
"""
from pathlib import Path

import graph_orchestrator as go

_GO_SRC = Path(go.__file__).read_text(encoding="utf-8")


def _mag(delta):
    # SUB-oferta (actual < expected): la única dirección material tras
    # P1-COHERENCE-SEVERE-DIRECTIONAL (la sobre-oferta de empaque nunca cuenta).
    return {"food": "X", "magnitude": True, "delta_pct": delta, "expected_qty": 100.0,
            "actual_qty": 100.0 * (1 - min(delta, 0.99))}


def test_two_noise_survivors_do_not_count(monkeypatch):
    # el caso vivo: 2 drifts marginales (12%, 18%) → material_count=0 con floor 0.25
    block = [_mag(0.12), _mag(0.18)]
    assert go._coherence_material_divergence_count(block, 0.25) == 0


def test_material_magnitude_counts():
    block = [_mag(0.30), _mag(0.12)]
    assert go._coherence_material_divergence_count(block, 0.25) == 1


def test_presence_always_counts():
    # entradas sin magnitude flag (presence/cap_swallowed: alimento AUSENTE) son materiales siempre
    block = [{"food": "Pollo", "hypothesis": "cap_swallowed_modifier", "expected_qty": 400.0, "actual_qty": 0.0}]
    assert go._coherence_material_divergence_count(block, 0.25) == 1


def test_floor_zero_preserves_previous_behavior():
    # floor 0.0 → toda magnitude cuenta (comportamiento pre-fix exacto)
    block = [_mag(0.11), _mag(0.11)]
    assert go._coherence_material_divergence_count(block, 0.0) == 2


def test_infinite_and_garbage_deltas_are_failsafe():
    block = [{"food": "Y", "magnitude": True, "delta_pct": float("inf")},
             {"food": "Z", "magnitude": True, "delta_pct": "garbage"}]
    # fail-safe: delta no-finito/no-parseable → 0.0 → no cuenta como material de CONTEO
    assert go._coherence_material_divergence_count(block, 0.25) == 0


def test_wired_into_review_severe_rule():
    # parser: la regla severe del review consume el conteo material + el knob existe
    assert "MEALFIT_REVIEW_COHERENCE_COUNT_MIN_DELTA" in _GO_SRC
    assert "_coherence_material_divergence_count(" in _GO_SRC
    i = _GO_SRC.index("MEALFIT_REVIEW_COHERENCE_SEVERE_MIN_COUNT")
    window = _GO_SRC[i:i + 2500]
    assert "_coherence_material_divergence_count" in window, (
        "la regla count>=MIN_COUNT del severe-only debe usar el conteo MATERIAL"
    )
