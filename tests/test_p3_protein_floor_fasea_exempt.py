"""[P3-PROTEIN-FLOOR-FASEA-EXEMPT · 2026-06-28] Corrección de regresión: el umbral mínimo del closer
(P3-PROTEIN-CLOSER-MIN-THRESHOLD, para evitar tack-ons triviales de 10g camarón) bloqueaba TAMBIÉN los cierres
pequeños del reparador del PISO de proteína (FASE A) → déficit clínico (corr=5a3f31d6: Día 65/80, Día 63/80,
band 0.42, protein 0.333). El run previo (band 0.92) era ANTES del umbral.

Fix: `_close_protein_gap_for_meal` acepta `enforce_min_threshold` (default True para el closer del motor = coherencia).
FASE A (`_repair_protein_floor_post_caps`) pasa `enforce_min_threshold=False` → CUMPLE el piso (prioridad clínica),
sin que el umbral de tamaño le bloquee cierres legítimos. La coherencia del tack-on la cubren scale-first + sweet-guard
+ wording, no el umbral.
"""
from __future__ import annotations

from pathlib import Path

import graph_orchestrator as g

_BACKEND = Path(g.__file__).resolve().parent


class _Info:
    def __init__(self, n, p, c=1.0, f=1.0, k=99.0):
        self.name, self.protein, self.carbs, self.fats, self.kcal = n, p, c, f, k


def _meal():
    return {"name": "Almuerzo", "protein": 5, "carbs": 10, "fats": 5, "cals": 150, "ingredients": ["base"]}


def test_engine_closer_enforces_threshold_by_default():
    # Gap pequeño (~1.4g prot) → tack-on trivial → bloqueado (coherencia del closer del motor).
    cands = [(0.2, "Camarones", _Info("Camarones", 20))]
    assert g._close_protein_gap_for_meal(_meal(), 7, None, cands) == 0


def test_floor_repair_exempt_closes_small_gap():
    # MISMO gap pequeño pero con enforce_min_threshold=False (lo que pasa FASE A) → SÍ cierra (cumple el piso).
    cands = [(0.2, "Camarones", _Info("Camarones", 20))]
    added = g._close_protein_gap_for_meal(_meal(), 7, None, cands, enforce_min_threshold=False)
    assert added > 0, "FASE A (exenta) debe poder cerrar gaps pequeños para cumplir el piso clínico"


def test_fasea_calls_closer_exempt():
    """Parser: FASE A (`_repair_protein_floor_post_caps`) DEBE pasar enforce_min_threshold=False al closer."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("def _repair_protein_floor_post_caps")
    nxt = src.index("\ndef ", i + 10)
    body = src[i:nxt]
    assert "enforce_min_threshold=False" in body, "FASE A debe eximir el umbral para cumplir el piso"


def test_signature_has_enforce_flag():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "enforce_min_threshold: bool = True" in src
