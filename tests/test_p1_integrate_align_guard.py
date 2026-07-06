"""[P1-INTEGRATE-ALIGN-GUARD · 2026-07-06]

Regresión VIVIDA (plan d4a001eb, ~40 min tras deployar P1-CLOSER-INTEGRATE): band protein 0.333
— la celda de proteína nunca había fallado en el día. Raíz: `_scale_congruent_protein_line`
sincroniza el raw por ÍNDICE, y el display↔raw viene DESALINEADO de forma crónica (7-10 de 12
meals en los 4 planes recientes: raw con "Sal al gusto"/"ajo" que display no tiene) → el idx
escalaba la línea EQUIVOCADA del raw y el truth-up (raw-preferido) revertía la proteína cerrada
(cena entregada con P=8). El append viejo era inmune (posición-independiente).

Guard: listas desalineadas → False (el caller appendea a ambos extremos, comportamiento seguro).
+ Telemetría [P1-RAW-MISALIGN] per-meal en el boundary — la data para cazar al pase culpable
(P1 de raíz ABIERTO, ver memoria batch 16).
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


class _ScaleDB:
    def grams_from_ingredient_string(self, s):
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", str(s).lower())
        return float(m.group(1).replace(",", ".")) if m else None


def test_misaligned_lists_refuse_scale():
    import graph_orchestrator as go
    meal = {"name": "Tostones Rellenos de Queso",
            "ingredients": ["25 g de queso", "½ plátano verde"],
            "ingredients_raw": ["25 g de queso mozzarella fresco en cubos",
                                "0.5 plátano verde", "Sal al gusto"]}  # raw 3 vs display 2
    assert go._scale_congruent_protein_line(meal, "queso", 150, _ScaleDB()) is False, \
        "raw desalineado → jamás escalar por idx (el truth-up raw-preferido revertía la proteína)"
    assert meal["ingredients"][0] == "25 g de queso", "display intacto (el caller appendea)"


def test_aligned_lists_still_scale():
    import graph_orchestrator as go
    meal = {"name": "Bollitos",
            "ingredients": ["15 g de queso blanco", "150 g de yuca"],
            "ingredients_raw": ["15 g de queso blanco", "150 g de yuca"]}
    assert go._scale_congruent_protein_line(meal, "queso", 185, _ScaleDB()) is True
    assert meal["ingredients"][0] == meal["ingredients_raw"][0], "alineado → integrate normal"


def test_misalign_telemetry_wired_at_boundary():
    assert "[P1-RAW-MISALIGN]" in _GO, "telemetría per-meal para cazar al pase culpable"
    i = _GO.index("[P1-INTEGRATE-ALIGN-GUARD · 2026-07-06] TELEMETRÍA")
    assert 'parts.append(f"raw_misalign=' in _GO[i:i + 1200], \
        "el boundary reporta el conteo en el summary del finalize"
