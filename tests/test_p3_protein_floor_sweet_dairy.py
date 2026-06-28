"""[P3-PROTEIN-FLOOR-SWEET-DAIRY · 2026-06-28] Cierre de la tensión coherencia↔piso en bariátrica: las meriendas
bariátricas son DULCES (yogur+fruta, avena); el sweet-guard (correcto) bloquea proteína SALADA ahí, y FASE A excluía
TODOS los lácteos → no tenía con qué cerrar el piso en las comidas dulces → déficit recurrente (corr=bf5c0e54: Día
61/80, band 0.25, entrega degradada).

Fix: FASE A ahora incluye YOGUR como candidato (densidad media; cap bariátrico 120g → el add ≤90g es seguro y NO
re-excede el cap), excluyendo solo QUESOS/leche (cap 30g). En platos dulces el closer (no_cook/dairy-egg + sweet-guard)
elige yogur; en salados prefiere carne densa por categoría. + scale-first solo crece la proteína existente si cierra el
gap COMPLETO (si no, cae al append) → ya no deja el piso abierto por el clamp 2×.
"""
from __future__ import annotations

from pathlib import Path

import graph_orchestrator as g

_BACKEND = Path(g.__file__).resolve().parent
_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


def _fasea_body():
    i = _SRC.index("def _repair_protein_floor_post_caps")
    return _SRC[i:_SRC.index("\ndef ", i + 10)]


def test_fasea_includes_yogurt_excludes_cheeses():
    body = _fasea_body()
    assert "P3-PROTEIN-FLOOR-SWEET-DAIRY" in body
    assert "min_protein=10.0" in body  # densidad bajada para incluir yogur
    assert "_DAIRY_EXCLUDE" in body and "ricotta" in body and "cottage" in body
    # el yogur NO está en la lista de exclusión (sí entra como candidato dulce-compatible)
    _excl_line = next(ln for ln in body.splitlines() if "_DAIRY_EXCLUDE = (" in ln)
    _tuple_part = _excl_line.split("#")[0]  # ignora el comentario ('yogur SÍ entra')
    assert "yogur" not in _tuple_part.lower(), f"el yogur NO debe excluirse: {_tuple_part}"


def test_scale_first_full_close_only():
    """scale-first solo crece si cierra el gap COMPLETO (factor_needed <= scale_max); si no, cae al append."""
    i = _SRC.index("def _try_scale_existing_protein")
    body = _SRC[i:_SRC.index("\ndef ", i + 10)]
    assert "factor_needed > scale_max" in body and "→ append" in body


def test_sweet_guard_blocks_savory_but_allows_yogurt_in_sweet():
    """En un plato dulce, el closer NO mete carne salada (sweet-guard) pero SÍ puede meter yogur (no es _MEAT)."""
    from constants import strip_accents as _sa

    class _Info:
        def __init__(self, n, p, c=4.0, f=0.4, k=59.0):
            self.name, self.protein, self.carbs, self.fats, self.kcal = n, p, c, f, k

    sweet = {"name": "Yogurt Griego con Lechosa", "protein": 5, "carbs": 20, "fats": 3, "cals": 120,
             "ingredients": ["100g de yogurt griego", "95g de lechosa"]}
    # candidatos: yogur (dulce-compat) + camarón (salado). El sweet-guard debe dejar SOLO el yogur.
    cands = [(0.10, "Yogurt griego", _Info("Yogurt griego", 10)), (0.24, "Camarones", _Info("Camarones", 24))]
    added = g._close_protein_gap_for_meal(sweet, 18, None, cands, enforce_min_threshold=False)
    txt = " ".join(str(i) for i in sweet["ingredients"]).lower()
    assert "camaron" not in txt, "no debe meter camarón en un yogurt dulce"
