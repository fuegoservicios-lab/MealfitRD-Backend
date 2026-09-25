# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-245 · 2026-09-25] Con «Nada» de tiempo, los cerradores de proteína eligen proteína lista.

Auditoría del 25-sep: los cerradores añadían pechuga cruda con «Cocina … a la plancha» a quien eligió «Nada» (≤10 min).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import proteina_lista as pl  # noqa: E402

CANDS = [(0.25, "Pechuga de pollo", SimpleNamespace(name="Pechuga de pollo")),
         (0.22, "Atún en agua", SimpleNamespace(name="Atún en agua")),
         (0.20, "Lentejas", SimpleNamespace(name="Lentejas")),
         (0.18, "Huevo", SimpleNamespace(name="Huevo"))]
LISTOS = {"atun en agua", "queso blanco"}


def test_con_nada_solo_lo_listo_y_el_huevo():
    out = pl.filtrar_listas(CANDS, {"cookingTime": "none"}, listos=LISTOS)
    assert [c[1] for c in out] == ["Atún en agua", "Huevo"], out


def test_con_tiempo_no_se_toca():
    assert pl.filtrar_listas(CANDS, {"cookingTime": "30min"}, listos=LISTOS) == CANDS
    assert pl.filtrar_listas(CANDS, {}, listos=LISTOS) == CANDS


def test_sin_nada_listo_se_conserva_la_lista():
    solo_crudo = CANDS[:1] + CANDS[2:3]
    assert pl.filtrar_listas(solo_crudo, {"cookingTime": "none"}, listos=LISTOS) == solo_crudo


def test_los_cuatro_cerradores_filtran():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert src.count('__import__("proteina_lista").filtrar_listas(_safe_high_density_proteins(') == 4


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 245
