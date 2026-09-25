# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-232 · 2026-09-25] Los rechazos («no me gusta») tienen guard determinista en el revisor.

Batería real del 25-sep: el tercer intento de «no me gusta el pescado» traía 200 g de atún en agua; solo lo paró el
revisor LLM. Alergia, dieta y mercurio tenían guard determinista; el rechazo no.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402

PLAN = {"days": [{"day": 2, "meals": [
    {"meal": "Cena", "name": "Bowl ligero de atún y bulgur con mango",
     "ingredients": ["200 g de atún en agua", "35 g de bulgur", "½ mango"]},
    {"meal": "Almuerzo", "name": "Salteado de pollo",
     "ingredients": ["150 g de pechuga de pollo", "80 g de champiñones", "½ taza de remolacha rallada",
                     "90 g de camarones"]},
]}]}


def _terminos(viol):
    return {t for _m, _i, t in viol}


def test_encuentra_chips_texto_libre_y_sinonimos():
    viol = go._scan_dislike_violations(PLAN, {"dislikes": ["Pescado", "Hongos"], "otherDislikes": "remolacha"})
    ings = {i for _m, i, _t in viol}
    assert "200 g de atún en agua" in ings
    assert "80 g de champiñones" in ings
    assert "½ taza de remolacha rallada" in ings
    assert "90 g de camarones" not in ings, "los mariscos no son pescado (lote 210): el usuario no los marcó"


def test_sin_rechazos_no_hay_nada():
    assert go._scan_dislike_violations(PLAN, {"dislikes": ["Ninguno"]}) == []
    assert go._scan_dislike_violations(PLAN, {}) == []
    assert go._scan_dislike_violations(PLAN, {"dislikes": ["Ninguno"], "otherDislikes": "atún"}) == [], \
        "el centinela es exclusivo, igual que en el generador (P0-FORM-1)"


def test_cableado_en_el_revisor():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("_allergen_viol = _scan_allergen_violations(plan, allergies)")
    j = src.index("_dl_viol = _scan_dislike_violations(plan, form_data)", i)
    k = src.index("_diet_viol = _scan_diet_violations(plan, form_data.get(\"dietType\"))", i)
    assert i < j < k
    assert 'severity = _severity_max(severity, "high")' in src[j:j + 900]
    assert go.DISLIKE_HARD_GUARD is True
    assert "tooltip-anchor: P1-PLAN-LOTE-232-RECHAZOS" in src


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 232
