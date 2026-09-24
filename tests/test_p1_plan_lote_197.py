# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-197 · 2026-09-24] Los consejos de micronutrientes no le recomiendan a nadie lo que no puede comer.

rd21 (alergia a lácteos y mariscos): plan limpio al primer intento, pero el panel decía «Calcio bajo — Refuerza con
lácteos (yogur/queso)», el consejo de suplemento «primero_alimentos: yogur/queso…» y la directiva del PROMPT le pedía
al modelo «Calcio → lácteos (yogur, queso)» y «Zinc → mariscos», contra la prohibición dura del mismo prompt."""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import micros_seguros as ms  # noqa: E402

_LM = ["Lacteos", "Mariscos"]


def test_la_nota_del_calcio_sin_lacteos_ofrece_las_alternativas():
    import micronutrients as mn
    original = mn._SUPPLEMENT_NOTE["calcium_mg"]
    nota = ms.nota_panel("calcium_mg", original, _LM)
    assert not re.search(r"l[aá]cteo|yogur|queso", nota, re.I), nota
    assert "bebidas vegetales fortificadas" in nota and "tofu" in nota, nota
    assert ms.nota_panel("calcium_mg", original, []) == original, "sin alergias, el texto de siempre"


def test_zinc_y_selenio_sin_mariscos():
    import micronutrients as mn
    for clave in ("zinc_mg", "selenium_mcg"):
        nota = ms.nota_panel(clave, mn._SUPPLEMENT_NOTE[clave], _LM)
        assert "marisco" not in nota.lower(), (clave, nota)


def test_los_alimentos_del_suplemento_tambien():
    original = "yogur/queso, sardina con espina, vegetales de hoja verde, sésamo/ajonjolí, tofu"
    nuevo = ms.alimentos_seguros("calcium_mg", original, ["Lacteos", "Pescado"])
    assert "yogur" not in nuevo and "sardina" not in nuevo, nuevo
    assert "bebidas vegetales fortificadas con calcio" in nuevo and nuevo.count("tofu") == 1, nuevo


def test_la_directiva_del_prompt_no_contradice_la_alergia():
    import micronutrients as mn
    sin = mn.build_micronutrient_targets_directive(sex="male", age=40, goal="gain_muscle")
    con = mn.build_micronutrient_targets_directive(sex="male", age=40, goal="gain_muscle", allergies=_LM)
    assert "lácteos (yogur, queso)" in sin and "mariscos" in sin, "la directiva de siempre sí los nombra"
    calcio = next(l for l in con.split("\n") if l.startswith("• Calcio"))
    zinc = next(l for l in con.split("\n") if l.startswith("• Zinc"))
    prioridad = next(l for l in con.split("\n") if l.startswith("PRIORIDAD"))
    assert "lácteos" not in calcio and "bebidas vegetales fortificadas" in calcio, calcio
    assert "mariscos" not in zinc, zinc
    assert "queso" not in prioridad, prioridad


def test_el_consejo_de_suplemento_recibe_las_alergias():
    import micronutrients as mn
    informe = {"gaps": [{"key": "calcium_mg", "nutriente": "Calcio", "status": "bajo", "valor": 400, "piso": 1000,
                         "unidad": "mg"}]}
    items = mn.build_supplement_recommendations(informe, sex="male", age=40, allergies=_LM)["items"]
    assert items and "yogur" not in items[0]["primero_alimentos"], items


def test_las_siete_llamadas_pasan_las_alergias():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    lineas = [l for l in src.splitlines()
              if 'allergies=__import__("constants").alergias_y_rechazos(' in l and "P1-PLAN-LOTE-197" in l]
    assert len(lineas) == 7, lineas


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 197 and m.group(2) >= "2026-09-24"
