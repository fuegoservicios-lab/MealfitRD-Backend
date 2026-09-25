# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-255 · 2026-09-25] «Posible reactividad cruzada» con un alimento no declarado es un aviso.

Batería rd252, perfil maní + sésamo + «piña» (texto libre): dos rechazos CRÍTICOS por linaza, edamame, lechosa y guineo
«con posible reactividad cruzada» y el usuario recibió el plan de emergencia. Los textos reales, abajo.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import revisor_no_defectos as rnd  # noqa: E402

REALES = [
    "Día 2: contiene semillas de linaza, con posible reactividad cruzada con la alergia declarada al sésamo.",
    "Día 2: contiene edamame (soja), con posible reactividad cruzada con la alergia declarada al maní.",
    "Día 3: la lechosa (papaya) y los guineos (plátano) se identifican en el reporte clínico como posibles reactivos "
    "cruzados de alto riesgo para la alergia a la piña. Evitarlos hasta una evaluación alergológica o provocación oral "
    "controlada.",
    "Día 3: las semillas de girasol presentan posible reactividad cruzada con el sésamo; el reporte recomienda "
    "permitirlas solo tras descartar sensibilización clínica.",
]


def test_los_criticos_reales_pasan_a_aviso():
    approved, issues, severity, avisos = rnd._downgrade_reviewer_non_issues(False, list(REALES), "critical")
    assert approved is True and issues == [] and len(avisos) == 4, (approved, issues, severity)


def test_un_defecto_real_se_queda():
    real = "Día 1: la cena contiene maní tostado, alérgeno declarado por el usuario."
    approved, issues, severity, avisos = rnd._downgrade_reviewer_non_issues(False, [real, REALES[0]], "critical")
    assert approved is False and issues == [real] and severity == "critical" and avisos == [REALES[0]]


def test_el_alergeno_declarado_lo_sigue_parando_la_guarda_determinista():
    # La rebaja solo toca el TEXTO del revisor; el escáner determinista corre después con la alergia declarada.
    import graph_orchestrator as go
    plan = {"days": [{"meals": [{"name": "x", "ingredients": ["1 cda de tahini", "20 g de linaza"]}]}]}
    v = go._scan_allergen_violations(plan, ["Sesamo"])
    assert [x[1] for x in v] == ["1 cda de tahini"], v


def test_ancla():
    src = (_BACKEND / "revisor_no_defectos.py").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-255-REACTIVIDAD-CRUZADA" in src and "_CROSS_REACTIVITY_RX.search(t)" in src


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 255
