# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-265 · 2026-09-25] Las restricciones duras del formulario llegan al evaluador y a los correctores.

Batería final (alergia al maní y al sésamo): la autocrítica sugirió «cambia la merienda por fruta+maní». Ni el evaluador
ni sus correctores recibían alergias, rechazos, dieta ni condiciones: sólo metas, tiempo de cocina y horario.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import horizon  # noqa: E402


def test_las_cuatro_restricciones_en_claro():
    fd = {"allergies": ["Maní", "Sésamo"], "otherAllergies": "piña", "dislikes": ["Berenjena"],
          "dietType": "vegetariana", "medicalConditions": ["Diabetes T2"]}
    txt = horizon.restrictions_rule(fd)
    assert "Maní" in txt and "Sésamo" in txt and "piña" in txt          # también lo tecleado en «Otra alergia»
    assert "Berenjena" in txt
    assert "vegetariana" in txt and "pescado" in txt
    assert "Diabetes T2" in txt
    assert "obligatorias también al corregir" in txt


def test_el_evaluador_lleva_su_propia_cabecera():
    txt = horizon.restrictions_rule({"allergies": ["Maní"]}, para="evaluador")
    assert "tus sugerencias NUNCA pueden proponer" in txt and "Maní" in txt


def test_sin_restricciones_no_hay_bloque():
    assert horizon.restrictions_rule({}) == ""
    assert horizon.restrictions_rule({"allergies": ["Ninguna"], "medicalConditions": ["Ninguna"],
                                      "dietType": "balanced"}) == ""
    assert horizon.restrictions_rule(None) == ""


def test_texto_libre_largo_se_acota():
    txt = horizon.restrictions_rule({"allergies": ["x" * 500]})
    assert "x" * 61 not in txt


def test_cableado_en_evaluador_y_correctores():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert 'restrictions_rule(state.get("form_data") or {})  # [P1-PLAN-LOTE-265]' in src   # corrector Flash (y Pro)
    assert 'restrictions_rule(form_data or {})  # [P1-PLAN-LOTE-265]' in src                # re-corrección quirúrgica
    assert "{gm_dinner_block}{user_context}{__import__('horizon').restrictions_rule(form_data, para='evaluador')}" in src
    # las hermanas siguen en su sitio (lotes 220 y 241)
    assert "HORARIO DEL USUARIO (obligatorio, también al corregir): {_sch_rule}" in src
    assert "P1-PLAN-LOTE-265-RESTRICCIONES-A-LOS-CORRECTORES" in (_BACKEND / "horizon.py").read_text(encoding="utf-8")


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 265
