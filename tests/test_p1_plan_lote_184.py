# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-184 · 2026-09-23] Batería rd14 (código 183):

  · Embarazo: rechazo CRÍTICO «el pollo al airfryer… no se indica que deba cocinarse completamente (74 °C)» — con la
    cláusula de carnes del 183 ya en vigor. La nota se omitía porque un PASO de la receta ya decía «74 °C» (lo pide el
    prompt, P0-DONENESS), pero el revisor sólo lee nombre, ingredientes y NOTAS: para la cláusula de carnes, sólo una
    nota la cubre.
  · Alergia a lácteos y mariscos: «verificar SUS etiquetas» (el patrón tenía «la/las») y «evitar pescado hasta confirmar
    con evaluación alergológica»: demandas de verificación que el menú no puede satisfacer con un dato.

Lo que NO entró (y por qué): en rd13 el cerrador pegaba huevo a una merienda dulce de un alérgico a lácteos aunque el
desayuno ya lo llevara (rechazo de variedad). Dejar la merienda sin proteína en ese caso se probó en rd14 y abrió
DÉFICIT DE PROTEÍNA en HTA (98/116 g) y embarazo (94/116 g): el diseño de julio («el piso gana», test de
P1-CLOSER-DAY-AWARE-PROTEIN) tenía razón. Se revirtió."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_PREG = {"medicalConditions": ["Embarazo"], "gender": "female"}


def _nota(meal):
    return next((s for s in meal.get("recipe") or [] if str(s).startswith("🤰")), "")


def test_la_clausula_de_carnes_no_se_omite_porque_un_paso_diga_74():
    import graph_orchestrator as go
    comida = {"meal": "Almuerzo", "name": "Pollo al airfryer con arroz integral",
              "ingredients": ["1 pechuga de pollo (≈150 g)", "½ taza de arroz integral"],
              "recipe": ["Cocina el pollo en el airfryer 18 minutos hasta que alcance 74 °C por dentro.", "Sirve."]}
    go._apply_pregnancy_food_safety_annotations({"days": [{"day": 1, "meals": [comida]}]}, _PREG)
    assert "sin partes rosadas" in _nota(comida), comida["recipe"]
    assert "sin partes rosadas" in go._meal_safety_notes_for_summary(comida), "el revisor la lee"


def test_una_nota_de_seguridad_con_74_si_la_cubre():
    import graph_orchestrator as go
    comida = {"meal": "Almuerzo", "name": "Pollo frío estilo ceviche",
              "ingredients": ["1 pechuga de pollo (≈150 g)", "1 limón"],
              "recipe": ["Hierve la pechuga.", "⚠️ Seguridad alimentaria: cocina la carne por completo (74 °C por dentro, "
                                               "sin partes rosadas) ANTES de marinarla en el limón."]}
    go._apply_pregnancy_food_safety_annotations({"days": [{"day": 1, "meals": [comida]}]}, _PREG)
    assert "cocina las carnes y el pollo" not in _nota(comida), "ya hay una NOTA que lo dice"


@pytest.mark.parametrize("issue", [
    "No se puede confirmar que el pan integral familiar (día 2) esté libre de leche y derivados. Verificar sus etiquetas "
    "y excluir cualquier producto que contenga leche.",
    "El plan incluye filetes de pescado y atún en varios días. El reporte clínico recomienda evitar pescado y atún hasta "
    "confirmar con evaluación alergológica la posible sensibilización asociada a la alergia declarada a mariscos.",
    "Día 3, desayuno: el pan integral puede contener leche. Con alergia declarada a lácteos, confirme la etiqueta o "
    "sustitúyalo por pan sin lácteos.",
])
def test_demandas_de_verificacion_son_aviso(issue):
    import graph_orchestrator as go
    aprobado, reales, _s, avisos = go._downgrade_reviewer_verification_demands(False, [issue], "critical")
    assert aprobado and reales == [] and avisos == [issue]


def test_confirmar_con_algo_que_habla_de_pasteurizar_sigue_siendo_rechazo():
    import graph_orchestrator as go
    issue = "Confirme con el vendedor que el queso fresco sea pasteurizado (embarazo)."
    aprobado, reales, _s, _a = go._downgrade_reviewer_verification_demands(False, [issue], "critical")
    assert not aprobado and reales == [issue]


def test_las_alternativas_sin_lacteo_no_empujan_huevo():
    from prompts.day_generator import allergy_hard_line
    linea = allergy_hard_line(["Lacteos"])
    assert "Sin lácteos" in linea and "huevo" not in linea.split("Sin lácteos", 1)[1]


def test_el_piso_de_proteina_sigue_ganando_en_la_merienda():
    """La regla revertida no debe volver: rd14 midió el déficit que abre."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-184-MERIENDA-SIN-REPETIR" not in src


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 184 and m.group(2) >= "2026-09-23"
