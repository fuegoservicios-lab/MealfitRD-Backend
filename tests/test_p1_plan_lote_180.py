# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-180 · 2026-09-23] Dos rechazos CRÍTICOS del revisor en la batería real que el reintento arreglaba, a
~4 min cada uno:

  · HTA — «el almuerzo del día 3 se describe como preparado con pollo crudo al estilo ceviche». El plato se llamaba
    «Pollo frío estilo ceviche…» y su receta cocina la pechuga, pero el revisor sólo lee nombre, ingredientes y notas
    de seguridad: la nota «cocina la carne por completo ANTES del limón» va ahora a todo ceviche de carne, para todos.
  · embarazo — «el casabe… sin indicar que sea de origen industrial/controlado». Demanda de CADENA DE SUMINISTRO que el
    generador no puede satisfacer con un dato: aviso, no rechazo (la familia de «marca regulada», lote 177)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import etiquetas_clinicas as etq  # noqa: E402


def _plan(nombre, ingredientes, pasos):
    return {"days": [{"day": 3, "meals": [{"meal": "Almuerzo", "name": nombre, "ingredients": list(ingredientes),
                                           "recipe": list(pasos), "_display": {"en-US": {}}}]}]}


def test_ceviche_de_pollo_lleva_la_nota_y_una_sola_vez():
    plan = _plan("Pollo frío estilo ceviche con espinacas", ["150 g de pechuga de pollo", "1 limón"],
                 ["Hierve la pechuga 15 min.", "Desmenúzala y marínala con limón."])
    assert etq.etiquetar(plan, {}) == 1, "sin condiciones también: es seguridad alimentaria, no una condición"
    comida = plan["days"][0]["meals"][0]
    assert comida["recipe"][-1].startswith("⚠️ Seguridad alimentaria:")
    assert "_display" not in comida, "la traducción se rehace con la nota"
    etq.etiquetar(plan, {})
    assert sum("el cítrico sólo da sabor" in p for p in comida["recipe"]) == 1, "idempotente"


def test_ceviche_de_pescado_no_lleva_la_nota_de_carne():
    plan = _plan("Ceviche de dorado", ["150 g de dorado", "1 limón"], ["Corta el pescado.", "Marínalo."])
    assert etq.etiquetar(plan, {}) == 0
    assert not any("cítrico" in p for p in plan["days"][0]["meals"][0]["recipe"])


def test_el_revisor_lee_la_nota_aunque_sea_la_tercera():
    import graph_orchestrator as go
    comida = {"name": "Ceviche de pollo", "recipe": [
        "Hierve el pollo.",
        "⚠️ Seguridad alimentaria: lava bien los vegetales de hoja.",
        "🩺 Nota clínica: prefiere el plátano verde.",
        etq._NOTA_CEVICHE]}
    assert "el cítrico sólo da sabor" in go._meal_safety_notes_for_summary(comida)


def test_origen_industrial_es_aviso_no_rechazo():
    import graph_orchestrator as go
    issue = ("El casabe aparece en los días 1, 2 y 3 sin indicar que sea de origen industrial/controlado o que haya "
             "sido procesado adecuadamente para reducir los glucósidos cianogénicos. Durante el embarazo, debe "
             "especificarse esa condición o excluirse.")
    aprobado, reales, sev, avisos = go._downgrade_reviewer_verification_demands(False, [issue], "critical")
    assert aprobado and reales == [] and avisos == [issue]


def test_pasteurizado_sigue_siendo_rechazo():
    import graph_orchestrator as go
    issue = "El queso fresco debe ser pasteurizado por el embarazo declarado."
    aprobado, reales, _s, _a = go._downgrade_reviewer_verification_demands(False, [issue], "critical")
    assert not aprobado and reales == [issue]


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 180 and m.group(2) >= "2026-09-23"
