# -*- coding: utf-8 -*-
"""[P1-SUBST-ORPHAN-STEP · 2026-09-05] El reescritor de pasos solo puede actuar si el alimento VIEJO se nombraba
en la receta. Cuando el LLM no lo nombra —merienda del día 3 del plan vivo 18326457: los pasos hablaban de
ciruelas, queso cottage y granola— el ingrediente NUEVO entraba en la lista del plato y en la de la compra sin
aparecer en ninguna instrucción. El retorno del reescritor ya distinguía el caso y se estaba tirando."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402

_VEGETARIANO = {"dietType": "vegetariana"}


def _meal(ingredientes, pasos):
    return {"days": [{"meals": [{"name": "Plato", "ingredients": list(ingredientes),
                                 "ingredients_raw": list(ingredientes), "recipe": list(pasos)}]}]}


def _receta(plan):
    return plan["days"][0]["meals"][0]["recipe"]


def test_la_nota_nombra_el_ingrediente_huerfano():
    """El caso vivo: los pasos no mencionan el alimento sustituido, así que nadie sabe qué hacer con el nuevo."""
    plan = _meal(["¼ taza de queso cottage", "120 g de jamón", "25 g de granola"],
                 ["Mise en place: mide el queso cottage y la granola.",
                  "Montaje: coloca el queso en un vaso y corona con la granola."])
    assert go._apply_diet_substitutions(plan, _VEGETARIANO) == 1
    nota = [s for s in _receta(plan) if "reemplaz" in s.lower()]
    assert nota, "la nota de procedencia sigue estando"
    assert "no lo nombran" in nota[0], nota[0]
    assert "Soya texturizada" in nota[0], f"la nota debe NOMBRAR el ingrediente: {nota[0]}"


def test_si_los_pasos_si_lo_nombran_la_nota_no_se_alarga():
    """Cuando el reescritor pudo colocar el ingrediente, añadir la coletilla sería ruido."""
    plan = _meal(["120 g de jamón", "25 g de granola"],
                 ["Saltea el jamón 3 minutos.", "Sirve con la granola por encima."])
    assert go._apply_diet_substitutions(plan, _VEGETARIANO) == 1
    nota = [s for s in _receta(plan) if "reemplaz" in s.lower()][0]
    assert "no lo nombran" not in nota, nota
    pasos = " ".join(_receta(plan)[:2]).lower()
    assert "soya texturizada" in pasos, "el paso quedó reescrito con el nuevo alimento"
    assert "jamón" not in pasos and "jamon" not in pasos


def test_la_coletilla_es_idempotente():
    """Las subs corren dos veces por plan (pasada + repaso tras las de condición): la nota no se duplica."""
    plan = _meal(["120 g de jamón"], ["Mezcla todo en un bol."])
    go._apply_diet_substitutions(plan, _VEGETARIANO)
    go._apply_diet_substitutions(plan, _VEGETARIANO)
    notas = [s for s in _receta(plan) if "no lo nombran" in s]
    assert len(notas) == 1, notas


def test_no_se_inventa_un_paso_nuevo():
    """Dónde va exactamente el ingrediente depende de la receta; afirmarlo sería adivinar. Va en la nota."""
    plan = _meal(["120 g de jamón"], ["Mezcla todo en un bol.", "Refrigera 10 minutos."])
    go._apply_diet_substitutions(plan, _VEGETARIANO)
    rec = _receta(plan)
    assert len(rec) == 3, f"dos pasos + la nota, sin instrucciones inventadas: {rec}"
    assert rec[0].startswith("Mezcla") and rec[1].startswith("Refrigera")


def test_el_retorno_del_reescritor_deja_de_tirarse():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("_steps_changed = False")
    bloque = src[i:i + 2200]
    assert "_steps_changed = bool(_rewrite_recipe_steps_after_subs(" in bloque
    assert "if note and swaps and not _steps_changed:" in bloque
