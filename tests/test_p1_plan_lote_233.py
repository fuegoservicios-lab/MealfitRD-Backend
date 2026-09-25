# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-233 · 2026-09-25] La última palabra de alergia/dieta/rechazo en el escudo.

Auditoría del 25-sep: cinco superficies del escudo añaden comida DESPUÉS del escaneo del revisor (rellenos fantasma que
meten maní/aguacate/pan cuando la palabra sale en el nombre o en un paso —«Galletas sin maní» → «10 g de maní»—, relleno
de ganar músculo, sustituto de la compra única, tope de pescado del embarazo) sin mirar las restricciones declaradas.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import restricciones_finales as rf  # noqa: E402

CTX = {"allergies": ["Mani"], "dietType": "vegan", "dislikes": ["Cilantro"], "otherDislikes": "remolacha"}
PLAN = {"days": [{"day": 1, "meals": [
    {"meal": "Almuerzo", "name": "Ensalada verde con garbanzos",
     "ingredients": ["100 g de garbanzos cocidos", "10 g de maní", "1 cda de cilantro picado", "30 g de queso fresco",
                     "½ taza de remolacha rallada", "1 cdta de aceite de oliva"],
     "ingredients_raw": ["1 cdta de aceite de oliva", "30g de queso fresco", "100g de garbanzos cocidos", "10g de maní"],
     "recipe": ["Mise en place: escurre los garbanzos.", "Montaje: sirve con el aceite.",
                "🥜 Espolvorea el maní por encima.", "⚕️ Alergia: sin maní ni derivados."],
     "protein": 99, "carbs": 99, "fats": 99, "cals": 999},
    {"meal": "Merienda", "name": "Galletas sin maní con fresas",
     "ingredients": ["2 galletas de arroz", "10 g de maní", "5 fresas"]},
    {"meal": "Cena", "name": "Pollo con maní y arroz",
     "ingredients": ["120 g de pechuga de pollo", "15 g de maní", "½ taza de arroz"]},
    {"meal": "Desayuno", "name": "Tostada", "ingredients": ["10 g de maní"]},
]}]}


class _DB:
    """Macros de juguete: garbanzos 7 g de proteína, aceite 5 g de grasa; lo demás no resuelve."""
    def macros_from_ingredient_string(self, s):
        s = s.lower()
        if "garbanzo" in s:
            return {"protein": 7.0, "carbs": 20.0, "fats": 2.0, "kcal": 130.0}
        if "aceite" in s:
            return {"protein": 0.0, "carbs": 0.0, "fats": 5.0, "kcal": 45.0}
        return None

    def lookup(self, s):
        return None


def test_retira_lo_anadido_y_respeta_la_identidad():
    plan = copy.deepcopy(PLAN)
    out = rf.retirar_prohibidos(plan, CTX, db=_DB(), surface="test")
    m1, m2, m3, m4 = plan["days"][0]["meals"]
    assert m1["ingredients"] == ["100 g de garbanzos cocidos", "1 cdta de aceite de oliva"], m1["ingredients"]
    assert m1["ingredients_raw"] == ["1 cdta de aceite de oliva", "100g de garbanzos cocidos"], "raw por ALIMENTO"
    assert (m1["protein"], m1["fats"]) == (7, 7), "macros re-sumados desde las líneas que quedan"
    assert m1["recipe"] == ["Mise en place: escurre los garbanzos.", "Montaje: sirve con el aceite.",
                            "⚕️ Alergia: sin maní ni derivados."], "la nota del añadido se va con él; la clínica se queda"
    assert m2["ingredients"] == ["2 galletas de arroz", "5 fresas"], "«sin maní» no es la identidad del plato"
    assert m3["ingredients"] == PLAN["days"][0]["meals"][2]["ingredients"], "la identidad no se toca (la rechaza el revisor)"
    assert m4["ingredients"] == ["10 g de maní"], "nunca deja una comida sin ingredientes"
    kinds = {(r["kind"], r["term"]) for r in out["retiradas"]}
    assert ("alergia", "mani") in {(k, t.replace("í", "i")) for k, t in kinds}
    assert any(k == "dieta" for k, _ in kinds) and any(k == "rechazo" for k, _ in kinds)
    assert plan["_restricciones_retiradas"] and plan["_restricciones_en_el_plato"]


def test_sin_contexto_no_hace_nada():
    plan = copy.deepcopy(PLAN)
    assert rf.retirar_prohibidos(plan, {}) == {"retiradas": [], "en_el_plato": []}
    assert plan == PLAN


def test_cableado_al_final_del_escudo():
    src = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
    i = src.index("_etq.etiquetar(_pd, _clin_ctx)")
    j = src.index("_rfin.retirar_prohibidos(_pd, _clin_ctx, db=_db_ins, surface=surface)", i)
    assert j > i
    assert "tooltip-anchor: P1-PLAN-LOTE-233-ULTIMA-PALABRA" in src


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 233
