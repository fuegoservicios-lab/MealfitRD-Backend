# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-182 · 2026-09-23] Batería real rd11: lo que llevó un plan al de EMERGENCIA y lo que costó reintentos.

  · Alergia a lácteos y a mariscos: el día traía «yogurt griego» en dos meriendas —la guarda lo cazó— y el propio prompt
    se lo sugería («…queso fresco, yogurt, frutos secos… estas son OK siempre»). El segundo intento lo rechazó CRÍTICO
    el revisor por «confirme si la alergia a mariscos incluye pescado» ⇒ plan de emergencia.
  · Embarazo: «habichuelas rojas secas sin indicar que deben hervirse al menos 10 minutos».
  · «corta 275 g de pechuga» con «1 pechuga de pollo (≈134 g)» en la lista (7 de 13 menciones descuadradas de rd9+rd10).
    Sólo se RECORTA: «el paso pide MENOS» es la decisión V7a del dueño (docs/decisiones_dueno_2026_09_14.md)."""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def _esqueleto():
    return {"brief_concept": "Día variado", "assigned_technique": "A la plancha",
            "protein_pool": ["Pollo"], "carb_pool": ["Avena", "Batata", "Yuca"], "fruit_pool": ["Mango"],
            "meal_types": ["Desayuno", "Almuerzo", "Merienda", "Cena"]}


# ─────────────── la alergia en la asignación del día ───────────────

def test_sin_alergias_el_texto_no_cambia():
    from prompts.day_generator import build_day_assignment_context
    ctx = build_day_assignment_context(_esqueleto(), 1)
    assert "ALERGIAS DECLARADAS" not in ctx
    assert "huevos, claras, queso fresco, yogurt, frutos secos, mantequilla de maní (estas son OK siempre" in ctx
    assert "fruta con lácteo, pan integral, casabe, tostada de maíz, frutos secos o yogur" in ctx
    assert "líquidos (aceite, leche, etc)" in ctx
    assert build_day_assignment_context(_esqueleto(), 1, allergies=["Ninguna"], dislikes=["Ninguno"]) == ctx


def test_alergia_a_lacteos_linea_dura_y_ninguna_sugerencia_lactea():
    from prompts.day_generator import build_day_assignment_context
    ctx = build_day_assignment_context(_esqueleto(), 1, allergies=["Lacteos", "Mariscos"], dislikes=["Ninguno"])
    assert "🚫 ALERGIAS DECLARADAS — PROHIBICIÓN ABSOLUTA: Lacteos, Mariscos" in ctx
    ok = re.search(r"Para diversificar desayuno/merienda usa: (.+?) \(estas son OK siempre", ctx).group(1)
    assert "yogurt" not in ok and "queso" not in ok and "huevos" in ok, ok
    assert "yogur)" not in ctx and "fruta con lácteo" not in ctx
    assert "líquidos (aceite, agua, caldo, etc)" in ctx


def test_alergia_al_huevo_y_rechazo_del_mani():
    from prompts.day_generator import build_day_assignment_context
    ctx = build_day_assignment_context(_esqueleto(), 1, allergies=["Huevo"], dislikes=["Maní"])
    ok = re.search(r"Para diversificar desayuno/merienda usa: (.+?) \(estas son OK siempre", ctx).group(1)
    assert "huevos" not in ok and "claras" not in ok and "mantequilla de maní" not in ok, ok
    assert "yogurt" in ok


def test_los_tres_llamadores_pasan_alergias_y_rechazos():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    llamadas = [m.end() for m in re.finditer(r"build_day_assignment_context\(", src)]
    assert len(llamadas) >= 3
    for i in llamadas:
        prof, j = 1, i
        while prof and j < len(src):                       # hasta el paréntesis que cierra la llamada
            prof += {"(": 1, ")": -1}.get(src[j], 0)
            j += 1
        assert "allergies=" in src[i:j] and "dislikes=" in src[i:j], src[i:i + 200]


# ─────────────── «confirme si…» es aviso; lo que habla de pasteurizar, nunca ───────────────

def test_confirmar_el_alcance_de_una_alergia_es_aviso():
    import graph_orchestrator as go
    for issue in ("El plan incluye pescado y atún. Confirme si la alergia declarada a mariscos también incluye pescado antes de consumirlos.",
                  "El atún aparece en dos comidas. La alergia declarada a «mariscos» puede incluir pescado en el uso coloquial; confirmar su alcance antes de servirlo."):
        aprobado, reales, _s, avisos = go._downgrade_reviewer_verification_demands(False, [issue], "critical")
        assert aprobado and reales == [] and avisos == [issue]


def test_pasteurizar_nunca_es_aviso():
    import graph_orchestrator as go
    for issue in ("Confirme si el queso fresco es pasteurizado (embarazo).",
                  "El queso fresco debe ser pasteurizado por el embarazo declarado."):
        aprobado, reales, _s, _a = go._downgrade_reviewer_verification_demands(False, [issue], "critical")
        assert not aprobado and reales == [issue]


# ─────────────── habichuelas secas ───────────────

def test_habichuelas_secas_llevan_la_nota_para_todos():
    import etiquetas_clinicas as etq
    plan = {"days": [{"day": 1, "meals": [{"meal": "Almuerzo", "name": "Habichuelas guisadas con arroz",
                                           "ingredients": ["½ taza de habichuelas rojas secas", "½ taza de arroz"],
                                           "recipe": ["Remoja las habichuelas la noche anterior.", "Guísalas con sofrito."]}]}]}
    assert etq.etiquetar(plan, {}) == 1
    pasos = plan["days"][0]["meals"][0]["recipe"]
    assert pasos[-1].startswith("⚠️ Seguridad alimentaria:") and "al menos 10 minutos" in pasos[-1]
    etq.etiquetar(plan, {})
    assert sum("al menos 10 minutos" in p for p in pasos) == 1, "idempotente"
    import graph_orchestrator as go
    assert "al menos 10 minutos" in go._meal_safety_notes_for_summary(plan["days"][0]["meals"][0]), "el revisor la lee"


@pytest.mark.parametrize("ing,pasos", [
    ("½ taza de habichuelas rojas cocidas", ["Calienta las habichuelas."]),
    ("½ taza de habichuelas rojas secas", ["Hierve las habichuelas a fuego fuerte 10 minutos y luego 1 hora a fuego bajo."]),
])
def test_sin_nota_si_no_son_secas_o_la_receta_ya_lo_dice(ing, pasos):
    import etiquetas_clinicas as etq
    plan = {"days": [{"day": 1, "meals": [{"name": "Habichuelas", "ingredients": [ing], "recipe": list(pasos)}]}]}
    assert etq.etiquetar(plan, {}) == 0


# ─────────────── el paso no pide más gramos de los que la lista pesa ───────────────

@pytest.fixture(scope="module")
def index():
    from culinary_coherence import build_culinary_index
    fs = sorted(glob.glob(str(_BACKEND / "scripts" / "data" / "catalogo_nutricion_*.json")))
    assert fs, "falta el catálogo congelado"
    cat = json.loads(Path(fs[-1]).read_text(encoding="utf-8"))
    return build_culinary_index(cat.get("filas") or cat.get("rows") or cat)


def test_el_paso_que_pide_mas_que_el_peso_de_la_lista_se_recorta(index):
    from recipe_contract import reconcile_step_quantities
    m = {"ingredients": ["1 pechuga de pollo (≈134 g)", "1 limón"],
         "recipe": ["Mise en place: corta 275 g de pechuga de pollo en tiras y exprime el limón.",
                    "Cocina la pechuga de pollo 8 minutos por lado."]}
    r = reconcile_step_quantities(m, index)
    assert m["recipe"][0] == "Mise en place: corta 134 g de pechuga de pollo en tiras y exprime el limón."
    assert r["reescritas"] == 1 and r["familias"] == {"g": 1}
    assert reconcile_step_quantities(m, index)["reescritas"] == 0, "idempotente"


def test_el_paso_que_pide_menos_se_queda_decision_v7a(index):
    from recipe_contract import reconcile_step_quantities
    paso = "Mise en place: corta 75 g de pechuga de pollo en cubos."
    m = {"ingredients": ["½ pechuga de pollo (≈100 g)"], "recipe": [paso]}
    assert reconcile_step_quantities(m, index)["reescritas"] == 0 and m["recipe"][0] == paso


def test_con_gramos_exactos_en_la_lista_manda_el_exacto(index):
    from recipe_contract import _cantidades_lista, _gramos_aproximados
    ings = ["150 g de pechuga de pollo", "1 pechuga de pollo (≈134 g)"]
    assert _gramos_aproximados(ings, index, _cantidades_lista(ings, index)) == {}


def test_knob_apagado_no_toca(index, monkeypatch):
    import recipe_contract as rc
    monkeypatch.setattr(rc, "approx_grams_on", lambda: False)
    paso = "Mise en place: corta 275 g de pechuga de pollo en tiras."
    m = {"ingredients": ["1 pechuga de pollo (≈134 g)"], "recipe": [paso]}
    rc.reconcile_step_quantities(m, index)
    assert m["recipe"][0] == paso


def test_la_decision_v7a_ya_se_corrige(index):
    """[P1-PLAN-LOTE-212] El dueño cambió la decisión V7a el 24-sep: el paso que pide menos se alinea, con plural."""
    from recipe_contract import reconcile_step_quantities
    m = {"ingredients": ["2 tortillas integrales"], "recipe": ["Mise en place: mide 1 tortilla integral."]}
    r = reconcile_step_quantities(m, index)
    assert m["recipe"][0] == "Mise en place: mide 2 tortillas integrales." and r["sin_reparar"] == {}


# ─────────────── DM2: la fruta dulce entra en el tope glucémico ───────────────

class _DB:
    def macros_from_ingredient_string(self, s):
        m = re.match(r"\s*([\d.]+)\s*g\s+de\s+(.+)$", str(s))
        if not m:
            return None
        g = float(m.group(1))
        return {"name": m.group(2), "grams": g, "kcal": 0.5 * g, "protein": 0.01 * g, "carbs": 0.12 * g, "fats": 0.0}

    def grams_from_ingredient_string(self, s):
        mac = self.macros_from_ingredient_string(s)
        return mac["grams"] if mac else None

    def __getattr__(self, _n):
        return lambda *a, **k: None


def _g(linea):
    return float(re.match(r"\s*([\d.]+)", linea).group(1))


def test_dm2_la_fruta_dulce_entra_en_el_tope():
    import graph_orchestrator as go
    comida = {"meal": "Merienda PM", "name": "Batido tropical de piña",
              "ingredients": ["300 g de piña", "60 g de yogurt griego", "100 g de guineo verde"], "recipe": []}
    n = go.cap_dm2_high_gi_portions([{"day": 2, "meals": [comida]}], {"medicalConditions": ["Diabetes T2"]}, db=_DB())
    assert n == 1, comida["ingredients"]
    assert _g(comida["ingredients"][0]) == 120 and "piña" in comida["ingredients"][0]
    assert _g(comida["ingredients"][2]) >= 100, "el guineo verde no es fruta dulce en DM2"


def test_sin_dm2_la_fruta_no_se_toca():
    import graph_orchestrator as go
    comida = {"meal": "Merienda", "name": "Batido de piña", "ingredients": ["300 g de piña"], "recipe": []}
    assert go.cap_dm2_high_gi_portions([{"day": 1, "meals": [comida]}], {"medicalConditions": ["Ninguna"]}, db=_DB()) == 0
    assert comida["ingredients"] == ["300 g de piña"]


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 182 and m.group(2) >= "2026-09-23"
