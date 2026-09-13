# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-35 · 2026-09-13] C7 en su parte medible: CUL-P2-01 (variedad perceptible), P2-02 (presupuesto de
reparación) y P2-03 (deriva por cohorte y cobertura). Tres instrumentos de solo lectura; ninguno cambia un veredicto ni
una comida.

  · `scripts/measure_variedad_perceptible.py`: nombres distintos frente a PLATOS distintos (firma de preparación) en
    ventanas de 7/15/30 días, renombrados y compra reusada con otra técnica.
  · `scripts/measure_presupuesto_reparacion.py`: cuántas capas reescriben cada comida (p50/p95), qué capas, y el coste por
    plan válido; propone un presupuesto (p95 + 1) que NO implementa.
  · `bench_superficies_culinarias.py --informe A --desglose` y `judge_violation_rate.py --por-pais`: la regresión de una
    cohorte o un país no se esconde en la media, y un plan sin medir se ve como menos cobertura.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


def _mod(nombre: str):
    spec = importlib.util.spec_from_file_location(nombre, _BACKEND / "scripts" / f"{nombre}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def var():
    return _mod("measure_variedad_perceptible")


@pytest.fixture(scope="module")
def vocab(var):
    return var._cargar_vocabularios()


def _comida(nombre, ings, pasos, franja="Almuerzo", **kw):
    return {"meal": franja, "name": nombre, "ingredients": ings, "recipe": pasos, **kw}


_POLLO_ARROZ = (["150 g de pechuga de pollo", "1 taza de arroz blanco", "2 hojas de lechuga"],
                ["Guisa el pollo con sazón 20 min.", "Hierve el arroz.", "Sirve con la lechuga."])
_NOMBRES = ["Pollo guisado con arroz y ensalada", "Pechuga criolla con arroz blanco", "Arroz con pollo casero y lechuga",
            "Pollo de la abuela con arroz", "Guiso de pollo con su arroz", "Pollo sabroso con arroz y verde",
            "Plato del día: pollo, arroz y ensalada"]


# ─────────────── CUL-P2-01 · variedad perceptible ───────────────

def test_el_mismo_plato_renombrado_siete_veces_es_un_plato(var, vocab):
    plan = {"days": [{"meals": [_comida(n, *_POLLO_ARROZ)]} for n in _NOMBRES]}
    r = var.medir_plan(plan, vocab, tope_7d=2)
    v = r["ventanas"]["7"]
    assert v["nombres_distintos"] == 7 and v["firmas_distintas"] == 1 and v["renombrados"] == 6
    assert r["ventanas"]["15"]["medida"] is False, "una ventana más larga que el plan no se mide: se dice"
    assert r["ventanas_7d_rotas"] == 1, "siete veces el mismo plato en 7 días rompe el tope de 2"


def test_la_misma_compra_con_otra_tecnica_es_variedad_real(var, vocab):
    guiso = _comida("Pollo guisado", ["150 g de pechuga de pollo", "1 taza de arroz"], ["Guisa el pollo 20 min."])
    plancha = _comida("Wrap de pollo", ["120 g de pechuga de pollo", "1 tortilla de harina"], ["Cocina el pollo a la plancha."])
    r = var.medir_plan({"days": [{"meals": [guiso]}] * 3 + [{"meals": [plancha]}] * 4}, vocab, tope_7d=2)
    v = r["ventanas"]["7"]
    assert v["firmas_distintas"] == 2 and "pollo" in v["proteina_con_varias_tecnicas"]


def test_un_plan_sin_comidas_no_mide_cero_platos(var, vocab):
    """La primera corrida del día determinista salió con 30 días vacíos (knobs fijados después del import) y el informe
    decía «0 nombres / 0 platos». Cero medido y nada medido no son lo mismo."""
    r = var.medir_plan({"days": [{"meals": []}] * 30}, vocab, tope_7d=2)
    assert all(v["medida"] is False and "sin comidas" in v["motivo"] for v in r["ventanas"].values())


def test_la_plantilla_de_la_biblioteca_manda_sobre_el_nombre(var, vocab):
    a = var.firma(_comida("Mangú con huevo", [], [], _template_id="do.desayuno.mangu"), vocab)
    b = var.firma(_comida("Mangú de la casa", [], [], _template_id="do.desayuno.mangu"), vocab)
    assert a == b == ("plantilla", "do.desayuno.mangu")


def test_res_no_se_encuentra_dentro_de_queso_fresco(var, vocab):
    f = var.firma(_comida("Queso fresco con fruta", ["60 g de queso fresco", "1 taza de lechosa"], ["Sirve."]), vocab)
    assert "res" not in f[1]


def test_la_medicion_es_determinista(var, vocab):
    plan = {"days": [{"meals": [_comida(n, *_POLLO_ARROZ)]} for n in _NOMBRES]}
    assert var.medir_plan(copy.deepcopy(plan), vocab, 2) == var.medir_plan(copy.deepcopy(plan), vocab, 2)


def test_el_artefacto_de_variedad_mide_30_dias_del_dia_determinista():
    a = json.loads((_BACKEND / "scripts" / "data" / "variedad_perceptible_2026_09_13.json").read_text(encoding="utf-8"))
    det = [v for k, v in a["planes"].items() if k.startswith("deterministico:")]
    assert det and all(det[0]["ventanas"][w]["medida"] for w in ("7", "15", "30"))
    assert any(k.startswith("corpus:") for k in a["planes"]) and any(k.startswith("real:") for k in a["planes"])
    assert a["tope_7d_balanced"] == 2


# ─────────────── CUL-P2-02 · presupuesto de reparación ───────────────

def test_las_capas_cuentan_reescrituras_no_diagnosticos():
    pr = _mod("measure_presupuesto_reparacion")
    m = {"_portion_floor_adjusted": True, "_protein_closed": True, "_solver_raw_by_food": {"x": 1},
         "_solver_not_converged": True, "_misalign_trace": [1], "_recipe_contract_final": {"reescritas": 2, "modo": "repair"}}
    assert sorted(pr.capas(m)) == ["_portion_floor_adjusted", "_protein_closed", "_recipe_contract_final"]
    assert pr.capas({"_recipe_contract_final": {"reescritas": 0, "modo": "repair"}}) == []
    r = pr.medir([{"days": [{"meals": [m, {"_egg_day_capped": True}, {}]}]}])
    assert r["comidas"] == 3 and r["capas_por_comida"]["max"] == 3 and r["reescrituras_contrato_final"] == 2
    assert r["comidas_con_solver"] == 1, "el solver es composición: se reporta aparte, no como reparación"


def test_el_presupuesto_se_propone_y_no_se_implementa():
    a = json.loads((_BACKEND / "scripts" / "data" / "presupuesto_reparacion_2026_09_13.json").read_text(encoding="utf-8"))
    p95 = max(a["fuentes"][k]["capas_por_comida"]["p95"] for k in ("corpus", "real"))
    assert a["propuesta"]["valor"] == p95 + 1 and "no implementada" in a["propuesta"]["estado"].lower()
    assert a["coste"]["planes"] == 3 and "válido" in a["coste"]["nota"]
    for f in ("graph_orchestrator.py", "recipe_contract.py", "knobs.py"):
        assert "MEALFIT_RECIPE_REPAIR_BUDGET" not in (_BACKEND / f).read_text(encoding="utf-8"), f"{f}: la propuesta no se cablea"


# ─────────────── CUL-P2-03 · deriva y cobertura ───────────────

def test_el_desglose_reparte_lo_medido_sin_cambiar_el_informe_base():
    bench = _mod("bench_superficies_culinarias")
    art = json.loads((_BACKEND / "scripts" / "data" / "bench_superficies_replay_2026_09_13_l31.json").read_text(encoding="utf-8"))
    base = bench.render(art)
    assert "desglose" not in base
    perfiles = bench._perfiles_de(art, _BACKEND / "scripts" / "data")
    assert set(perfiles.values()) == {"perfil_baseline_m", "perfil_dm2_metformina", "perfil_vegetariana"}
    con = bench.render(art, desglose=True, perfiles=perfiles)
    assert con.startswith(base), "el desglose se AÑADE; no reescribe el informe"
    assert con.count("cobertura ") == len(art["superficies"]) and "perfil_dm2_metformina" in con


def test_el_juez_se_desglosa_por_pais():
    src = (_BACKEND / "scripts" / "judge_violation_rate.py").read_text(encoding="utf-8")
    assert "--por-pais" in src and "plan_data->>'_country'" in src and "P1-PLAN-LOTE-35" in src


def test_docs_plan_marker():
    import re
    doc = (_BACKEND / "docs" / "culinary_coherence.md").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-35" in doc and "variedad perceptible" in doc.lower() and "presupuesto" in doc.lower()
    assert "P1-PLAN-LOTE-35" in (_BACKEND / "docs" / "plan_pendientes_2026_09_11.md").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 35
