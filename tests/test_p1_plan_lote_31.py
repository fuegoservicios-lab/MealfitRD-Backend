# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-31 · 2026-09-13] Los V7e «de fábrica» del bench real: el paso pide más piezas de las que la lista compra.

Diez V7e en los 3 planes recién generados (lote 30). Medidos uno a uno eran tres cosas distintas:
  · siete: la lista bajó de 2 a ½ (o a 1) y el paso siguió diciendo «los 2 plátanos verdes» — el contrato de C2 no tocaba
    piezas que cruzan el singular/plural (`gramatical`). Ahora, cuando el paso pide MÁS, reescribe con concordancia: número,
    artículo, sustantivo (del nombre canónico, con sus tildes), adjetivos y clíticos de la cláusula;
  · tres: V7e SUMABA las menciones de un mismo paso («6 claras de huevo reservando 6 claras» = 12). Ahora manda la mayor;
  · y el LLM repitiéndose («6 claras de huevo y 6 claras de huevo y 6 claras»): paso (6) del contrato, se deja una.
La dirección contraria (el paso pide MENOS: «pica 1 tomate» con 3) sigue en `gramatical`: es la decisión V7a del dueño.
"""
from __future__ import annotations

import copy
import glob
import inspect
import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


def _ultimo(patron: str) -> Path:
    fs = sorted(glob.glob(str(_BACKEND / "scripts" / "data" / patron)))
    assert fs, f"falta {patron}"
    return Path(fs[-1])


@pytest.fixture(scope="module")
def index():
    from culinary_coherence import build_culinary_index
    cat = json.loads(_ultimo("catalogo_nutricion_*.json").read_text(encoding="utf-8"))
    return build_culinary_index(cat.get("filas") or cat.get("rows") or cat)


def _real() -> dict:
    return json.loads((_BACKEND / "scripts" / "data" / "bench_superficies_real_2026_09_13.json").read_text(encoding="utf-8"))


# ─────────────── (3) la concordancia singular/plural cuando el paso pide MÁS ───────────────

@pytest.mark.parametrize("ings, paso, esperado", [
    (["½ plátanos verde", "3 huevos"], "Mise en place: pela los 2 plátanos verdes y córtalos en 4 trozos cada uno; pela los huevos.",
     "Mise en place: pela el ½ plátano verde y córtalo en 4 trozos; pela los huevos."),
    (["1 ciruela"], "Divide las 2 ciruelas por la mitad.", "Divide la ciruela por la mitad."),
    (["1 ciruela"], "Divide 2 ciruelas por la mitad.", "Divide 1 ciruela por la mitad."),
    (["½ ciruela"], "Acompaña con las 2 ciruelas frescas de postre.", "Acompaña con la ½ ciruela fresca de postre."),
    (["½ ciruela"], "Lava las 2 ciruelas y resérvalas enteras para el plato.", "Lava la ½ ciruela y resérvala entera para el plato."),
    (["1 tortilla integral"], "Calienta las 2 tortillas integrales 30 segundos por lado.", "Calienta la tortilla integral 30 segundos por lado."),
    (["½ cebollines"], "Pica fino los 2 cebollines.", "Pica fino el ½ cebollín."),
    (["1 casabe"], "Montaje: unta las 2 tostadas de casabe con la mantequilla de maní.", "Montaje: unta la tostada de casabe con la mantequilla de maní."),
    (["½ ciruela"], "Sirve unas 2 ciruelas al lado.", "Sirve ½ ciruela al lado."),
])
def test_concordancia_numero_articulo_sustantivo_adjetivos_y_cliticos(index, ings, paso, esperado):
    from recipe_contract import reconcile_step_quantities
    m = {"ingredients": ings, "recipe": [paso]}
    r = reconcile_step_quantities(m, index)
    assert m["recipe"][0] == esperado
    assert r["concordancia"] == 1 and r["reescritas"] == 1 and r["familias"] == {"pieza": 1} and r["sin_reparar"] == {}
    assert r["cambios"][0]["concordancia"] is True and r["cambios"][0]["a"] <= 1.0 < r["cambios"][0]["de"]
    r2 = reconcile_step_quantities(m, index)
    assert r2["reescritas"] == 0 and m["recipe"][0] == esperado, "idempotente: la segunda pasada no cambia nada"


def test_la_direccion_v7a_y_lo_que_no_sabe_concordar_siguen_en_gramatical(index):
    from culinary_coherence import build_culinary_index
    from recipe_contract import _singular_span, reconcile_step_quantities
    # el paso pide MENOS de lo comprado: decisión V7a del dueño, se declara y no se toca (misma conducta que el lote 23)
    m = {"ingredients": ["3 tomates"], "recipe": ["Mise en place: pica 1 tomate."]}
    r = reconcile_step_quantities(m, index)
    assert m["recipe"][0] == "Mise en place: pica 1 tomate." and r["sin_reparar"] == {"gramatical": 1} and r["concordancia"] == 0
    # el singular pediría una tilde que el texto no trae («tostón»): no se inventa
    m = {"ingredients": ["1 casabe"], "recipe": ["Mise en place: prepara 1½ tostones de casabe (aprox. 30 g) para tostar."]}
    r = reconcile_step_quantities(m, index)
    assert "1½ tostones de casabe" in m["recipe"][0] and r["sin_reparar"] == {"gramatical": 1}
    # «2 de ciruelas» no es una forma que sepa leer
    m = {"ingredients": ["½ ciruela"], "recipe": ["Sirve 2 de ciruelas al lado."]}
    assert reconcile_step_quantities(m, index)["sin_reparar"] == {"gramatical": 1} and m["recipe"][0] == "Sirve 2 de ciruelas al lado."
    # un alias que ya es plural no dice cuál es su singular; el canónico sí, con su tilde
    idx = build_culinary_index([{"name": "Plátano verde", "aliases": ["guineos"]}, {"name": "Cebollín"}])
    assert _singular_span("guineos", idx, "Plátano verde") is None
    assert _singular_span("plátanos verdes", idx, "Plátano verde") == "plátano verde"
    assert _singular_span("Cebollines", idx, "Cebollín") == "Cebollín"
    assert _singular_span("cebollines", idx, "Cebollín") == "cebollín"
    m = {"ingredients": ["½ plátanos verde"], "recipe": ["Pela los 2 guineos."]}
    assert reconcile_step_quantities(m, idx)["sin_reparar"] == {"gramatical": 1} and m["recipe"][0] == "Pela los 2 guineos."


def test_la_concordancia_no_toca_la_clausula_si_habla_de_otro_alimento(index):
    from recipe_contract import reconcile_step_quantities
    # «Sal» es alimento del catálogo: la cláusula no es sólo del plátano ⇒ los clíticos no se tocan (conservador)
    m = {"ingredients": ["½ plátanos verde", "Sal al gusto"], "recipe": ["Hierve los 2 plátanos verdes y escúrrelos bien junto con la sal."]}
    reconcile_step_quantities(m, index)
    assert m["recipe"][0] == "Hierve el ½ plátano verde y escúrrelos bien junto con la sal."
    # y si el otro alimento va en OTRA cláusula, los clíticos de la del plátano sí se concuerdan
    m = {"ingredients": ["½ plátanos verde", "Sal al gusto"], "recipe": ["Hierve los 2 plátanos verdes y escúrrelos bien; añade la sal."]}
    reconcile_step_quantities(m, index)
    assert m["recipe"][0] == "Hierve el ½ plátano verde y escúrrelo bien; añade la sal."


# ─────────────── el detector V7e: la mención mayor del paso, no la suma ───────────────

def test_v7e_no_suma_las_menciones_del_mismo_paso_y_la_lista_y_v7a_siguen_sumando(index):
    import culinary_coherence as cc
    dia = {"day": 1}
    m = {"ingredients": ["3 huevos", "6 claras de huevo"], "recipe": ["Separa 3 huevos y 6 claras de huevo reservando 6 claras en un bol."]}
    assert cc._v7e_paso_pide_mas_piezas(dia, m, index) == [], "las mismas seis claras, no doce"
    m["recipe"] = ["Casca 12 claras de huevo sobre el guiso."]
    v = cc._v7e_paso_pide_mas_piezas(dia, m, index)
    assert len(v) == 1 and v[0]["food"] == "Clara de huevo" and "pide 12 y la lista compra 6" in v[0]["detail"]
    assert cc._v7_piezas("6 claras de huevo y 6 claras", index) == {"Clara de huevo": 12.0}
    assert cc._v7_piezas("6 claras de huevo y 6 claras", index, agregar="max") == {"Clara de huevo": 6.0}
    # dos líneas del mismo alimento compran la SUMA (la lista sigue sumando), y V7a también
    src_e = inspect.getsource(cc._v7e_paso_pide_mas_piezas)
    src_a = inspect.getsource(cc._v7a_lista_compra_de_mas)
    assert 'agregar="max"' in src_e and 'agregar="max"' not in src_a and "P1-PLAN-LOTE-31-V7E-MAX" in src_e
    cuerpo = src_e.split('"""')[2]                       # el código, sin el docstring
    assert "_v7_piezas(ing, index).items()" in cuerpo and 'agregar="max").items()' in cuerpo, "la lista suma; el paso no"


# ─────────────── (6) el LLM que se repite ───────────────

def test_colapsar_repeticiones_deja_una_mencion_y_respeta_lo_que_no_es_cadena(index):
    from recipe_repair import colapsar_repeticiones
    m = {"ingredients": ["3 huevos", "6 claras de huevo"],
         "recipe": ["Mise en place: separa 3 huevos y 6 claras de huevo y 6 claras de huevo y 6 claras.",
                    "Casca 3 huevos y 6 claras de huevo, 6 claras de huevo sobre el guiso.",
                    "Separa 3 huevos y 6 claras de huevo reservando 6 claras en un bol.",
                    "Hierve 6 claras y 6 claras de huevo en agua.",
                    "⚠️ Seguridad alimentaria: 6 claras de huevo y 6 claras de huevo bien cocidas."]}
    r = colapsar_repeticiones(m, index)
    assert r["aplicado"] == ["Clara de huevo"] and r["descartado"] == [] and len(r["cambios"]) == 3
    assert m["recipe"] == ["Mise en place: separa 3 huevos y 6 claras de huevo.",
                           "Casca 3 huevos y 6 claras de huevo sobre el guiso.",
                           "Separa 3 huevos y 6 claras de huevo reservando 6 claras en un bol.",
                           "Hierve 6 claras en agua.",
                           "⚠️ Seguridad alimentaria: 6 claras de huevo y 6 claras de huevo bien cocidas."]
    assert m["ingredients"] == ["3 huevos", "6 claras de huevo"], "la lista no se toca"
    assert colapsar_repeticiones(m, index) == {"aplicado": [], "descartado": [], "cambios": []}, "idempotente"
    # cantidades distintas no son una repetición
    m = {"ingredients": ["3 tomates"], "recipe": ["Pica 1 tomate y 2 tomates."]}
    assert colapsar_repeticiones(m, index)["aplicado"] == [] and m["recipe"] == ["Pica 1 tomate y 2 tomates."]


def test_colapsar_repeticiones_se_deshace_si_abre_un_hallazgo(index, monkeypatch):
    import recipe_repair as rr
    llamadas = {"n": 0}

    def _falso(meal, idx):
        llamadas["n"] += 1
        return set() if llamadas["n"] == 1 else {("V3", "Clara de huevo")}

    monkeypatch.setattr(rr, "_hallazgos", _falso)
    antes = ["Separa 3 huevos y 6 claras de huevo y 6 claras de huevo."]
    m = {"ingredients": ["3 huevos", "6 claras de huevo"], "recipe": list(antes)}
    r = rr.colapsar_repeticiones(m, index)
    assert r["aplicado"] == [] and r["descartado"] == ["Clara de huevo"] and m["recipe"] == antes


# ─────────────── el contrato entero: orden, sellos, agregado ───────────────

def test_el_contrato_corre_el_paso_6_y_sella_repeticiones_y_concordancia(index):
    import recipe_contract as rc
    src = inspect.getsource(rc.reconcile_meal)
    assert src.index("_retirar_sin_lista(meal, index)") < src.index("_colapsar_repeticiones(meal, index)"), "(6) va tras (5)"
    m = {"ingredients": ["½ ciruela", "3 huevos", "6 claras de huevo"],
         "recipe": ["Mise en place: separa 3 huevos y 6 claras de huevo y 6 claras de huevo; lava las 2 ciruelas y resérvalas enteras."]}
    n = rc._aplicar_meal(m, index, "repair")
    sello = m[rc.TELEMETRIA_KEY]
    assert n == 1 and sello["repeticiones"] == 1 and sello["concordancia"] == 1 and sello["familias"] == {"pieza": 1}
    assert m["recipe"] == ["Mise en place: separa 3 huevos y 6 claras de huevo; lava la ½ ciruela y resérvala entera."]
    # en sombra no toca y anota lo que habría hecho
    s = {"ingredients": ["½ ciruela"], "recipe": ["Lava las 2 ciruelas."]}
    assert rc._aplicar_meal(s, index, "shadow") == 0 and s["recipe"] == ["Lava las 2 ciruelas."] and s[rc.TELEMETRIA_KEY]["concordancia"] == 1
    # un plato sin nada que decir sigue sin sello
    q = {"ingredients": ["1 ciruela"], "recipe": ["Lava la ciruela."]}
    rc._aplicar_meal(q, index, "repair")
    assert rc.TELEMETRIA_KEY not in q
    agg = rc.reconcile_days([{"meals": [{"ingredients": ["½ ciruela"], "recipe": ["Lava las 2 ciruelas."]}]}], index)
    assert agg["concordancia"] == 1 and agg["reescritas"] == 1


# ─────────────── los planes reales: 7 → 0 sin abrir nada ───────────────

def test_los_planes_reales_quedan_sin_v7e_y_sin_hallazgos_nuevos(index):
    import culinary_coherence as cc
    from recipe_contract import reconcile_meal
    from recipe_repair import _hallazgos
    antes = despues = 0
    for pg in _real()["planes_generados"]:
        for d in pg["plan_data"]["days"]:
            for meal in d["meals"]:
                m = copy.deepcopy(meal)
                dia = {"day": d.get("day", 1)}
                antes += len(cc._v7e_paso_pide_mas_piezas(dia, m, index))
                h0 = _hallazgos(m, index)
                reconcile_meal(m, index)
                despues += len(cc._v7e_paso_pide_mas_piezas(dia, m, index))
                assert _hallazgos(m, index) - h0 == set(), meal.get("name")
    assert antes >= 7 and despues == 0, (antes, despues)


def test_el_replay_pareado_de_los_planes_reales_baja_los_v7e_a_cero():
    real = _real()
    rep = json.loads((_BACKEND / "scripts" / "data" / "bench_superficies_replay_2026_09_13_l31.json").read_text(encoding="utf-8"))
    assert rep["modo"] == "real-replay" and rep["planes_de"] == "bench_superficies_real_2026_09_13.json" and rep["n_planes"] == 3
    for s, r in rep["superficies"].items():
        assert r["n_nuevos"] == 0, s
    v7e_real = sum(p["etapas"].get("salida", {}).get("V7e", 0) for p in real["superficies"]["insert"]["planes"])
    v7e_rep = sum(p["etapas"].get("salida", {}).get("V7e", 0) for p in rep["superficies"]["insert"]["planes"])
    assert v7e_real >= 9 and v7e_rep == 0, (v7e_real, v7e_rep)
    assert rep["superficies"]["insert"]["resueltos"] > real["superficies"]["insert"]["resueltos"]


# ─────────────── docs, plan, marcador ───────────────

def test_docs_plan_y_marcador():
    doc = (_BACKEND / "docs" / "culinary_coherence.md").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-31" in doc and "concordancia" in doc and "colapsar_repeticiones" in doc and "V7e" in doc
    plan = (_BACKEND / "docs" / "plan_pendientes_2026_09_11.md").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-31" in plan
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', app)
    assert m and int(m.group(1)) >= 31 and m.group(2) >= "2026-09-13"
    assert len((_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8").splitlines()) <= 53100
    for anchor, f in (("P1-PLAN-LOTE-31-CONCORDANCIA", "recipe_contract.py"), ("P1-PLAN-LOTE-31-REPETICION", "recipe_repair.py"),
                      ("P1-PLAN-LOTE-31-V7E-MAX", "culinary_coherence.py")):
        assert anchor in (_BACKEND / f).read_text(encoding="utf-8"), anchor
