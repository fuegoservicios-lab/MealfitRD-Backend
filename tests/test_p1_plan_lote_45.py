# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-45 · 2026-09-14] La prueba RD del dueño (plan 40535829): lo que la generación hizo mal y la máquina no vio.

El dueño generó un plan con su perfil dominicano (ganar músculo, 30 días, compra mensual, presupuesto bajo, tiempo de
cocina «Nada») y los tres días salieron del día determinista con recetas de la biblioteca. Cuatro defectos, cada uno con
su prueba aquí:

  1. El reparador del contrato EXTRAÍA las oraciones de cocción de las recetas narrativas a un único «El Toque de Fuego»
     final: 149 de las 193 recetas DO salían con 281 oraciones fuera de su orden. Ahora el pilar se rotula en su sitio
     (`recipe_order`) y `repair_stage_diff` cuenta el desorden.
  2. El pool del planificador rechazaba HIGH al día determinista («omitió tilapia…»), que el reintento vuelve a armar
     igual: 3 intentos. El día determinista queda fuera de ese conteo; su familia sale del blueprint.
  3. El día rehecho en el reintento no veía los reciclados: Mofongo con pollo los días 1 y 2.
  4. El tiempo de cocina del formulario no llegaba al selector: 10 de 12 platos por encima de 10 min.

Y la cebolla sofrita encima de un mangú dejó de ser «guiso».
"""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]

import deterministic_day as dd  # noqa: E402
import dish_registry as dr  # noqa: E402
import graph_orchestrator as go  # noqa: E402
import recipe_order as ro  # noqa: E402
import repair_stage_diff as rsd  # noqa: E402
from constants import strip_accents  # noqa: E402

_LIB = json.loads((_BACKEND / "data" / "registry" / "recipe_library_do_v1.json").read_text(encoding="utf-8"))["por_id"]
_REG = {t["template_id"]: t for t in json.loads(
    (_BACKEND / "data" / "registry" / "dish_registry_do_v1.json").read_text(encoding="utf-8"))["templates"]}
MOFONGO = "tpl_ce7b17d2082d"       # Mofongo de plátano verde al horno con pollo guisado (almuerzo D1 y D2 del plan)
MANGU = "tpl_1876d7cc7a09"         # Mangú de guineo verde con queso frito ligero al airfryer y arenque (cena D3)
AREPITAS = "tpl_614403a321f1"      # Arepitas de maíz con queso gouda (horneadas): el último paso hornea


def _meal_de(tid):
    return {"name": _REG[tid]["name"], "ingredients": ["225 g de muslo de pollo", "1 plátano verde"],
            "recipe": list(_LIB[tid]["pasos"]), "_recipe_source": "library", "_template_id": tid}


def _reparar(meal):
    issues = go._recipe_step_contract_issues(meal)
    return go._repair_recipe_contract(meal, issues) if issues else []


def _oraciones_de(pasos):
    return sum(len(ro._oraciones(p)) for p in pasos)


# ─────────────── 1. el pilar se rotula en su sitio ───────────────
def test_el_mofongo_se_rotula_en_su_sitio_y_conserva_cada_oracion_en_su_orden():
    meal = _meal_de(MOFONGO)
    assert _reparar(meal) == []
    rec = meal["recipe"]
    assert rec[0].startswith("Mise en place:")
    assert rec[1].startswith("El Toque de Fuego: Precalienta el horno a 200 °C")
    assert rec[-1].startswith("Montaje: Sirve el mofongo caliente")
    assert sum(1 for p in rec if p.startswith("Montaje")) == 1, "un segundo «sirve» genérico detrás del de la receta"
    pasos = _LIB[MOFONGO]["pasos"]
    assert ro.fuera_de_orden(rec, pasos) == {"emparejadas": _oraciones_de(pasos), "fuera": 0}


def test_con_el_knob_apagado_la_extraccion_desordena_y_la_medida_lo_ve(monkeypatch):
    monkeypatch.setattr(go, "RECIPE_TDF_IN_PLACE_ENABLED", False)
    meal = _meal_de(MOFONGO)
    _reparar(meal)
    rec = meal["recipe"]
    i_escurre = next(i for i, p in enumerate(rec) if p.startswith("Escúrrelos bien"))
    i_fuego = next(i for i, p in enumerate(rec) if p.startswith("El Toque de Fuego"))
    assert i_escurre < i_fuego, "el defecto del plan 40535829: escurrir antes de hervir"
    assert ro.fuera_de_orden(rec, _LIB[MOFONGO]["pasos"])["fuera"] >= 3


def test_las_193_recetas_de_la_biblioteca_salen_en_su_orden():
    desordenadas, montaje_en_sitio, fuera_de_orden_lint, con_residual = [], 0, 0, []
    for tid, r in _LIB.items():
        meal = {"name": (_REG.get(tid) or {}).get("name", tid), "ingredients": ["100 g de algo"],
                "recipe": list(r["pasos"]), "_recipe_source": "library", "_template_id": tid}
        residual = _reparar(meal)
        fuera_de_orden_lint += "prefijos fuera de orden" in residual
        if residual:
            con_residual.append((tid, residual))
        if ro.fuera_de_orden(meal["recipe"], r["pasos"])["fuera"]:
            desordenadas.append(tid)
        montaje_en_sitio += meal["recipe"][-1].startswith("Montaje:") and not meal["recipe"][-1].startswith("Montaje: Emplata «")
    assert len(_LIB) >= 190
    assert desordenadas == [], f"recetas de biblioteca con oraciones fuera de su orden tras el reparador: {desordenadas[:5]}"
    assert fuera_de_orden_lint == 0
    assert con_residual == [], f"recetas de biblioteca que el reparador deja con badge: {con_residual[:3]}"
    assert montaje_en_sitio >= 70, montaje_en_sitio      # 73 terminan sirviendo: esas no reciben un segundo «sirve»


def test_el_mise_de_plantilla_no_vuelve_cocinado_a_un_plato_frio():
    """Las 15 recetas sin fuego (batidas, casabe con aguacate) recibían el Mise «…ten todo listo antes de COCINAR», que
    `_meal_is_no_cook` lee como señal de fuego: el re-lint pedía un Toque de Fuego y el plato frío salía con badge."""
    frias = [tid for tid, r in _LIB.items()
             if not any(go._NOCOOK_COOK_SIGNAL_RE.search(strip_accents(p.lower())) for p in r["pasos"])
             and not go._NOCOOK_COOK_SIGNAL_RE.search(strip_accents(str((_REG.get(tid) or {}).get("name", "")).lower()))]
    assert len(frias) >= 10, len(frias)
    for tid in frias:
        meal = _meal_de(tid)
        assert _reparar(meal) == [], tid
        assert not any("toque de fuego" in p.lower() for p in meal["recipe"]), f"fuego inventado en {tid}"
        assert go._meal_is_no_cook(meal), tid


def test_el_ultimo_paso_que_cocina_recibe_la_plantilla_de_montaje():
    meal = _meal_de(AREPITAS)
    _reparar(meal)
    assert meal["recipe"][-1].startswith("Montaje: Emplata «")
    assert ro.fuera_de_orden(meal["recipe"], _LIB[AREPITAS]["pasos"])["fuera"] == 0


def test_montaje_en_su_sitio_salta_notas_pero_no_parches_de_accion():
    rec = ["Pica la cebolla.", "Sofríe la cebolla 5 min.", "Sirve caliente.", "💪 Cocina camarones a la plancha y sírvelos."]
    assert ro.rotular_montaje_en_su_sitio(rec) is None and rec[2] == "Sirve caliente."
    rec = ["Pica la cebolla.", "Sirve caliente.", "🌱 Nota del Nutricionista AI: acompaña con zanahoria rallada."]
    assert ro.rotular_montaje_en_su_sitio(rec) == 1 and rec[1] == "Montaje: Sirve caliente."


def test_el_fuego_prefiere_el_primer_paso_con_tiempo_y_no_toca_parches():
    es_c = lambda s: bool(go._NOCOOK_COOK_SIGNAL_RE.search(strip_accents(s.lower())))
    t_ok = lambda s: bool(go._CONTRACT_TIME_RE.search(s))
    rec = ["Sazona el pollo mientras preparas el sofrito.", "Sofríe la cebolla 4-5 minutos.", "💪 Cocina el huevo 8 min."]
    assert ro.rotular_fuego_en_su_sitio(rec, es_c, t_ok) == 1
    assert rec[1].startswith("El Toque de Fuego: Sofríe") and rec[2].startswith("💪")
    assert ro.rotular_fuego_en_su_sitio(["💪 Cocina camarones 5 min."], es_c, t_ok) is None


def test_el_medidor_de_la_cadena_cuenta_el_desorden(caplog, monkeypatch):
    cat = [{"name": "Muslo de pollo", "aliases": ["pollo"], "category": "Proteínas", "prep_methods": ["guisado"]},
           {"name": "Plátano verde", "aliases": ["platano verde"], "category": "Víveres", "prep_methods": ["hervido"]}]
    plan = {"days": [{"day": 1, "meals": [dict(_meal_de(MOFONGO), meal="Almuerzo")]}]}
    ctx = rsd.start(plan, surface="t", catalog=cat)
    assert ctx is not None
    monkeypatch.setattr(go, "RECIPE_TDF_IN_PLACE_ENABLED", False)
    _reparar(plan["days"][0]["meals"][0])
    with caplog.at_level("WARNING"):
        inf = rsd.finish(ctx, plan)
    assert inf["orden"]["etapas"]["entrada"] == 0 and inf["orden"]["etapas"]["salida"] >= 3
    assert inf["orden"]["medidas"] == 1 and inf["orden"]["desordenadas"] == 1
    assert inf["orden"]["detalle"][0]["template_id"] == MOFONGO
    assert any("desordenó" in r.message and "P1-PLAN-LOTE-45" in r.message for r in caplog.records)
    assert ro.medir_plan(plan)["desordenadas"] == 1


# ─────────────── 2 y 3. el día determinista: fidelidad y memoria ───────────────
def _validar(dias):
    res = {"days": dias}
    sk = {"days": [{"day": d["day"], "protein_pool": ["tilapia", "yogurt griego natural", "mantequilla de maní"]} for d in dias]}
    go._run_assembly_validations(res, sk, set())
    return res.get("_skeleton_fidelity_errors") or []


def _dia(n, determinista):
    d = {"day": n, "meals": [{"name": "Mofongo", "ingredients": ["200 g de muslo de pollo"], "recipe": ["Hierve 10 min."]}]}
    if determinista:
        d["_day_source"] = "deterministic"
    return d


def test_el_pool_del_planificador_no_juzga_al_dia_determinista(monkeypatch):
    errs = _validar([_dia(1, True), _dia(2, False)])
    assert len(errs) == 1 and errs[0].startswith("Día 2 omitió"), "el día del modelo SÍ sigue juzgado"
    monkeypatch.setattr(go, "SKELETON_FIDELITY_SKIP_DETERMINISTIC", False)
    assert len(_validar([_dia(1, True), _dia(2, False)])) == 2


def test_la_memoria_del_reintento_ve_los_dias_reciclados():
    src = inspect.getsource(go.generate_days_parallel_node)
    i_mem = src.index("_det_prev: list = []")
    i_seed = src.index("_det_prev.extend(recycled_days_cache[_k] for _k in sorted(recycled_days_cache))")
    i_gen = src.index("async def _safe_gen(")
    assert i_mem < i_seed < i_gen, "sembrar la memoria antes de lanzar los días"
    assert "DETERMINISTIC_MEMORY_SEES_RECYCLED and surgical_mode and recycled_days_cache" in src
    # y lo sembrado cuenta: la plantilla del día reciclado entra en la ventana de repetición
    assert dd._conteo_ventana([{"day": 1, "meals": [{"_template_id": MOFONGO}]}], 0) == {MOFONGO: 1}


def test_la_familia_del_dia_sale_del_blueprint(monkeypatch):
    sk = {"protein_pool": ["tilapia", "Muslo de pollo", "mantequilla de maní"]}
    sl = {"days": [{"day_index": 0, "protein": "Pollo"}, {"day_index": 1, "protein": "Pescado"}]}
    assert dd._familias_para(sk, sl, 1) == ["Pescado"]
    assert dd._familias_para({"protein": "Res", "protein_pool": ["tilapia"]}, sl, 1) == ["Res"]
    monkeypatch.setenv("MEALFIT_DETERMINISTIC_DAY_BLUEPRINT_FAMILY", "false")
    assert dd._familias_para(sk, sl, 1) == ["pescado", "Muslo de pollo", "mantequilla de maní"]
    monkeypatch.setenv("MEALFIT_DETERMINISTIC_DAY_POOL_FAMILY_CANON", "false")
    assert dd._familias_para(sk, sl, 1) == ["tilapia", "Muslo de pollo", "mantequilla de maní"]


def test_un_alimento_del_pool_se_lleva_a_su_familia_solo_si_no_se_entiende():
    assert dd._familia_canonica("tilapia") == "pescado"
    assert dd._familia_canonica("Merluza") == "pescado"
    for tal_cual in ("Atún en agua", "Pechuga de pollo", "Lentejas", "Pescado", "mantequilla de maní"):
        assert dd._familia_canonica(tal_cual) == tal_cual
    # medido sobre el registro DO: «tilapia» como familia sólo alcanzaba la plantilla sin proteína
    assert len(dr.template_candidates("DO", "almuerzo", "tilapia", k=200)) <= 1
    assert len(dr.template_candidates("DO", "almuerzo", "pescado", k=200)) >= 10


def test_los_fijados_con_la_franja_del_motor_quedan_apagados_por_medicion(monkeypatch):
    fij = {"1:lunch": ["tpl_a"], "1:snack": ["tpl_b"]}
    assert dd._fijados_para(fij, 1, "almuerzo") is None
    assert dd._fijados_para({"1:almuerzo": ["tpl_c"]}, 1, "almuerzo") == ["tpl_c"]
    monkeypatch.setenv("MEALFIT_DETERMINISTIC_DAY_PINNED_SLOT_ALIAS", "true")
    assert dd._fijados_para(fij, 1, "almuerzo") == ["tpl_a"] and dd._fijados_para(fij, 1, "merienda") == ["tpl_b"]
    assert dd._fijados_para(fij, 2, "almuerzo") is None


# ─────────────── 4. el tiempo de cocina ───────────────
def _plantilla(tid, minutos, fuente="receta"):
    return {"template_id": tid, "logistics": {"prep_minutes_est": minutos, "prep_minutes_source": fuente}}


def test_el_tiempo_de_cocina_ordena_por_tramos(monkeypatch):
    por_id = {"lento": _plantilla("lento", 45), "rapido": _plantilla("rapido", 10), "medio": _plantilla("medio", 20)}
    llamadas = []

    def _falso(tids, objetivo, catalogo, por_id_, slot="", **kw):
        llamadas.append(list(tids))
        return [(por_id_[t], 1.0) for t in tids]

    monkeypatch.setattr(dd, "elegir_plantillas", _falso)
    orden = lambda r: [p[0]["template_id"] for p in r]
    tids = ["lento", "rapido", "medio"]
    assert orden(dd.elegir_con_tiempo(tids, {}, {}, por_id, "almuerzo", 10)) == ["rapido", "medio", "lento"]
    assert orden(dd.elegir_con_tiempo(tids, {}, {}, por_id, "almuerzo", 10, saturados={"rapido": 2}, max_rep=2)) \
        == ["medio", "lento", "rapido"], "la cuota de repetición del usuario va antes que el tiempo"
    llamadas.clear()
    assert orden(dd.elegir_con_tiempo(tids, {}, {}, por_id, "almuerzo", None)) == tids and llamadas == [tids]
    assert orden(dd.elegir_con_tiempo(tids, {}, {}, por_id, "almuerzo", 30)) == ["rapido", "medio", "lento"]


def test_el_presupuesto_sale_del_formulario_y_el_relleno_no_cuenta(monkeypatch):
    assert dd._presupuesto_minutos({"cookingTime": "none"}) == 10
    assert dd._presupuesto_minutos({"health_profile": {"cookingTime": "30min"}}) == 30
    assert dd._presupuesto_minutos({"cookingTime": "plenty"}) is None and dd._presupuesto_minutos({}) is None
    assert dd._minutos_de(_plantilla("x", 15, "defecto")) is None and dd._minutos_de(_plantilla("x", 15)) == 15
    monkeypatch.setenv("MEALFIT_DETERMINISTIC_DAY_COOKING_TIME", "false")
    assert dd._presupuesto_minutos({"cookingTime": "none"}) is None


def test_el_armador_usa_familia_fijados_y_tiempo():
    src = inspect.getsource(dd.build_day_for_skeleton)
    for ancla in ("_familias_para(skeleton_day, _sl, day_index)", "_fijados_para(_fijados, day_index, slot)",
                  "elegir_con_tiempo(tids, obj, catalogo, por_id, slot, _presup_min", "_presupuesto_minutos(_fd)"):
        assert ancla in src, ancla


# ─────────────── la cebolla sofrita no es guiso ───────────────
def test_la_cebolla_sofrita_encima_de_un_mangu_no_es_guiso():
    mangu = {"name": _REG[MANGU]["name"], "recipe": list(_LIB[MANGU]["pasos"])}
    assert go._meal_is_stewy(mangu, strip_accents) is False
    assert "al guiso" not in go._closer_protein_step_text("arenque", False, stewy=go._meal_is_stewy(mangu, strip_accents))
    assert go._meal_is_stewy({"name": "Habichuelas con sofrito criollo", "recipe": ["Prepara el sofrito."]}, strip_accents)


# ─────────────── knobs, docs, marcador ───────────────
def test_knobs_docs_y_marcador():
    go_src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    dd_src = (_BACKEND / "deterministic_day.py").read_text(encoding="utf-8")
    for knob, src in (('_env_bool("MEALFIT_RECIPE_TDF_IN_PLACE", True)', go_src),
                      ('_env_bool("MEALFIT_SKELETON_FIDELITY_SKIP_DETERMINISTIC", True)', go_src),
                      ('_env_bool("MEALFIT_DETERMINISTIC_MEMORY_SEES_RECYCLED", True)', go_src),
                      ('_knob_on("MEALFIT_DETERMINISTIC_DAY_BLUEPRINT_FAMILY")', dd_src),
                      ('_knob_on("MEALFIT_DETERMINISTIC_DAY_POOL_FAMILY_CANON")', dd_src),
                      ('_knob_on("MEALFIT_DETERMINISTIC_DAY_PINNED_SLOT_ALIAS", False)', dd_src),
                      ('_knob_on("MEALFIT_DETERMINISTIC_DAY_COOKING_TIME")', dd_src)):
        assert knob in src, knob
    knobs_doc = (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    for k in ("MEALFIT_RECIPE_TDF_IN_PLACE", "MEALFIT_SKELETON_FIDELITY_SKIP_DETERMINISTIC",
              "MEALFIT_DETERMINISTIC_MEMORY_SEES_RECYCLED", "MEALFIT_DETERMINISTIC_DAY_BLUEPRINT_FAMILY",
              "MEALFIT_DETERMINISTIC_DAY_POOL_FAMILY_CANON", "MEALFIT_DETERMINISTIC_DAY_PINNED_SLOT_ALIAS",
              "MEALFIT_DETERMINISTIC_DAY_COOKING_TIME"):
        assert k in knobs_doc, k
    for doc in ("knobs_reference.md", "plan_pendientes_2026_09_11.md", "deterministic_day.md", "culinary_coherence.md",
                "plan_agente_lotes_38_43_2026_09_14.md"):
        assert "P1-PLAN-LOTE-45" in (_BACKEND / "docs" / doc).read_text(encoding="utf-8"), doc
    for f, ancla in (("recipe_order.py", "P1-PLAN-LOTE-45-ORDEN"), ("graph_orchestrator.py", "P1-PLAN-LOTE-45-TDF-EN-SU-SITIO"),
                     ("graph_orchestrator.py", "P1-PLAN-LOTE-45-FIDELIDAD-DETERMINISTA"),
                     ("graph_orchestrator.py", "P1-PLAN-LOTE-45-MEMORIA-RECICLADOS"),
                     ("deterministic_day.py", "P1-PLAN-LOTE-45-FAMILIA-DEL-BLUEPRINT"),
                     ("deterministic_day.py", "P1-PLAN-LOTE-45-TIEMPO-DE-COCINA"), ("repair_stage_diff.py", "P1-PLAN-LOTE-45")):
        assert ancla in (_BACKEND / f).read_text(encoding="utf-8"), (f, ancla)
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', app)
    assert m and int(m.group(1)) >= 45
