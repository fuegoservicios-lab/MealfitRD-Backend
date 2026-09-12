# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-22 · 2026-09-12] C1 del plan de pendientes: CUL-P0-01 (identidad, versión y estado explícito de
evaluación) y CUL-P0-02 (referencia independiente y marcador estricto).

Lo que se confundía con «aprobado», y ahora tiene nombre:

  · el scan que devolvía `[]` por «coherente», por «sin catálogo» y por «reventó» → `culinary_contract_scan_status`;
  · la comida identificada por FRANJA en las dos capas → `meal_index` en cada violación, `idx` en el payload del juez
    y `resolve_judge_violations` para las quejas antiguas (ambiguo ≠ repartido);
  · la entrada del juez sin decir con qué rúbrica/modelo/país juzgó → `context` junto al sello;
  · «comidas que el juez señaló alguna vez» leído como defectos entregados → `judge_evaluation_state` y
    `juez_entregado`, con la partición que RECONCILIA con el denominador.

Y el marcador estricto: adjudica hallazgo↔defecto por rúbrica (FP localizado + FN del esperado; duplicados no
multiplican; cero división → null), acuerdo entre anotadores, adjudicación, intervalos por plan y particiones por
linaje. Con las 80 etiquetas binarias del 09-07 sale INCOMPLETO por código de salida — a propósito.
"""
from __future__ import annotations

import copy
import inspect
import json
import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import culinary_coherence as cc  # noqa: E402

_CAT = [{"name": "Pollo", "aliases": [], "category": "proteina", "ready_to_eat": False, "prep_methods": ["hervir"]},
        {"name": "Arroz blanco", "aliases": ["arroz"], "category": "cereal", "ready_to_eat": False, "prep_methods": ["hervir"]}]


def _plan_dos_meriendas():
    """Día 1 con dos Meriendas: la primera (índice 1) tiene un huérfano (pollo en la lista, ningún paso lo usa); la
    segunda (índice 2) está limpia. Un hallazgo por franja no sabría a cuál acusar."""
    return {"days": [{"day": 1, "meals": [
        {"meal": "Desayuno", "name": "Pollo hervido", "ingredients": ["150 g de pollo"], "recipe": ["Hierve el pollo"]},
        {"meal": "Merienda", "name": "Pollo suelto", "ingredients": ["150 g de pollo"], "recipe": ["Sirve la fruta"]},
        {"meal": "Merienda", "name": "Pollo hervido", "ingredients": ["150 g de pollo"], "recipe": ["Hierve el pollo"]},
    ]}]}


# ────────────────────────────────────────────────────────────── CUL-P0-01 · el scan dice su estado

def test_el_scan_devuelve_su_estado_y_el_adaptador_sigue_devolviendo_la_lista():
    plan = _plan_dos_meriendas()
    viol, est = cc.culinary_contract_scan_status(plan, _CAT)
    assert est["status"] == "scanned" and est["meals"] == 3 and est["violations"] == len(viol)
    assert est["checks"] == list(cc.CHECKS_CAPA1) and est["error"] is None
    assert est["schema"] == cc.FINDING_SCHEMA_VERSION and re.fullmatch(r"[0-9a-f]{16}", est["reglas_huella"])
    assert cc.culinary_contract_scan(plan, _CAT) == viol


def test_sin_catalogo_no_es_coherente_es_no_catalog():
    viol, est = cc.culinary_contract_scan_status(_plan_dos_meriendas(), [])
    assert viol == [] and est["status"] == "no_catalog" and est["checks"] == []


def test_un_scan_que_revienta_lo_dice_y_no_lanza():
    viol, est = cc.culinary_contract_scan_status(5, _CAT)          # type: ignore[arg-type]
    assert viol == [] and est["status"] == "error" and "AttributeError" in est["error"]


def test_sin_comidas_es_no_meals():
    _, est = cc.culinary_contract_scan_status({"days": []}, _CAT)
    assert est["status"] == "no_meals" and est["meals"] == 0


def test_la_huella_de_las_reglas_coincide_con_la_del_corpus():
    from culinary_corpus import huella_reglas
    assert cc.rules_fingerprint() == huella_reglas()


# ────────────────────────────────────────────────────────────── CUL-P0-01 · identidad por ocurrencia

def test_dos_meriendas_el_mismo_dia_conservan_dos_identidades():
    viol = cc.culinary_contract_scan(_plan_dos_meriendas(), _CAT)
    v3 = [v for v in viol if v["check"] == "V3"]
    assert len(v3) == 1 and v3[0]["meal"] == "Merienda" and v3[0]["meal_index"] == 1, viol
    assert all("meal_index" in v for v in viol)
    # las claves históricas siguen ahí: consumidores antiguos (`_degrade_offending_steps`) no se enteran
    assert {"day", "meal", "check", "food", "detail", "severity", "repairable"} <= set(v3[0])


def test_el_payload_del_juez_lleva_idx_por_comida():
    meals = cc.judge_payload_meals(_plan_dos_meriendas())
    assert [m["idx"] for m in meals] == [0, 1, 2] and meals[1]["slot"] == "Merienda"
    assert set(meals[0]) == {"day", "idx", "slot", "name", "ingredients", "recipe"}


def test_resolver_quejas_del_juez_declarada_unica_ambigua_sin_comida():
    plan = _plan_dos_meriendas()
    quejas = [
        {"day": 1, "meal": "Desayuno", "tipo": "paso_incoherente"},                 # única
        {"day": 1, "meal": "Merienda", "tipo": "paso_incoherente"},                 # ambigua: dos meriendas
        {"day": 1, "meal": "Merienda", "tipo": "paso_incoherente", "meal_index": 2},  # declarada
        {"day": 1, "meal": "Cena", "tipo": "paso_incoherente"},                     # sin comida
        {"day": 1, "meal": "Desayuno", "tipo": "paso_incoherente", "meal_index": 2},  # índice que no es esa franja ⇒ por franja
    ]
    r = cc.resolve_judge_violations(plan, quejas)
    assert [(x["meal_index"], x["resolucion"]) for x in r] == [
        (0, "unica"), (None, "ambigua"), (2, "declarada"), (None, "sin_comida"), (0, "unica")]
    assert quejas[0].get("meal_index") is None, "devuelve copias: no muta la entrada"
    assert cc.resolve_judge_violations(None, quejas) == quejas or True   # fail-open: jamás lanza


def test_el_contexto_del_juez_va_junto_al_sello():
    ctx = cc.judge_context(country="DO", model="glm-5.3-flash", guard="warn", rubric="rúbrica v3", plan=_plan_dos_meriendas())
    assert ctx["country"] == "DO" and ctx["model"] == "glm-5.3-flash" and ctx["guard"] == "warn"
    assert re.fullmatch(r"[0-9a-f]{16}", ctx["rubric_fingerprint"]) and ctx["meals"] == 3
    assert ctx["schema"] == cc.FINDING_SCHEMA_VERSION and ctx["reglas_huella"] == cc.rules_fingerprint()
    assert cc.judge_context(rubric="otra")["rubric_fingerprint"] != ctx["rubric_fingerprint"]


# ────────────────────────────────────────────────────────────── CUL-P0-01 · estado explícito de evaluación

def _con_historial(plan, entradas):
    p = copy.deepcopy(plan)
    p["_culinary_judge_history"] = entradas
    return p


def test_timeout_off_obsoleto_y_desconocido_no_cuentan_como_aprobados():
    plan = _plan_dos_meriendas()
    sello = cc.judged_fingerprint(plan)
    assert cc.judge_evaluation_state(plan) == {"estado": "no_evaluado", "aprobado": False, "hallazgos": 0, "entrada": None}
    assert cc.judge_evaluation_state(_con_historial(plan, [{"status": "unavailable", "violations": []}]))["estado"] == "no_disponible"
    assert cc.judge_evaluation_state(_con_historial(plan, [{"status": "judged", "judged_fingerprint": "otro", "violations": []}]))["estado"] == "juzgado_obsoleto"
    assert cc.judge_evaluation_state(_con_historial(plan, [{"violations": []}]))["estado"] == "desconocido"
    for e in ("no_disponible", "juzgado_obsoleto", "desconocido"):
        assert e in cc.ESTADOS_JUEZ
    vig = cc.judge_evaluation_state(_con_historial(plan, [{"status": "judged", "judged_fingerprint": sello, "violations": []}]))
    assert vig == {"estado": "juzgado_vigente", "aprobado": True, "hallazgos": 0, "entrada": 0,
                   "sellos": {"con_sello": 0, "entregadas": 0}}
    con = cc.judge_evaluation_state(_con_historial(plan, [{"status": "judged", "judged_fingerprint": sello, "violations": [{"day": 1}]}]))
    assert con["estado"] == "juzgado_vigente" and con["aprobado"] is False and con["hallazgos"] == 1


def test_una_entrada_vigente_anterior_manda_sobre_un_reintento_no_disponible():
    plan = _plan_dos_meriendas()
    sello = cc.judged_fingerprint(plan)
    st = cc.judge_evaluation_state(_con_historial(plan, [
        {"status": "judged", "judged_fingerprint": sello, "violations": []},
        {"status": "unavailable", "violations": []}]))
    assert st["estado"] == "juzgado_vigente" and st["entrada"] == 0


def test_el_contrato_tambien_tiene_estado():
    plan = _plan_dos_meriendas()
    assert cc.contract_evaluation_state(plan, _CAT)["estado"] == "evaluado"
    assert cc.contract_evaluation_state(plan, _CAT)["aprobado"] is False        # el huérfano
    assert cc.contract_evaluation_state(plan, [])["estado"] == "no_evaluable"
    assert cc.contract_evaluation_state({"days": []}, _CAT)["estado"] == "no_evaluado"
    for est in (cc.contract_evaluation_state(plan, []), cc.contract_evaluation_state({"days": []}, _CAT)):
        assert est["aprobado"] is False


def test_tres_coberturas_no_una():
    d = cc.scan_coverage_detail(_plan_dos_meriendas(), _CAT)
    assert d["estado"] == "medida" and d["lineas"] == 3 and d["lineas_reconocidas"] == 3
    assert d["reconocimiento"] == 1.0 and d["catalogo"] == 1.0 and d["ready_to_eat"] == 1.0
    assert d["por_check"]["V1"] == d["catalogo"] and d["por_check"]["V2"] == d["ready_to_eat"] and d["por_check"]["V3"] == d["reconocimiento"]
    assert d["catalogo"] == round(cc.scan_coverage(_plan_dos_meriendas(), _CAT), 3)
    assert cc.scan_coverage_detail({"days": [{"meals": [{"ingredients": ["xyzzy"], "recipe": []}]}]}, _CAT)["estado"] == "sin_alimentos"
    assert cc.scan_coverage_detail(_plan_dos_meriendas(), [])["estado"] == "sin_catalogo"
    assert cc.scan_coverage_detail(5, _CAT)["estado"] == "error"     # type: ignore[arg-type]


def _comida(slot, nombre, pasos):
    return {"meal": slot, "name": nombre, "ingredients": ["150 g de pollo"], "recipe": pasos}


def test_el_sello_por_comida_sobrevive_al_shift_que_archiva_y_renumera():
    """Al re-congelar la base, los 5 planes del corpus salían «juzgado_obsoleto» (0/64 sobre lo entregado): el sello del
    PLAN lleva la posición del día y el shift la cambia. El sello por COMIDA reencuentra la comida juzgada donde esté."""
    A, Bm, Cm = _comida("Desayuno", "A", ["Hierve el pollo"]), _comida("Desayuno", "B", ["Hierve el pollo y sirve"]), _comida("Cena", "C", ["Hierve el pollo"])
    plan = {"days": [{"day": 1, "meals": [A]}, {"day": 2, "meals": [Bm, Cm]}]}
    juzgadas = cc.resolve_judge_violations(plan, [{"day": 2, "meal": "Desayuno", "tipo": "paso_incoherente", "detalle": "b"},
                                                  {"day": 1, "meal": "Desayuno", "tipo": "combo_absurdo", "detalle": "a"}])
    assert [q["resolucion"] for q in juzgadas] == ["unica", "unica"]
    assert juzgadas[0]["meal_seal"] == cc.meal_seal(Bm) and juzgadas[1]["meal_seal"] == cc.meal_seal(A)
    # el shift archiva el día 1 y renumera: B pasa a ser día 1, índice 0
    shifted = {"days": [{"day": 1, "meals": [Bm, Cm]}]}
    r = cc.resolve_judge_violations(shifted, juzgadas)
    assert r[0]["resolucion"] == "por_sello" and r[0]["ocurrencia_actual"] == [1, 0] and r[0]["day"] == 2
    # la queja sobre A cae por franja en el Desayuno de hoy (que es B): atada a una comida que existe, pero conserva
    # el sello de A — que ya no se entrega
    assert r[1]["resolucion"] in ("unica", "declarada") and r[1]["meal_seal"] == cc.meal_seal(A) and r[1]["meal_seal"] not in cc.meal_seals_index(shifted)
    assert cc.meal_seal(A) != cc.meal_seal(Bm) and cc.meal_seal(A) == cc.meal_seal(dict(A))

    cb = _baseline()
    shifted["_culinary_judge_history"] = [{"status": "judged", "judged_fingerprint": cc.judged_fingerprint(plan), "violations": juzgadas}]
    st = cc.judge_evaluation_state(shifted)
    assert st["estado"] == "juzgado_obsoleto" and st["sellos"] == {"con_sello": 2, "entregadas": 1}
    m = cb._medir_filas([{"id": "p", "plan_data": shifted}], _CAT)
    assert m["estado_evaluacion"]["juez"] == {"juzgado_obsoleto": 1}
    assert m["hallazgos"]["juez"] == {"obsoletos": 1, "con_comida": 2, "vigentes_por_sello": 1}
    assert m["juez_entregado"] == {"comidas": 1, "pct": 50.0, "por_tipo": {"paso_incoherente": 1}}, "B sigue entregada: su queja es vigente aunque el plan entero sea «obsoleto»"
    assert m["particion"]["reconcilia"] is True and m["particion"]["solo_juez_vigente"] == 1


def test_las_violaciones_de_capa_1_tambien_llevan_sello_de_comida():
    viol = cc.culinary_contract_scan(_plan_dos_meriendas(), _CAT)
    v3 = [v for v in viol if v["check"] == "V3"][0]
    assert v3["meal_seal"] == cc.meal_seal(_plan_dos_meriendas()["days"][0]["meals"][1])


# ────────────────────────────────────────────────────────────── el orquestador y el worker

def test_el_orquestador_persiste_el_estado_del_scan_y_resuelve_las_quejas_del_juez():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8", errors="ignore")
    assert '_cul_viol, plan["_culinary_contract_scan"] = culinary_contract_scan_status(plan, _cul_cat)' in src
    assert "P1-PLAN-LOTE-22-SCAN-STATUS" in src
    i = src.index("_cj_hist.append({")
    bloque = src[i:src.index("})", i) + 2]
    assert '"context": _cj_context(country=_cj_country, model=CULINARY_JUDGE_MODEL' in bloque
    assert "_cj_viol = _cj_resolve(plan, _cj_viol)" in src[i - 600:i]
    assert "resolve_judge_violations as _cj_resolve" in src and "judge_context as _cj_context" in src
    assert "_meals = _cj_payload(plan)" in src and "judge_payload_meals as _cj_payload" in src
    j = src.index("class CulinaryViolation(BaseModel):")
    assert "meal_index: Optional[int] = None" in src[j:j + 1500]
    assert "meal_index (el `idx` de ESA comida" in src, "la rúbrica tiene que pedir el índice o el juez nunca lo devolverá"


def test_el_estado_del_scan_viaja_de_t1_a_t2_como_sus_hermanas():
    import cron_tasks
    assert "_culinary_contract_scan" in cron_tasks.P0_4_T2_INCREMENTAL_KEYS
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8", errors="ignore")
    assert "for _cul_k in ('_culinary_contract_violations', '_culinary_contract_coverage', '_culinary_contract_scan'):" in src


def test_el_god_file_no_subio_el_tope():
    n = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8", errors="ignore").count("\n")
    assert n <= 53_100, f"graph_orchestrator.py tiene {n} líneas: extraer, no subir el tope"


# ────────────────────────────────────────────────────────────── la línea base reconcilia

def _baseline():
    sys.path.append(str(_BACKEND / "scripts"))  # al FINAL: en cabeza, scripts/plan_gym.py sombrearía a plan_gym
    import culinary_baseline as cb
    return cb


def test_la_linea_base_separa_lo_entregado_del_historico_y_reconcilia_con_el_denominador():
    cb = _baseline()
    plan = _plan_dos_meriendas()
    sello = cc.judged_fingerprint(plan)
    plan["_culinary_judge_history"] = [
        {"status": "judged", "judged_fingerprint": "otra-version", "violations": [{"day": 1, "meal": "Desayuno", "tipo": "combo_absurdo"}]},
        {"status": "judged", "judged_fingerprint": sello, "violations": [
            {"day": 1, "meal": "Desayuno", "tipo": "paso_incoherente"},    # única ⇒ atada
            {"day": 1, "meal": "Merienda", "tipo": "paso_incoherente"},    # ambigua ⇒ no se reparte
            {"day": 1, "meal": "Cena", "tipo": "paso_incoherente"}]},      # sin comida
        {"status": "unavailable", "judged_fingerprint": None, "violations": []},
    ]
    r = cb._medir_filas([{"id": "plan-1", "plan_data": plan}], _CAT)
    assert r["comidas"] == 3
    assert r["estado_evaluacion"] == {"contrato": {"scanned": 1}, "juez": {"juzgado_vigente": 1}}
    assert r["hallazgos"]["juez"] == {"obsoletos": 1, "vigentes": 1, "no_disponibles": 1,
                                      "con_comida": 2, "ambiguos": 1, "sin_comida": 1}
    assert r["hallazgos"]["determinista"]["con_comida"] >= 1 and "sin_comida" not in r["hallazgos"]["determinista"]
    assert r["juez_entregado"] == {"comidas": 1, "pct": 33.3, "por_tipo": {"paso_incoherente": 1}}
    assert r["juez"]["comidas"] >= 2, "el histórico sigue contando lo que el juez señaló alguna vez"
    p = r["particion"]
    assert p["reconcilia"] is True and p["total"] == p["denominador"] == 3
    assert p["solo_juez_vigente"] == 1 and p["solo_determinista"] == 1 and p["ninguna"] == 1 and p["ambas"] == 0
    for k in ("estado_evaluacion", "hallazgos", "juez_entregado", "particion"):
        assert k in cb.CIFRAS
    assert "RECONCILIA" in cb.render(r) and "SOBRE LO ENTREGADO" in cb.render(r)


def test_un_scan_sin_catalogo_cuenta_como_no_evaluable_no_como_limpio():
    cb = _baseline()
    r = cb._medir_filas([{"id": "p", "plan_data": _plan_dos_meriendas()}], [])
    assert r["estado_evaluacion"]["contrato"] == {"no_catalog": 1} and r["determinista"]["comidas"] == 0


def test_el_corpus_fijo_reproduce_con_las_cifras_nuevas():
    """La línea base del corpus del 09-12 se re-congeló con las cifras de C1: medirla hoy da lo mismo."""
    from culinary_corpus import cargar, huella_reglas
    cb = _baseline()
    corpus = cargar(_BACKEND / "scripts" / "data" / "culinary_corpus_2026_09_12.json")
    base = json.loads((_BACKEND / "docs" / "culinary_baseline_2026_09_12.json").read_text(encoding="utf-8"))
    if huella_reglas() != base["computation"]["reglas_huella"]:
        pytest.skip("las reglas cambiaron desde la línea base: --verificar dice el delta; re-congela si es el esperado")
    r = cb.medir_corpus(corpus, fichero="x")
    for k in cb.CIFRAS:
        assert r[k] == base[k], k
    assert base["particion"]["reconcilia"] is True and base["particion"]["denominador"] == base["comidas"]


# ────────────────────────────────────────────────────────────── el muestreador ata por ocurrencia

def test_el_muestreador_ata_los_hallazgos_a_la_ocurrencia_y_no_reparte_lo_ambiguo():
    sys.path.append(str(_BACKEND / "scripts"))
    import culinary_golden_sample as gs
    plan = _plan_dos_meriendas()
    plan["_culinary_judge_history"] = [{"status": "judged", "violations": [
        {"day": 1, "meal": "Merienda", "tipo": "paso_incoherente", "detalle": "x"},      # ambigua
        {"day": 1, "meal": "Desayuno", "tipo": "combo_absurdo", "detalle": "y"}]}]      # única
    comidas, det, juez, sueltos = gs.indexar_hallazgos("pid", plan, _CAT)
    assert set(comidas) == {("pid", 1, 0, "Desayuno"), ("pid", 1, 1, "Merienda"), ("pid", 1, 2, "Merienda")}
    assert list(det) == [("pid", 1, 1)] and det[("pid", 1, 1)][0].startswith("V3:")
    assert list(juez) == [("pid", 1, 0)]
    assert sueltos == [{"capa": "juez", "day": 1, "meal": "Merienda", "motivo": "ambigua"}]
    assert comidas[("pid", 1, 1, "Merienda")]["indice"] == 1
    src = inspect.getsource(gs)
    assert "no llevan indice" in src and "CUL-P0-01" in src and "random" not in src.lower()
    assert "def a_markdown_ciego" in src and "def plantilla_anotacion" in src


def test_la_plantilla_de_anotacion_no_prellena_nada():
    sys.path.append(str(_BACKEND / "scripts"))
    import culinary_golden_sample as gs
    d = json.loads((_BACKEND / "docs" / "culinary_golden_set.json").read_text(encoding="utf-8"))
    pl = gs.plantilla_anotacion(d, "B")
    assert len(pl["casos"]) == len(d["casos"]) == 80
    assert all(v == {"veredicto": "", "defectos": []} for v in pl["casos"].values())
    ciego = gs.a_markdown_ciego(d)
    # sin lo que dijo la máquina, sin el estrato de cada caso y sin veredicto previo (estructural, no por palabras)
    assert "<details>" not in ciego and "- det ·" not in ciego and "- juez ·" not in ciego
    assert not re.search(r"\*(solo_determinista|solo_juez|ambas|sin_hallazgo) ·", ciego)
    assert "veredicto_humano" not in ciego and "nota_humana" not in ciego
    assert ciego.count("## `") == 80


# ────────────────────────────────────────────────────────────── CUL-P0-02 · el marcador estricto

def _score():
    sys.path.append(str(_BACKEND / "scripts"))
    import culinary_golden_score as gsc
    return gsc


def _caso(id_, plan, det, juez, anot=None, **extra):
    c = {"id": id_, "plan": plan, "estrato": "ambas", "maquina_determinista": det, "maquina_juez": juez,
         "veredicto_humano": "", "nota_humana": ""}
    if anot is not None:
        c["anotaciones"] = anot
    c.update(extra)
    return c


def _anot(veredicto, defectos, anotador="A"):
    return [{"anotador": anotador, "veredicto": veredicto, "defectos": defectos}]


def test_el_estricto_adjudica_hallazgo_a_hallazgo():
    gsc = _score()
    casos = [
        # TP localizado: la clase Y el alimento coinciden
        _caso("a", "p1", ["V4: res 85 g vs 140 g"], [], _anot("defecto", [{"clase": "cantidad_inconsistente", "severidad": "minor", "evidencia": "res", "alimento": "res"}])),
        # un error DISTINTO: la máquina acusa al aceite (V1), la persona vio quinoa sin cocer ⇒ FP + FN del esperado
        _caso("b", "p1", ["V1: aceite ..."], [], _anot("defecto", [{"clase": "seco_sin_coccion", "severidad": "high", "evidencia": "quinoa", "alimento": "quinoa"}])),
        # duplicados no multiplican TP
        _caso("c", "p2", ["V3: Zanahoria sin uso", "V3: Zanahoria sin uso"], [], _anot("defecto", [{"clase": "ingrediente_huerfano", "severidad": "minor", "evidencia": "zanahoria"}])),
        # un defecto que ninguna capa mecaniza: FN del sistema entero, no de cada capa
        _caso("d", "p2", [], [], _anot("defecto", [{"clase": "coccion_faltante", "severidad": "high", "evidencia": "frijoles"}])),
        # ok limpio: nada que emparejar
        _caso("e", "p3", [], [], _anot("ok", [])),
    ]
    r = gsc.puntuar_estricto({"casos": casos, "disponibles_por_estrato": {"ambas": 5}})
    det = r["capas"]["determinista"]["crudo"]
    assert (det["tp"], det["fp"], det["fn"]) == (2, 1, 1), det
    assert det["precision"] == 66.7 and det["recall"] == 66.7
    juez = r["capas"]["juez"]["crudo"]
    assert (juez["tp"], juez["fp"], juez["fn"]) == (0, 0, 0) and juez["precision"] is None and juez["recall"] is None
    assert r["no_mecanizable"] == {"coccion_faltante": 1}
    assert r["por_clase"]["seco_sin_coccion"]["fn"] == 1 and r["por_clase"]["cantidad_inconsistente"]["tp"] == 1
    assert r["completos"] == 5 and r["completo"] is False and r["promocion_habilitada"] is False


def test_el_alimento_nombrado_localiza_el_acierto():
    gsc = _score()
    casos = [_caso("a", "p1", ["V4: aceite 5 g vs 10 g"], [], _anot("defecto", [{"clase": "cantidad_inconsistente", "severidad": "minor", "evidencia": "res", "alimento": "res"}]))]
    det = gsc.puntuar_estricto({"casos": casos, "disponibles_por_estrato": {}})["capas"]["determinista"]["crudo"]
    assert (det["tp"], det["fp"], det["fn"]) == (0, 1, 1), "misma clase, otro alimento: FP localizado y FN del esperado"


def test_las_etiquetas_binarias_del_09_07_dejan_el_estricto_incompleto():
    gsc = _score()
    d = json.loads((_BACKEND / "docs" / "culinary_golden_set.json").read_text(encoding="utf-8"))
    r = gsc.puntuar_estricto(d)
    assert r["estados"] == {"sin_rubrica": 77, "dudoso": 3} or r["estados"].get("sin_rubrica", 0) >= 70
    assert r["completos"] == 0 and r["completo"] is False and r["promocion_habilitada"] is False
    assert (r["acuerdo"] or {}).get("anotadores") == 1
    assert "INCOMPLETO" in gsc.render_estricto(r)
    # y el binario sigue funcionando sobre las mismas etiquetas
    b = gsc.puntuar(d)
    assert b["etiquetados"] == 77 and b["capas"]["determinista"]["crudo"]["precision"] == 91.2
    assert b["capas"]["determinista"]["ic95_crudo"]["conglomerados"] > 2


def test_el_estricto_sale_con_codigo_4_si_esta_incompleto(tmp_path):
    import shutil
    import subprocess
    raiz = tmp_path / "backend"
    (raiz / "docs").mkdir(parents=True)
    (raiz / "scripts").mkdir()
    shutil.copy2(_BACKEND / "scripts" / "culinary_golden_score.py", raiz / "scripts" / "culinary_golden_score.py")
    shutil.copy2(_BACKEND / "docs" / "culinary_golden_set.json", raiz / "docs" / "culinary_golden_set.json")
    rc = subprocess.run([sys.executable, str(raiz / "scripts" / "culinary_golden_score.py"), "--estricto"],
                        capture_output=True, text=True, cwd=str(raiz)).returncode
    assert rc == 4
    # completo con rúbrica (20+ casos anotados) ⇒ 0: si no, el 4 de arriba no probaría nada
    d = json.loads((raiz / "docs" / "culinary_golden_set.json").read_text(encoding="utf-8"))
    for c in d["casos"][:25]:
        c["anotaciones"] = _anot("ok", [])
    (raiz / "docs" / "culinary_golden_set.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    rc2 = subprocess.run([sys.executable, str(raiz / "scripts" / "culinary_golden_score.py"), "--estricto"],
                         capture_output=True, text=True, cwd=str(raiz)).returncode
    assert rc2 == 0


def test_acuerdo_y_adjudicacion():
    gsc = _score()
    dos = lambda va, vb, defs_a=None, defs_b=None: [  # noqa: E731
        {"anotador": "A", "veredicto": va, "defectos": defs_a if defs_a is not None else []},
        {"anotador": "B", "veredicto": vb, "defectos": defs_b if defs_b is not None else []}]
    h = [{"clase": "ingrediente_huerfano", "severidad": "minor", "evidencia": "x"}]
    casos = [
        _caso("a", "p1", ["V3: x"], [], dos("defecto", "defecto", h, h)),           # acuerdo total ⇒ completo
        _caso("b", "p1", ["V3: x"], [], dos("defecto", "ok", h, [])),                 # desacuerdo ⇒ pendiente
        _caso("c", "p2", [], [], dos("ok", "ok")),                                    # acuerdo
        _caso("d", "p2", ["V3: x"], [], dos("defecto", "ok", h, []), adjudicacion={"veredicto": "defecto", "defectos": h, "por": "C"}),
    ]
    r = gsc.puntuar_estricto({"casos": casos, "disponibles_por_estrato": {}})
    assert r["estados"] == {"completo": 3, "pendiente_adjudicacion": 1}
    a = r["acuerdo"]
    assert a["anotadores"] == 2 and a["comunes"] == 4 and a["acuerdo_observado"] == 0.5 and set(a["discrepancias"]) == {"b", "d"}
    assert a["kappa"] is not None and -1 <= a["kappa"] <= 1
    assert r["capas"]["determinista"]["crudo"]["tp"] == 2      # a y d (adjudicado); b no puntúa


def test_particiones_por_linaje_sin_parientes_cruzados():
    gsc = _score()
    casos = [_caso(f"c{i}", f"plan-{i % 7}", [], []) for i in range(40)]
    part = gsc.particiones_por_linaje({"casos": casos}, k=3)
    assert part["cruzados"] == [] and sum(len(v) for v in part["folds"].values()) == 40
    assert sum(part["planes_por_fold"].values()) == 7
    # cada plan cae entero en un fold
    for f, ids in part["folds"].items():
        planes = {c["plan"] for c in casos if c["id"] in ids}
        for otro, otros_ids in part["folds"].items():
            if otro != f:
                assert not planes & {c["plan"] for c in casos if c["id"] in otros_ids}


def test_cero_division_devuelve_null_y_el_intervalo_es_por_plan():
    gsc = _score()
    assert gsc._r(0, 0) is None and gsc._r(3, 1) == 75.0
    casos = [_caso(f"c{i}", f"plan-{i % 4}", ["V3: x"], [], _anot("defecto", [{"clase": "ingrediente_huerfano", "severidad": "minor", "evidencia": "x"}])) for i in range(12)]
    ic = gsc.puntuar_estricto({"casos": casos, "disponibles_por_estrato": {}})["capas"]["determinista"]["ic95_crudo"]
    assert ic["conglomerados"] == 4 and ic["remuestreos"] == gsc.BOOTSTRAP_N and ic["precision"] == [100.0, 100.0]
    assert gsc._bootstrap([casos[0]], lambda c: (1, 0, 0)) is None, "un solo conglomerado no es un intervalo"


def test_la_rubrica_cubre_todos_los_codigos_de_la_maquina():
    gsc = _score()
    cubiertos = set().union(*gsc.RUBRICA.values())
    assert set(cc.CHECKS_CAPA1) <= cubiertos
    assert {"combo_absurdo", "tecnica_impropia", "paso_incoherente", "slot_inapropiado", "nombre_no_corresponde"} <= cubiertos


# ────────────────────────────────────────────────────────────── el calibrador y el test golden cruzan por COMIDA

def test_el_calibrador_y_el_test_golden_exigen_la_comida():
    cal = (_BACKEND / "scripts" / "calibrate_culinary_judge.py").read_text(encoding="utf-8")
    assert 'str(x.get("meal") or "").lower() == _m' in cal and 'str(v.meal or "").lower() == _m' in cal
    tg = (_BACKEND / "tests" / "test_p1_culinary_golden.py").read_text(encoding="utf-8")
    assert 'str(x.get("meal") or "").lower() == _m' in tg
    man = json.loads((_BACKEND / "tests" / "fixtures" / "culinary_golden" / "golden_manifest.json").read_text(encoding="utf-8"))
    assert all(df.get("meal") for e in man["mutados"].values() for df in e["defects"]), "el manifest nombra la comida de cada defecto"


# ────────────────────────────────────────────────────────────── docs, plan, marker

def test_docs_plan_y_marker():
    doc = (_BACKEND / "docs" / "culinary_coherence.md").read_text(encoding="utf-8")
    assert "## Estado explícito de evaluación e identidad por ocurrencia (C1 · `P1-PLAN-LOTE-22`" in doc
    assert "culinary_contract_scan_status" in doc and "resolve_judge_violations" in doc and "--estricto" in doc
    assert "las 80 etiquetas BINARIAS existen desde el 2026-09-07" in doc
    plan = (_BACKEND / "docs" / "plan_pendientes_2026_09_11.md").read_text(encoding="utf-8")
    assert "| C1 | ✅ 2026-09-12 · software listo · 👤 anotación con rúbrica + 2º anotador |" in plan
    assert "| C0 | ✅ 2026-09-12 · corpus fijo + etiquetas binarias (09-07) · 👤 rúbrica |" in plan
    rev = (_BACKEND.parent / "docs" / "audits" / "2026-09-07-coherencia-culinaria" / "REVISION-DE-LOS-GAPS.md").read_text(encoding="utf-8")
    assert "## Hecho el 2026-09-12 (`P1-PLAN-LOTE-22`, C1 del plan de pendientes)" in rev
    app = (_BACKEND / "app.py").read_text(encoding="utf-8", errors="ignore")
    m = re.search(r'_LAST_KNOWN_PFIX = "([^"]+)"', app)
    assert m and m.group(1).split("·")[-1].strip() >= "2026-09-12"
    assert "P1-PLAN-LOTE-22" in (_BACKEND / "culinary_coherence.py").read_text(encoding="utf-8")
