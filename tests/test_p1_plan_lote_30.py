# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-30 · 2026-09-12] El benchmark culinario de superficies en MODO REAL, ejecutado con presupuesto pequeño sobre
planes recién generados por el pipeline (CUL-P1-07, segunda mitad: la que el lote 28 dejó escrita pero nunca corrió).

Lo que se prueba:
  · el modo real sustituye las escrituras de `db_core` por dobles que cuentan por tabla y no ejecutan (las lecturas siguen),
    ANTES de importar al orquestador; abre los pools como el arranque de la app y estampa un id de correlación por generación;
  · el coste se suma en proceso con la tarifa del propio emisor (`compute_llm_cost_micros`) y, sin permiso explícito, la fila
    de `llm_usage_events` no sale del proceso;
  · el formulario que recibe el pipeline es el que `/analyze` construye (sin `_` del cliente, `_plan_start_date`,
    `_days_to_generate = PLAN_CHUNK_SIZE`);
  · el ciclo para en el presupuesto y anota errores/emergencias como estado, no como excepción;
  · el artefacto real existe, se reconstruye y declara lo que suprimió; docs, plan y marcador.
"""
from __future__ import annotations

import asyncio
import glob
import importlib.util
import json
import re
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_SRC = (_BACKEND / "scripts" / "bench_superficies_culinarias.py").read_text(encoding="utf-8")


def _mod(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _BACKEND / rel)   # por ruta: `scripts/` no va a sys.path (lote 13)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def bench():
    return _mod("scripts/bench_superficies_culinarias.py", "bench_superficies_l30")


def _cuerpo(nombre: str) -> str:
    return _SRC.split(f"\ndef {nombre}(")[1].split("\ndef ")[0]


# ─────────────── no escribir en la base del dueño ───────────────

def test_el_bloqueo_de_escrituras_cuenta_por_tabla_y_no_ejecuta(bench):
    fake = types.SimpleNamespace(__name__="fake", execute_sql_write=None, aexecute_sql_write=None, execute_sql_transaction=None)
    c = bench.bloquear_escrituras(fake, {})
    assert fake.execute_sql_write("INSERT INTO pipeline_metrics (a) VALUES (1)", (1,)) is True
    assert fake.execute_sql_write("UPDATE app_kv_store SET v = 1 WHERE k = 'x'", returning=True) == []
    assert asyncio.run(fake.aexecute_sql_write("DELETE FROM system_alerts WHERE 1 = 1")) is True
    assert fake.execute_sql_transaction([("insert into llm_usage_events (a) values (1)", ()), ("UPDATE plan_generation_runs SET a = 1", ())]) is True
    assert c == {"pipeline_metrics": 1, "app_kv_store": 1, "system_alerts": 1, "llm_usage_events": 1, "plan_generation_runs": 1}


def test_el_modo_real_bloquea_antes_del_orquestador_abre_los_pools_y_correlaciona():
    g = _cuerpo("generar_real")
    assert g.index("bloquear_escrituras()") < g.index("from graph_orchestrator import arun_plan_pipeline"), "los módulos enlazan el nombre al importar"
    assert g.index("_abrir_pool_sync()") < g.index("from graph_orchestrator import arun_plan_pipeline")
    assert "from db_core import async_connection_pool as pool" in g and "await pool.open()" in g and "await pool.close()" in g
    assert "set_correlation_id(corr)" in g and "reset_correlation_id(tok)" in g
    assert "WindowsSelectorEventLoopPolicy" in g
    assert 'if getattr(mod, "__name__", "") == "db_core"' in _cuerpo("bloquear_escrituras") and 'sys.modules.get("db")' in _SRC
    assert not re.search(r"\b(INSERT INTO|UPDATE meal_plans|DELETE FROM)\b", _SRC), "el benchmark no escribe planes de usuarios (lote 28)"
    assert "P1-PLAN-LOTE-30-BENCH-REAL" in _SRC


def test_el_contador_de_coste_usa_la_tarifa_del_emisor_y_no_escribe_sin_permiso(bench, monkeypatch):
    import db_profiles as dp
    llamadas = []

    def original(**kw):
        llamadas.append(kw)

    monkeypatch.setattr(dp, "log_llm_usage_event", original)
    _log, est = bench.contar_coste(telemetria_prod=False, original=original)
    dp.log_llm_usage_event(model="glm-5.3-flash", node="day_generator", input_tokens=10000, output_tokens=2000, cached_tokens=0)
    dp.log_llm_usage_event(model="gpt-5.6-luna", node="reviewer", input_tokens=3000, output_tokens=500, cached_tokens=1000)
    esperado = dp.compute_llm_cost_micros("glm-5.3-flash", 10000, 2000, 0) + dp.compute_llm_cost_micros("gpt-5.6-luna", 3000, 500, 1000)
    assert est["micros"] == esperado > 0 and est["llamadas"] == 2 and est["por_nodo"] == {"day_generator": 1, "reviewer": 1}
    assert est["modelos"] == ["glm-5.3-flash", "gpt-5.6-luna"] and llamadas == [], "sin --telemetria-prod la fila no sale del proceso"
    _log2, est2 = bench.contar_coste(telemetria_prod=True, original=original)
    dp.log_llm_usage_event(model="glm-5.3-flash", node="planner", input_tokens=100, output_tokens=10)
    assert len(llamadas) == 1 and llamadas[0]["node"] == "planner" and est2["micros"] == dp.compute_llm_cost_micros("glm-5.3-flash", 100, 10, 0)


# ─────────────── imitar a /analyze ───────────────

def test_form_para_pipeline_imita_al_router(bench):
    import graph_orchestrator as go
    from constants import PLAN_CHUNK_SIZE
    perfil = {"_id": 1, "_label": "x", "_expect": {}, "age": 35, "dietType": "vegetarian", "totalDays": 30, "user_id": "guest"}
    pd = bench.form_para_pipeline(perfil, hoy="2026-09-12")
    assert "_id" not in pd and "_label" not in pd and "_expect" not in pd and pd["age"] == 35 and pd["dietType"] == "vegetarian"
    assert pd["_days_to_generate"] == PLAN_CHUNK_SIZE == 3 and pd["_plan_start_date"] == "2026-09-12"
    assert {"_plan_start_date", "_days_to_generate"} <= set(go._TRUSTED_INTERNAL_FORM_KEYS), "las mismas claves que el router inyecta"
    router = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    assert 'pipeline_data["_days_to_generate"] = PLAN_CHUNK_SIZE' in router and 'pipeline_data["_plan_start_date"] = start_date_iso' in router
    assert bench.form_para_pipeline({"age": 1})["_days_to_generate"] == PLAN_CHUNK_SIZE, "sin totalDays también"


def test_el_ciclo_para_en_el_presupuesto_y_anota_errores_y_emergencias(bench):
    async def gen(i, fd):
        if fd == "error":
            raise ValueError("boom")
        if fd == "vacio":
            return {"plan_data": {"days": []}, "coste_usd_est": 0.01}
        if fd == "emergencia":
            return {"plan_data": {"days": [{"meals": []}], "_p1_5_emergency_return": True}, "coste_usd_est": 0.30}
        return {"plan_data": {"days": [{"meals": [{}]}]}, "coste_usd_est": 0.30, "perfil": fd}

    out = asyncio.run(bench._ciclo(["a", "emergencia", "b", "c"], 0.50, gen))
    assert [o["estado"] for o in out] == ["generado", "generado:emergencia", "no_generado:presupuesto_agotado", "no_generado:presupuesto_agotado"]
    assert out[0]["plan_data"]["days"] and out[2]["plan_data"] is None and out[0]["perfil"] == "a"
    out2 = asyncio.run(bench._ciclo(["error", "vacio", "z"], 10.0, gen))
    assert out2[0]["estado"] == "error:ValueError" and out2[0]["plan_data"] is None and "boom" in out2[0]["detalle"]
    assert out2[1]["estado"] == "sin_dias" and out2[1]["plan_data"] is None and out2[2]["estado"] == "generado"


def test_el_modo_real_sigue_negandose_sin_presupuesto_y_admite_telemetria_explicita(bench, capsys):
    assert bench.main(["--real", "--sin-guardar"]) == 2 and "presupuesto" in capsys.readouterr().out
    assert bench.main(["--real", "--perfil", "x.json", "--presupuesto-usd", "0", "--sin-guardar"]) == 2
    assert '"--telemetria-prod", action="store_true"' in _SRC and "telemetria_prod=a.telemetria_prod" in _SRC
    assert 'sufijo = "real_" if a.real else ("replay_" if a.planes_de else "")' in _SRC, "real y replay no pisan al offline del mismo día"


# ─────────────── el artefacto de la corrida real ───────────────

def _artefacto_real() -> dict:
    fs = sorted(glob.glob(str(_BACKEND / "scripts" / "data" / "bench_superficies_real_*.json")))
    assert fs, "falta el artefacto de la corrida real"
    return json.loads(Path(fs[-1]).read_text(encoding="utf-8"))


def test_el_artefacto_real_existe_se_reconstruye_y_declara_lo_que_suprimio(bench):
    art = _artefacto_real()
    assert art["modo"] == "real" and art["n_planes"] >= 1 and art["modo_catalogo"] == "nutricion"
    assert set(art["superficies"]) == set(bench.SUPERFICIES)
    gens = art["generaciones"]
    assert gens and all("coste_usd_est" in g and "estado" in g and "perfil" in g for g in gens)
    ok = [g for g in gens if g["estado"].startswith("generado")]
    assert ok and all(g["dias"] >= 1 and g["comidas"] >= 1 and g["llamadas_llm"] >= 1 for g in ok)
    assert art["telemetria_prod"] is False and isinstance(art["escrituras_suprimidas"], dict)
    assert 0 < art["coste_usd_est"] <= art["presupuesto_usd"], "el presupuesto se respetó y el coste no es cero"
    assert all(p["plan_data"]["days"] for p in art["planes_generados"]) and len(art["planes_generados"]) == art["n_planes"]
    txt = bench.render(art)
    assert "modo real" in txt and "escrituras suprimidas" in txt and "coste estimado" in txt
    for s, r in art["superficies"].items():
        assert set(r) >= {"planes", "n_nuevos", "resueltos", "contrato_estampado", "ms"}, s


# ─────────────── lo que la corrida real encontró: la lista pierde una línea y el paso la sigue nombrando ───────────────

def _catalogo():
    fs = sorted(glob.glob(str(_BACKEND / "scripts" / "data" / "catalogo_nutricion_*.json")))
    return json.loads(Path(fs[-1]).read_text(encoding="utf-8"))["filas"]


def _smoothie_sin_granola() -> dict:
    """La comida del plan vegetariano real tras el piso de porciones (`_floor_subservible_portions` borró «15 g de Granola»)."""
    art = _artefacto_real()
    pd = next(p["plan_data"] for p in art["planes_generados"] if p["perfil"] == "perfil_vegetariana")
    meal = json.loads(json.dumps(pd["days"][0]["meals"][0]))
    assert any("granola" in i.lower() for i in meal["ingredients"]) and "granola" in " ".join(meal["recipe"]).lower()
    meal["ingredients"] = [i for i in meal["ingredients"] if "granola" not in i.lower()]
    return meal


def test_quitar_mencion_retira_el_item_el_complemento_o_la_frase_y_respeta_el_resto():
    import recipe_repair as rr
    q = rr.quitar_mencion
    assert q("Montaje: vierte el smoothie en un bowl, corona con la granola y unas cubitos de lechosa reservados. Sirve bien frío.", "granola") == \
        "Montaje: vierte el smoothie en un bowl, corona con unas cubitos de lechosa reservados. Sirve bien frío."
    assert q("Mise en place: lava las espinacas; mide ¼ cda de linaza y 15 g de granola.", "granola") == "Mise en place: lava las espinacas; mide ¼ cda de linaza."
    assert q("licúa el yogurt con la lechosa, las espinacas, linaza, la canela y el agua fría hasta espesar.", "canela") == \
        "licúa el yogurt con la lechosa, las espinacas, linaza y el agua fría hasta espesar."
    assert q("Licúa la lechosa, las espinacas y la granola hasta obtener una crema.", "granola") == "Licúa la lechosa y las espinacas hasta obtener una crema."
    assert q("El Toque de Fuego: Hornea la batata y el jamón 20 min.", "jamon") == "El Toque de Fuego: Hornea la batata 20 min."
    assert q("Saltea la cebolla, el jamón picado y el pimiento 3 min.", "jamon") == "Saltea la cebolla y el pimiento 3 min."
    assert q("Tuesta la granola 2 min. Sirve el yogurt frío.", "granola") == "Sirve el yogurt frío."
    assert q("Corona con la granola.", "granola") == "" and q("Añade el jamón picado y mezcla bien.", "jamon") == "", "sin el alimento no queda nada que decir"
    assert q("Sirve con arroz.", "granola") == "Sirve con arroz." and q("Sirve la granolada.", "granola") == "Sirve la granolada.", "no toca lo que no lo nombra"
    assert q("Añade el Jamón.", "jamon") == "" and q("Añade los jamones y la cebolla.", "jamon") == "Añade la cebolla.", "acentos, mayúsculas y plural"
    assert q("Mise en place: corta 270 g de mango en cubos y mide 10 g de yogur de coco, 65 ml de leche, 1.5 cdta de mantequilla de maní (5 g) y 1 cdta de canela en polvo.", "yogur de coco") == \
        "Mise en place: corta 270 g de mango en cubos y mide 65 ml de leche, 1.5 cdta de mantequilla de maní (5 g) y 1 cdta de canela en polvo.", "el nombre ENTERO"
    assert q("Montaje: coloca el mango, el yogur de coco, la leche y la canela en la licuadora; procesa hasta espesar.", "yogur de coco") == \
        "Montaje: coloca el mango, la leche y la canela en la licuadora; procesa hasta espesar."
    assert q("Sirve el pan con mantequilla de maní.", "mantequilla de mani") == "Sirve el pan." and q("Añade el yogur y sirve.", "yogur de coco") == "", "si el entero no está, su cabeza"


def test_retirar_sin_lista_cierra_el_v5_nacido_en_la_cadena_sin_tocar_la_lista_y_es_idempotente():
    import culinary_coherence as cc
    import recipe_repair as rr
    filas = _catalogo()
    index = cc.build_culinary_index(filas)
    meal = _smoothie_sin_granola()
    lista = list(meal["ingredients"])
    assert [v["food"] for v in cc._v5_paso_usa_lo_que_no_esta({"day": 1}, meal, index)] == ["granola"], "el caso real: V5 tras borrar la línea"
    r = rr.retirar_sin_lista(meal, index)
    assert r["aplicado"] == ["granola"] and r["descartado"] == [] and len(r["cambios"]) == 2
    assert meal["ingredients"] == lista, "la lista no se toca jamás"
    assert "granola" not in " ".join(meal["recipe"]).lower() and len(meal["recipe"]) == 3
    assert cc._v5_paso_usa_lo_que_no_esta({"day": 1}, meal, index) == []
    assert not cc.culinary_contract_scan({"days": [{"day": 1, "meals": [meal]}]}, filas), "no abre otro hallazgo"
    r2 = rr.retirar_sin_lista(meal, index)
    assert r2["aplicado"] == [] and r2["descartado"] == [] and r2["cambios"] == []
    assert rr.retirar_sin_lista({"recipe": []}, index) == {"aplicado": [], "descartado": [], "cambios": []}
    assert rr.retirar_sin_lista(meal, {}) == {"aplicado": [], "descartado": [], "cambios": []}, "sin índice, nada"


def test_retirar_sin_lista_se_deshace_si_la_mencion_sobrevive_o_la_receta_queda_vacia(monkeypatch):
    import culinary_coherence as cc
    import recipe_repair as rr
    index = cc.build_culinary_index(_catalogo())
    meal = {"name": "Yogurt", "ingredients": ["1 taza de yogurt natural"], "recipe": ["Corona con la granola."]}
    antes = list(meal["recipe"])
    r = rr.retirar_sin_lista(meal, index)
    assert r["aplicado"] == [] and meal["recipe"] == antes, "la receta se quedaría vacía: se deshace y se declara"
    assert r["descartado"] == ["granola"] or r["descartado"] == [], r
    meal2 = _smoothie_sin_granola()
    antes2 = list(meal2["recipe"])
    monkeypatch.setattr(rr, "quitar_mencion", lambda paso, alimento: paso.replace("15 g de granola", "15 g de granola tostada"))
    r2 = rr.retirar_sin_lista(meal2, index)
    assert r2["aplicado"] == [] and r2["descartado"] == ["granola"] and meal2["recipe"] == antes2, "si V5 sigue acusando, se deshace"
    # el corpus fijo: el batido con «yogur de coco» fuera de la lista; tirar la frase entera dejaría mango, leche y maní sin
    # paso (V3) — así fue la primera versión, medida: 3 hallazgos nuevos. La retirada que abre otro hallazgo se deshace.
    from culinary_corpus import cargar, filas_para_medir
    corpus = sorted(glob.glob(str(_BACKEND / "scripts" / "data" / "culinary_corpus_*.json")))[-1]
    batido = next(m for pl in filas_para_medir(cargar(corpus)) for d in (pl.get("plan_data") or pl)["days"] for m in d["meals"]
                  if "yogur de coco" in " ".join(m.get("recipe") or []).lower())
    batido = json.loads(json.dumps(batido))
    antes3 = list(batido["recipe"])
    monkeypatch.setattr(rr, "quitar_mencion", lambda paso, alimento: "" if "yogur de coco" in paso.lower() else paso)
    r3 = rr.retirar_sin_lista(batido, index)
    assert r3["aplicado"] == [] and r3["descartado"] == ["yogur de coco"] and batido["recipe"] == antes3, "abrir un V3 no es reparar"
    monkeypatch.undo()
    r4 = rr.retirar_sin_lista(batido, index)
    assert r4["aplicado"] == ["yogur de coco"] and "yogur de coco" not in " ".join(batido["recipe"]).lower()
    assert "coloca el mango, la leche, la mantequilla de maní y la canela" in " ".join(batido["recipe"]), "el ítem se va; sus vecinos se quedan"


def test_el_contrato_final_corre_el_paso_5_y_sella_sin_lista_sin_borrar_sellos_anteriores():
    import culinary_coherence as cc
    import recipe_contract as rc
    index = cc.build_culinary_index(_catalogo())
    meal = _smoothie_sin_granola()
    r = rc.reconcile_meal(meal, index)
    assert r["sin_lista"] == 1 and r["cambios_sin_lista"][0]["food"] == "granola" and r["sin_reparar"] == {}
    meal = _smoothie_sin_granola()
    assert rc._aplicar_meal(meal, index, "repair", None) == 0
    assert meal[rc.TELEMETRIA_KEY]["sin_lista"] == 1 and meal[rc.TELEMETRIA_KEY]["modo"] == "repair"
    sello = dict(meal[rc.TELEMETRIA_KEY])
    rc._aplicar_meal(meal, index, "repair", None)
    assert meal[rc.TELEMETRIA_KEY] == sello, "una pasada sin nada que decir no borra lo que dijo la anterior"
    limpio = {"name": "Agua", "ingredients": ["1 vaso de agua"], "recipe": ["Sirve el agua."]}
    rc._aplicar_meal(limpio, index, "repair", None)
    assert rc.TELEMETRIA_KEY not in limpio, "un plato que nunca necesitó nada sigue sin sello (lotes 23/29)"
    sombra = _smoothie_sin_granola()
    pasos = list(sombra["recipe"])
    rc._aplicar_meal(sombra, index, "shadow", None)
    assert sombra["recipe"] == pasos and sombra[rc.TELEMETRIA_KEY]["sin_lista"] == 1 and sombra[rc.TELEMETRIA_KEY]["modo"] == "shadow"
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert 'sl = _retirar_sin_lista(meal, index)' in src and 'if r.get("sin_lista"):' in src and "P1-PLAN-LOTE-30" in src
    assert "P1-PLAN-LOTE-30-SIN-LISTA" in (_BACKEND / "recipe_repair.py").read_text(encoding="utf-8")


def test_el_replay_pareado_de_los_planes_reales_baja_el_hallazgo_de_la_cadena_a_cero(bench):
    real = _artefacto_real()
    fs = sorted(glob.glob(str(_BACKEND / "scripts" / "data" / "bench_superficies_replay_*.json")))
    assert fs, "falta el artefacto del replay (mismos planes, código con el paso 5)"
    rep = json.loads(Path(fs[-1]).read_text(encoding="utf-8"))
    assert rep["modo"] == "real-replay" and rep["planes_de"] == "bench_superficies_real_2026_09_13.json" and rep["n_planes"] == real["n_planes"]
    assert real["superficies"]["insert"]["n_nuevos"] == 1 and real["superficies"]["quality"]["n_nuevos"] == 1, "el antes: V5 granola nacido en la cadena"
    assert real["superficies"]["insert"]["planes"][2]["nuevos"][0]["check"] == "V5"
    for s in bench.SUPERFICIES:
        assert rep["superficies"][s]["n_nuevos"] == 0, s
    assert rep["superficies"]["insert"]["contrato_estampado"] >= real["superficies"]["insert"]["contrato_estampado"], "los sellos ya no se pierden al persistir"
    txt = bench.comparar(real, rep)
    assert "insert" in txt and "1 → 0" in txt
    assert '"--planes-de"' in _SRC and 'modo="real-replay"' in _SRC


# ─────────────── docs, plan, marcador ───────────────

def test_docs_plan_y_marcador():
    doc = (_BACKEND / "docs" / "culinary_coherence.md").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-30" in doc and "modo real" in doc and "escritura" in doc and "retirar_sin_lista" in doc
    plan = (_BACKEND / "docs" / "plan_pendientes_2026_09_11.md").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-30" in plan
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', app)
    assert m and int(m.group(1)) >= 30 and m.group(2) >= "2026-09-12"
    assert len((_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8").splitlines()) <= 53100
