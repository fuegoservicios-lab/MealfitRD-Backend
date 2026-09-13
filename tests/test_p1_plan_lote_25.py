# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-25 · 2026-09-12] C4 del plan de pendientes = H8 de la auditoría del 09-11: asignación paso↔ingrediente
en la receta congelada (`usa[i]`, paralelo a `pasos[i]`) y V6/V7a/V3 de «puede repartir» a SUMA EXACTA.

Lo que se prueba, por capas:
  · el reparto lee las pistas del texto («la mitad», «el resto», «parte de», «restante», reservas, referencias) y cierra la
    suma a 1 — o dice exactamente por qué no cierra (a medias, tres mitades, comprado sin usar);
  · la asignación va ATADA al texto por hash: un paso editado la caduca sola;
  · sobre la biblioteca real (193 recetas) el snapshot existe, valida, reproduce y su lista `revisar` es un ratchet;
  · en el escáner, una comida de receta congelada con asignación vigente lee las cuentas en vez de adivinar por texto,
    el knob la apaga, y las comidas del LLM siguen con su heurística intacta.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]

import recipe_usage as ru  # noqa: E402
import culinary_coherence as cc  # noqa: E402
import dish_registry as dr  # noqa: E402


def _c(cid, nombre, grams=50.0):
    return {"ingredient_id": cid, "canonical": nombre, "name": nombre, "grams": grams}


_CONSTS = [_c("pechuga_de_pollo", "Pechuga de pollo", 140), _c("aji_morron", "Ají morrón"), _c("cebolla", "Cebolla", 40),
           _c("yuca", "Yuca", 200), _c("sal", "Sal", 0.5)]


def _cuentas(entry):
    return {k: round(v, 3) for k, v in entry["cuentas"].items()}


def _fr(entry, i):
    return {x["ingredient_id"]: round(x["fraccion"], 3) for x in entry["usa"][i]}


# ─────────────── el vocabulario cerrado de la plantilla ───────────────

def test_las_formas_son_el_nombre_completo_y_los_tokens_que_solo_ese_constituyente_tiene():
    formas, ambiguos = ru._formas(_CONSTS)
    n = ru._norm_clausulas("Corta la pechuga de pollo en cubos y ensarta alternando pollo, morrón y cebolla")
    assert any(rx.search(n) for rx in formas["pechuga_de_pollo"]), "«pollo» a solas es la pechuga: nadie más lo tiene"
    assert any(rx.search(n) for rx in formas["aji_morron"]), "«morrón» a solas es el ají morrón"
    assert not ambiguos


def test_un_token_compartido_por_dos_constituyentes_no_decide_y_se_declara_ambiguo():
    consts = [_c("aceite_vegetal", "Aceite vegetal"), _c("aceite_de_oliva", "Aceite de oliva"), _c("papa", "Papa")]
    formas, ambiguos = ru._formas(consts)
    assert any(k.startswith("aceite") for k in ambiguos)
    e = ru.derivar_uso(["Fríe la papa en el aceite."], consts)
    tipos = {(h["tipo"], h["ingredient_id"]) for h in e["hallazgos"]}
    assert ("ambigua", "aceite_vegetal") in tipos and ("ambigua", "aceite_de_oliva") in tipos
    assert e["estado"] == "revisar"
    e2 = ru.derivar_uso(["Fríe la papa en el aceite de oliva y guarda el aceite vegetal para otro día."], consts)
    assert _cuentas(e2)["aceite_de_oliva"] == 1.0, "el nombre completo sí decide"


def test_plurales_y_singulares_casan_en_ambas_direcciones():
    consts = [_c("huevo", "Huevo"), _c("habichuelas_rojas", "Habichuelas rojas")]
    e = ru.derivar_uso(["Bate los huevos.", "Añade la habichuela roja guisada."], consts)
    assert _fr(e, 0) == {"huevo": 1.0} and _fr(e, 1) == {"habichuelas_rojas": 1.0}


# ─────────────── el reparto ───────────────

def test_sin_pistas_toda_la_cantidad_entra_en_el_primer_paso_y_lo_demas_son_referencias():
    e = ru.derivar_uso(["Corta la pechuga de pollo en cubos.", "El pollo está seguro cuando el termómetro marca 74 °C."],
                       [_c("pechuga_de_pollo", "Pechuga de pollo")])
    assert _fr(e, 0) == {"pechuga_de_pollo": 1.0} and _fr(e, 1) == {}
    assert e["estado"] == "exacta" and not e["hallazgos"]


def test_la_mitad_y_la_otra_mitad_cierran_exactas():
    e = ru.derivar_uso(["Sazona con la mitad de la sal.", "Sazona con la otra mitad de la sal."], [_c("sal", "Sal")])
    assert _fr(e, 0) == {"sal": 0.5} and _fr(e, 1) == {"sal": 0.5} and e["estado"] == "exacta"


def test_el_resto_del_morron_y_la_cebolla_reparte_a_los_dos_encadenados():
    pasos = ["Corta la cebolla y el ají morrón en trozos y ensarta en los pinchos.",
             "Sofríe el resto del morrón y la cebolla con un chorrito de agua."]
    e = ru.derivar_uso(pasos, [_c("aji_morron", "Ají morrón"), _c("cebolla", "Cebolla")])
    assert _fr(e, 0) == {"aji_morron": 0.5, "cebolla": 0.5} and _fr(e, 1) == {"aji_morron": 0.5, "cebolla": 0.5}
    assert e["estado"] == "estimada", "la primera parte no trae cifra: se estima, y se dice"


def test_una_fraccion_fija_heredada_por_conjuncion_no_parte_al_alimento_de_al_lado():
    e = ru.derivar_uso(["Sazona la pechuga con la mitad de la sal y el orégano, y déjala reposar.",
                        "Termina con la otra mitad de la sal."], [_c("sal", "Sal"), _c("oregano", "Orégano dominicano")])
    assert _cuentas(e)["oregano"] == 1.0 and _fr(e, 0)["oregano"] == 1.0
    assert _cuentas(e)["sal"] == 1.0 and e["estado"] == "exacta"


def test_la_pista_propaga_por_una_lista_con_comas():
    e = ru.derivar_uso(["Pica la mitad de la cebolla y la mitad del ajo.", "Pica la otra mitad de la cebolla con el resto del ají, el ajo y el cilantro."],
                       [_c("cebolla", "Cebolla"), _c("ajo", "Ajo"), _c("aji", "Ají cubanela"), _c("cilantro", "Cilantro")])
    assert _cuentas(e) == {"cebolla": 1.0, "ajo": 1.0, "aji": 1.0, "cilantro": 1.0}
    assert _fr(e, 1)["ajo"] == 0.5 and _fr(e, 1)["cilantro"] == 1.0


def test_una_mitad_sola_es_usado_a_medias_la_cuarta_clase_de_las_notas_humanas():
    e = ru.derivar_uso(["Sofríe la cebolla con la mitad del cilantro.", "Sirve caliente."],
                       [_c("cebolla", "Cebolla"), _c("cilantro", "Cilantro")])
    h = [x for x in e["hallazgos"] if x["tipo"] == "a_medias"]
    assert h and h[0]["ingredient_id"] == "cilantro" and _cuentas(e)["cilantro"] == 0.5
    assert e["estado"] == "revisar"


def test_tres_mitades_son_sobre_asignado():
    e = ru.derivar_uso(["Amasa con la mitad del aceite.", "Báñalas con la otra mitad del aceite.",
                        "Sofríe la cebolla con la otra mitad del aceite."], [_c("aceite_vegetal", "Aceite vegetal"), _c("cebolla", "Cebolla")])
    h = [x for x in e["hallazgos"] if x["tipo"] == "sobre_asignado"]
    assert h and "dos veces" in h[0]["detalle"] and e["estado"] == "revisar"


def test_fracciones_fijas_por_encima_de_uno_son_sobre_asignado():
    e = ru.derivar_uso(["Usa la mitad del ajo.", "Usa la mitad del ajo.", "Usa la mitad del ajo."], [_c("ajo", "Ajo")])
    assert [x["tipo"] for x in e["hallazgos"]] == ["sobre_asignado"] and _cuentas(e)["ajo"] == 1.5


def test_una_mencion_sin_pista_seguida_de_el_resto_fue_una_parte():
    e = ru.derivar_uso(["Licúa la avena con la leche y el queso blanco.", "Rellena cada crepe con el resto del queso blanco."],
                       [_c("avena", "Avena"), _c("leche", "Leche"), _c("queso_blanco", "Queso blanco")])
    assert _fr(e, 0)["queso_blanco"] == 0.5 and _fr(e, 1)["queso_blanco"] == 0.5 and e["estado"] == "estimada"


def test_el_todo_entro_antes_y_las_mitades_posteriores_son_del_producto_intermedio():
    e = ru.derivar_uso(["Hierve la yuca pelada y májala.", "Toma la mitad de la masa de yuca para la base y cubre con la otra mitad."],
                       [_c("yuca", "Yuca")])
    assert _cuentas(e)["yuca"] == 1.0 and not e["hallazgos"] and e["estado"] == "estimada"
    assert _fr(e, 0)["yuca"] == 0.5 and _fr(e, 1)["yuca"] == 0.5, "lo que la mitad no nombra se queda donde entró; se estima y se dice"


def test_la_pista_detras_del_alimento_tambien_cierra():
    e = ru.derivar_uso(["Cocina las tortitas con la mitad del aceite de oliva.", "Aliña con el aceite de oliva restante y la sal que queda."],
                       [_c("aceite_de_oliva", "Aceite de oliva"), _c("sal", "Sal")])
    assert _cuentas(e)["aceite_de_oliva"] == 1.0 and _fr(e, 1)["aceite_de_oliva"] == 0.5


def test_lo_que_queda_para_despues_no_consume_y_las_referencias_tampoco():
    pasos = ["Mezcla el huevo con la mitad del aguacate majado. La otra mitad del aguacate queda para untar el pan.",
             "Unta la otra mitad del aguacate sobre el pan, corrige la sal y sirve sin sal extra."]
    e = ru.derivar_uso(pasos, [_c("aguacate", "Aguacate"), _c("huevo", "Huevo")])
    assert _fr(e, 0)["aguacate"] == 0.5 and _fr(e, 1)["aguacate"] == 0.5 and e["estado"] == "exacta"
    e2 = ru.derivar_uso(pasos, [_c("aguacate", "Aguacate"), _c("huevo", "Huevo"), _c("sal", "Sal")])
    assert "sal" not in _fr(e2, 1), "«corrige la sal» y «sin sal» no gastan sal"
    assert [h["tipo"] for h in e2["hallazgos"]] == ["sin_uso"], "la sal se nombra pero nadie la usa: comprada sin uso"
    e3 = ru.derivar_uso(pasos, [_c("aguacate", "Aguacate"), _c("huevo", "Huevo"), _c("sal", "Sal")], condimentos=lambda n: n == "Sal")
    assert e3["estado"] == "exacta" and [h["tipo"] for h in e3["hallazgos"]] == ["sin_uso_condimento"]
    assert ru.validar_uso(e3, pasos, [_c("aguacate", "Aguacate"), _c("huevo", "Huevo"), _c("sal", "Sal")]) == []


def test_una_parte_sin_cifra_y_lo_que_entro_sin_pista_se_reparten_a_partes_iguales():
    e = ru.derivar_uso(["Cocina las papas en el aceite.", "Aliña el repollo con un chorrito del aceite."],
                       [_c("aceite_de_oliva", "Aceite de oliva"), _c("papa", "Papa"), _c("repollo", "Repollo")])
    assert _fr(e, 0)["aceite_de_oliva"] == 0.5 and _fr(e, 1)["aceite_de_oliva"] == 0.5
    assert all(x.get("estimada") for u in e["usa"] for x in u if x["ingredient_id"] == "aceite_de_oliva")


def test_la_pista_no_cruza_la_clausula():
    e = ru.derivar_uso(["Mezcla el puré con la mitad del huevo. Añade sal y revuelve.", "Barniza con el resto del huevo."],
                       [_c("huevo", "Huevo"), _c("sal", "Sal")])
    assert _fr(e, 0)["sal"] == 1.0 and _cuentas(e) == {"huevo": 1.0, "sal": 1.0}


def test_comprado_sin_nombrar_es_sin_uso_salvo_que_sea_condimento():
    consts = [_c("lentejas", "Lentejas"), _c("auyama", "Auyama"), _c("sal", "Sal")]
    pasos = ["Enjuaga las lentejas.", "Cocina 20 minutos y sirve."]
    e = ru.derivar_uso(pasos, consts)
    assert {h["tipo"] for h in e["hallazgos"]} == {"sin_uso"} and e["estado"] == "revisar"
    e2 = ru.derivar_uso(pasos, consts, condimentos=lambda n: n.lower() == "sal")
    tipos = {(h["tipo"], h["ingredient_id"]) for h in e2["hallazgos"]}
    assert ("sin_uso_condimento", "sal") in tipos and ("sin_uso", "auyama") in tipos


# ─────────────── el hash y el validador ───────────────

def test_el_hash_cambia_con_una_letra_y_el_validador_lo_caduca():
    pasos = ["Sazona con la mitad de la sal.", "Sazona con la otra mitad de la sal."]
    consts = [_c("sal", "Sal")]
    e = ru.derivar_uso(pasos, consts)
    assert ru.validar_uso(e, pasos, consts) == []
    assert ru.pasos_hash(pasos) != ru.pasos_hash([pasos[0], pasos[1] + "!"])
    assert any("caducada" in p for p in ru.validar_uso(e, [pasos[0], pasos[1] + "!"], consts))


def test_el_validador_acusa_ids_ajenos_sumas_que_no_cierran_y_cuentas_que_mienten():
    pasos = ["Sazona con la sal."]
    consts = [_c("sal", "Sal")]
    e = ru.derivar_uso(pasos, consts)
    malo = json.loads(json.dumps(e))
    malo["usa"][0][0]["ingredient_id"] = "pimienta"
    assert any("no es constituyente" in p for p in ru.validar_uso(malo, pasos, consts))
    malo2 = json.loads(json.dumps(e))
    malo2["usa"][0][0]["fraccion"] = 0.5
    problemas = ru.validar_uso(malo2, pasos, consts)
    assert any("cuentas dice" in p for p in problemas) and any("Σ = 0.5" in p for p in problemas)
    assert ru.validar_uso({"pasos_hash": "x", "usa": []}, pasos, consts), "usa sin un elemento por paso"


# ─────────────── la biblioteca real: snapshot, ratchet, reproducibilidad ───────────────

def _corpus():
    fs = sorted(glob.glob(str(_BACKEND / "scripts" / "data" / "culinary_corpus_*.json")))
    if not fs:
        pytest.skip("sin corpus culinario fijo en scripts/data")
    return json.loads(Path(fs[-1]).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def snap():
    ru.clear_caches()
    s = ru.cargar_uso("DO")
    assert s and s.get("por_id"), "falta data/registry/recipe_usage_do_v1.json: python scripts/asignar_uso_pasos.py --write"
    return s


def test_hay_una_asignacion_por_receta_congelada_y_todas_validan(snap):
    pasos = ru._pasos_index("DO")
    tpl = dr.templates_by_id("DO")
    assert set(snap["por_id"]) == {t for t in pasos if t in tpl}
    for tid, e in snap["por_id"].items():
        assert ru.validar_uso(e, pasos[tid], tpl[tid]["constituents"]) == [], tid
        assert len(e["usa"]) == len(pasos[tid])
        assert ru.usage_for_template(tid, "DO") is e or ru.usage_for_template(tid, "DO") == e


def test_el_snapshot_reproduce_desde_el_texto_y_el_catalogo_congelado(snap):
    c = _corpus()
    index = cc.build_culinary_index(c["catalogo_filas"])
    fresco = ru.derivar_biblioteca("DO", index, indice_huella=c["catalogo"]["huella"], generado_el="x")
    assert fresco["por_id"] == snap["por_id"], "el snapshot no reproduce: python scripts/asignar_uso_pasos.py --write"
    assert snap.get("indice_huella") == c["catalogo"]["huella"]


def test_ratchet_de_la_biblioteca_lo_que_la_maquina_no_decide_es_poco_y_tiene_nombre(snap):
    r = snap["resumen"]
    assert r["recetas"] == 193
    assert r["estados"].get("exacta", 0) >= 150 and r["estados"].get("revisar", 0) <= 3
    assert r["constituyentes_con_suma_1"] >= 1000 and r["constituyentes"] == 1017
    tpl = dr.templates_by_id("DO")
    revisar = {tpl[t]["name"] for t, e in snap["por_id"].items() if e["estado"] == "revisar"}
    assert revisar <= {"Yaniqueques horneados (no fritos) con huevo revuelto", "Pollo al horno con batata y ensalada de repollo",
                       "Lentejas guisadas con auyama y batata"}, revisar
    tipos = r["hallazgos"]
    assert tipos.get("a_medias", 0) == 0 and tipos.get("sobre_asignado", 0) <= 2 and tipos.get("sin_uso", 0) <= 1
    assert tipos.get("ambigua", 0) == 0, "un token compartido sin decidir dentro de una plantilla"


def test_una_receta_editada_caduca_su_asignacion_sola(snap, monkeypatch):
    tid = next(iter(snap["por_id"]))
    pasos = dict(ru._pasos_index("DO"))
    pasos[tid] = list(pasos[tid]) + ["Un paso nuevo que nadie asignó."]
    monkeypatch.setattr(ru, "_pasos_index", lambda country="DO": pasos)
    assert ru.usage_for_template(tid, "DO") is None
    assert ru.cuentas_para_comida({"_recipe_source": "library", "_template_id": tid, "recipe": pasos[tid]}) is None


# ─────────────── el escáner: cuentas en vez de texto ───────────────

def _meal_de(tid, **extra):
    t = dr.templates_by_id("DO")[tid]
    ings = [f"{float(x['grams']):g} g de {x['canonical']}" for x in t["constituents"]]
    m = {"meal": "Almuerzo", "name": t["name"], "ingredients": ings, "ingredients_raw": list(ings),
         "recipe": list(ru._pasos_index("DO")[tid]), "_recipe_source": "library", "_template_id": tid}
    m.update(extra)
    return m


def _tid(prefix):
    return next(t for t, x in dr.templates_by_id("DO").items() if x["name"].startswith(prefix))


@pytest.fixture(scope="module")
def rows():
    return _corpus()["catalogo_filas"]


def test_tres_mitades_que_la_heuristica_no_veia_salen_como_V6_exacto(snap, rows, monkeypatch):
    monkeypatch.delenv("MEALFIT_RECIPE_USAGE_EXACT", raising=False)
    plan = {"days": [{"day": 1, "meals": [_meal_de(_tid("Yaniqueques horneados")), _meal_de(_tid("Lentejas guisadas con auyama y batata")),
                                          _meal_de(_tid("Pinchos de pollo"))]}]}
    viol, est = cc.culinary_contract_scan_status(plan, rows)
    assert est["status"] == "scanned" and est["exactas"] == 3
    por = {(v["check"], v["meal_index"], v["food"]) for v in viol}
    assert ("V6", 0, "Aceite vegetal") in por and ("V3", 1, "Auyama") in por
    assert all("asignación exacta" in v["detail"] for v in viol if v["check"] in ("V3", "V6", "V7a"))
    assert not [v for v in viol if v["meal_index"] == 2], "los pinchos reparten bien: cero hallazgos, y evaluados"


def test_con_el_knob_apagado_vuelve_la_heuristica(snap, rows, monkeypatch):
    monkeypatch.setenv("MEALFIT_RECIPE_USAGE_EXACT", "0")
    plan = {"days": [{"day": 1, "meals": [_meal_de(_tid("Yaniqueques horneados")), _meal_de(_tid("Lentejas guisadas con auyama y batata"))]}]}
    viol, est = cc.culinary_contract_scan_status(plan, rows)
    assert est["exactas"] == 0
    assert not [v for v in viol if v["check"] == "V6"], "sin cifras en los pasos, la heurística no ve las tres mitades"
    assert [v for v in viol if v["check"] == "V3" and v["food"] == "Auyama"], "V3 por texto sigue viendo la auyama"


def test_una_receta_reescrita_en_el_plato_o_del_llm_sigue_con_la_heuristica(snap, rows, monkeypatch):
    monkeypatch.delenv("MEALFIT_RECIPE_USAGE_EXACT", raising=False)
    m = _meal_de(_tid("Yaniqueques horneados"))
    m["recipe"] = m["recipe"][:2]
    assert ru.cuentas_para_comida(m) is None
    llm = {"meal": "Cena", "name": "Ajo", "ingredients": ["½ diente de ajo", "100 g de pollo"],
           "recipe": ["Pica 1 diente de ajo y dóralo con el pollo a la plancha."]}
    viol, est = cc.culinary_contract_scan_status({"days": [{"day": 1, "meals": [llm, m]}]}, rows)
    assert est["exactas"] == 0 and {v["check"] for v in viol if v["meal_index"] == 0} >= {"V6"}


def test_los_cuatro_checks_se_ceden_desde_dentro_y_la_cadena_del_escaner_no_cambia():
    src = (_BACKEND / "culinary_coherence.py").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-25-SCAN-EXACT" in src
    for fn in ("_v3_huerfanos", "_v6_paso_pide_mas_que_la_lista", "_v7a_lista_compra_de_mas", "_v7e_paso_pide_mas_piezas"):
        cuerpo = src.split(f"def {fn}(day, meal, index)")[1].split("\ndef ")[0]
        assert "_cuentas_exactas(meal)" in cuerpo, fn
    cadena = src.split("def culinary_contract_scan(")[1]
    for lit in ("out.extend(_v3_huerfanos(day, meal, index))", "out.extend(_v6_paso_pide_mas_que_la_lista(day, meal, index))",
                "out.extend(_v7a_lista_compra_de_mas(day, meal, index))", "out.extend(_v7e_paso_pide_mas_piezas(day, meal, index))"):
        assert lit in cadena, lit
    assert '"exactas": 0' in cadena
    assert cc._v6_paso_pide_mas_que_la_lista(1, {}, {}) == [] and cc._v3_huerfanos(1, {"ingredients": [], "recipe": []}, {}) == []


def test_el_traductor_solo_lleva_al_escaner_los_tres_hallazgos_de_cantidad():
    cuentas = {"estado": "revisar", "cuentas": {"a": 0.0, "b": 0.5, "c": 1.5}, "nombres": {"a": "Auyama", "b": "Cilantro", "c": "Ajo"},
               "hallazgos": [{"tipo": "sin_uso", "ingredient_id": "a", "detalle": "x"}, {"tipo": "a_medias", "ingredient_id": "b", "detalle": "y"},
                             {"tipo": "sobre_asignado", "ingredient_id": "c", "detalle": "z"}, {"tipo": "fuera_de_plantilla", "alimento": "Sofrito", "detalle": "w"},
                             {"tipo": "sin_uso_condimento", "ingredient_id": "s", "detalle": "v"}, {"tipo": "ambigua", "ingredient_id": "b", "detalle": "u"}]}
    out = ru.hallazgos_para_scanner(cuentas)
    assert [(c, f) for c, f, _ in out] == [("V3", "Auyama"), ("V7a", "Cilantro"), ("V6", "Ajo")]
    assert all(d.startswith("asignación exacta (revisar)") for _, _, d in out)


# ─────────────── el script, la biblioteca intacta, docs, knob y marcador ───────────────

def _script():
    p = _BACKEND / "scripts" / "asignar_uso_pasos.py"
    spec = importlib.util.spec_from_file_location("asignar_uso_pasos", p)   # por ruta: `scripts/` no va a sys.path (lote 13)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, p.read_text(encoding="utf-8")


def test_el_script_verifica_el_snapshot_vigente_y_solo_escribe_con_write(capsys):
    mod, src = _script()
    assert not re.search(r"\b(INSERT|UPDATE|DELETE|psycopg)\b", src), "el script no toca la base"
    assert mod.main(["--verificar"]) == 0
    assert "vigente" in capsys.readouterr().out
    assert "--write" in src and "escribir_snapshot" in src


def test_la_biblioteca_congelada_no_cambia_y_solo_la_lee_quien_la_leia():
    lib = json.loads((_BACKEND / "data" / "registry" / "recipe_library_do_v1.json").read_text(encoding="utf-8"))
    assert all(set(r) <= {"franja", "pasos"} for r in lib["por_id"].values()), "la asignación vive APARTE, atada por hash"
    src = (_BACKEND / "recipe_usage.py").read_text(encoding="utf-8")
    assert "recipe_library_do" not in src and "P1-PLAN-LOTE-25-RECIPE-USAGE" in src
    assert re.search(r"def usage_exact_enabled\(\).*?MEALFIT_RECIPE_USAGE_EXACT.*?True", src, re.S)


def test_docs_plan_knob_y_marcador():
    doc = (_BACKEND / "docs" / "culinary_coherence.md").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-25" in doc and "paso↔ingrediente" in doc and "1004" in doc and "tres mitades" in doc
    knobs = (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    assert "| `MEALFIT_RECIPE_USAGE_EXACT` | `True` |" in knobs
    plan = (_BACKEND / "docs" / "plan_pendientes_2026_09_11.md").read_text(encoding="utf-8")
    assert re.search(r"^\| C4 \| ✅ 2026-09-12", plan, re.M) and "P1-PLAN-LOTE-25" in plan
    aud = (_BACKEND / "docs" / "auditoria_arquitectura_verificacion_2026_09_11.md").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-25" in aud, "el H8 de la auditoría enlaza con lo hecho"
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', app)
    assert m and int(m.group(1)) >= 25 and m.group(2) >= "2026-09-12"
