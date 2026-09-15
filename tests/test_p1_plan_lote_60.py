# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-60 · 2026-09-15] (lote 38 del plan 38-44 · C1 cierre) La anotación del dueño entra al instrumento.

Con la anotación con rúbrica del dueño (80/80, Hoja del dueño 2026-09-13) el marcador estricto daba 0 aciertos POR
CONSTRUCCIÓN: exigía el `alimento` del defecto como subcadena del texto de la máquina, y V4 no nombra alimento. Aquí:

(a) el adjudicador empareja el defecto del dueño con el hallazgo aunque el texto no nombre alimento (caso real
    `0108f857ae`) y sigue separando dos alimentos distintos cuando el hallazgo sí nombra uno (caso real `098d23388f`);
(b) «defecto» sin ningún defecto de la rúbrica queda `sin_rubrica`, no puntúa y el informe lo nombra;
(c) el fichero del dueño carga con `cargar_anotaciones`: 80 casos con veredicto;
(d) el refresco escribe la columna con fecha AL LADO y jamás toca las del 09-06 (funcional y por el fuente);
(e) la línea base estricta publicada (md + json coherentes entre sí), docs, marker ≥ 60 y anclas.
"""
from __future__ import annotations

import collections
import importlib.util
import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_GOLDEN = _BACKEND / "docs" / "culinary_golden_set.json"
_ANOT = _BACKEND / "docs" / "culinary_golden_anotaciones_angelo.json"
_FECHA = "2026-09-15"
_BASE_MD = _BACKEND / "docs" / f"culinary_baseline_estricto_{_FECHA}.md"
_BASE_JSON = _BACKEND / "docs" / f"culinary_baseline_estricto_{_FECHA}.json"
_REFRESH = _BACKEND / "scripts" / "culinary_golden_refresh.py"


def _mod(rel: str, nombre: str):
    spec = importlib.util.spec_from_file_location(nombre, _BACKEND / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def gsc():
    return _mod("scripts/culinary_golden_score.py", "culinary_golden_score_l60")


@pytest.fixture(scope="module")
def rf():
    return _mod("scripts/culinary_golden_refresh.py", "culinary_golden_refresh_l60")


def _golden() -> dict:
    return json.loads(_GOLDEN.read_text(encoding="utf-8"))


def _caso(id_, ingredientes, det, juez, veredicto, defectos):
    return {"id": id_, "plan": f"plan-{id_}", "estrato": "ambas", "ingredientes": ingredientes,
            "maquina_determinista": det, "maquina_juez": juez,
            "anotaciones": [{"anotador": "A", "veredicto": veredicto, "defectos": defectos}]}


# ─────────────────────────────── (a) el adjudicador

def test_el_caso_real_0108f857ae_es_un_acierto_por_codigo(gsc):
    """«Lista: 85 g de res; paso 1: porción de 140 g» (dueño) y «V4: ingrediente declara 85 g, pasos declaran 140 g»
    (máquina) son el MISMO defecto. Antes: FN + FP, porque «carne de res magra» no es subcadena del texto de V4."""
    d = _golden()
    caso = next(c for c in d["casos"] if c["id"] == "0108f857ae")
    ext = gsc.cargar_anotaciones([_ANOT])
    defecto = ext["0108f857ae"][0]["defectos"][0]
    assert defecto["alimento"] == "Carne de res magra" and defecto["clase"] == "cantidad_inconsistente"
    assert caso["maquina_determinista"] == ["V4: ingrediente declara 85 g, pasos declaran 140 g"]
    r = gsc.puntuar_estricto({"casos": [caso], "disponibles_por_estrato": d["disponibles_por_estrato"]}, ext)
    det = r["capas"]["determinista"]
    assert (det["crudo"]["tp"], det["crudo"]["fp"], det["crudo"]["fn"]) == (1, 0, 0)
    assert det["emparejados"] == {"codigo": 1}
    assert [x["emparejado_por"] for x in r["detalle"] if x["resultado"] == "tp"] == ["codigo"]


def test_si_el_hallazgo_nombra_otro_alimento_no_es_acierto(gsc):
    caso = _caso("x", ["85 g de carne de res magra", "1 cdta de aceite de oliva"],
                 ["V3: listado ('1 cdta de aceite de oliva') pero ningún paso lo menciona"], [], "defecto",
                 [{"clase": "ingrediente_huerfano", "severidad": "minor", "evidencia": "e", "alimento": "Carne de res magra"}])
    det = gsc.puntuar_estricto({"casos": [caso]})["capas"]["determinista"]["crudo"]
    assert (det["tp"], det["fp"], det["fn"]) == (0, 1, 1), "misma clase, otro alimento: FP localizado y FN del esperado"


def test_plural_y_acentos_no_separan_el_mismo_alimento(gsc):
    caso = _caso("y", ["⅔ taza de rábanos", "1½ tomates medianos"],
                 ["V3: listado ('⅔ taza de rábanos') pero ningún paso lo menciona"], [], "defecto",
                 [{"clase": "ingrediente_huerfano", "severidad": "minor", "evidencia": "e", "alimento": "Rabano."}])
    capa = gsc.puntuar_estricto({"casos": [caso]})["capas"]["determinista"]
    assert (capa["crudo"]["tp"], capa["crudo"]["fp"], capa["crudo"]["fn"]) == (1, 0, 0)
    assert capa["emparejados"] == {"codigo+alimento": 1}


def test_el_juez_que_se_queja_de_otro_alimento_no_acierta(gsc):
    """Caso real `098d23388f`: el dueño vio el merey que el título promete tostado y ningún paso tuesta; el juez se quejó
    del queso cottage. Sin filtro contaba como acierto."""
    caso = next(c for c in _golden()["casos"] if c["id"] == "098d23388f")
    juez = gsc.puntuar_estricto({"casos": [caso]}, gsc.cargar_anotaciones([_ANOT]))["capas"]["juez"]["crudo"]
    assert (juez["tp"], juez["fp"], juez["fn"]) == (0, 1, 1)


def test_nombrar_un_alimento(gsc):
    voc = gsc.vocabulario_de_comida({"ingredientes": ["85 g de carne de res magra", "½ taza de arroz blanco cocido"]})
    assert {"carne", "res", "arroz"} <= voc and not {"taza", "blanco", "cocido", "magra"} & voc
    assert not gsc._nombra_alimento("V4: ingrediente declara 85 g, pasos declaran 140 g", voc)
    assert gsc._nombra_alimento("V3: listado ('½ taza de arroz blanco cocido') pero ningún paso lo menciona", voc)
    # sin lista (fixtures): cualquier palabra no genérica cuenta, y el código del hallazgo no es un alimento
    assert gsc._nombra_alimento("V4: aceite 5 g vs 10 g", set())
    assert not gsc._nombra_alimento("V4: ingrediente declara 85 g, pasos declaran 140 g", set())
    assert not gsc._nombra_alimento("paso_incoherente: x", set())
    assert gsc._palabras_de_alimento("Nueces, limones y tomates") == {"nuez", "limon", "tomate"}


def test_el_hallazgo_refrescado_declara_el_alimento_que_acusa(gsc, rf):
    """Caso real `119f0e6a06`: el V7e refrescado acusa al casabe (1½ tortas en la lista, 1¾ en el paso) y su detalle cita
    el paso, que nombra la ciruela y el yogurt. Sin el alimento declarado, el filtro lo descartaba como otro alimento."""
    v = {"check": "V7e", "food": "Casabe",
         "detail": "el paso pide 1.75 y la lista compra 1.5: Mise en place: lava y corta 65 g de ciruela en trozos, "
                   "mide ¾ taza de yogurt griego sin azúcar"}
    t = rf.texto_determinista(v)
    assert t.startswith("V7e: ") and t.endswith(" (alimento: Casabe)")
    assert rf.texto_determinista({"check": "V8a", "food": "tiempo", "detail": "x"}) == "V8a: x", "V8a no acusa un alimento"
    assert rf.texto_juez({"tipo": "paso_incoherente", "detalle": "d", "componente": "la masa (de trigo)"}) == \
        "paso_incoherente: d (componente: la masa de trigo)"
    assert rf.texto_juez({"tipo": "combo_absurdo", "certeza": "dudosa", "detalle": "d"}) == "combo_absurdo [dudosa]: d"
    assert gsc._alimento_explicito(t) == {"casabe"} and gsc._alimento_explicito("V4: sin nada") == set()
    ingredientes = ["¾ taza de yogurt griego sin azúcar", "65 g de ciruela en trozos", "1½ tortas pequeño de casabe"]
    casabe = _caso("z", ingredientes, [t], [], "defecto",
                   [{"clase": "cantidad_inconsistente", "severidad": "minor", "evidencia": "e", "alimento": "Casabe"}])
    capa = gsc.puntuar_estricto({"casos": [casabe]})["capas"]["determinista"]
    assert (capa["crudo"]["tp"], capa["crudo"]["fp"], capa["crudo"]["fn"]) == (1, 0, 0)
    assert capa["emparejados"] == {"codigo+alimento": 1}
    ciruela = _caso("w", ingredientes, [t], [], "defecto",
                    [{"clase": "cantidad_inconsistente", "severidad": "minor", "evidencia": "e", "alimento": "Ciruela"}])
    c2 = gsc.puntuar_estricto({"casos": [ciruela]})["capas"]["determinista"]["crudo"]
    assert (c2["tp"], c2["fp"], c2["fn"]) == (0, 1, 1), "acusa al casabe: el defecto de la ciruela no es ése"


# ─────────────────────────────── (b) «defecto» sin defectos

def test_defecto_sin_defectos_queda_sin_rubrica_y_no_puntua(gsc):
    vacio = _caso("v", [], ["V3: listado ('x') pero ningún paso lo menciona"], [], "defecto", [])
    ok = _caso("o", [], [], [], "ok", [])
    r = gsc.puntuar_estricto({"casos": [vacio, ok]})
    assert r["estados"] == {"sin_rubrica": 1, "completo": 1}
    assert r["sin_rubrica_casos"] == ["v"]
    det = r["capas"]["determinista"]["crudo"]
    assert (det["tp"], det["fp"], det["fn"]) == (0, 0, 0), "el V3 de un «defecto» vacío no es un FP"
    txt = gsc.render_estricto(r)
    assert "sin ningun defecto de la rubrica" in txt and "v" in txt.split("rubrica (no puntuan")[1]
    raro = _caso("r", [], [], [], "defecto", [{"clase": "inventada", "severidad": "minor", "evidencia": "e"}])
    assert gsc.puntuar_estricto({"casos": [raro]})["sin_rubrica_casos"] == ["r"]


def test_la_anotacion_del_dueno_no_deja_ningun_defecto_sin_rubrica(gsc):
    """El único que había, `060b4fda6a`, lo transcribió el agente de la nota del dueño del 09-07 (decisión delegada)."""
    r = gsc.puntuar_estricto(_golden(), gsc.cargar_anotaciones([_ANOT]))
    assert r["sin_rubrica_casos"] == [] and r["estados"] == {"completo": 75, "dudoso": 5}
    assert r["completo"] is True and r["promocion_habilitada"] is False, "un solo anotador: no promociona"


# ─────────────────────────────── (c) el fichero del dueño

def test_el_fichero_del_dueno_carga_con_80_casos_con_veredicto(gsc):
    ext = gsc.cargar_anotaciones([_ANOT])
    ids = {c["id"] for c in _golden()["casos"]}
    anotados = {k: v for k, v in ext.items() if isinstance(k, str)}
    assert set(anotados) == ids and len(ids) == 80
    assert all(len(v) == 1 and v[0]["anotador"] == "angelo" for v in anotados.values())
    veredictos = collections.Counter(str(v[0]["veredicto"]).strip() for v in anotados.values())
    assert veredictos == {"defecto": 66, "ok": 9, "dudoso": 5}


# ─────────────────────────────── (d) el refresco

def test_el_refresco_escribe_al_lado_y_no_toca_las_del_09_06(rf):
    d = {"generado": "2026-09-06", "casos": [
        {"id": "a", "maquina_determinista": ["V4: x"], "maquina_juez": ["combo_absurdo: y"], "veredicto_humano": "defecto"},
        {"id": "b", "maquina_determinista": [], "maquina_juez": [], "veredicto_humano": "ok"}]}
    copia = json.loads(json.dumps(d))
    nuevo = rf.aplicar(d, "determinista", "2030-01-01", {"a": ["V7c: z"], "b": []}, {"reglas_huella": "h"})
    assert d == copia, "no muta la entrada"
    for antes, despues in zip(d["casos"], nuevo["casos"]):
        assert despues["maquina_determinista"] == antes["maquina_determinista"]
        assert despues["maquina_juez"] == antes["maquina_juez"]
    a = nuevo["casos"][0]
    assert a["maquina_determinista_2030-01-01"] == ["V7c: z"]
    assert list(a).index("maquina_determinista_2030-01-01") == list(a).index("maquina_juez") + 1
    assert list(nuevo)[-2:] == ["refrescos", "casos"]
    assert nuevo["refrescos"][0]["columna"] == "maquina_determinista_2030-01-01" and nuevo["refrescos"][0]["casos"] == 2
    with pytest.raises(ValueError):
        rf.aplicar(nuevo, "determinista", "2030-01-01", {"a": []}, {})      # misma fecha: no se pisa
    with pytest.raises(ValueError):
        rf.columna("determinista", "")                                      # sin fecha no hay columna
    # un caso sin juzgar no lleva la columna (el marcador lo deja fuera de la capa, no lo cuenta como limpio)
    parcial = rf.aplicar(d, "juez", "2030-01-01", {"a": ["tecnica_impropia: w"]}, {})
    assert "maquina_juez_2030-01-01" in parcial["casos"][0] and "maquina_juez_2030-01-01" not in parcial["casos"][1]


def test_el_fuente_del_refresco_nunca_asigna_ni_borra_una_columna_base():
    src = _REFRESH.read_text(encoding="utf-8")
    for base in ("maquina_determinista", "maquina_juez"):
        assert not re.search(r'\[\s*["\']' + base + r'["\']\s*\]\s*=(?!=)', src), f"asigna {base}"
        assert not re.search(r'\.pop\(\s*["\']' + base + r'["\']', src), f"borra {base}"
        assert not re.search(r'^\s*del\s+[^\n]*["\']' + base + r'["\']', src, re.M), f"borra {base}"
    assert "tooltip-anchor: P1-PLAN-LOTE-60-REFRESH" in src
    assert 'os.environ["MEALFIT_CULINARY_JUDGE_GUARD"] = "warn"' in src
    i_guard = src.index('os.environ["MEALFIT_CULINARY_JUDGE_GUARD"] = "warn"')
    i_bloq = src.index("bench.bloquear_escrituras()")
    i_orq = src.index("import graph_orchestrator as go")
    assert i_guard < i_bloq < i_orq, "guard y dobles de escritura ANTES de importar el orquestador"
    assert "conn.read_only = True" in src and "TOPE_USD = 1.00" in src


def test_el_escaner_del_refresco_corre_sin_base(rf):
    cat = [{"name": "Arroz blanco", "aliases": ["arroz"], "category": "Granos", "ready_to_eat": False,
            "prep_methods": ["hervir"]},
           {"name": "Huevo", "aliases": ["huevos"], "category": "Proteínas", "ready_to_eat": False,
            "prep_methods": ["hervir", "freir"]}]
    caso = {"id": "k", "dia": 3, "franja": "Desayuno", "nombre": "Huevo hervido",
            "ingredientes": ["2 huevos", "40 g de arroz blanco"], "pasos": ["Hierve los huevos 10 minutos y sirve."]}
    por_caso, resumen = rf.escanear([caso], cat)
    assert resumen["estado_scan"] == {"scanned": 1} and resumen["reglas_huella"] and resumen["no_aplica"]
    assert por_caso["k"] and all(re.match(r"^V\d[a-e]?: ", t) for t in por_caso["k"])
    assert any(t.startswith("V3: ") and "arroz" in t for t in por_caso["k"]), por_caso["k"]
    assert rf.plan_de(caso)["days"][0]["day"] == 3 and rf.comida_de(caso)["meal"] == "Desayuno"


def test_el_golden_set_trae_las_columnas_refrescadas_y_conserva_las_del_09_06():
    d = _golden()
    det, juez = f"maquina_determinista_{_FECHA}", f"maquina_juez_{_FECHA}"
    assert all(det in c for c in d["casos"])
    con_alimento = {"V1", "V2", "V3", "V4", "V5", "V6", "V7a", "V7b", "V7c", "V7d", "V7e"}
    for c in d["casos"]:
        for t in c[det]:
            assert ("(alimento: " in t) == (t.split(":", 1)[0] in con_alimento), t
    base = collections.Counter(t.split(":", 1)[0] for c in d["casos"] for t in c["maquina_determinista"])
    assert base == {"V1": 12, "V2": 2, "V3": 16, "V4": 12, "V5": 2}, "la columna del 09-06, intacta"
    ref = {r["columna"]: r for r in d["refrescos"]}
    assert ref[det]["catalogo"]["filas"] > 300 and ref[det]["reglas_huella"] and ref[det]["no_aplica"]
    j = ref[juez]
    assert j["modelo"] == "glm-5.3-flash" and j["una_comida_por_llamada"] is True
    assert 0 < j["coste_usd_est"] <= j["presupuesto_usd"] <= 1.0
    assert j["casos_juzgados"] + len(j["casos_sin_juzgar"]) + len(j["fallos"]) == 80
    assert sum(1 for c in d["casos"] if juez in c) == j["casos_juzgados"]
    assert set(j["escrituras_suprimidas"]) <= {"llm_usage_events", "pipeline_metrics", "app_kv_store", "system_alerts"}


# ─────────────────────────────── (e) la línea base, docs y marker

def test_la_linea_base_estricta_esta_publicada_y_es_coherente():
    assert _BASE_MD.exists() and _BASE_JSON.exists()
    j = json.loads(_BASE_JSON.read_text(encoding="utf-8"))
    md = _BASE_MD.read_text(encoding="utf-8")
    assert j["antes"]["columnas"] == {"determinista": "maquina_determinista", "juez": "maquina_juez"}
    assert j["despues"]["columnas"] == {"determinista": f"maquina_determinista_{_FECHA}", "juez": f"maquina_juez_{_FECHA}"}
    for k in ("antes", "despues"):
        assert j[k]["completo"] is True and j[k]["sin_rubrica_casos"] == []
        for capa in ("determinista", "juez"):
            c = j[k]["capas"][capa]["crudo"]
            assert f"{c['tp']}/{c['fp']}/{c['fn']}" in md, (k, capa)
    assert "no se toca" in md and "P1-PLAN-LOTE-60" in md and "culinary_golden_anotaciones_angelo.json" in md


def test_docs_marker_y_anclas():
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'^_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-\d\d-\d\d"', app, re.M)
    assert m and int(m.group(1)) >= 60
    assert "P1-PLAN-LOTE-60" in (_BACKEND / "docs" / "plan_pendientes_2026_09_11.md").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-60" in (_BACKEND / "docs" / "plan_agente_lotes_38_43_2026_09_14.md").read_text(encoding="utf-8")
    assert "culinary_golden_refresh.py" in (_BACKEND / "docs" / "culinary_coherence.md").read_text(encoding="utf-8")
    score = (_BACKEND / "scripts" / "culinary_golden_score.py").read_text(encoding="utf-8")
    assert "tooltip-anchor: P1-PLAN-LOTE-60-ADJUDICADOR" in score and "--comparar-maquina" in score
