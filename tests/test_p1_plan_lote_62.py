# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-62 · 2026-09-15] (lote 39 del plan 38-44 · C5 / CUL-P1-05) Lo que el dueño más marcó y la máquina no veía.

Contra la línea base del lote 38 (`maquina_determinista_2026-09-15`, adjudicador estricto, anotación del dueño):
`coccion_faltante` 0/8 → 8/8, `seco_sin_coccion` 3/11 → 11/11, `usa_lo_que_no_esta` 1/9 → 2/9, y ningún hallazgo sobre
los 9 `ok`. Aquí:

(a) V7f `coccion_faltante` (capa 1, warn, knob `MEALFIT_CULINARY_V7F`): dispara en un plato mínimo y nombra el alimento,
    calla con la cocción escrita (también la que llega por pronombre o por la preparación), calla sobre un listo para
    comer, acusa lo que un paso usa ya cocido, da UN hallazgo por comida y el knob apagado lo quita;
(b) V7c amplía la FORMA, no la severidad: «crudo» como «seco» (el plato mínimo del plan: arroz crudo + «incorpora»), y
    «cocina 3 minutos» no cuece una legumbre seca;
(c) V5 mira la frase del alimento, no ±28 caracteres que cruzan a la vecina, y el sofrito que hace la receta no falta en
    la lista;
(d) el instrumento: la RUBRICA trae V7f, `--desde` compara con la línea base del lote 38 y la capa que no se re-corrió cae
    a esa misma columna; el escáner de hoy sostiene las cifras sobre los casos del dueño;
(e) docs, knob documentado y marker ≥ 62.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

import culinary_coherence as cc

_BACKEND = Path(__file__).resolve().parents[1]
_GOLDEN = _BACKEND / "docs" / "culinary_golden_set.json"
_ANOT = _BACKEND / "docs" / "culinary_golden_anotaciones_angelo.json"
_CORPUS = _BACKEND / "scripts" / "data" / "culinary_corpus_2026_09_12.json"
_COL = "maquina_determinista_2026-09-15-lote62"

# Ficha FIJADA en la prueba: la prueba no depende de la base.
_CAT = [
    {"name": "Yuca", "aliases": ["yucas"], "category": "Víveres", "ready_to_eat": False, "prep_methods": ["hervido", "frito"]},
    {"name": "Pechuga de pollo", "aliases": ["pechugas de pollo"], "category": "Proteínas", "ready_to_eat": False,
     "prep_methods": ["a la plancha", "hervido", "horneado"]},
    {"name": "Huevo", "aliases": ["huevos"], "category": "Proteínas", "ready_to_eat": False, "prep_methods": ["hervido", "frito"]},
    {"name": "Jamón de pavo", "aliases": [], "category": "Embutidos", "ready_to_eat": True, "prep_methods": ["ninguno"]},
    {"name": "Lentejas", "aliases": ["lenteja"], "category": "Legumbres", "ready_to_eat": False, "prep_methods": ["hervido"]},
    {"name": "Arroz blanco", "aliases": ["arroz"], "category": "Granos", "ready_to_eat": False, "prep_methods": ["hervido"]},
    {"name": "Arándanos", "aliases": [], "category": "Frutas", "ready_to_eat": True, "prep_methods": ["crudo"]},
    {"name": "Almendras", "aliases": ["almendra"], "category": "Frutos secos", "ready_to_eat": True, "prep_methods": ["crudo"]},
    {"name": "Yogur griego", "aliases": [], "category": "Lácteos", "ready_to_eat": True, "prep_methods": ["ninguno"]},
    {"name": "Cebolla", "aliases": ["cebollas"], "category": "Vegetales", "ready_to_eat": False, "prep_methods": ["crudo", "sofrito"]},
    {"name": "Tomate", "aliases": ["tomates"], "category": "Vegetales", "ready_to_eat": True, "prep_methods": ["crudo"]},
    {"name": "Sofrito", "aliases": [], "category": "Condimentos", "ready_to_eat": True, "prep_methods": ["ninguno"]},
]
_INDEX = cc.build_culinary_index(_CAT)


def _meal(ings, pasos, meal="Almuerzo"):
    return {"meal": meal, "name": "Plato", "ingredients": list(ings), "ingredients_raw": list(ings), "recipe": list(pasos)}


def _v7f(ings, pasos):
    return cc._v7f_coccion_faltante(1, _meal(ings, pasos), _INDEX)


def _v7c(ings, pasos):
    return cc._v7c_seco_sin_coccion(1, _meal(ings, pasos), _INDEX)


def _v5(ings, pasos):
    return cc._v5_paso_usa_lo_que_no_esta(1, _meal(ings, pasos), _INDEX)


def _mod(rel: str, nombre: str):
    spec = importlib.util.spec_from_file_location(nombre, _BACKEND / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def gsc():
    return _mod("scripts/culinary_golden_score.py", "culinary_golden_score_l62")


@pytest.fixture(scope="module")
def rf():
    return _mod("scripts/culinary_golden_refresh.py", "culinary_golden_refresh_l62")


# ─────────────────────────────── (a) V7f

def test_v7f_dispara_en_un_plato_minimo_y_nombra_el_alimento():
    out = _v7f(["200 g de yuca"], ["Calienta la yuca en la plancha 4 minutos, girándola para dorarla.", "Sirve la yuca con cebolla."])
    assert len(out) == 1 and out[0]["check"] == "V7f" and out[0]["food"] == "Yuca", out
    assert "«Yuca»" in out[0]["detail"] and "ningún paso lo cuece" in out[0]["detail"]
    assert out[0]["severity"] == "minor" and out[0]["repairable"] is False, "warn: ni bloquea ni se repara solo"


def test_v7f_calla_con_la_coccion_escrita():
    assert _v7f(["200 g de yuca"], ["Pela la yuca y hiérvela en agua con sal 15 minutos.", "Sirve la yuca con cebolla."]) == []
    # por pronombre en la cláusula siguiente
    assert _v7f(["200 g de yuca"], ["Pela la yuca y córtala en trozos; hiérvelos 20 minutos.", "Sírvela con cebolla."]) == []
    # lo que cuece la preparación cuece lo que lleva
    assert _v7f(["200 g de yuca", "½ cebolla"],
                ["Ralla la yuca y mézclala con la cebolla; forma tortitas y fríelas 4 minutos por lado."]) == []


def test_v7f_el_participio_describe_salvo_el_de_resultado():
    assert [v["food"] for v in _v7f(["200 g de yuca"], ["Sirve la yuca horneada con cebolla."])] == ["Yuca"]
    assert _v7f(["2 huevos", "½ cebolla"],
                ["Bate los huevos con la cebolla; remueve 3 minutos hasta que estén cuajados."]) == []
    assert [v["food"] for v in _v7f(["2 huevos", "½ cebolla"], ["Bate los huevos con la cebolla; sírvelos cuajados."])] == ["Huevo"]


def test_v7f_calla_sobre_un_listo_para_comer():
    assert _v7f(["60 g de jamón de pavo"], ["Coloca el jamón de pavo sobre el pan y sirve."]) == []


def test_v7f_acusa_lo_que_un_paso_usa_ya_cocido():
    out = _v7f(["150 g de pechuga de pollo", "½ cebolla"], ["Desmenuza la pechuga y mézclala con la cebolla picada.", "Sirve."])
    assert len(out) == 1 and out[0]["food"] == "Pechuga de pollo" and "lo usa ya cocido" in out[0]["detail"], out


def test_v7f_un_hallazgo_por_comida_con_los_dos_alimentos():
    out = _v7f(["200 g de yuca", "150 g de pechuga de pollo"],
               ["Desmenuza la pechuga.", "Calienta la yuca 3 minutos en la sartén y sírvela con la pechuga."])
    assert len(out) == 1, out
    assert "«Yuca»" in out[0]["detail"] and "«Pechuga de pollo»" in out[0]["detail"]


def test_v7f_knob_apagado_cero_hallazgos(monkeypatch):
    pasos = ["Calienta la yuca en la plancha 4 minutos.", "Sirve la yuca."]
    plan = {"days": [{"day": 1, "meals": [_meal(["200 g de yuca"], pasos)]}]}
    monkeypatch.setenv("MEALFIT_CULINARY_V7F", "false")
    assert _v7f(["200 g de yuca"], pasos) == []
    assert not [v for v in cc.culinary_contract_scan(plan, _CAT) or [] if v["check"] == "V7f"]
    monkeypatch.setenv("MEALFIT_CULINARY_V7F", "true")
    assert [v for v in cc.culinary_contract_scan(plan, _CAT) or [] if v["check"] == "V7f"]


def test_v7f_knob_registrado_y_en_la_capa_1():
    from knobs import get_knobs_registry_snapshot
    assert cc._v7f_enabled() is True
    assert "MEALFIT_CULINARY_V7F" in get_knobs_registry_snapshot()
    assert "V7f" in cc.CHECKS_CAPA1 and cc.CHECKS_CAPA1.index("V7f") == cc.CHECKS_CAPA1.index("V7e") + 1
    assert cc.CHECKS_CAPA1[-1] == "V9"


# ─────────────────────────────── (b) V7c: la forma, no la severidad

def test_el_plato_minimo_del_plan_arroz_crudo_incorporado_es_de_v7c():
    ings = ["40 g de arroz blanco crudo", "1 tomate"]
    out = _v7c(ings, ["Incorpora el arroz al bol con el tomate picado y sirve."])
    assert [v["food"] for v in out] == ["Arroz blanco"] and out[0]["severity"] == "minor", out
    assert _v7f(ings, ["Incorpora el arroz al bol con el tomate picado y sirve."]) == [], "lo seco es de V7c: sin doble hallazgo"
    assert _v7c(ings, ["Hierve el arroz en agua 15 minutos y sírvelo con el tomate."]) == []


def test_v7c_el_verbo_generico_de_3_minutos_no_cuece_una_legumbre_seca():
    ings = ["½ taza de lentejas secas (100 g)"]
    assert [v["check"] for v in _v7c(ings, ["Agrega las lentejas y cocina 3 minutos."])] == ["V7c"]
    assert _v7c(ings, ["Agrega las lentejas y cocina 25 minutos."]) == []
    assert cc._duracion_max_min("dora 2 minutos por lado") == 4 and cc._duracion_max_min("sin tiempo") is None


def test_v7c_si_ningun_paso_lo_nombra_es_de_v3():
    assert _v7c(["½ taza de lentejas secas", "1 tomate"], ["Sirve el tomate en rodajas."]) == []


# ─────────────────────────────── (c) V5

def test_v5_la_frase_del_alimento_y_no_la_vecina():
    out = _v5(["100 g de arándanos", "150 g de yogur griego"],
              ["Lava los arándanos; separa las almendras fileteadas y sírvelas encima del yogur."])
    assert len(out) == 1 and "almendra" in cc._norm(out[0]["food"]), out


def test_v5_un_hallazgo_por_alimento_y_comida():
    out = _v5(["100 g de arándanos"], ["Separa las almendras.", "Sirve los arándanos con las almendras encima."])
    assert len(out) == 1, out


def test_v5_el_sofrito_que_hace_la_receta_no_falta():
    # Frase AISLADA de la lista a propósito: con la ventana nueva, aquí lo que calla es la exención, no una vecina.
    ings = ["½ cebolla", "40 g de arroz blanco"]
    pasos = ["Sofríe la cebolla 3 minutos; agrega el sofrito y el arroz, y hiérvelo 18 minutos."]
    assert not [v for v in _v5(ings, pasos) if "sofrito" in cc._norm(v["food"])]
    # sin «sofríe» en los pasos, el sofrito sí es un alimento que la lista no trae
    pasos = ["Calienta la cebolla 3 minutos; agrega el sofrito y el arroz, y hiérvelo 18 minutos."]
    assert [v for v in _v5(ings, pasos) if "sofrito" in cc._norm(v["food"])]


# ─────────────────────────────── (d) el instrumento

def test_la_rubrica_trae_v7f(gsc, rf):
    assert gsc.RUBRICA["coccion_faltante"] == {"V7f"}
    assert "V7f" in gsc._CODIGOS_DET and "V7f" in rf.CHECKS_CON_ALIMENTO


def test_desde_y_la_capa_no_recorrida_cae_a_su_columna(gsc):
    d = {"casos": [{"maquina_determinista_A": [], "maquina_juez_A": [], "maquina_determinista_B": []}]}
    antes, _ = gsc.columnas_de(d, "A")
    assert antes == {"determinista": "maquina_determinista_A", "juez": "maquina_juez_A"}
    despues, notas = gsc.columnas_de(d, "B", antes)
    assert despues == {"determinista": "maquina_determinista_B", "juez": "maquina_juez_A"}
    assert any("--desde" in n for n in notas), notas
    assert gsc.columnas_de(d, "B")[0]["juez"] == "maquina_juez", "sin --desde, la del 09-06 como siempre"
    assert '"--desde"' in (_BACKEND / "scripts" / "culinary_golden_score.py").read_text(encoding="utf-8")


def test_la_columna_del_lote_62_esta_escrita_al_lado():
    g = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    assert all(_COL in c and "maquina_determinista_2026-09-15" in c for c in g["casos"]), "la línea base del lote 38, intacta"
    ref = {r["columna"]: r for r in g["refrescos"]}
    assert ref[_COL]["catalogo"]["filas"] > 300 and ref[_COL]["reglas_huella"]
    v7f = [t for c in g["casos"] for t in c[_COL] if t.startswith("V7f: ")]
    assert v7f and all("(alimento: " in t for t in v7f)


def test_el_escaner_de_hoy_sostiene_las_cifras_sobre_los_casos_del_dueno(rf):
    g = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    anot = json.loads(_ANOT.read_text(encoding="utf-8"))["casos"]
    cat = json.loads(_CORPUS.read_text(encoding="utf-8"))["catalogo_filas"]

    def clase(c, k):
        return any(d.get("clase") == k for d in (anot.get(str(c["id"])) or {}).get("defectos") or [])

    coc = [c for c in g["casos"] if clase(c, "coccion_faltante")]
    seco = [c for c in g["casos"] if clase(c, "seco_sin_coccion")]
    ok = [c for c in g["casos"] if (anot.get(str(c["id"])) or {}).get("veredicto") == "ok"]
    assert (len(coc), len(seco), len(ok)) == (8, 11, 9)
    por_caso, _ = rf.escanear(coc + seco + ok, cat)

    def con(casos, cod):
        return [c["id"] for c in casos if any(t.startswith(cod + ": ") for t in por_caso[c["id"]])]

    assert len(con(coc, "V7f")) >= 6, con(coc, "V7f")      # objetivo del plan; hoy 8/8
    assert len(con(seco, "V7c")) >= 10, con(seco, "V7c")   # hoy 11/11
    for c in ok:
        assert not [t for t in por_caso[c["id"]] if t.split(":", 1)[0] in ("V5", "V7c", "V7f")], (c["id"], por_caso[c["id"]])


# ─────────────────────────────── (e) docs, knob y marker

def test_docs_knob_y_marker():
    doc = (_BACKEND / "docs" / "culinary_coherence.md").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-62" in doc and "| `coccion_faltante` (8) | 0/8 | 8/0 |" in doc and "18/60/37" in doc
    knobs = (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    assert "| `MEALFIT_CULINARY_V7F` | `True` |" in knobs
    assert "P1-PLAN-LOTE-62" in (_BACKEND / "docs" / "plan_pendientes_2026_09_11.md").read_text(encoding="utf-8")
    agente = (_BACKEND / "docs" / "plan_agente_lotes_38_43_2026_09_14.md").read_text(encoding="utf-8")
    assert re.search(r"^\| 39 \| C5 / CUL-P1-05 \| ✅ HECHO \(`P1-PLAN-LOTE-62`", agente, re.M)
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'^_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', app, re.M)
    assert m and int(m.group(1)) >= 62 and m.group(2) >= "2026-09-15"
    src = (_BACKEND / "culinary_coherence.py").read_text(encoding="utf-8")
    assert "tooltip-anchor: P1-PLAN-LOTE-62-V7F" in src and '_env_bool("MEALFIT_CULINARY_V7F", True)' in src
