# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-853 · 2026-09-29] Un español, un mexicano o un colombiano leen su palabra, no la dominicana.

G24 (29-sep, 6 planes reales con el código de P1-PLAN-LOTE-815): «guineo», «lechosa», «auyama», «ají morrón», «queso
blanco», «habichuelas» y «funda» salían en nombres, descripciones, pasos y lista de ES/MX/CO/US. Son NOMBRES DEL
CATÁLOGO —el identificador con que resuelven la Nevera, el guard de coherencia y el backstop de alergias—, así que el
modelo está obligado a usarlos y el plan (`plan_data`) no se toca. El lote 649 los GLOSA al leer («guineo (plátano)»);
este lote los SUSTITUYE al pintar, por país de mercado y con un léxico que es DATA (`data/lexico_vista_pais.json`):

  - solo si la frase sigue siendo gramatical: si el género cambia (habichuelas→frijoles) se concuerdan el determinante
    y los adjetivos vecinos, y si algo más adelante sigue refiriéndose a la palabra («añádelas») se glosa, como el 649;
  - «funda» solo como envase de la lista («1 funda (1 Lb)»): en un paso es el verbo («que el queso funda»);
  - donde el 649 glosa el mismo nombre del catálogo, dicen lo mismo;
  - RD no cambia NADA, y el knob `MEALFIT_COUNTRY_DISPLAY_LEXICON` apagado devuelve el texto tal cual.

La tabla vive aquí (SSOT) y el frontend la lleva en espejo (`src/data/lexicoVistaPais.json`); los `casos` del JSON
los corren las dos implementaciones (este test y `lexicoDelPais.p1_plan_lote_853.test.js`).

tooltip-anchor: P1-PLAN-LOTE-853
"""
import copy
import json
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_DATOS = _BACKEND / "data" / "lexico_vista_pais.json"
_ESPEJO = _BACKEND.parent / "frontend" / "src" / "data" / "lexicoVistaPais.json"


@pytest.fixture(autouse=True)
def _knob_encendido(monkeypatch):
    monkeypatch.delenv("MEALFIT_COUNTRY_DISPLAY_LEXICON", raising=False)


def _lx():
    import lexico_vista_pais
    return lexico_vista_pais


# ── los datos ─────────────────────────────────────────────────────────────────────────────────────────────────────

def test_paises_beta_y_nunca_rd():
    from constants import COUNTRY_PROFILES
    datos = json.loads(_DATOS.read_text(encoding="utf-8"))
    assert set(datos["paises"]) == {"ES", "MX", "CO", "US", "PR"}
    assert set(datos["paises"]) <= set(COUNTRY_PROFILES)
    assert "DO" not in datos["paises"]
    for pais, filas in datos["paises"].items():
        for f in filas:
            assert len(f["de"]) == 2 and len(f["a"]) == 2, (pais, f)
            assert f["genero"][0] in "mf" and f["genero"][1] in "mf", (pais, f)
            assert f.get("ambito", "texto") in ("texto", "envase"), (pais, f)
            if f["genero"][0] != f["genero"][1]:
                assert f"{f['genero'][0]}>{f['genero'][1]}" in datos["concordancia"], (pais, f)


def test_dice_lo_mismo_que_la_glosa_del_649():
    """Donde el 649 glosa un nombre del catálogo, la sustitución da ese mismo nombre (sin mayúsculas)."""
    from food_names_i18n import nombres_por_pais
    lx = _lx()
    comprobados = 0
    for pais, tabla in nombres_por_pais().items():
        for canon, local in tabla.items():
            leido = lx.localizar_texto(canon, pais)
            if leido != canon:
                comprobados += 1
                assert leido.lower() == local.lower(), (pais, canon, leido, local)
    assert comprobados >= 20


def test_el_espejo_del_frontend_es_identico():
    if not _ESPEJO.exists():
        pytest.skip("el frontend de este checkout aún no trae el espejo (el CI del backend clona su `main`)")
    assert json.loads(_ESPEJO.read_text(encoding="utf-8")) == json.loads(_DATOS.read_text(encoding="utf-8"))


def test_los_casos_compartidos_con_el_frontend():
    lx = _lx()
    casos = json.loads(_DATOS.read_text(encoding="utf-8"))["casos"]
    assert len(casos) >= 20
    fallos = []
    for c in casos:
        f = {"texto": lx.texto_para_leer, "lista": lx.nombre_de_lista_para_leer,
             "envase": lx.envase_para_leer}[c["funcion"]]
        got = f(c["entrada"], c["pais"], *([c["siguientes"]] if "siguientes" in c else []))
        if got != c["lectura"]:
            fallos.append((c["pais"], c["entrada"], got, c["lectura"]))
    assert not fallos, fallos


# ── la sustitución, con criterio ─────────────────────────────────────────────────────────────────────────────────

def test_las_palabras_de_g24_por_pais():
    lx = _lx()
    assert lx.texto_para_leer("½ lechosa mediana (405g)", "MX") == "½ papaya mediana (405g)"
    assert lx.texto_para_leer("Batido cremoso de guineo, manzana y leche de soya", "MX") == \
        "Batido cremoso de plátano, manzana y leche de soya"
    assert lx.texto_para_leer("Batido cremoso de guineo, manzana y leche de soya", "ES") == \
        "Batido cremoso de plátano, manzana y leche de soja"
    assert lx.texto_para_leer("pela el guineo cortándolo en rodajas", "CO") == "pela el banano cortándolo en rodajas"
    assert lx.texto_para_leer("Auyama guisada con huevo", "CO") == "Ahuyama guisada con huevo"
    assert lx.texto_para_leer("Auyama guisada con huevo", "ES") == "Calabaza guisada con huevo"
    assert lx.texto_para_leer("pica ½ ají morrón y 1 tomate", "ES") == "pica ½ pimiento y 1 tomate"
    assert lx.texto_para_leer("pica ½ ají morrón y 1 tomate", "MX") == "pica ½ pimiento morrón y 1 tomate"
    assert lx.texto_para_leer("pica ½ ají morrón y 1 tomate", "CO") == "pica ½ pimentón y 1 tomate"
    assert lx.texto_para_leer("desmenuza 60 g de queso blanco", "CO") == "desmenuza 60 g de queso campesino"
    assert lx.texto_para_leer("Mango con queso blanco fresco", "ES") == "Mango con queso fresco"


def test_la_caja_se_respeta():
    lx = _lx()
    assert lx.texto_para_leer("Atún a la Criolla con Apio, Ají Morrón y Sofrito", "MX") == \
        "Atún a la Criolla con Apio, Pimiento Morrón y Sofrito"
    assert lx.texto_para_leer("Atún a la Criolla con Apio, Ají Morrón y Sofrito", "ES") == \
        "Atún a la Criolla con Apio, Pimiento y Sofrito"
    assert lx.texto_para_leer("GUINEO", "ES") == "PLÁTANO"


def test_el_platano_de_espana_no_choca_con_el_platano_dominicano():
    lx = _lx()
    # el plátano de cocinar dominicano se lee «macho»; el guineo, «plátano»; nada se encadena
    assert lx.texto_para_leer("1 guineo verde y ½ plátano verde", "ES") == "1 plátano verde y ½ plátano macho verde"
    assert lx.texto_para_leer("Tortitas de quinoa y plátano", "ES") == "Tortitas de quinoa y plátano"


def test_habichuelas_a_frijoles_concuerda_o_glosa():
    lx = _lx()
    assert lx.nombre_de_lista_para_leer("Habichuelas negras", "MX") == "Frijoles negros"
    assert lx.nombre_de_lista_para_leer("Habichuelas rojas", "CO") == "Fríjoles rojos"
    assert lx.nombre_de_lista_para_leer("Habichuelas rojas", "ES") == "Alubias rojas"
    assert lx.texto_para_leer("Añade las habichuelas rojas cocidas y el cilantro.", "MX") == \
        "Añade los frijoles rojos cocidos y el cilantro."
    assert lx.texto_para_leer("sirve con una habichuela guisada", "US") == "sirve con un frijol guisado"
    # algo más adelante sigue hablando de ellas en femenino: no se sustituye, se glosa (como el 649)
    assert lx.texto_para_leer("Escurre las habichuelas y májalas con el tenedor.", "MX") == \
        "Escurre las habichuelas (frijoles) y májalas con el tenedor."
    # en Puerto Rico también se dice habichuelas
    assert lx.texto_para_leer("Añade las habichuelas rojas cocidas.", "PR") == "Añade las habichuelas rojas cocidas."


def test_funda_solo_como_envase_de_la_lista():
    lx = _lx()
    assert lx.envase_para_leer("1 funda (1 Lb)", "ES") == "1 bolsa (1 Lb)"
    assert lx.envase_para_leer("2 fundas (Selecto 1 Lb · Wala c/u)", "MX") == "2 bolsas (Selecto 1 Lb · Wala c/u)"
    assert lx.envase_para_leer("1 funda (1 Lb)", "PR") == "1 funda (1 Lb)"
    assert lx.envase_para_leer("1 funda (1 Lb)", "DO") == "1 funda (1 Lb)"
    # el verbo fundir no es un envase
    assert lx.texto_para_leer("tapa 2 minutos más para que el queso funda.", "CO") == \
        "tapa 2 minutos más para que el queso funda."


def test_con_la_glosa_del_649_en_un_solo_paso():
    lx = _lx()
    # la batata no está en el léxico (batata→boniato cambia de género): el 649 la sigue glosando
    assert lx.texto_para_leer("Wok de pollo con batata, auyama y ají morrón", "ES") == \
        "Wok de pollo con batata (boniato), calabaza y pimiento"


def test_rd_y_otros_paises_no_cambian_nada():
    lx = _lx()
    for pais in ("DO", None, "", "XX"):
        for t in ("1 guineo mediano", "Añade las habichuelas", "1 funda (1 Lb)", "60 g de queso blanco"):
            assert lx.texto_para_leer(t, pais) == t
            assert lx.envase_para_leer(t, pais) == t
            assert lx.nombre_de_lista_para_leer(t, pais) == t


def test_el_knob_apagado_deja_la_conducta_del_649(monkeypatch):
    lx = _lx()
    monkeypatch.setenv("MEALFIT_COUNTRY_DISPLAY_LEXICON", "false")
    assert lx.activo() is False
    assert lx.texto_para_leer("1 guineo mediano", "ES") == "1 guineo (plátano) mediano"
    assert lx.envase_para_leer("1 funda (1 Lb)", "ES") == "1 funda (1 Lb)"
    assert lx.nombre_de_lista_para_leer("Lechosa", "MX") == "Lechosa"


def test_el_knob_esta_registrado_y_documentado():
    from knobs import _KNOBS_REGISTRY
    _lx().activo()
    assert _KNOBS_REGISTRY["MEALFIT_COUNTRY_DISPLAY_LEXICON"]["default"] is True
    doc = (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    assert "MEALFIT_COUNTRY_DISPLAY_LEXICON" in doc and "VITE_COUNTRY_DISPLAY_LEXICON" in doc


# ── display-only: el plan nunca se toca ──────────────────────────────────────────────────────────────────────────

def test_la_comida_para_leer_es_una_copia():
    lx = _lx()
    meal = {"name": "Batido de guineo y lechosa", "desc": "Con guineo.", "meal": "Desayuno",
            "ingredients": ["1 guineo", "½ lechosa"], "recipe": ["Licúa el guineo y la lechosa."],
            "ingredients_raw": ["1 guineo", "½ lechosa"]}
    antes = copy.deepcopy(meal)
    leida = lx.comida_para_leer(meal, "MX")
    assert meal == antes
    assert leida is not meal
    assert leida["name"] == "Batido de plátano y papaya"
    assert leida["ingredients"] == ["1 plátano", "½ papaya"]
    assert leida["recipe"] == ["Licúa el plátano y la papaya."]
    # lo que no se pinta (y lo que el motor lee) no se toca
    assert leida["ingredients_raw"] == ["1 guineo", "½ lechosa"]
    assert lx.comida_para_leer(meal, "DO") is meal


def test_los_pasos_se_leen_con_los_siguientes():
    """[ronda 1 del revisor] Un paso nombra las habichuelas y el SIGUIENTE vuelve sobre ellas («Májalas»): leerlo
    solo sustituía y dejaba «Escurre los frijoles» + «Májalas». Los pasos se leen con los que vienen detrás, hasta
    que vuelven a nombrarlas (desde ahí el pronombre es de esa mención)."""
    lx = _lx()
    meal = {"name": "Habichuelas majadas", "recipe": ["Escurre las habichuelas rojas.", "Májalas con un tenedor."],
            "ingredients": ["1 taza de habichuelas rojas", "2 cucharadas de agua"]}
    leida = lx.comida_para_leer(meal, "MX")
    assert leida["recipe"] == ["Escurre las habichuelas rojas (frijoles rojos).", "Májalas con un tenedor."]
    # los ingredientes son renglones sueltos: cada uno se lee solo
    assert leida["ingredients"] == ["1 taza de frijoles rojos", "2 cucharadas de agua"]
    assert leida["name"] == "Frijoles majados"


def test_la_doc_no_promete_lo_que_la_heuristica_no_cumple():
    """[ronda 1 del revisor] «Ante la duda glosa, el peor caso es la conducta del 649» era falso: la concordancia es
    una heurística. La doc dice sus límites medidos en vez de prometer."""
    doc = (_BACKEND / "docs" / "lexico_vista_pais.md").read_text(encoding="utf-8")
    assert "peor caso es la conducta del 649" not in doc
    assert "Límites conocidos" in doc
