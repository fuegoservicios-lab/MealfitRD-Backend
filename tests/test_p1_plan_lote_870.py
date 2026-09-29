"""[P1-PLAN-LOTE-870 · 2026-09-29] La ola de países (850-855) como un TODO: ninguna superficie beta vuelve a hablar en
dominicano y la República Dominicana no cambia ni un byte.

La validación beta con 6 planes reales (G24, 29-sep) mostró que el vocabulario dominicano no entraba por UN sitio sino
por cinco a la vez: la asignación previa de carbos y técnicas y el prompt (850), los productos del súper de RD en la
lista (852), los nombres del catálogo a la vista (853), el metalenguaje del prompt en la descripción (854) y los
instrumentos que contaban «res» dentro de «fresco» (855). Cada lote tiene su test; éste ancla las cinco costuras juntas
para que un lote futuro que reabra UNA no pase por verde porque las otras cuatro siguen cerradas, y fija el control:
con TODO encendido, el prompt, la lista y la vista de un plan dominicano son los de antes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_BETA = ("ES", "US", "MX", "CO")          # PR comparte léxico criollo a propósito (habichuelas, guineo): test propio en 850
_CULTURA = {"DO": "dominican_criolla", "ES": "spain_mediterranea", "US": "us_everyday", "MX": "mexico_casera",
            "PR": "puertorico_criolla", "CO": "colombia_casera"}
_KNOBS_DE_LA_OLA = ("MEALFIT_BETA_CULTURAL_ASSIGNMENT", "MEALFIT_BETA_NO_DO_SUPERMARKET_PRODUCTS",
                    "MEALFIT_BETA_METRIC_PACKAGE_LABELS", "MEALFIT_COUNTRY_DISPLAY_LEXICON",
                    "MEALFIT_DESCRIPTION_TRUTH", "MEALFIT_CROSS_DAY_PROTEIN_TOKEN_MATCH")


@pytest.fixture
def ola(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    for k in _KNOBS_DE_LA_OLA + ("MEALFIT_MARKET_POOL_UNIVERSAL", "MEALFIT_CULTURAL_PROFILES",
                                 "MEALFIT_DESCRIPTION_TRUTH_DO"):
        monkeypatch.delenv(k, raising=False)
    yield monkeypatch


def _form(cc):
    return {"country": cc, "cultureProfiles": {"main": _CULTURA[cc], "secondary": []}, "dietType": "balanced",
            "allergies": ["Ninguna"], "dislikes": ["Ninguno"], "cookingTime": "30min", "mainGoal": "maintenance",
            "budget": "medium", "gender": "female", "age": "30",
            "user_id": "749a56da-5817-4921-9df2-08b6ae28f7f6"}


def _superficies_de_prompt(cc):
    """El texto que el modelo lee de la cocina, por las dos puertas que G24 midió: el prompt de sistema del generador
    de días y el bloque de micronutrientes del día."""
    import graph_orchestrator as go
    from micronutrients import build_micronutrient_targets_directive as micros
    return {"sistema": go._day_system_instruction_for_diet(_form(cc)),
            "micros": micros(sex="female", age=30, cocina=cc)}


# ─── 1. ninguna superficie beta habla en dominicano ──────────────────────────────────────────────────────────────────
# Lo que SÍ puede quedar: la PROHIBICIÓN («los platos dominicanos NO son requisito ni default», es el ancla de F1) y
# la regla de técnica del casabe (P1-CASABE-NO-BOIL: dice cómo NO cocinarlo; el 850 la dejó fuera de alcance a
# propósito). Lo que no puede volver es un EMPUJÓN: «comidas dominicanas», el mangú, las habichuelas, el guineo.
_PERMITIDO = (r"los platos dominicanos NO son requisito ni default\.?",
              r"TÉCNICA CORRECTA POR ALIMENTO \[P1-CASABE-NO-BOIL[^\n]*")


@pytest.mark.parametrize("cc", _BETA)
def test_el_prompt_beta_no_pide_comida_dominicana(ola, cc):
    import re
    for nombre, texto in _superficies_de_prompt(cc).items():
        for rx in _PERMITIDO:
            texto = re.sub(rx, "", texto)
        t = re.sub(r"\n  - habichuela: [^\n]*", "", texto.lower())   # la tabla de macros del MOCK es un dato
        for w in ("dominican", "mangú", "casabe", "habichuela", "guineo", "gandul"):
            assert w not in t, (cc, nombre, w, t[max(0, t.find(w) - 100): t.find(w) + 60])


def _lista_saneada(cc):
    """La cola del agregador (`sanear_lista_beta`) dentro del país de la lista, como la corre `envase_pais`."""
    import lista_sin_super_rd as ls
    from envase_pais import lista_de_pais
    items = [{"name": "Sal", "brand_product_id": "do-123", "market_pkg_price_rd": 25.0}]
    with lista_de_pais(cc):
        ls.sanear_lista_beta(items)
    return items[0]


@pytest.mark.parametrize("cc", _BETA + ("PR",))
def test_la_lista_beta_no_lleva_productos_del_super_de_rd(ola, cc):
    it = _lista_saneada(cc)
    assert "brand_product_id" not in it and "market_pkg_price_rd" not in it, (cc, it)


def test_do_la_lista_conserva_su_supermercado(ola):
    it = _lista_saneada("DO")
    assert it.get("brand_product_id") == "do-123" and it.get("market_pkg_price_rd") == 25.0, it


# ─── 2. el control: RD con TODO encendido es la RD de antes ──────────────────────────────────────────────────────────
def test_do_el_prompt_de_sistema_es_el_de_siempre(ola):
    import graph_orchestrator as go
    assert go._day_system_instruction_for_diet(_form("DO")) == go._DAY_SYSTEM_INSTRUCTION_CACHED


def test_do_el_bloque_de_micros_es_el_de_siempre(ola):
    from micronutrients import build_micronutrient_targets_directive as micros
    assert micros(sex="female", age=30, cocina="DO") == micros(sex="female", age=30)


def test_do_la_vista_no_toca_nada(ola):
    import lexico_vista_pais as lv
    texto = "Guineo maduro con habichuelas y lechosa, queso blanco y ají morrón en una funda."
    for fn in ("localizar", "localizar_texto", "vista"):
        f = getattr(lv, fn, None)
        if callable(f):
            try:
                assert f(texto, "DO") == texto
            except TypeError:
                continue
            return
    pytest.skip("lexico_vista_pais no expone una función de texto con (texto, país)")


def test_do_la_descripcion_se_encendio_aparte_con_su_palanca(ola):
    """Esta ola dejó la descripción veraz APAGADA en RD; la encendió aparte el lote 858/890, tras leer el replay de DO.
    Lo que este test fija es que en RD sigue habiendo una palanca propia para volver a apagarla sin redeploy."""
    import descripcion_veraz as dv
    assert dv.incluye_do() is True, "RD encendido por defecto desde el lote 858 (marcador 890)"
    ola.setenv("MEALFIT_DESCRIPTION_TRUTH_DO", "false")
    assert dv.incluye_do() is False, "MEALFIT_DESCRIPTION_TRUTH_DO=false vuelve a dejar RD como antes"


# ─── 3. cada costura con su palanca de vuelta atrás, registrada y documentada ───────────────────────────────────────
def test_las_palancas_de_la_ola_estan_documentadas():
    doc = (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    faltan = [k for k in _KNOBS_DE_LA_OLA if k not in doc]
    assert not faltan, faltan


def test_marker():
    src = Path(__file__).read_text(encoding="utf-8")
    assert "[P1-PLAN-LOTE-870 · 2026-09-29]" in src
