# backend/tests/test_p1_plan_lote_622.py
# [P1-PLAN-LOTE-622 · 2026-09-27] El aviso de comida del coach, en inglés, decía «Your 3:45 merienda is just about
# here—go ahead and merendar now» (producción, 27-sep). El prompt le daba al modelo el NOMBRE y los VERBOS de la comida
# en español («Merienda», «merendar») y la directiva de idioma le manda dejar en español los nombres de ALIMENTOS: el
# modelo trató «merienda» como uno de ellos. El nombre de la comida no es un identificador del motor: va en el idioma
# del usuario. En español no cambia ni un byte.
import inspect
import re

import pytest


LOCALES = ("en-US", "pt-BR", "fr-FR", "it-IT")
COMIDAS = ("Desayuno", "Almuerzo", "Merienda", "Cena")
ESPANOL = re.compile(r"desayun|almorz|almuerzo|merend|merienda|\bcena[rs]?\b|cenaste", re.I)


def test_en_espanol_las_palabras_son_las_de_siempre():
    import proactive_agent as pa
    for meal in COMIDAS:
        p = pa.palabras_de_la_comida(meal, "es-DO")
        assert p == {"missing_meal": meal, "verbo": pa.VERBO_DE_COMIDA[meal],
                     "infinitivo": pa.INFINITIVO_DE_COMIDA[meal]}
    # sin idioma o con uno desconocido: español (nunca lanza)
    assert pa.palabras_de_la_comida("Merienda", None)["infinitivo"] == "merendar"
    assert pa.palabras_de_la_comida("Merienda", "de-DE")["infinitivo"] == "merendar"


@pytest.mark.parametrize("locale", LOCALES)
def test_en_otro_idioma_ni_el_nombre_ni_los_verbos_quedan_en_espanol(locale):
    import proactive_agent as pa
    for meal in COMIDAS:
        p = pa.palabras_de_la_comida(meal, locale)
        assert set(p) == {"missing_meal", "verbo", "infinitivo"}
        for k, v in p.items():
            assert v and isinstance(v, str), (locale, meal, k)
            # la italiana «cena» es italiano; lo que no puede quedar es el verbo ni la comida española
            if locale != "it-IT":
                assert not ESPANOL.search(v), (locale, meal, k, v)


def test_en_ingles_la_merienda_es_un_snack():
    import proactive_agent as pa
    p = pa.palabras_de_la_comida("Merienda", "en-US")
    assert "snack" in p["missing_meal"] and "snack" in p["infinitivo"]


def test_el_prompt_del_aviso_usa_las_palabras_en_su_idioma():
    import proactive_agent as pa
    src = inspect.getsource(pa)
    llamada = src[src.index("prompt = PROACTIVE_PROMPT.format("):]
    llamada = llamada[:llamada.index(")\n")]
    assert "palabras_de_la_comida(meal_to_check, _nudge_locale)" in llamada
    # los tres huecos salen de la misma función, no de las tablas en español
    assert "VERBO_DE_COMIDA.get" not in llamada and "INFINITIVO_DE_COMIDA.get" not in llamada


def test_el_prompt_en_ingles_no_lleva_la_merienda():
    from prompts.proactive import PROACTIVE_PROMPT
    import proactive_agent as pa
    prompt = PROACTIVE_PROMPT.format(**pa.palabras_de_la_comida("Merienda", "en-US"), trigger_time="15:45",
                                     diet_type="balanced", goals="lose fat", tone_instruction="", style_instruction="")
    # la única «merienda» que queda es el ejemplo de lo que NO se hace («cenar tu merienda»)
    assert "merendar" not in prompt
    assert prompt.lower().count("merienda") == 1
