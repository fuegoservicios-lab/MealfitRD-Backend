# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-524 · 2026-09-27] El duradero que el plato YA traía no se repite tras la sustitución de compra única.

Replay forzado de los días 21+ (322 planes, 3.519 comidas con sustitución): en ~620 el duradero quedaba DOS veces en la
lista y el paso lo nombraba dos veces — «½ zanahoria» + «170 g de zanahoria», «1 cdta de orégano» + «Orégano
dominicano al gusto», «65 g de casabe» + «1½ tortas pequeñas de casabe»; «añade las sardinas, la zanahoria, la
zanahoria y la manzana» (adulto mayor con HTA, día 1). Ahora la nueva se suma a la que había (misma medida o gramos del
catálogo), una «al gusto» cede su sitio a la que dice cuánto y el texto nombra el alimento una vez.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import compra_unica  # noqa: E402
import sustitucion_fresca as sf  # noqa: E402

_GRAMOS = {"½ zanahoria rallada": 30.0, "170 g de zanahoria": 170.0, "65 g de casabe": 65.0,
           "1½ tortas pequeñas de casabe": 45.0, "125 g de manzana": 125.0, "1 cda de vinagre de manzana": 15.0}


@pytest.fixture(autouse=True)
def _gramos_del_catalogo(monkeypatch):
    monkeypatch.setattr(compra_unica, "_gramos_de_linea", lambda t: _GRAMOS.get(str(t)))


def _lineas(meal, palabra):
    return [x for x in meal["ingredients"] if palabra in sf._sa(x)]


def test_bowl_del_adulto_mayor_una_zanahoria_y_un_oregano():
    # batería real del 27-sep (adulto mayor con HTA, día 1): pepino y tomate → zanahoria; cilantro → orégano
    m = {"name": "Bowl caribeño de sardinas con yautía y vegetales frescos",
         "desc": "Sardinas sobre yautía tierna, con pepino, tomate y limón para un almuerzo fresco.",
         "ingredients": ["150 g de sardinas en lata", "¾ pedazo de yautía (≈188 g)", "½ pepino", "1 tomate mediano",
                         "½ limón", "1 cda de cilantro picado", "¾ cdta de aceite de oliva", "Ajo",
                         "Orégano dominicano al gusto"],
         "recipe": ["Mise en place: pela y corta ¾ pedazo de yautía en trozos; lava y corta ½ pepino y 1 tomate mediano; "
                    "exprime ½ limón; pica 1 cda de cilantro; sazona 150 g de sardinas en lata con ajo y orégano "
                    "dominicano.",
                    "Montaje: coloca la yautía en un bowl, añade las sardinas, el pepino y el tomate; termina con el jugo "
                    "de limón y el cilantro. Acompaña con agua."]}
    sf.sustituir_en_plato(m, 2, "½ pepino", "100 g de zanahoria", "zanahoria")
    sf.sustituir_en_plato(m, 3, "1 tomate mediano", "150 g de zanahoria", "zanahoria")
    sf.sustituir_en_plato(m, 5, "1 cda de cilantro picado", "1 cdta de orégano", "oregano")
    assert _lineas(m, "zanahoria") == ["250 g de zanahoria"], m["ingredients"]
    assert _lineas(m, "oregano") == ["1 cdta de orégano"], m["ingredients"]
    texto = " ".join(m["recipe"]).lower()
    assert "la zanahoria, la zanahoria" not in texto and "la zanahoria y la zanahoria" not in texto, m["recipe"]
    assert "100 g de zanahoria y 150 g de zanahoria" not in texto, m["recipe"]
    assert texto.count("250 g de zanahoria") == 1, m["recipe"]


def test_la_zanahoria_que_ya_estaba_se_suma_en_gramos():
    m = {"ingredients": ["3 huevos", "1 taza de repollo rallado", "½ zanahoria rallada", "170 g de pepino",
                         "½ cebolla picada"],
         "ingredients_raw": ["3 huevos", "1 taza de repollo rallado", "½ zanahoria rallada", "170 g de pepino",
                             "½ cebolla picada"],
         "recipe": ["Mise en place: lava y corta 1 taza de repollo, ½ zanahoria, 170 g de pepino en cubos y ½ cebolla.",
                    "Montaje: mezcla los huevos con el repollo, la zanahoria, el pepino, la cebolla y la pimienta."]}
    sf.sustituir_en_plato(m, 3, "170 g de pepino", "170 g de zanahoria", "zanahoria")
    assert _lineas(m, "zanahoria") == ["200 g de zanahoria rallada"], m["ingredients"]
    assert [x for x in m["ingredients_raw"] if "zanahoria" in x] == ["200 g de zanahoria rallada"], m["ingredients_raw"]
    mise, montaje = m["recipe"]
    assert "200 g de zanahoria rallada" in mise and mise.count("zanahoria") == 1, mise
    assert montaje.count("zanahoria") == 1, montaje


def test_casabe_en_tortas_se_suma_al_casabe_del_plato():
    m = {"ingredients": ["2 huevos", "65 g de casabe", "1 rebanada de pan integral"],
         "recipe": ["Montaje: sirve los huevos con el casabe y el pan integral."]}
    sf.sustituir_en_plato(m, 2, "1 rebanada de pan integral", "1½ tortas pequeñas de casabe", "casabe")
    assert _lineas(m, "casabe") == ["110 g de casabe"], m["ingredients"]


def test_misma_cuchara_o_taza_se_suma():
    # rd10 (adulto mayor con HTA): «1 cdta de orégano» + «½ cdta de orégano dominicano»; familia de 4: repollo 3 + 2 tazas
    m = {"ingredients": ["150 g de sardinas en lata", "1 cda de cilantro picado", "½ cdta de orégano dominicano"],
         "recipe": ["Montaje: termina con el cilantro y el orégano dominicano."]}
    sf.sustituir_en_plato(m, 1, "1 cda de cilantro picado", "1 cdta de orégano", "oregano")
    assert _lineas(m, "oregano") == ["1½ cdtas de orégano dominicano"], m["ingredients"]
    m = {"ingredients": ["3 tazas de repollo", "2 tazas de lechuga"], "recipe": ["Montaje: mezcla el repollo y la lechuga."]}
    sf.sustituir_en_plato(m, 1, "2 tazas de lechuga", "2 tazas de repollo", "repollo")
    assert _lineas(m, "repollo") == ["5 tazas de repollo"], m["ingredients"]


def test_cucharada_y_cucharadita_se_suman_en_cucharaditas():
    # rd227b (rechaza pescado) y rd11 (embarazo): antes salían «2 g de orégano» y «1 pizca de orégano dominicano»
    m = {"ingredients": ["130 g de atún en agua", "1 cda de cilantro picado", "½ cdta de orégano"],
         "recipe": ["Mise en place: mide 1 cda de cilantro y ½ cdta de orégano."]}
    sf.sustituir_en_plato(m, 1, "1 cda de cilantro picado", "1 cda de orégano", "oregano")
    assert _lineas(m, "oregano") == ["3½ cdtas de orégano"], m["ingredients"]
    m = {"ingredients": ["2 huevos", "1 cda de cilantro picado", "¼ cucharadita de orégano dominicano"],
         "recipe": ["Montaje: termina con el cilantro."]}
    sf.sustituir_en_plato(m, 1, "1 cda de cilantro picado", "¾ cdta de orégano", "oregano")
    assert _lineas(m, "oregano") == ["1 cdta de orégano dominicano"], m["ingredients"]


def test_la_linea_en_gramos_no_lleva_tamano_ni_plural_y_el_paso_conserva_su_corte():
    _GRAMOS.update({"1 zanahoria mediana": 80.0, "200 g de zanahoria": 200.0, "3 zanahorias": 240.0,
                    "250 g de zanahoria": 250.0})
    # rd252 (embarazo): «corta 280 g de zanahoria mediana finas»
    m = {"ingredients": ["120 g de sardinas en lata", "1 zanahoria mediana", "200 g de pepino"],
         "recipe": ["Mise en place: corta 200 g de pepino y 1 zanahoria en rodajas finas.",
                    "Montaje: mezcla el pepino y la zanahoria con el limón."]}
    sf.sustituir_en_plato(m, 2, "200 g de pepino", "200 g de zanahoria", "zanahoria")
    assert _lineas(m, "zanahoria") == ["280 g de zanahoria"], m["ingredients"]
    assert "corta 280 g de zanahoria en rodajas finas." in m["recipe"][0], m["recipe"]
    # rd230 (gastritis): «490 g de zanahorias»
    m = {"ingredients": ["3 zanahorias", "250 g de pepino"], "recipe": ["Montaje: sirve la zanahoria con el pepino."]}
    sf.sustituir_en_plato(m, 1, "250 g de pepino", "250 g de zanahoria", "zanahoria")
    assert _lineas(m, "zanahoria") == ["490 g de zanahoria"], m["ingredients"]


def test_dos_preparaciones_conservan_sus_dos_lineas():
    # rd240/rd2 (rechaza pescado): «ralla 1 taza de repollo y ½ zanahoria; corta 100 g de zanahoria» con la lista en
    # «140 g de zanahoria»: el paso contaba ½ zanahoria + 140 g
    _GRAMOS.update({"½ zanahoria": 40.0, "100 g de zanahoria": 100.0})
    m = {"ingredients": ["130 g de atún en agua", "1 taza de repollo", "½ zanahoria", "100 g de pepino"],
         "recipe": ["Mise en place: ralla 1 taza de repollo y ½ zanahoria; corta 100 g de pepino en rodajas.",
                    "Montaje: añade el repollo, la zanahoria y el pepino."]}
    lista_del_llamador = m["ingredients"]         # el bucle de `_single_trip_fresh_substitute` guarda esta referencia
    sf.sustituir_en_plato(m, 3, "100 g de pepino", "100 g de zanahoria", "zanahoria")
    assert m["ingredients"] is lista_del_llamador
    assert _lineas(m, "zanahoria") == ["½ zanahoria", "100 g de zanahoria"], m["ingredients"]
    assert "½ zanahoria; corta 100 g de zanahoria" in m["recipe"][0], m["recipe"]
    assert m["recipe"][1].count("zanahoria") == 1, m["recipe"]


def test_la_coordinacion_conserva_su_y():
    t = "Montaje: acompaña con el repollo morado, la zanahoria y la zanahoria. Termina con limón."
    assert sf._sin_repetir(t, "zanahoria") == "Montaje: acompaña con el repollo morado y la zanahoria. Termina con limón."
    t = "Tostadas integrales con huevo y manzana, con manzana"
    assert sf._sin_repetir(t, "manzana") == "Tostadas integrales con huevo y manzana"


def test_el_vinagre_de_manzana_no_es_la_manzana():
    m = {"ingredients": ["1 cda de vinagre de manzana", "½ aguacate", "2 tazas de lechuga"],
         "recipe": ["Montaje: mezcla la lechuga con el aguacate y el vinagre de manzana."]}
    sf.sustituir_en_plato(m, 1, "½ aguacate", "125 g de manzana", "manzana")
    assert _lineas(m, "manzana") == ["1 cda de vinagre de manzana", "125 g de manzana"], m["ingredients"]


def test_sin_gramos_queda_la_que_habia_y_sus_macros(monkeypatch):
    monkeypatch.setattr(compra_unica, "_gramos_de_linea", lambda t: None)

    class _Db:
        def macros_from_ingredient_string(self, t):
            return {"protein": 1, "carbs": 16, "fats": 0, "kcal": 70} if "zanahoria" in t else \
                {"protein": 1, "carbs": 6, "fats": 0, "kcal": 26}

    m = {"ingredients": ["½ zanahoria rallada", "170 g de pepino"], "protein": 10, "carbs": 40, "fats": 5, "cals": 245,
         "recipe": ["Montaje: mezcla la zanahoria y el pepino."]}
    sf.sustituir_en_plato(m, 1, "170 g de pepino", "170 g de zanahoria", "zanahoria", db=_Db())
    assert m["ingredients"] == ["½ zanahoria rallada"], m["ingredients"]
    assert m["carbs"] == 40 - 6 and m["cals"] == 245 - 26, (m["carbs"], m["cals"])
    assert m["recipe"][0].lower().count("zanahoria") == 1, m["recipe"]
