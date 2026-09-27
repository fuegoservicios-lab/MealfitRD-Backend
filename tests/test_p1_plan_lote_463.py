# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-463 · 2026-09-27] La sustitución de la compra única: «la carne», el corte que sigue y el pronombre
que la temperatura escondía.

Leído en el replay forzado del 460 (322 planes): «corta la carne de res en piezas de parrilla» salía «escurre el atún de
parrilla»; «cocina la carne… hasta alcanzar 63 °C en el centro y déjala reposar» dejaba «déjala» (el pronombre quedaba
fuera de alcance por «el centro», que no es un alimento); «sazona la carne y las espinacas» seguía nombrando la carne; y
«sirve la res en rebanadas» cortaba en rebanadas un atún en lata.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import sustitucion_fresca as sf  # noqa: E402


def _plato():
    return {"name": "Res a la parrilla con batata asada y espinacas al ajo",
            "ingredients": ["190 g de atún en agua", "1¼ batatas medianas", "2½ tazas de espinacas"],
            "recipe": [
                "Mise en place: corta la carne de res en piezas de parrilla, lava y corta 275 g de batata en gajos, "
                "rebana ½ cebolla.",
                "El Toque de Fuego: asa la batata en una parrilla o sartén grill tapada a fuego medio-alto durante 18-22 "
                "min, volteándola; cocina la carne de res a la parrilla a fuego medio-alto hasta alcanzar 63 °C en el "
                "centro y déjala reposar 3 min. Saltea la cebolla y el ajo con el aceite de oliva a fuego medio durante 2 "
                "min, agrega las espinacas y cocina 2-3 min; sazona la carne y las espinacas con Sal al gusto.",
                "Montaje: sirve la res en rebanadas junto a los gajos de batata y las espinacas salteadas."]}


def test_la_carne_el_corte_y_el_pronombre():
    m = _plato()
    sf.reescribir_plato(m, "190 g de carne de res magra", "190 g de atún en agua", "atun en agua")
    assert m["name"] == "Atún con batata asada y espinacas al ajo", m["name"]
    assert m["recipe"][0] == ("Mise en place: escurre el atún, lava y corta 275 g de batata en gajos, rebana ½ "
                              "cebolla."), m["recipe"][0]
    assert "calienta el atún a fuego medio-alto y déjalo reposar 3 min." in m["recipe"][1], m["recipe"][1]
    assert "sazona el atún y las espinacas" in m["recipe"][1], m["recipe"][1]
    assert m["recipe"][2] == "Montaje: sirve el atún junto a los gajos de batata y las espinacas salteadas.", m["recipe"][2]
    texto = " ".join(m["recipe"]).lower()
    assert "carne" not in texto and "°c" not in texto and "rebanadas" not in texto, texto


def _reescribe(viejo, nueva, sub, desc="", pasos=()):
    m = {"name": "x", "desc": desc, "ingredients": [nueva, "1 taza de arroz"], "recipe": list(pasos)}
    sf.reescribir_plato(m, viejo, nueva, sub)
    return m


def test_la_bandeja_hornea_los_vegetales_y_el_duradero_entra_al_final():
    """«Coloca la pechuga, el brócoli y la cebolla en una bandeja… Hornea 20-25 min»: la primera versión dejaba las
    sardinas 20-25 min en el horno o bajaba la bandeja entera (con sus vegetales) a 5-8."""
    m = _reescribe("2 pechugas de pollo", "340 g de sardinas en lata", "sardinas en lata", pasos=[
        "El Toque de Fuego: calienta el horno a 200 °C. Coloca la pechuga de pollo, el brócoli, la berenjena y la cebolla "
        "en una bandeja; distribuye el ajo y la sal. Hornea 20-25 minutos, hasta que el centro de la parte más gruesa del "
        "pollo alcance 74 °C. Mientras tanto, cocina el bulgur 12-15 minutos."])
    assert m["recipe"][0] == (
        "El Toque de Fuego: calienta el horno a 200 °C. Coloca el brócoli, la berenjena y la cebolla en una bandeja; "
        "distribuye el ajo y la sal. Hornea 20-25 minutos. Añade las sardinas los últimos 5 minutos, solo para "
        "calentarlas. Mientras tanto, cocina el bulgur 12-15 minutos."), m["recipe"][0]


def test_el_tiempo_baja_solo_si_el_verbo_es_del_duradero():
    m = _reescribe("1 pechuga de pollo (≈204 g)", "204 g de sardinas en lata", "sardinas en lata", pasos=[
        "El Toque de Fuego: calienta la mitad del aceite en una sartén; añade el pollo y el ajo, agrega agua y cocina "
        "tapado 15-18 minutos."])
    assert "cocina tapado 15-18 minutos" in m["recipe"][0], "el guiso conserva su tiempo"
    m = _reescribe("1½ filetes de pescado", "225 g de sardinas en lata", "sardinas en lata", pasos=[
        "El Toque de Fuego: unta el pescado con aceite, ajo y sal; hornéalo a 200 °C durante 12-15 min, hasta que alcance "
        "63 °C en la parte más gruesa."])
    assert m["recipe"][0] == ("El Toque de Fuego: unta las sardinas con aceite, ajo y sal; hornéalas a 200 °C durante "
                              "5-8 min."), m["recipe"][0]


def test_comprobando_con_termometro_no_queda_colgando():
    m = _reescribe("½ pechuga de pollo (≈77 g)", "77 g de sardinas en lata", "sardinas en lata", pasos=[
        "El Toque de Fuego: sofríe la cebolla 3 min; añade el pollo y un poco de agua, tapa y guisa a fuego bajo 10-12 "
        "min, comprobando con termómetro que alcance 74 °C en la parte más gruesa. Incorpora el perejil al final."])
    assert m["recipe"][0] == ("El Toque de Fuego: sofríe la cebolla 3 min; añade las sardinas y un poco de agua, tapa "
                              "y guisa a fuego bajo 10-12 min. Incorpora el perejil al final."), m["recipe"][0]


def test_la_descripcion_pierde_la_coccion_del_crudo():
    m = _reescribe("150 g de filete de tilapia fresca", "180 g de garbanzos cocidos", "garbanzos cocidos",
                   desc="Filete blanco de tilapia estofado suavemente con cebolla, servido con plátano verde tierno.")
    assert m["desc"] == "Garbanzos estofados suavemente con cebolla, servidos con plátano verde tierno.", m["desc"]
    m = _reescribe("175 g de carne de res magra en tiras", "175 g de atún en agua", "atun en agua",
                   desc="Res magra marinada en limón y marcada a la plancha, servida sobre bulgur caliente.")
    assert m["desc"] == "Atún marinado en limón, servido sobre bulgur caliente.", m["desc"]
    m = _reescribe("2 pechugas de pollo", "340 g de sardinas en lata", "sardinas en lata",
                   desc="Pechuga sazonada con ajo y orégano, horneada junto a berenjena; con arepitas horneadas.")
    assert m["desc"] == "Sardinas sazonadas con ajo y orégano junto a berenjena; con arepitas horneadas.", m["desc"]


def test_hasta_que_este_firme_tambien_sale():
    """`firme` no es raíz + o/a: «(opac|cocid|…|firme)[oa]s?» no casaba nunca «firme»."""
    m = _reescribe("½ pechuga de pollo (≈100 g)", "100 g de sardinas en lata", "sardinas en lata", pasos=[
        "El Toque de Fuego: escalfa ½ pechuga de pollo (≈100 g) en agua caliente durante 3-4 minutos, hasta que el pollo "
        "esté firme y el centro aún suave."])
    assert m["recipe"][0] == ("El Toque de Fuego: escalfa 100 g de sardinas en lata en agua caliente durante 3-4 "
                              "minutos."), m["recipe"][0]


def test_el_salmon_previamente_congelado_tambien_se_parea():
    """Lote 289: «previamente congelado» ya no es despensa; la pareja en raw lo trataba como duradero y la lista seguía
    comprando 250 g de salmón para un plato de garbanzos."""
    m = {"ingredients": ["250 g de garbanzos cocidos", "1 limón"],
         "ingredients_raw": ["250 g de salmon previamente congelado y apto para consumo crudo", "1 limón"]}
    assert sf.parear_raw(m, "250 g de salmón previamente congelado y apto para consumo crudo",
                         "250 g de garbanzos cocidos") == "pareado"
    assert m["ingredients_raw"] == ["250 g de garbanzos cocidos", "1 limón"]


def test_el_alcance_del_pronombre_no_lo_corta_un_utensilio():
    assert sf._fin_ambito("… a fuego medio-alto en el centro y déjala reposar; sirve", 0) == \
        "… a fuego medio-alto en el centro y déjala reposar; sirve".index(";")
    assert sf._fin_ambito("y mézclala con el tomate", 0) == "y mézclala con el tomate".index("el tomate")
