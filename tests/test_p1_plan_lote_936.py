# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-936 · 2026-09-30] La compra única no manda toda la fruta a manzana: una rueda de duraderos, y aceitunas
por el aguacate de los platos salados.

Batería real del PLATO con el 932 vivo (owner_like, días 8-11 de 30): la tabla `compra_unica.SUSTITUTOS` cambia toda
fruta que no aguanta —y el aguacate— por «manzana»; había manzana en 9 de 16 comidas y «Sardinas con casabe, queso
blanco y manzana» donde iba aguacate. El dueño (30-sep) aprobó la rueda.

La rueda (manzana 45 días, naranja 30, pera 21 en nevera, según `pantry_durability`) gira por día y comida como la de
las proteínas, salta lo que no aguanta hasta ese día y lo que no es seguro (alergia, rechazo), y deja para el final lo
que el día ya lleva. El aguacate de un plato salado pasa a aceitunas (despensa, misma función: grasa), con el peso
de la línea; en un plato dulce o licuado sigue a la rueda de fruta.
"""
from __future__ import annotations

import pathlib

import compra_unica as cu
import fruta_duradera as fd

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
_REQ = {"need_days": 8, "allow_frozen": False, "freezer_mode": "none", "freeze_window_days": 0}


def _sub(texto, dia, semilla=0, evitar=(), plato="Ensalada de sardinas con casabe", alergias=None, contexto=None):
    return cu.sustituir_linea(texto, dia, _REQ, semilla=semilla, evitar=set(evitar), plato=plato, alergias=alergias,
                              contexto=contexto)


def test_la_fruta_gira_por_dia_y_comida():
    vistos = {_sub("100 g de lechosa", 7, semilla=s)[1] for s in (0, 1, 2)}
    assert vistos == {"manzana", "naranja", "pera"}, vistos
    linea, sub, hit = _sub("100 g de lechosa", 7, semilla=1)
    assert hit == "lechosa" and linea == f"100 g de {sub}"


def test_lo_que_no_aguanta_hasta_ese_dia_no_entra():
    """La pera aguanta 21 días en nevera: en el día 25 la rueda la salta."""
    assert all(_sub("100 g de lechosa", 24, semilla=s)[1] in ("manzana", "naranja") for s in range(6))
    assert all(_sub("100 g de lechosa", 40, semilla=s)[1] == "manzana" for s in range(6))


def test_lo_que_el_dia_ya_lleva_va_al_final():
    assert _sub("100 g de lechosa", 7, semilla=1, evitar={"naranja"})[1] != "naranja"
    assert _sub("100 g de lechosa", 7, semilla=1, evitar={"naranja", "pera"})[1] == "manzana"


def test_lo_que_no_es_seguro_no_entra():
    assert all(_sub("100 g de lechosa", 7, semilla=s, alergias=["Naranja"])[1] != "naranja" for s in range(6))


def test_el_aguacate_de_un_plato_salado_pasa_a_aceitunas_con_su_peso():
    linea, sub, hit = _sub("35 g de aguacate", 7, plato="Sardinas con casabe, queso blanco y aguacate")
    assert (linea, sub, hit) == ("35 g de aceitunas", "aceitunas", "aguacate")
    assert _sub("145 g de aguacate", 7, plato="Casabe tostado con queso blanco y aguacate")[0] == "40 g de aceitunas", "tope"
    linea, sub, _ = _sub("½ aguacate", 7, plato="Casabe con sardinas al limón y ensalada")
    assert sub == "aceitunas" and linea.endswith("g de aceitunas") and linea[0].isdigit(), linea


def test_el_aguacate_de_un_batido_sigue_a_la_fruta():
    linea, sub, _ = _sub("70 g de aguacate", 7, plato="Batido cremoso de aguacate y avena con yogurt")
    assert sub in ("manzana", "naranja", "pera") and linea == f"70 g de {sub}"


def test_el_plato_entero_deja_de_nombrar_el_aguacate():
    import sustitucion_fresca as sf
    m = {"meal": "Desayuno", "name": "Revoltillo de coliflor con aguacate fresco", "desc": "Huevos con coliflor y aguacate.",
         "ingredients": ["2 huevos", "70 g de aguacate"], "ingredients_raw": ["2 huevos", "70 g de aguacate"],
         "recipe": ["Mise en place: corta 70 g de aguacate en láminas.", "Montaje: sirve con el aguacate."]}
    sf.sustituir_en_plato(m, 1, "70 g de aguacate", "40 g de aceitunas", "aceitunas")
    texto = " | ".join([m["name"], m["desc"]] + m["recipe"])
    assert "aguacate" not in texto.lower() and "aceitunas" in m["name"].lower(), texto
    for sub in ("naranja", "pera"):
        assert sub in sf._DURADERO


def test_la_naranja_se_pela_en_gajos_no_se_corta_en_cubos():
    """El plato hereda el corte del fresco: «corta 100 g de naranja en cubos» (replay de los bloques)."""
    import sustitucion_fresca as sf
    m = {"meal": "Merienda", "name": "Lechosa con maní tostado", "desc": "Merienda fresca: lechosa en cubos con maní.",
         "ingredients": ["100 g de lechosa", "20 g de maní"], "ingredients_raw": ["100 g de lechosa", "20 g de maní"],
         "recipe": ["Mise en place: corta 100 g de lechosa en cubos y mide 20 g de maní.",
                    "Montaje: sirve la lechosa en un vaso y termina con el maní."]}
    sf.sustituir_en_plato(m, 0, "100 g de lechosa", "100 g de naranja", "naranja")
    assert m["recipe"][0] == "Mise en place: pela 100 g de naranja en gajos y mide 20 g de maní.", m["recipe"]
    assert "naranja en gajos" in m["desc"] and "cubos" not in m["desc"], m["desc"]
    assert fd.gajos("corta la naranja en láminas finas") == "pela la naranja en gajos"
    assert fd.gajos("pela y corta 95 g de naranja en cubos, y mide") == "pela 95 g de naranja en gajos, y mide", "no «pela y pela»"
    assert fd.gajos("mide 30 g de aceitunas y corta la pera en cubos") == "mide 30 g de aceitunas y corta la pera en cubos"


def test_las_frutas_y_las_aceitunas_cuentan_como_presentes_en_el_dia():
    assert cu.duraderos_del_dia(["100 g de naranja", "30 g de aceitunas", "1 pera"]) >= {"naranja", "aceitunas", "pera"}


def test_con_el_knob_apagado_todo_a_manzana(monkeypatch):
    monkeypatch.setenv("MEALFIT_SINGLE_TRIP_FRUIT_WHEEL", "false")
    assert all(_sub("100 g de lechosa", 7, semilla=s)[1] == "manzana" for s in range(3))
    assert _sub("70 g de aguacate", 7, plato="Sardinas con casabe y aguacate")[1] == "manzana"


def test_anclas():
    src = (_BACKEND / "compra_unica.py").read_text(encoding="utf-8")
    assert '__import__("fruta_duradera").elegir(' in src
    mod = (_BACKEND / "fruta_duradera.py")
    assert "tooltip-anchor: P1-PLAN-LOTE-936" in mod.read_text(encoding="utf-8")
    assert b"\x08" not in mod.read_bytes() and b"\r" not in mod.read_bytes()
    assert fd.RUEDA == ("manzana", "naranja", "pera")
