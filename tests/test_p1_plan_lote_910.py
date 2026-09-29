# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-910 · 2026-09-29] El marisco COCIDO que añade el cerrador recibe su paso de calentar.

Leídas enteras las 5 baterías reales del 28/29-sep (rdv801-868, 60 comidas): en dos almuerzos de EMBARAZO el cerrador de
proteína añadió «100 g de camarones cocidos» y el único texto que los nombra es «Acompaña con camarones.» — mientras la
nota 🤰 del mismo plato dice «cocina el pescado y los mariscos POR COMPLETO». El edamame cocido sí trae su «Calienta el
edamame cocido en agua hirviendo 2-3 minutos»; el camarón, no. Corpus (5.336 comidas, salida de la cola con el 888): 50
con camarones cocidos en la lista, 23 sin ningún paso que los caliente, 6 de embarazo/lactancia.
"""
from __future__ import annotations

import pathlib

import marisco_del_cerrador as mdc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
_NOTA_EMB = ("🤰 Seguridad alimentaria (embarazo/lactancia): lava y desinfecta las frutas, verduras y hierbas frescas antes "
             "de usarlas (aunque se vayan a cocinar); cocina el pescado y los mariscos POR COMPLETO (opacos y firmes; nada "
             "crudo ni a medio cocer).")
_TDF = ("El Toque de Fuego: cocina el arroz blanco según las indicaciones del paquete, hasta que esté tierno. En una "
        "sartén, calienta el aceite de oliva a fuego medio; sofríe la cebolla morada, el ajo y la zanahoria durante 5 min.")
_CALIENTA = "Calienta los camarones cocidos en una sartén 2-3 minutos, hasta que humeen."


def _meal(*, nombre="Arroz blanco con lentejas guisadas, zanahoria y limón", lista=None, tdf=_TDF,
          montaje="Montaje: sirve el guiso de lentejas sobre el arroz blanco. Acompaña con camarones.", notas=(_NOTA_EMB,)):
    rec = ["Mise en place: enjuaga 40 g de arroz blanco; pela y corta la zanahoria en cubitos."]
    if tdf:
        rec.append(tdf)
    rec.append(montaje)
    rec.extend(notas)
    return {"name": nombre, "recipe": rec,
            "ingredients": list(lista or ["40 g de arroz blanco crudo", "1 taza de lentejas de lata escurridas",
                                          "½ zanahoria pequeña", "100 g de camarones cocidos"])}


def test_los_camarones_cocidos_del_cerrador_se_calientan_al_final_del_toque_de_fuego():
    m = _meal()
    assert mdc.calentar(m) == 1
    assert m["recipe"][1] == _TDF + " " + _CALIENTA
    assert m["recipe"][2] == "Montaje: sirve el guiso de lentejas sobre el arroz blanco. Acompaña con camarones."
    assert mdc.calentar(m) == 0, "idempotente"


def test_sin_toque_de_fuego_nace_uno_antes_del_montaje():
    m = _meal(tdf=None)
    assert mdc.calentar(m) == 1
    assert m["recipe"][1] == "El Toque de Fuego: " + _CALIENTA
    assert m["recipe"][2].startswith("Montaje:")


def test_un_marisco_en_singular_concuerda():
    m = _meal(lista=["40 g de arroz blanco crudo", "120 g de pulpo cocido"],
              montaje="Montaje: sirve el arroz. Acompaña con pulpo.")
    assert mdc.calentar(m) == 1
    assert m["recipe"][1].endswith("Calienta el pulpo cocido en una sartén 2-3 minutos, hasta que humee.")


def test_lo_que_un_paso_ya_pone_al_fuego_no_se_toca():
    tdf = _TDF + " Saltea los camarones cocidos 2 minutos con el ajo."
    m = _meal(tdf=tdf)
    assert mdc.calentar(m) == 0 and m["recipe"][1] == tdf


def test_ni_el_camaron_crudo_ni_el_de_lata():
    for linea in ("150 g de camarones", "150 g de camarones frescos", "1 lata de camarones cocidos"):
        m = _meal(lista=["40 g de arroz blanco crudo", linea])
        antes = list(m["recipe"])
        assert mdc.calentar(m) == 0 and m["recipe"] == antes, linea


def test_un_plato_frio_sin_embarazo_los_sirve_frios():
    m = _meal(nombre="Ensalada fría de camarones con aguacate y pepino", notas=(),
              tdf=None, montaje="Montaje: mezcla el pepino y el aguacate con el limón. Acompaña con camarones.")
    antes = list(m["recipe"])
    assert mdc.calentar(m) == 0 and m["recipe"] == antes


def test_en_embarazo_tambien_el_plato_frio_los_calienta():
    m = _meal(nombre="Ensalada fría de camarones con aguacate y pepino", tdf=None,
              montaje="Montaje: mezcla el pepino y el aguacate con el limón. Acompaña con camarones.")
    assert mdc.calentar(m) == 1
    assert m["recipe"][1] == "El Toque de Fuego: " + _CALIENTA


def test_con_el_knob_apagado_nada(monkeypatch):
    monkeypatch.setenv("MEALFIT_COOKED_SEAFOOD_HEAT_STEP", "false")
    m = _meal()
    antes = list(m["recipe"])
    assert mdc.calentar(m) == 0 and m["recipe"] == antes


def _row(name, aliases=(), category="vegetal", rte=False, prep=("hervir", "saltear", "crudo")):
    return {"name": name, "aliases": list(aliases), "category": category, "ready_to_eat": rte, "prep_methods": list(prep)}


def test_la_cola_entera_lo_deja_y_no_lo_repite(monkeypatch):
    """Por el contrato de verdad, dos pasadas como en producción (grafo + escudo): el paso entra una vez, ningún otro
    reparador lo quita y el Montaje los sigue sirviendo."""
    import culinary_coherence as cc
    import recipe_contract as rc
    cat = [_row("Arroz blanco", ["arroz"], "cereal"), _row("Lentejas", ["lentejas de lata"], "legumbre"), _row("Zanahoria"),
           _row("Camarones", ["camarones cocidos", "camaron"], "proteina", True, ["saltear", "hervir"])]
    monkeypatch.setattr(rc, "_index_default", lambda db=None: cc.build_culinary_index(cat))
    monkeypatch.setenv("MEALFIT_RECIPE_FINAL_CONTRACT", "repair")
    m = _meal()
    rc.apply_final_contract_meal(m, None)
    texto = " ".join(m["recipe"])
    assert texto.count("Calienta los camarones cocidos en una sartén 2-3 minutos, hasta que humeen.") == 1, m["recipe"]
    assert "Acompaña con camarones" in texto, m["recipe"]
    antes = list(m["recipe"])
    rc.apply_final_contract_meal(m, None)
    assert m["recipe"] == antes


def test_ancla_tras_los_acompanamientos_del_449():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    i449 = src.index('__import__("pasos_cerrador").acompanamientos_en_una_frase(meal)  # [P1-PLAN-LOTE-449]')
    i910 = src.index('__import__("marisco_del_cerrador").calentar(meal)  # [P1-PLAN-LOTE-910]')
    i634 = src.index('__import__("doble_punto").limpiar(meal)  # [P1-PLAN-LOTE-634]')
    assert i449 < i910 < i634
    assert "tooltip-anchor: P1-PLAN-LOTE-910" in (_BACKEND / "marisco_del_cerrador.py").read_text(encoding="utf-8")
