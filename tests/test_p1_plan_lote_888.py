# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-888 · 2026-09-29] El ave cruda que un paso sólo calienta un momento pasa a cocción real, en el paso.

Copia de un plan real (nocturno sin tiempo, «pescado->pollo»): «Seca 255 g de pechuga de pollo…» y «Calienta pechuga de
pollo en una sartén a fuego bajo durante 1-2 minutos» — el tiempo de la sardina de lata. Revisor del 857 (6d): «añade el
pollo al guiso en los últimos minutos». El 737 sólo añadía al final «⚠️ … debe cocinarse por completo».
"""
from __future__ import annotations

import pathlib

import ave_tiempo_seguro as ats

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def _meal(fuego, lista=("255 g de pechuga de pollo", "1 pedazo de yautía"), extra=()):
    return {"ingredients": list(lista),
            "recipe": ["Mise en place: pela la yautía y seca 255 g de pechuga de pollo con papel de cocina.", fuego, *extra,
                       "Montaje: sirve la pechuga de pollo sobre la yautía majada."]}


def test_calentar_1_2_minutos_la_pechuga_cruda_pasa_a_cocinarla():
    m = _meal("El Toque de Fuego: cocina la yautía en el microondas 6-8 minutos y májala. Calienta pechuga de pollo en "
              "una sartén a fuego bajo durante 1-2 minutos.")
    assert ats.asegurar(m) == 1
    assert m["recipe"][1] == ("El Toque de Fuego: cocina la yautía en el microondas 6-8 minutos y májala. Cocina pechuga "
                              "de pollo en una sartén a fuego medio durante 8-10 minutos, hasta que el pollo alcance 74 °C "
                              "por dentro.")
    assert ats.asegurar(m) == 0, "idempotente"


def test_los_ultimos_minutos_del_pescado_y_su_razon_se_van():
    m = _meal("El Toque de Fuego: sofríe la cebolla 3 minutos, añade el tomate y el agua. Añade la pechuga de pollo al "
              "guiso en los últimos minutos para que no se deshaga.")
    assert ats.asegurar(m) == 1
    assert m["recipe"][1].endswith("Añade la pechuga de pollo al guiso y cocínala 10-12 minutos, hasta que el pollo "
                                   "alcance 74 °C por dentro.")


def test_por_lado_y_la_senal_del_pescado():
    m = _meal("El Toque de Fuego: calienta el aceite a fuego medio-alto y sella pechuga de pollo 3-4 minutos por lado "
              "hasta que se desmenuce fácilmente.")
    assert ats.asegurar(m) == 1
    assert m["recipe"][1] == ("El Toque de Fuego: calienta el aceite a fuego medio-alto y sella pechuga de pollo 6-7 "
                              "minutos por lado, hasta que el pollo alcance 74 °C por dentro.")


def test_un_hasta_que_debil_se_une_sin_repetirse():
    m = _meal("El Toque de Fuego: incorpora la pechuga de pollo y el repollo y saltea 3-4 min hasta que el pollo esté "
              "humeante; sazona con sal.")
    assert ats.asegurar(m) == 1
    assert "saltea 8-10 minutos, hasta que el pollo alcance 74 °C por dentro y el pollo esté humeante; sazona" in m["recipe"][1]


def test_lo_que_ya_esta_bien_no_se_toca(monkeypatch):
    casos = [
        # el mismo paso termina en 74 °C (sella y luego guisa)
        _meal("El Toque de Fuego: cocina la pechuga de pollo con la cebolla 5 min. Agrega el tomate, tapa y guisa 8-10 "
              "min, hasta que el pollo alcance 74 °C en la parte más gruesa."),
        # el guiso tapado lo termina
        _meal("El Toque de Fuego: dora el pollo 3 min por lado; añade ½ taza de agua y guisa tapado 15-18 minutos."),
        # …también en la misma frase
        _meal("El Toque de Fuego: añade el pollo y dóralo 4 minutos, agrega ½ taza de agua y el orégano, y guisa tapado "
              "a fuego medio-bajo por 15 minutos."),
        # ya dice cuándo está hecho
        _meal("El Toque de Fuego: saltea la pechuga de pollo en tiras 4-5 min hasta que esté completamente cocida."),
        # cocción previa del ave
        _meal("El Toque de Fuego: calienta la pechuga de pollo 2 minutos con el ajo.",
              extra=()) | {"recipe": ["💡 Cocción previa: cocina la pechuga de pollo en agua 15-18 min, hasta 74 °C.",
                                      "El Toque de Fuego: calienta la pechuga de pollo 2 minutos con el ajo."]},
        # la lista la compra cocida
        _meal("El Toque de Fuego: calienta la pechuga de pollo 2 minutos.", lista=("150 g de pechuga de pollo cocida",)),
    ]
    for m in casos:
        antes = list(m["recipe"])
        assert ats.asegurar(m) == 0 and m["recipe"] == antes, antes
    monkeypatch.setenv("MEALFIT_POULTRY_SAFE_TIME", "false")
    m = _meal("El Toque de Fuego: calienta pechuga de pollo en una sartén durante 1-2 minutos.")
    assert ats.asegurar(m) == 0


def test_ancla_entre_el_441_y_el_737():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    i441 = src.index('__import__("pasos_cantidades").ave_hasta_74(meal)  # [P1-PLAN-LOTE-441]')
    i888 = src.index('__import__("ave_tiempo_seguro").asegurar(meal)  # [P1-PLAN-LOTE-888]')
    i737 = src.index('__import__("ave_con_su_punto").asegurar(meal)  # [P1-PLAN-LOTE-737]')
    assert i441 < i888 < i737
    assert "tooltip-anchor: P1-PLAN-LOTE-888" in (_BACKEND / "ave_tiempo_seguro.py").read_text(encoding="utf-8")
