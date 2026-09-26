# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-390 · 2026-09-26] Las claras duras se hierven dentro de su huevo.

Plan de la batería real (adulto mayor con HTA): «hierve 3 huevos y 3 claras de huevo 10-11 minutos, enfríalos y
pélalos» — una clara suelta no se hierve ni se pela (9 comidas en el corpus de 315 planes)."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_la_clara_dura_se_hierve_en_su_huevo():
    m = {"ingredients": ["3 huevos", "3 claras de huevo"],
         "recipe": ["El Toque de Fuego: hierve la yautía 15-18 minutos. En otra olla, hierve 3 huevos y 3 claras de huevo "
                    "10-11 minutos, enfríalos y pélalos.", "Montaje: corona con los huevos duros."]}
    assert pc.claras_en_su_huevo(m) == 1
    assert m["recipe"][0] == ("El Toque de Fuego: hierve la yautía 15-18 minutos. En otra olla, hierve 3 huevos y 3 claras "
                              "de huevo (hiérvelas dentro de su huevo entero, con cáscara) 10-11 minutos, enfríalos y "
                              "pélalos; quita la yema de cada huevo pelado: usa solo la clara."), m["recipe"][0]
    assert pc.claras_en_su_huevo(m) == 0                                   # idempotente
    s = {"ingredients": ["6 claras de huevo"],
         "recipe": ["Mise en place: enjuaga 50 g de cebada; hierve 6 claras de huevo 8-10 min, pélalos y córtalos en mitades."]}
    assert pc.claras_en_su_huevo(s) == 1
    assert s["recipe"][0].endswith("hierve 6 claras de huevo (hiérvelas dentro de su huevo entero, con cáscara) 8-10 min, "
                                   "pélalos y córtalos en mitades; quita la yema de cada huevo pelado: usa solo la clara.")
    u = {"ingredients": ["3 huevos", "1 clara de huevo"],
         "recipe": ["El Toque de Fuego: hierve 3 huevos y 1 clara de huevo (se pesan sin cáscara) en agua hirviendo unos "
                    "10-12 minutos, hasta que no escurran al pelarlos."]}
    assert pc.claras_en_su_huevo(u) == 1
    assert "1 clara de huevo (hiérvela dentro de su huevo entero, con cáscara) en agua" in u["recipe"][0]
    assert u["recipe"][0].endswith("; quita la yema del huevo pelado: usa solo la clara."), u["recipe"][0]
    c = {"ingredients": ["1 huevo", "4 claras de huevo"],
         "recipe": ["El Toque de Fuego: hornea la avena 12-14 minutos. En paralelo, cuece el huevo entero y 4 claras en agua "
                    "hirviendo 9-10 minutos; enfría, pela y reserva."]}
    assert pc.claras_en_su_huevo(c) == 1
    assert c["recipe"][0].endswith("cuece el huevo entero y 4 claras (hiérvelas dentro de su huevo entero, con cáscara) en "
                                   "agua hirviendo 9-10 minutos; enfría, pela y reserva; quita la yema de cada huevo pelado: "
                                   "usa solo la clara."), c["recipe"][0]


def test_sin_pelar_en_la_misma_frase():
    # batería real sobre el 379 (estatina): la frase hierve; el Montaje «pela y rebana el huevo»
    m = {"ingredients": ["1 huevo", "3 claras de huevo"],
         "recipe": ["El Toque de Fuego: hierve 1 huevo y 3 claras de huevo durante 10 minutos; tuesta el pan integral en "
                    "sartén seca 2-3 minutos por lado.", "Montaje: pela y rebana el huevo."]}
    assert pc.claras_en_su_huevo(m) == 1
    assert m["recipe"][0].startswith("El Toque de Fuego: hierve 1 huevo y 3 claras de huevo (hiérvelas dentro de su huevo "
                                     "entero, con cáscara) durante 10 minutos; tuesta el pan"), m["recipe"][0]
    # «hierve el agua…» no rige a las claras
    n = {"ingredients": ["3 claras de huevo"],
         "recipe": ["El Toque de Fuego: hierve el agua para el té y pon 3 claras de huevo a temperatura ambiente."]}
    antes = list(n["recipe"])
    assert pc.claras_en_su_huevo(n) == 0 and n["recipe"] == antes


def test_la_regla_de_la_forma_del_huevo_no_lo_deshace():
    import recipe_contract as rc
    from culinary_coherence import build_culinary_index
    idx = build_culinary_index([
        {"name": "Huevo", "aliases": ["huevos", "huevos enteros"], "category": "Proteínas", "prep_methods": ["cocido"]},
        {"name": "Clara de huevo", "aliases": ["claras de huevo", "clara de huevo", "claras", "clara de huevos"],
         "category": "Proteínas", "prep_methods": ["cocido"]},
    ])
    for lista, paso in ((["6 claras de huevo"], "Mise en place: hierve 6 claras de huevo 8-10 min, pélalos y córtalos."),
                        (["3 huevos", "3 claras de huevo"], "El Toque de Fuego: hierve 3 huevos y 3 claras de huevo 10 min, "
                                                            "enfríalos y pélalos.")):
        m = {"ingredients": list(lista), "recipe": [paso]}
        assert pc.claras_en_su_huevo(m) == 1
        antes = list(m["recipe"])
        assert rc.egg_forms_in_list(m["ingredients"], idx)[rc.CLARA] > 0      # la regla 2 SÍ ve las claras de la lista
        rc.egg_forms_step_sync(m, idx)
        assert m["recipe"] == antes, m["recipe"]


def test_las_claras_batidas_no_se_tocan():
    casos = [
        ["El Toque de Fuego: mientras hierve el agua, bate 3 claras de huevo y pela el ajo."],
        ["El Toque de Fuego: bate 4 claras de huevo y cuájalas en la sartén 3-4 minutos."],
        ["El Toque de Fuego: hierve la yautía; en un bol, mezcla 2 claras de huevo con la avena y pélalas… no."],
    ]
    for pasos in casos:
        m = {"ingredients": ["4 claras de huevo"], "recipe": list(pasos)}
        assert pc.claras_en_su_huevo(m) == 0 and m["recipe"] == pasos, pasos


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("pasos_cantidades").claras_en_su_huevo(meal)  # [P1-PLAN-LOTE-390]' in src
