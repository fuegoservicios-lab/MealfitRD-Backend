# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-857 · 2026-09-29] Dos instrumentos de variedad que no veían lo que había en el plato.

(1) Pescados de los países beta. `_MAIN_PROTEIN_ALIASES['pescado']` (el mapa del contador cross-día, del gate
same-day, de `reeleccion_dia` y del armador determinista) no conocía trucha, sardina, mojarra, bagre,
huachinango, lenguado, caballa ni boquerón, y SÍ contaba «dorado», que en el plato es casi siempre un adjetivo:
G24 CO D1 contaba pescado por «Plátano Maduro Dorado», no por sus sardinas, y la trucha de D1 y D3 no existía.
Las especies salen del vocabulario de pescado del escáner (`_ALLERGEN_SYNONYMS['pescado']`, que ya incluye
`vocabulario_mar.PESCADOS_EXTRA`), no de otra tabla. Knob `MEALFIT_BETA_FISH_SPECIES_COUNT`.

(2) `_count_staple_repetitions` casaba los básicos por subcadena: «pina» dentro de «espinaca» (G24 DO: `pina: 2`
sin una piña). Ahora con el resolvedor del gate same-day (`culinary_context._name_has_token`), y su espejo del
armador determinista (`deterministic_day._basicos_de`) con él. Knob `MEALFIT_STAPLE_TOKEN_MATCH`.
"""
from __future__ import annotations

import copy
import logging
import re
import subprocess
import sys
from pathlib import Path

import pytest

import graph_orchestrator as go

_BACKEND = Path(__file__).resolve().parent.parent


def _plan(textos_por_dia):
    """Un plan de N días, una comida por día cuyo nombre e ingrediente es el texto dado."""
    return [{"day": i + 1, "meals": [{"meal": "Almuerzo", "name": t, "ingredients": [f"120 g de {t}"]}]}
            for i, t in enumerate(textos_por_dia)]


# ── (1) pescados ─────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("especie", ["Trucha", "Sardinas", "Mojarra", "Bagre", "Huachinango", "Lenguado",
                                     "Caballa", "Boquerones", "Filete de dorada"])
def test_las_especies_beta_son_pescado(especie):
    assert "pescado" in go._protein_gate_labels_in_text(f"150 g de {especie}"), especie


def test_trucha_y_sardina_cuentan_entre_dias():
    dias = _plan(["Trucha a la plancha con patacones", "Ensalada estilo ceviche con sardinas",
                  "Trucha al horno con limón"])
    assert go._count_cross_day_heavy_protein_repetition(dias).get("pescado") == 3
    assert go.build_variety_report({"days": dias})["cross_day_proteins"].get("pescado") == 3


def test_dorado_adjetivo_no_es_pescado():
    """G24 CO D1: el «pescado» salía del plátano, no de las sardinas."""
    dias = _plan(["Plátano maduro dorado con queso", "Tostadas doradas con aguacate", "Papas doradas al horno"])
    assert "pescado" not in go._count_cross_day_heavy_protein_repetition(dias)
    assert "pescado" not in go._protein_gate_labels_in_text("Plátano Maduro Dorado")
    assert "pescado" in go._protein_gate_labels_in_text("150 g de filete de dorado"), "la frase inequívoca se queda"


def test_sardinas_en_el_almuerzo_y_trucha_en_la_cena_el_mismo_dia():
    dia = {"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Ensalada fresca con sardinas y garbanzos",
         "ingredients": ["120 g de Sardinas en lata", "80 g de Garbanzos"]},
        {"meal": "Cena", "name": "Arepa asada con ensalada fresca y trucha",
         "ingredients": ["1 Arepa de maíz", "100 g de Trucha"]},
    ]}
    assert go.build_variety_report({"days": [dia]})["same_day_protein_repeats"] >= 1
    assert go._days_with_same_day_protein_repeat({"days": [dia]}) == [1]


@pytest.mark.parametrize("texto", ["Ensalada César con pollo", "1 taza de fumet", "1 cda de salsa inglesa",
                                   "Carpaccio de remolacha", "Arroz con atún"])
def test_lo_que_no_es_un_pez_de_plato_no_cuenta(texto):
    """Del escáner salen sólo las especies: el aderezo César, el fondo, la salsa inglesa y los homónimos
    («carpa» ⊂ «carpaccio») se quedan fuera; el atún conserva su etiqueta."""
    assert "pescado" not in go._protein_gate_labels_in_text(texto), texto


def test_el_armador_determinista_ve_las_mismas_especies():
    import deterministic_day as dd
    assert "pescado" in dd._pesadas_de({"name": "Trucha al horno", "ingredients": ["150 g de Trucha"]})
    assert "pescado" not in dd._pesadas_de({"name": "Plátano maduro dorado", "ingredients": ["1 Plátano maduro"]})


def test_knob_de_especies_apagado_deja_el_mapa_como_estaba(monkeypatch):
    import pescado_especies as pe
    base = {"pescado": ["pescado", "tilapia", "dorado"], "atun": ["atun"], "pollo": ["pollo"]}
    vocab = ["pescado", "atun", "trucha", "sardina", "dorado", "cesar"]
    monkeypatch.setattr(pe, "BETA_FISH_SPECIES_COUNT", False)
    apagado = copy.deepcopy(base)
    assert pe.extender_pescado(apagado, vocab) == ([], [])
    assert apagado == base
    monkeypatch.setattr(pe, "BETA_FISH_SPECIES_COUNT", True)
    encendido = copy.deepcopy(base)
    anadidos, quitados = pe.extender_pescado(encendido, vocab)
    assert anadidos == ["trucha", "sardina", "filete de dorada"] and quitados == ["dorado"]
    assert encendido["pescado"] == ["pescado", "tilapia", "trucha", "sardina", "filete de dorada"]
    assert encendido["atun"] == ["atun"], "el atún no se muda de etiqueta"


# ── (2) básicos entre días ───────────────────────────────────────────────────────────────────────────────


def test_espinaca_no_es_pina():
    dias = _plan(["Ensalada de espinacas con uva", "Revoltillo con espinaca"])
    assert "pina" not in go._count_staple_repetitions(dias)


def test_pina_si_cuenta():
    dias = _plan(["Piña picada con yogurt", "Batido de piña"])
    assert go._count_staple_repetitions(dias).get("pina") == 2


def test_espejo_del_armador_determinista():
    import deterministic_day as dd
    assert "pina" not in dd._basicos_de({"name": "Ensalada de espinacas", "ingredients": ["2 tazas de Espinacas"]})
    assert "pina" in dd._basicos_de({"name": "Piña con yogurt", "ingredients": ["1 taza de Piña"]})


def test_knob_de_basicos_apagado_vuelve_a_la_subcadena(monkeypatch):
    import basicos_por_token as bpt
    dias = _plan(["Ensalada de espinacas con uva", "Revoltillo con espinaca"])
    monkeypatch.setattr(bpt, "STAPLE_TOKEN_MATCH", False)
    assert go._count_staple_repetitions(dias).get("pina") == 2      # la conducta previa, medida
    monkeypatch.setattr(bpt, "STAPLE_TOKEN_MATCH", True)
    assert "pina" not in go._count_staple_repetitions(dias)


# ── ronda 6 · detectar sí, reescribir sólo lo de antes y nunca un pez listo, crudo o frío ─────────────────────────
#
# Cinco rondas intentando que el autofix REESCRIBIERA bien las especies nuevas seguían dejando pollo crudo en casos
# límite (la lata que la línea no llama «en lata», el ceviche blanqueado 1-2 minutos). Regla nueva: (1) las especies
# nuevas cuentan para DETECTAR la repetición, pero el autofix no toca una comida que las lleve (decide el gate, que
# regenera el día) y reescribe con el conjunto de alias de antes del lote; (2) para cualquier pescado o marisco —
# también el atún de la base—, el autofix nunca reescribe una comida cuyo pez es conserva o precocido, crudo o
# servido frío.


def _tilapia():
    return {"meal": "Almuerzo", "name": "Tilapia al horno con arroz",
            "ingredients": ["150 g de Filete de tilapia", "1 taza de Arroz"],
            "recipe": ["Hornea la tilapia 20 minutos a 200 °C."]}


def _comida(nombre, lineas, pasos, slot="Cena"):
    return {"meal": slot, "name": nombre, "ingredients": list(lineas), "ingredients_raw": list(lineas),
            "recipe": list(pasos)}


def _dia_dos_pescados(nombre, linea, pasos):
    return [{"day": 1, "meals": [_tilapia(), _comida(nombre, [linea, "1/2 Aguacate"], pasos)]}]


def _textos(meal):
    return [meal["name"], *meal["ingredients"], *(meal.get("ingredients_raw") or []), *(meal.get("recipe") or [])]


_BLANQUEO = ("Blanquea {a} en agua hirviendo 1-2 minutos y escúrrelo bien antes de marinar (el cítrico solo marina, "
             "no cuece). Marina {a} en jugo de limón 20 minutos.")

# Las formas del revisor r5, con especie NUEVA: la comida no se reescribe nunca.
_ESPECIES_NUEVAS = {
    "sardinas-lata-en-el-paso": _comida("Arroz con sardinas", ["90 g de Sardinas", "1 taza de Arroz"],
                                        ["Escurre las sardinas de la lata y mézclalas con el arroz caliente."]),
    "sardinas-abre-la-lata": _comida("Casabe con sardinas", ["90 g de Sardinas", "1 Casabe"],
                                     ["Abre la lata y escurre las sardinas.", "Colócalas sobre el casabe."]),
    "sardinas-ya-viene-cocido": _comida("Arroz con vegetales y sardinas", ["1 taza de Arroz", "90 g de Sardinas"],
                                        ["Cocina el arroz 18 minutos.",
                                         "Escurre e incorpora sardinas (ya viene cocido) a la preparación."]),
    "lata-pequena-de-sardinas": _comida("Ensalada de sardinas", ["1 lata pequeña de sardinas", "1 tomate"],
                                        ["Desmenuza las sardinas y mézclalas con el tomate."]),
    "caballa-de-su-lata": _comida("Casabe con caballa", ["90 g de Caballa", "1 Casabe"],
                                  ["Escurre la caballa de su lata y colócala sobre el casabe."]),
    "ceviche-de-trucha-blanqueado": _comida("Ceviche de trucha", ["150 g de Trucha", "1 limón"],
                                            [_BLANQUEO.format(a="la trucha"), "Sirve frío."]),
    "trucha-cocida": _comida("Trucha a la plancha con papas", ["150 g de Trucha", "200 g de Papa"],
                             ["Cocina la trucha a la plancha 4 minutos por lado.", "Hierve las papas 15 minutos."]),
    "bagre-guisado": _comida("Bagre guisado con arroz", ["150 g de Bagre", "1 taza de Arroz"],
                             ["Guisa el bagre 20 minutos en salsa de tomate."]),
}


@pytest.mark.parametrize("tilapia_primero", [True, False], ids=["tilapia-primero", "especie-nueva-primero"])
@pytest.mark.parametrize("clave", list(_ESPECIES_NUEVAS))
def test_la_especie_nueva_cuenta_pero_su_comida_no_se_reescribe(clave, tilapia_primero):
    otra = copy.deepcopy(_ESPECIES_NUEVAS[clave])
    antes = copy.deepcopy(otra)
    dias = [{"day": 1, "meals": [_tilapia(), otra] if tilapia_primero else [otra, _tilapia()]}]
    assert go._days_with_same_day_protein_repeat({"days": dias}) == [1], "la repetición se detecta"
    go._protein_repeat_autofix(dias, {"dietType": "balanced", "country": "ES"}, None)
    assert otra == antes, "la comida con la especie nueva no se toca"


def test_la_especie_nueva_va_al_gate_con_su_motivo(caplog):
    """Tilapia delante (la guardiana de la base): la comida de sardinas no se reescribe, la repetición sigue y decide el
    gate; el log dice por qué."""
    dias = [{"day": 1, "meals": [_tilapia(), copy.deepcopy(_ESPECIES_NUEVAS["sardinas-lata-en-el-paso"])]}]
    antes = copy.deepcopy(dias)
    with caplog.at_level(logging.INFO):
        assert go._protein_repeat_autofix(dias, {"dietType": "balanced"}, None) == 0
    assert dias == antes
    assert "reason=especie_nueva" in caplog.text
    assert go._days_with_same_day_protein_repeat({"days": dias}) == [1]


@pytest.mark.parametrize("sardinas_primero", [True, False], ids=["orden-real", "trucha-primero"])
def test_g24_co_sardinas_y_trucha_van_al_gate(sardinas_primero):
    """G24 CO D1: las dos son especies nuevas; la ronda 5 reescribía la trucha a pollo. Ahora ninguna se toca."""
    sard = {"meal": "Almuerzo", "name": "Ensalada Fresca Estilo Ceviche con Sardinas, Garbanzos y Plátano Maduro Dorado",
            "ingredients": ["110 g de sardinas en lata escurridas",
                            "½ taza de garbanzos cocidos (de lata, enjuagados y escurridos)", "½ plátano maduro"],
            "recipe": ["Mise en place: escurre bien los 110 g de sardinas en lata y desmenúzalas.",
                       "Montaje: marina las sardinas 5 minutos con el jugo de limón y sirve."]}
    tru = {"meal": "Cena", "name": "Arepas de Maíz Asadas Rellenas de Ricotta con Ensalada Fresca y Trucha",
           "ingredients": ["35 g de harina de maíz precocida", "180 g de queso ricotta", "50 g de trucha cocida"],
           "recipe": ["El Toque de Fuego: forma 2 arepas y ásalas a fuego medio 5-6 minutos por lado. Cocina trucha "
                      "a la plancha o hervida y sírvela como proteína del plato."]}
    dias = [{"day": 1, "meals": [sard, tru] if sardinas_primero else [tru, sard]}]
    antes = copy.deepcopy(dias)
    assert go._protein_repeat_autofix(dias, {"dietType": "balanced", "country": "CO"}, None) == 0
    assert dias == antes


def test_el_reescritor_usa_los_alias_de_antes_del_lote():
    """Revisión r5 (no bloquea): «emplatado bonito» salía «emplatado pechuga de pollo». El reescritor ya no conoce las
    especies nuevas: la tilapia pasa a pollo y el adjetivo se queda."""
    dias = [{"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Mero a la plancha", "ingredients": ["150 g de Filete de mero"],
         "recipe": ["Cocina el mero a la plancha 4 minutos por lado."]},
        {"meal": "Cena", "name": "Tilapia al horno con papas", "ingredients": ["150 g de Filete de tilapia"],
         "recipe": ["Hornea la tilapia 20 minutos.", "Corta las papas para un emplatado bonito y sirve."]},
    ]}]
    assert go._protein_repeat_autofix(dias, {"dietType": "balanced"}, None) == 1
    cena = dias[0]["meals"][1]
    assert cena["_protein_autofix_applied"] == "pescado->pollo"
    assert cena["recipe"] == ["Hornea pechuga de pollo 20 minutos.", "Corta las papas para un emplatado bonito y sirve."]
    assert {"trucha", "sardina", "bonito", "caballa", "anchoa"} <= set(go._PEZ_NUEVO)
    assert not {"tilapia", "mero", "corvina", "pescado", "bacalao"} & set(go._PEZ_NUEVO), "la base no es nueva"


# (2) Guarda general: la base de antes del lote (atún, corvina, mero, camarones…) tampoco se reescribe cuando el pez es
# conserva/precocido, crudo o servido frío. Formas del revisor y de la base.
_PEZ_LISTO_CRUDO_FRIO = {
    # la base: «Ensalada de atún… sirve frío» salía «Escurre pechuga de pollo y mézclala… Sirve frío»
    "atun-ensalada-sirve-frio": ("atun", _comida("Ensalada de atún", ["120 g de Atún", "1 taza de Lechuga"],
                                                 ["Escurre el atún y mézclalo con la lechuga.", "Sirve frío."])),
    "atun-en-agua": ("atun", _comida("Wrap de atún", ["120 g de atún en agua", "1 tortilla"],
                                     ["Rellena la tortilla con el atún."])),
    "lata-de-atun": ("atun", _comida("Arroz con atún", ["1 lata de atún (120 g)", "1 taza de Arroz"],
                                     ["Mezcla el atún con el arroz caliente."])),
    "atun-en-conserva": ("atun", _comida("Pasta con atún", ["120 g de Atún en conserva", "1 taza de Pasta"],
                                         ["Incorpora el atún a la pasta."])),
    # el revisor: ceviche de corvina (especie de la base) blanqueado 1-2 min antes de marinar
    "ceviche-de-corvina-blanqueado": ("pescado", _comida("Ceviche de corvina", ["150 g de Corvina", "1 limón"],
                                                         [_BLANQUEO.format(a="la corvina"), "Sirve frío."])),
    "tiradito-de-corvina": ("pescado", _comida("Tiradito de corvina", ["120 g de Corvina"],
                                               ["Corta la corvina en láminas finas y báñala con leche de tigre."])),
    "mero-marinado-sin-fuego": ("pescado", _comida("Mero marinado al limón", ["150 g de Filete de mero marinado"],
                                                   ["Marina el mero en jugo de limón 30 minutos y sirve."])),
    "salmon-ahumado": ("pescado", _comida("Tosta de salmón ahumado", ["80 g de Salmón ahumado", "1 pan"],
                                          ["Coloca el salmón ahumado sobre la tosta."])),
    "pescado-ya-viene-cocido": ("pescado", _comida("Arroz con pescado", ["120 g de Filete de pescado", "1 taza de Arroz"],
                                                   ["Escurre e incorpora filete de pescado (ya viene cocido) al arroz."])),
    "ensalada-fria-de-merluza": ("pescado", _comida("Ensalada fría de merluza", ["120 g de Merluza", "1 tomate"],
                                                    ["Cocina la merluza al vapor 8 minutos.", "Mezcla con el tomate."])),
    "coctel-de-camarones-frio": ("camarones", _comida("Cóctel de camarones", ["120 g de Camarones cocidos"],
                                                      ["Mezcla los camarones con la salsa rosada y sirve frío."])),
}
_GUISOS = {"atun": _comida("Atún guisado con arroz", ["150 g de Atún fresco", "1 taza de Arroz"],
                           ["Guisa el atún 15 minutos en salsa de tomate."], slot="Almuerzo"),
           "pescado": {**_tilapia()},
           "camarones": _comida("Camarones al ajillo", ["150 g de Camarones", "1 taza de Arroz"],
                                ["Saltea los camarones con ajo 5 minutos."], slot="Almuerzo")}


@pytest.mark.parametrize("otro_primero", [True, False], ids=["cocinado-primero", "listo-primero"])
@pytest.mark.parametrize("clave", list(_PEZ_LISTO_CRUDO_FRIO))
def test_nunca_se_reescribe_un_pez_listo_crudo_o_frio(clave, otro_primero):
    etiqueta, comida = _PEZ_LISTO_CRUDO_FRIO[clave]
    comida = copy.deepcopy(comida)
    antes = copy.deepcopy(comida)
    guiso = copy.deepcopy(_GUISOS[etiqueta])
    dias = [{"day": 1, "meals": [guiso, comida] if otro_primero else [comida, guiso]}]
    assert go._days_with_same_day_protein_repeat({"days": dias}) == [1]
    go._protein_repeat_autofix(dias, {"dietType": "balanced", "country": "DO"}, None)
    assert comida == antes, "el pez listo, crudo o frío no se reescribe"


_FORMAS_PELIGROSAS = [c for c in _ESPECIES_NUEVAS.values()] + [c for _, c in _PEZ_LISTO_CRUDO_FRIO.values()]


@pytest.mark.parametrize("i", range(len(_FORMAS_PELIGROSAS)))
def test_nunca_sale_pollo_con_lata_frio_o_marinado_crudo(i):
    """La propiedad que las cinco rondas perseguían: en ninguna combinación sale «pechuga de pollo» (o pavo) en una
    comida que habla de lata, de servir frío o de marinar sin fuego."""
    comida = _FORMAS_PELIGROSAS[i]
    for otro in _GUISOS.values():
        for orden in (0, 1):
            m = [copy.deepcopy(otro), copy.deepcopy(comida)]
            dias = [{"day": 1, "meals": m if orden == 0 else m[::-1]}]
            go._protein_repeat_autofix(dias, {"dietType": "balanced"}, None)
            for meal in dias[0]["meals"]:
                if not meal.get("_protein_autofix_applied"):
                    continue
                blob = " ".join(str(t) for t in _textos(meal)).lower()
                assert not re.search(r"\blatas?\b|enlatad|ya viene cocid|\bfr[ií][oa]s?\b|\bmarin|ceviche|tiradito", blob), \
                    (meal.get("_protein_autofix_applied"), blob[:300])


@pytest.mark.parametrize("etiqueta,comida,motivo", [
    ("pescado", _ESPECIES_NUEVAS["trucha-cocida"], "especie_nueva"),
    ("pescado", _comida("Sardinas", ["90 g de Sardinas"], ["Sirve."]), "especie_nueva"),
    ("atun", _PEZ_LISTO_CRUDO_FRIO["atun-en-agua"][1], "pez_en_conserva"),
    ("atun", _PEZ_LISTO_CRUDO_FRIO["lata-de-atun"][1], "pez_en_conserva"),
    ("pescado", _PEZ_LISTO_CRUDO_FRIO["salmon-ahumado"][1], "pez_en_conserva"),
    ("pescado", _PEZ_LISTO_CRUDO_FRIO["pescado-ya-viene-cocido"][1], "pez_en_conserva"),
    ("pescado", _PEZ_LISTO_CRUDO_FRIO["ceviche-de-corvina-blanqueado"][1], "pez_crudo"),
    ("pescado", _PEZ_LISTO_CRUDO_FRIO["mero-marinado-sin-fuego"][1], "pez_crudo"),
    ("atun", _PEZ_LISTO_CRUDO_FRIO["atun-ensalada-sirve-frio"][1], "pez_servido_frio"),
    ("pescado", _PEZ_LISTO_CRUDO_FRIO["ensalada-fria-de-merluza"][1], "pez_servido_frio"),
    ("camarones", _PEZ_LISTO_CRUDO_FRIO["coctel-de-camarones-frio"][1], "pez_servido_frio"),
    # lo que se reescribe como en la base
    ("pescado", _tilapia(), None),
    ("pescado", _comida("Tilapia marinada al horno", ["150 g de tilapia marinada"], ["Hornea 20 minutos a 200 °C."]),
     None),
    ("pescado", _comida("Tilapia con arroz", ["150 g de Filete de tilapia marinada"],
                        ["Marina la tilapia con limón 10 minutos; luego hornéala 20 minutos a 200 °C."]), None),
    ("pescado", _comida("Merluza en salsa de tomate", ["150 g de Merluza"],
                        ["Cocina la merluza en salsa de tomate 10 minutos."]), None),
    # réplica 63eedc6b (locrio): el «crudo» es del arroz, no del pez
    ("pescado", _comida("Locrio de mero", ["150 g de Filete de mero"], [
        "Incorpora el mero en trozos grandes y agrega el arroz, que se pesa en crudo. Cubre con agua y cocina a fuego "
        "alto hasta que rompa hervor; tapa y cocina unos 15-18 minutos."]), None),
    ("atun", _GUISOS["atun"], None),
    ("camarones", _GUISOS["camarones"], None),
    # carne: la guarda es del mar
    ("pollo", _comida("Ensalada fría de pollo", ["120 g de pollo en lata"], ["Sirve frío."]), None),
])
def test_motivo_para_no_reescribir(etiqueta, comida, motivo):
    import pescado_especies as pe
    assert pe.motivo_para_no_reescribir(etiqueta, comida, go._MAIN_PROTEIN_ALIASES.get(etiqueta, ()), go._PEZ_NUEVO,
                                        go._PRECOOKED_PROTEIN_HINT, go._diet_pool_item_banned) == motivo


def test_la_conserva_sale_de_la_lista_del_cerrador():
    """SSOT: `_PRECOOKED_PROTEIN_HINT` (el cerrador escribe «ya viene cocido» con ella). «sardina» sin «fresca» es
    conserva, como anchoa y mojama; «sardinas frescas» no (y además es especie nueva)."""
    import pescado_especies as pe
    pez = go._MAIN_PROTEIN_ALIASES["pescado"]
    for linea in ("90 g de Sardinas", "40 g de Anchoas", "30 g de Mojama", "120 g de sardinas en aceite"):
        assert pe.motivo_para_no_reescribir("pescado", _comida("X", [linea], ["Sirve."]), pez, (),
                                            go._PRECOOKED_PROTEIN_HINT, go._diet_pool_item_banned) == "pez_en_conserva", linea
    assert pe.motivo_para_no_reescribir("pescado", _comida("X", ["150 g de Sardinas frescas"],
                                                           ["Cocina las sardinas a la plancha 3 minutos."]),
                                        pez, (), go._PRECOOKED_PROTEIN_HINT, go._diet_pool_item_banned) is None
    src = (_BACKEND / "pescado_especies.py").read_text(encoding="utf-8")
    assert "_v7f_evidencia" in src, "la cocción se mide con el criterio de V7f"
    for retirado in ("def pez_en_conserva", "def pez_crudo", "def guardiana_y_conservas", "PECES_DE_LATA"):
        assert retirado not in src, retirado


def test_el_autofix_llama_a_la_guarda_con_el_ssot():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    cuerpo = src.split("def _protein_repeat_autofix", 1)[1].split("\ndef ", 1)[0]
    assert "motivo_para_no_reescribir" in cuerpo and "_PRECOOKED_PROTEIN_HINT" in cuerpo and "_PEZ_NUEVO" in cuerpo
    assert "guardiana_y_conservas" not in cuerpo
    # la guardiana es la de la base: la primera comida con la proteína en el nombre
    assert "_keep_idx = next((i for i, (_, _in_name) in enumerate(hits) if _in_name), 0)" in cuerpo


def test_la_guarda_general_tiene_knob(monkeypatch):
    """`MEALFIT_PROTEIN_AUTOFIX_FISH_READY_GUARD=false` ⇒ la conducta de la base: el atún frío se reescribe."""
    import pescado_especies as pe
    ens = _PEZ_LISTO_CRUDO_FRIO["atun-ensalada-sirve-frio"][1]
    monkeypatch.setattr(pe, "PEZ_LISTO_GUARD", False)
    dias = [{"day": 1, "meals": [copy.deepcopy(_GUISOS["atun"]), copy.deepcopy(ens)]}]
    assert go._protein_repeat_autofix(dias, {"dietType": "balanced"}, None) == 1
    assert dias[0]["meals"][1]["_protein_autofix_applied"] == "atun->pollo"
    monkeypatch.setattr(pe, "PEZ_LISTO_GUARD", True)
    dias = [{"day": 1, "meals": [copy.deepcopy(_GUISOS["atun"]), copy.deepcopy(ens)]}]
    assert go._protein_repeat_autofix(dias, {"dietType": "balanced"}, None) == 0


def test_knob_de_especies_apagado_no_marca_especie_nueva(monkeypatch):
    import pescado_especies as pe
    monkeypatch.setattr(pe, "BETA_FISH_SPECIES_COUNT", False)
    assert pe.motivo_para_no_reescribir("pescado", _ESPECIES_NUEVAS["trucha-cocida"], ["pescado", "trucha"],
                                        ["trucha"], go._PRECOOKED_PROTEIN_HINT, go._diet_pool_item_banned) is None


def test_el_pez_cocinado_de_la_base_se_reescribe_como_antes():
    dias = [{"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Mero a la plancha", "ingredients": ["150 g de Filete de mero"],
         "recipe": ["Cocina el mero a la plancha 4 minutos por lado."]},
        _comida("Tilapia con arroz", ["150 g de Filete de tilapia marinada"],
                ["Marina la tilapia con limón 10 minutos; luego hornéala 20 minutos a 200 °C."])]}]
    assert go._protein_repeat_autofix(dias, {"dietType": "balanced"}, None) == 1
    assert dias[0]["meals"][1]["_protein_autofix_applied"] == "pescado->pollo"


def test_sin_compuestos_de_conserva_en_el_reescritor():
    assert go._PROTEIN_SOURCE_COMPOUNDS["pescado"] == ("filete de pescado blanco", "pescado blanco", "filete de pescado")


# ── ronda 4 · (2) la escalera respeta la dieta ───────────────────────────────────────────────────────────


_PESCETARIANOS = ("pescetariano", "Pescetariana", "pescatarian")


@pytest.mark.parametrize("dieta", _PESCETARIANOS)
@pytest.mark.parametrize("cena", [
    ("Mero a la plancha con ensalada", "150 g de Filete de mero", ["Cocina el mero a la plancha 4 minutos por lado."]),
    ("Ensalada de sardinas en lata con aguacate", "120 g de Sardinas en lata",
     ["Escurre las sardinas en lata y mézclalas con el aguacate.", "Montaje: sirve frío."]),
], ids=["mero", "sardinas-en-lata"])
def test_pescetariano_nunca_recibe_carne(dieta, cena):
    """Base: tilapia + mero con dieta pescetariana → «Pechuga de pollo a la plancha». El pescado sólo se cambia por
    otra proteína del mar; si la escalera no tiene ninguna, no se reescribe (decide el gate)."""
    dias = _dia_dos_pescados(*cena)
    antes = copy.deepcopy(dias)
    assert go._protein_repeat_autofix(dias, {"dietType": dieta, "country": "DO"}, None) == 0
    assert dias == antes
    assert go._scan_diet_violations({"days": dias}, dieta) == []


def test_pescetariano_atun_repetido_no_pasa_a_carne_ni_a_legumbre():
    dias = [{"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Ensalada de atún", "ingredients": ["120 g de Atún en agua", "1 taza de Lechuga"]},
        {"meal": "Cena", "name": "Arroz con vegetales", "ingredients": ["120 g de atún en agua", "1 taza de Arroz"]},
    ]}]
    go._protein_repeat_autofix(dias, {"dietType": "pescetariano"}, None)
    assert go._scan_diet_violations({"days": dias}, "pescetariano") == []
    assert "atún" in dias[0]["meals"][1]["ingredients"][0].lower()


def test_pescetariano_marisco_repetido_si_pasa_a_pescado():
    dias = [{"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Camarones al ajillo", "ingredients": ["150 g de Camarones", "1 taza de Arroz"]},
        {"meal": "Cena", "name": "Ensalada con camarones", "ingredients": ["120 g de Camarones", "1 taza de Lechuga"]},
    ]}]
    assert go._protein_repeat_autofix(dias, {"dietType": "pescetariano"}, None) == 1
    assert dias[0]["meals"][1]["_protein_autofix_applied"] == "camarones->pescado"


@pytest.mark.parametrize("dieta", _PESCETARIANOS)
def test_pescetariano_merienda_ligera_pasa_a_queso_como_en_la_base(dieta):
    """Revisión r4 (no bloquea): «del mar sólo a otra del mar» quitaba un arreglo que la base hacía bien — «Casabe con
    tilapia desmenuzada» en la merienda pasaba a queso, que la dieta permite. En la comida ligera o dulce el queso es el
    único destino que el autofix ya admite; en la principal sigue sin reescribirse (la legumbre de respaldo salía
    «50 g de habichuelas rojas guisadas cocida»)."""
    dias = [{"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Tilapia al horno con arroz",
         "ingredients": ["150 g de Filete de tilapia", "1 taza de Arroz"], "recipe": ["Hornea la tilapia 20 min."]},
        {"meal": "Merienda", "name": "Casabe con tilapia desmenuzada",
         "ingredients": ["1 pieza de Casabe", "60 g de tilapia desmenuzada"],
         "recipe": ["Coloca la tilapia desmenuzada sobre el casabe."]},
    ]}]
    assert go._protein_repeat_autofix(dias, {"dietType": dieta}, None) == 1
    assert dias[0]["meals"][1]["_protein_autofix_applied"] == "pescado->queso"
    assert go._scan_diet_violations({"days": dias}, dieta) == []


def test_omnivoro_sigue_como_antes():
    dias = _dia_dos_pescados("Mero a la plancha con ensalada", "150 g de Filete de mero", ["Cocina el mero."])
    assert go._protein_repeat_autofix(dias, {"dietType": "balanced"}, None) == 1
    assert dias[0]["meals"][1]["name"] == "Pechuga de pollo a la plancha con ensalada"


def test_la_dieta_sale_del_ssot():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    cuerpo = src.split("def _protein_repeat_autofix", 1)[1].split("\ndef ", 1)[0]
    assert "destino_apto_para_la_dieta" in cuerpo and "_diet_pool_item_banned" in cuerpo
    pe_src = (_BACKEND / "pescado_especies.py").read_text(encoding="utf-8")
    assert "canonicalize_diet_type" in pe_src


# ── ronda 4 · (3) las anchoas ────────────────────────────────────────────────────────────────────────────


def test_anchoas_son_pescado():
    """Datos: la única comida del corpus con anchoas (ES, «Coca de vegetales con anchoas») lleva 40 g — la mitad de la
    proteína del plato — y el catálogo la tiene en «Proteínas». Anchoas y boquerones el mismo día son el mismo pez."""
    assert "pescado" in go._protein_gate_labels_in_text("Coca de vegetales con anchoas")
    dia = {"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Coca de vegetales con anchoas", "ingredients": ["40 g de Anchoas"]},
        {"meal": "Cena", "name": "Boquerones fritos", "ingredients": ["150 g de Boquerones"]},
    ]}
    assert go._days_with_same_day_protein_repeat({"days": [dia]}) == [1]
    assert "atun" not in go._protein_gate_labels_in_text("40 g de Anchoas")

# ── contrato ─────────────────────────────────────────────────────────────────────────────────────────────


def test_knobs_documentados():
    doc = (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    for knob in ("MEALFIT_BETA_FISH_SPECIES_COUNT", "MEALFIT_STAPLE_TOKEN_MATCH",
                 "MEALFIT_PROTEIN_AUTOFIX_FISH_READY_GUARD"):
        assert knob in doc, knob
    assert "P1-PLAN-LOTE-857" in doc


def test_knobs_en_el_inventario_del_arranque():
    """Ronda 4: `basicos_por_token` se importaba en la primera llamada, así que su knob faltaba en el
    `[KNOBS/INVENTORY]` que `graph_orchestrator` emite al terminar de importarse. Se mide en un proceso limpio y sin
    importar antes los módulos del lote (importarlos primero es justo lo que escondía el fallo)."""
    codigo = (
        "import logging, os, sys\n"
        "os.environ['MEALFIT_DISABLE_SEMANTIC_CACHE'] = 'true'\n"
        "logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(message)s')\n"
        "import graph_orchestrator\n"
    )
    out = subprocess.run([sys.executable, "-c", codigo], cwd=_BACKEND, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    inventario = [ln for ln in out.stdout.splitlines() if "[KNOBS/INVENTORY]" in ln]
    assert inventario, "el arranque no emitió el inventario de knobs"
    for knob in ("MEALFIT_BETA_FISH_SPECIES_COUNT", "MEALFIT_STAPLE_TOKEN_MATCH",
                 "MEALFIT_PROTEIN_AUTOFIX_FISH_READY_GUARD"):
        assert knob + "=" in inventario[-1], knob


def test_marcador_y_anclas():
    for mod in ("pescado_especies.py", "basicos_por_token.py"):
        src = (_BACKEND / mod).read_text(encoding="utf-8")
        assert "[P1-PLAN-LOTE-857 · 2026-09-29]" in src and "tooltip-anchor: P1-PLAN-LOTE-857" in src, mod
    src_go = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    cuerpo = src_go.split("def _count_staple_repetitions", 1)[1].split("\ndef ", 1)[0]
    assert "basicos_por_token" in cuerpo and "any(a in text_norm for a in alias_list)" not in cuerpo
    assert 'pescado_especies").extender_pescado(_MAIN_PROTEIN_ALIASES, _ALLERGEN_SYNONYMS["pescado"])' in src_go
    src_dd = (_BACKEND / "deterministic_day.py").read_text(encoding="utf-8")
    assert "basicos_por_token" in src_dd.split("def _basicos_de", 1)[1].split("\ndef ", 1)[0]
