# -*- coding: utf-8 -*-
"""[P1-CULINARY-V6-STEP-OVERASK · 2026-09-06] El paso pide MÁS de lo que la lista compra.

V4 compara **gramos**. Nadie miraba las **piezas**: la lista dice «½ diente de ajo» y el paso
«pica 1 diente de ajo»; «3 rebanadas de pan» y «mide 4 rebanadas»; «½ cda de aceite» y «mide
1 cda». Medido sobre 96 planes vivos: **37 hallazgos en 33 comidas**, 13 alimentos, y el patrón es
uno solo — el modelo **redondea las fracciones hacia arriba** al recitar la lista en el «Mise en
place». Casos verificados a mano, con el texto completo delante:

    lista «3 rebanadas de pan»   → paso «mide 4 rebanadas»          (una rebanada sin comprar)
    lista «½ hoja de repollo»    → paso «separa 6 hojas grandes»     (el plato son *canoas*)
    lista «½ cda de aceite»      → paso «mide 1 cda»                 (el doble de grasa)
    lista «½ pedazo de yuca»     → paso «corta ¾ pedazo (255 g)»     (+55 g)

## Las dos decisiones que lo hacen medible

1. **Solo se acusa cuando el paso pide MÁS.** Un paso que usa MENOS que el total puede estar
   repartiendo el ingrediente entre pasos —«calienta 1 cda» de las 2 que compra, el resto
   después—; contarlo castigaría a la receta bien escrita. Uno que pide más no tiene de dónde
   sacarlo. **La dirección es lo que separa el defecto del reparto**, no el alimento.

2. **La unidad es obligatoria.** Se probó admitir la mención sin unidad («2 guineítos» contra
   «½ guineíto», que es un defecto real) y sube de 37 a 153 hallazgos con ruido demostrable: sin
   una unidad que ancle el número al alimento se le pega cualquier cifra vecina — «coloca el
   Batata como base» heredó un «3» de otra frase, y un «huevo 4.0» salió de «2 minutos por lado».
   **Descartado MEDIDO**, no por prudencia.

Y una que no es un umbral sino un límite de lo que la comprobación afirma: **V6 no decide de qué
LADO está el error**. En «Canoas de repollo» la equivocada era la lista —con media hoja no hay
canoas—, no el paso. Lo único que afirma es que los dos se contradicen.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import culinary_coherence as cc  # noqa: E402

_CAT = [
    {"name": "Ajo", "aliases": ["ajos"], "category": "condimento",
     "ready_to_eat": False, "prep_methods": ["picado"]},
    {"name": "Pan integral familiar", "aliases": [], "category": "cereal",
     "ready_to_eat": True, "prep_methods": ["tostado"]},
    {"name": "Aceite de oliva", "aliases": [], "category": "grasa",
     "ready_to_eat": True, "prep_methods": []},
    {"name": "Cebollin", "aliases": ["cebollín"], "category": "vegetal",
     "ready_to_eat": True, "prep_methods": ["picado"]},
    {"name": "Lechosa", "aliases": [], "category": "fruta",
     "ready_to_eat": True, "prep_methods": ["crudo"]},
]


def _plan(ingredientes, pasos):
    return {"days": [{"meals": [{"meal": "almuerzo", "name": "Prueba",
                                 "ingredients": ingredientes, "recipe": pasos}]}]}


def _v6(ingredientes, pasos):
    return [v for v in cc.culinary_contract_scan(_plan(ingredientes, pasos), _CAT)
            if v["check"] == "V6"]


# ── el caso canónico ──────────────────────────────────────────────────────────────────────────
def test_el_paso_pide_el_doble_de_lo_que_la_lista_compra():
    v = _v6(["½ diente de ajo"], ["Mise en place: pica 1 diente de ajo"])
    assert len(v) == 1, v
    assert v[0]["food"] == "Ajo" and v[0]["check"] == "V6"
    assert "pide 1 diente" in v[0]["detail"] and "compra 0.5" in v[0]["detail"]


def test_una_rebanada_de_pan_que_nadie_compro():
    v = _v6(["3 rebanadas de pan integral familiar"],
            ["Mise en place: mide 4 rebanadas de pan integral familiar"])
    assert len(v) == 1 and v[0]["food"] == "Pan integral familiar"


def test_la_fraccion_unicode_se_entiende_igual_que_el_decimal():
    """«½», «1½» y «0,5» son el mismo número; que uno de los tres se escape sería un agujero
    silencioso, porque el modelo usa los tres."""
    assert cc._v6_valor("½") == 0.5
    assert cc._v6_valor("1½") == 1.5
    assert cc._v6_valor("0,33") == 0.33
    assert cc._v6_valor("no soy un numero") is None


# ── la dirección: lo que separa el defecto del reparto ────────────────────────────────────────
def test_un_paso_que_usa_MENOS_puede_estar_repartiendo_y_no_se_acusa():
    """«calienta 1 cda» de las 2 que compra, y el resto en otro paso. Es la receta BIEN escrita:
    acusarla convertiría a V6 en un impuesto sobre el buen estilo."""
    assert _v6(["2 cdas de aceite de oliva"],
               ["Calienta 1 cda de aceite de oliva en el sartén"]) == []


def test_usar_exactamente_lo_que_compra_no_es_un_hallazgo():
    assert _v6(["1 diente de ajo"], ["pica 1 diente de ajo"]) == []


def test_la_tolerancia_absuelve_a_un_tercio_escrito_como_decimal():
    """«⅓ taza» y «0.33 taza» son la misma cantidad escrita de dos formas; sin tolerancia, V6
    acusaría a la conversión en vez de al defecto."""
    assert _v6(["⅓ taza de lechosa"], ["mide 0.34 taza de lechosa"]) == []


# ── la unidad obligatoria: la decisión medida ─────────────────────────────────────────────────
def test_sin_unidad_no_se_compara_aunque_el_defecto_sea_real():
    """«½ guineíto» contra «2 guineítos» ES un defecto y V6 lo deja pasar a sabiendas: admitir la
    mención sin unidad sube de 37 a 153 hallazgos y mete ruido demostrable (un número vecino se
    pega a cualquier alimento). Se prefiere perder un caso real a ganar cien inventados."""
    assert _v6(["½ lechosa"], ["corta 2 lechosas en cubos"]) == []


def test_los_gramos_se_reconocen_para_descartarlos_no_para_compararlos():
    """Si `g` no estuviera en el vocabulario, «355 g de lechosa» caería al cubo de las piezas y se
    compararía contra unidades. Se reconoce y se descarta: la coherencia en gramos es de V4."""
    assert _v6(["200 g de lechosa"], ["corta 355 g de lechosa en cubos"]) == []
    assert cc._V6_MASA_RE.fullmatch("g") and cc._V6_MASA_RE.fullmatch("ml")


def test_dos_unidades_distintas_no_son_comparables_sin_densidad():
    """«1 taza» contra «½ diente» no dice nada: convertirlas exigiría una densidad que V6 no
    tiene, e inventarla daría un veredicto con cara de medición."""
    assert _v6(["½ diente de ajo"], ["añade 1 taza de ajo"]) == []


# ── el vocabulario de unidades ────────────────────────────────────────────────────────────────
def test_el_tallo_cuenta_porque_produjo_hallazgos_reales():
    """«½ tallo de cebollín» en la lista y «pica 2 tallos» en el paso: 2 casos vivos."""
    v = _v6(["½ tallo de cebollin picado"], ["pica 2 tallos de cebollin"])
    assert len(v) == 1 and v[0]["food"] == "Cebollin"


def test_una_clara_de_huevo_NO_es_una_unidad():
    """El prototipo trataba «claras» como unidad y ganaba un hallazgo; una clara es un ALIMENTO.
    Aceptar alimentos como unidades convierte el vocabulario en una lista abierta sin criterio."""
    assert "claras" not in cc._V6_CONTABLE and "huevos" not in cc._V6_CONTABLE


# ── fail-open y cableado ──────────────────────────────────────────────────────────────────────
def test_jamas_lanza_pase_lo_que_pase():
    """Como el resto del contrato: un escáner que revienta bloquearía la entrega de un plan por un
    problema de OBSERVACIÓN."""
    assert cc._v6_paso_pide_mas_que_la_lista(1, {}, {}) == []
    assert cc._v6_paso_pide_mas_que_la_lista(1, {"ingredients": None, "recipe": 7}, {}) == []
    assert cc.culinary_contract_scan({"days": [{"meals": [{"ingredients": ["x"]}]}]}, _CAT) == []


def test_v6_esta_cableado_en_el_escaner():
    """Una comprobación que nadie invoca es indistinguible de una que no existe — la lección de
    P1-G (mode=block no-op) y del catálogo INERTE de F2."""
    src = (_BACKEND / "culinary_coherence.py").read_text(encoding="utf-8")
    cuerpo = src.split("def culinary_contract_scan")[1]
    assert "_v6_paso_pide_mas_que_la_lista(day, meal, index)" in cuerpo


def test_el_doc_registra_la_decision_medida_de_no_admitir_la_mencion_sin_unidad():
    """Sin la cifra escrita, el próximo que lea el código verá una restricción arbitraria y la
    quitará — es exactamente cómo volvió `P1-VIEWPORT-ZOOM-LOCK`."""
    doc = (_BACKEND / "docs" / "culinary_coherence.md").read_text(encoding="utf-8")
    assert "V6" in doc
    assert "153" in doc, "la cifra del experimento descartado tiene que estar en el doc"
