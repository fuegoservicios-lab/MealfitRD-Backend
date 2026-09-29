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


@pytest.mark.parametrize("texto", ["Ensalada César con pollo", "1 taza de fumet", "Coca de vegetales con anchoas",
                                   "Carpaccio de remolacha", "Arroz con atún"])
def test_lo_que_no_es_un_pez_de_plato_no_cuenta(texto):
    """Del escáner salen sólo las especies: el aderezo César, el fondo, la anchoa de adorno y los homónimos
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


# ── contrato ─────────────────────────────────────────────────────────────────────────────────────────────


def test_knobs_registrados_y_documentados():
    import basicos_por_token  # noqa: F401
    import pescado_especies  # noqa: F401
    from knobs import _KNOBS_REGISTRY
    doc = (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    for knob in ("MEALFIT_BETA_FISH_SPECIES_COUNT", "MEALFIT_STAPLE_TOKEN_MATCH"):
        assert knob in _KNOBS_REGISTRY, knob
        assert knob in doc, knob
    assert "P1-PLAN-LOTE-857" in doc


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
