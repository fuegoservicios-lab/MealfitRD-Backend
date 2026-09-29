# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-855 · 2026-09-29] Instrumentos que mienten sobre variedad y cocina local.

(1) `_count_cross_day_heavy_protein_repetition` (el `cross_day_proteins` del `variety_report`, la señal de
monotonía de la autocrítica y de `reeleccion_dia`) casaba los alias por SUBCADENA: «res» dentro de «queso
fresco» / «piña fresca» / «cilantro fresco» y «pollo» dentro de «repollo». G24 (29-sep): PR y CO salieron con
`res: 3` sin un gramo de carne de res. Ahora resuelve con la misma frontera de palabra que el gate same-day
(`culinary_context._name_has_token`), y el espejo del armador determinista (`deterministic_day._pesadas_de`)
con él. Knob `MEALFIT_CROSS_DAY_PROTEIN_TOKEN_MATCH`.

(2) `scripts/vocabulario_ajeno_bibliotecas.py` mide cuántas plantillas de las bibliotecas beta usan palabras
de la mesa dominicana que en ese país no se dicen así. Sólo mide: no toca bibliotecas ni catálogo (los
nombres del catálogo son identificadores del motor).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import graph_orchestrator as go

_BACKEND = Path(__file__).resolve().parent.parent


def _plan(textos_por_dia):
    """Un plan de N días, una comida por día cuyo nombre e ingrediente es el texto dado."""
    return [{"day": i + 1, "meals": [{"meal": "Almuerzo", "name": t, "ingredients": [f"30 g de {t}"]}]}
            for i, t in enumerate(textos_por_dia)]


# ── (1) el instrumento de variedad ───────────────────────────────────────────────────────────────────────


def test_queso_fresco_no_cuenta_res():
    dias = _plan(["Arepa con queso fresco", "Piña fresca con yogur", "Ensalada con cilantro fresco"])
    assert "res" not in go._count_cross_day_heavy_protein_repetition(dias)
    assert "res" not in go.build_variety_report({"days": dias})["cross_day_proteins"]


def test_carne_de_res_si_cuenta():
    dias = _plan(["Carne de res guisada", "Bistec encebollado", "Res molida con papa"])
    assert go._count_cross_day_heavy_protein_repetition(dias).get("res") == 3
    assert go.build_variety_report({"days": dias})["cross_day_proteins"].get("res") == 3


def test_los_tres_dias_reales_de_colombia_no_son_res():
    """Las tres comidas de G24 CO que daban `res: 3` (ninguna lleva carne de res)."""
    dias = [
        {"day": 1, "meals": [{"meal": "Almuerzo", "name": "Ensalada Fresca Estilo Ceviche con Sardinas",
                              "ingredients": ["120 g de Sardinas en lata", "1 Limón"]}]},
        {"day": 2, "meals": [{"meal": "Merienda", "name": "Copa con mango, queso fresco y yogurt griego",
                              "ingredients": ["40 g de Queso fresco", "150 g de Yogurt griego sin azúcar"]}]},
        {"day": 3, "meals": [{"meal": "Cena", "name": "Trucha al horno con queso fresco, palmito tibio",
                              "ingredients": ["150 g de Trucha", "30 g de Queso fresco"]}]},
    ]
    assert "res" not in go._count_cross_day_heavy_protein_repetition(dias)


def test_repollo_no_es_pollo():
    dias = _plan(["Ensalada de repollo", "Tacos de pescado con repollo", "Locrio de bacalao con repollo"])
    assert "pollo" not in go._count_cross_day_heavy_protein_repetition(dias)


def test_pavochon_sigue_siendo_pavo():
    """El arreglo es FRONTERA INICIAL (la del gate same-day), no palabra exacta: «pavochón» es pavo."""
    dias = _plan(["Pavochón con puré de auyama", "Pechuga de pavo a la plancha", "Pavo molido guisado"])
    assert go._count_cross_day_heavy_protein_repetition(dias).get("pavo") == 3


def _alias_de_una_palabra():
    out = []
    for lbl in sorted(go._HEAVY_PROTEIN_LABELS):
        for a in go._MAIN_PROTEIN_ALIASES.get(lbl, ()):
            if re.fullmatch(r"[a-z]+", str(a)):
                out.append((lbl, a))
    return out


@pytest.mark.parametrize("label,alias", _alias_de_una_palabra())
def test_barrido_ningun_alias_cuenta_pegado_dentro_de_otra_palabra(label, alias):
    """Barrido de TODAS las claves de proteína pesada: el alias pegado detrás de otras letras («fresco» ⊃ res,
    «repollo» ⊃ pollo) no es esa proteína; suelto, sí."""
    pegado = _plan([f"Ensalada de zu{alias}"] * 3)
    assert label not in go._count_cross_day_heavy_protein_repetition(pegado), (label, alias)
    suelto = _plan([f"Plato de {alias}"] * 3)
    assert go._count_cross_day_heavy_protein_repetition(suelto).get(label) == 3, (label, alias)


@pytest.mark.parametrize("label,alias", [(lbl, a) for lbl in sorted(go._HEAVY_PROTEIN_LABELS)
                                         for a in go._MAIN_PROTEIN_ALIASES.get(lbl, ())])
def test_barrido_todo_alias_suelto_sigue_contando(label, alias):
    dias = _plan([f"Plato con {alias} a la plancha"] * 3)
    assert go._count_cross_day_heavy_protein_repetition(dias).get(label) == 3, (label, alias)


def test_el_espejo_del_armador_determinista_usa_el_mismo_criterio():
    import deterministic_day as dd
    assert "res" not in dd._pesadas_de({"name": "Arepa con queso fresco", "ingredients": ["30 g de Queso fresco"]})
    assert "pollo" not in dd._pesadas_de({"name": "Ensalada de repollo", "ingredients": ["1 taza de Repollo"]})
    assert "res" in dd._pesadas_de({"name": "Carne de res guisada", "ingredients": ["150 g de Carne de res"]})


def test_knob_apagado_vuelve_a_la_subcadena(monkeypatch):
    import proteina_por_token as ppt
    dias = _plan(["Arepa con queso fresco", "Piña fresca con yogur", "Ensalada con cilantro fresco"])
    monkeypatch.setattr(ppt, "CROSS_DAY_PROTEIN_TOKEN_MATCH", False)
    assert go._count_cross_day_heavy_protein_repetition(dias).get("res") == 3   # la conducta previa, medida
    monkeypatch.setattr(ppt, "CROSS_DAY_PROTEIN_TOKEN_MATCH", True)
    assert "res" not in go._count_cross_day_heavy_protein_repetition(dias)


def test_knob_registrado_y_documentado():
    import proteina_por_token  # noqa: F401
    from knobs import _KNOBS_REGISTRY
    assert "MEALFIT_CROSS_DAY_PROTEIN_TOKEN_MATCH" in _KNOBS_REGISTRY
    doc = (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    assert "MEALFIT_CROSS_DAY_PROTEIN_TOKEN_MATCH" in doc and "P1-PLAN-LOTE-855" in doc


def test_marcador_y_anclas():
    src_mod = (_BACKEND / "proteina_por_token.py").read_text(encoding="utf-8")
    assert "[P1-PLAN-LOTE-855 · 2026-09-29]" in src_mod
    assert "tooltip-anchor: P1-PLAN-LOTE-855" in src_mod
    src_go = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    cuerpo = src_go.split("def _count_cross_day_heavy_protein_repetition", 1)[1].split("\ndef ", 1)[0]
    assert "proteina_por_token" in cuerpo and "P1-PLAN-LOTE-855" in cuerpo
    assert "any(a in text_norm for a in alias_list)" not in cuerpo, "la subcadena volvió al instrumento"
    src_dd = (_BACKEND / "deterministic_day.py").read_text(encoding="utf-8")
    cuerpo_dd = src_dd.split("def _pesadas_de", 1)[1].split("\ndef ", 1)[0]
    assert "proteina_por_token" in cuerpo_dd


# ── (2) medición del vocabulario ajeno en las bibliotecas beta ───────────────────────────────────────────


def test_vocabulario_ajeno_por_pais():
    from scripts import vocabulario_ajeno_bibliotecas as va
    assert va.palabras_ajenas("Guineo", "US") == ["guineo"]
    assert va.palabras_ajenas("Batido de guineo", "PR") == []           # en Puerto Rico se dice así
    assert va.palabras_ajenas("Auyama asada", "CO") == []               # «ahuyama/auyama»: colombiano
    assert va.palabras_ajenas("Auyama asada", "MX") == ["auyama"]
    assert va.palabras_ajenas("Habichuelas negras", "MX") == ["habichuelas negras"]
    assert va.palabras_ajenas("Salteado de tofu con habichuelas", "CO") == []   # en CO: la vaina verde
    assert va.palabras_ajenas("Habichuelas rojas guisadas", "CO") == ["habichuelas rojas"]
    assert va.palabras_ajenas("Cayeye de guineo verde", "CO") == []             # costa colombiana
    assert va.palabras_ajenas("Batido de guineo", "CO") == ["guineo"]
    assert va.palabras_ajenas("Batata asada", "ES") == []               # batata/boniato en España
    assert va.palabras_ajenas("Batata asada", "MX") == ["batata"]       # camote
    assert va.palabras_ajenas("Queso fresco con pollo", "ES") == []
    assert va.palabras_ajenas("Guineo con habichuelas y auyama", "DO") == []   # control: RD nunca es ajeno
    assert va.palabras_ajenas("Devuélvelas al sartén para que el queso funda", "CO") == []   # el verbo, no el envase


def test_la_medicion_de_bibliotecas_es_coherente():
    from scripts import vocabulario_ajeno_bibliotecas as va
    m = va.medir_bibliotecas()
    assert set(m) == {"ES", "US", "MX", "PR", "CO"}
    for cc, r in m.items():
        assert 0 <= r["plantillas_con_ajeno"] <= r["plantillas"], cc
        assert r["plantillas"] > 0, cc
        for ej in r["ejemplos"]:
            assert ej["palabras"], (cc, ej)
