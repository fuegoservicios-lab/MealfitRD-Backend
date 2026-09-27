# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-469 · 2026-09-27] Batería REAL del bloque 3 (días 8-11 de 30 sin congelador, perfil del dueño).

La IA escribió el pescado con dos nombres: display «1 filete de pescado», raw «155 g de carne de Filete de pescado blanco
cocida». El reconciliador raw→display buscaba la PRIMERA palabra («carne») en el display, no la encontraba y añadía la
línea: el plato llevaba el pescado dos veces, la compra única sustituía cada copia por un duradero distinto y el paso
quedaba «escurre las sardinas de atún» / «verifica que los garbanzos de sardinas esté cocida». Además:
- «la carne de filete de pescado» → «la carne de atún»: la «carne de» se queda sola;
- «verifica que … esté cocida y lista para consumir» sobre lo que viene en lata;
- «1 rebanada de pan integral familiar» → «1 de casabe» (display «1 casabe»), paso «1 rebanada de casabe familiar».
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import compra_unica as cu  # noqa: E402
import graph_orchestrator as go  # noqa: E402
import sustitucion_fresca as sf  # noqa: E402

_CATALOGO = {
    "155 g de carne de filete de pescado blanco cocida": ("filete de pescado blanco", 155.0),
    "1 filete de pescado": ("filete de pescado blanco", 150.0),
    "75 g de bulgur de coccion rapida": ("bulgur", 75.0),
    "2 tortas pequenas de casabe": ("casabe", 60.0),
    "40 g de casabe": ("casabe", 40.0),
    "0.5 diente de ajo picado": ("ajo", None),
}


def _resolver(monkeypatch):
    from constants import strip_accents

    def _r(line, *, cheap=False):
        return _CATALOGO.get(strip_accents(str(line)).lower(), (None, None))
    monkeypatch.setattr(go, "_resolve_line_food_grams", _r)


def _plato():
    return {"name": "Bulgur caribeño al wok con filete de pescado blanco y repollo",
            "ingredients": ["1 filete de pescado", "75 g de bulgur de cocción rápida", "2 tortas pequeñas de casabe"],
            "ingredients_raw": ["155 g de carne de Filete de pescado blanco cocida", "75 g de bulgur de cocción rápida",
                                "40 g de casabe"]}


def test_el_raw_con_otro_nombre_no_se_duplica_en_el_display(monkeypatch):
    _resolver(monkeypatch)
    m = _plato()
    assert go._reconcile_raw_missing_in_display([{"meals": [m]}]) == 0
    assert m["ingredients"] == ["1 filete de pescado", "75 g de bulgur de cocción rápida", "2 tortas pequeñas de casabe"]


def test_el_display_con_otro_nombre_no_se_duplica_en_el_raw(monkeypatch):
    _resolver(monkeypatch)
    m = _plato()
    assert go._reconcile_display_missing_in_raw([{"meals": [m]}]) == 0
    assert len(m["ingredients_raw"]) == 3, m["ingredients_raw"]


def test_lo_que_de_verdad_falta_se_sigue_anadiendo(monkeypatch):
    _resolver(monkeypatch)
    m = {"name": "Puré de Sardinas", "ingredients": ["1 filete de pescado"],
         "ingredients_raw": ["1 filete de pescado", "0.5 diente de ajo picado"]}
    assert go._reconcile_raw_missing_in_display([{"meals": [m]}]) == 1
    assert m["ingredients"][-1] == "0.5 diente de ajo picado"


def test_la_carne_de_se_va_con_el_fresco():
    m = {"name": "Bulgur caribeño al wok con filete de pescado blanco y repollo",
         "ingredients": ["1 filete de pescado"], "ingredients_raw": ["155 g de carne de Filete de pescado blanco cocida"],
         "recipe": ["Mise en place: mide 75 g de bulgur y prepáralo según las instrucciones del envase; escurre la carne "
                    "de filete de pescado blanco cocida; corta 1 taza de repollo."]}
    sf.sustituir_en_plato(m, 0, "1 filete de pescado", "155 g de atún en agua", "atun en agua")
    assert "escurre el atún;" in m["recipe"][0], m["recipe"][0]
    assert "carne de" not in m["recipe"][0]


def test_lo_listo_no_se_verifica_cocido():
    m = {"name": "Arroz cítrico con filete de pescado blanco y lechuga fresca", "ingredients": ["1¼ filetes de pescado"],
         "ingredients_raw": ["205 g de carne de Filete de pescado blanco cocida y lista para consumir"],
         "recipe": ["Mise en place: verifica que la carne de filete de pescado blanco esté cocida y lista para consumir; "
                    "mide 2 tazas de arroz blanco ya cocido."]}
    sf.sustituir_en_plato(m, 0, "1¼ filetes de pescado", "190 g de sardinas en lata", "sardinas en lata")
    assert m["recipe"][0].startswith("Mise en place: escurre las sardinas; mide 2 tazas"), m["recipe"][0]


def test_el_pan_en_rebanadas_pasa_a_tortas_de_casabe():
    req = {"need_days": 21, "allow_frozen": False}
    r = cu.sustituir_linea("1 rebanada de pan integral familiar", 20, req, gramos_de=lambda _t: None)
    assert r and r[0] == "1 torta pequeña de casabe", r
    r = cu.sustituir_linea("2 rebanadas de pan de agua", 20, req, gramos_de=lambda _t: None)
    assert r and r[0] == "2 tortas pequeñas de casabe", r
    r = cu.sustituir_linea("1 pan de agua", 20, req, gramos_de=lambda _t: 60.0)
    assert r and r[0] == "60 g de casabe", "con peso del catálogo manda el peso"
    m = {"name": "Tostadas dominicanas con huevo, palmito y queso cottage",
         "ingredients": ["1 rebanada de pan integral familiar"], "ingredients_raw": ["1 rebanada de pan integral familiar"],
         "recipe": ["Mise en place: ten listas 1 rebanada de pan integral familiar, 1 huevo y sal al gusto."]}
    sf.sustituir_en_plato(m, 0, "1 rebanada de pan integral familiar", "1 torta pequeña de casabe", "casabe")
    assert m["recipe"][0] == "Mise en place: ten listas 1 torta pequeña de casabe, 1 huevo y sal al gusto.", m["recipe"]
    assert m["ingredients_raw"] == ["1 torta pequeña de casabe"]
