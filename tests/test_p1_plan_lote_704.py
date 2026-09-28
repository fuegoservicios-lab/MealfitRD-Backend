# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-704 · 2026-09-28] (G75) La cobertura de plantillas del seeder se mide contra la biblioteca del PAÍS.

`_dish_template_class_counts` leía siempre `dish_templates.json` (87 plantillas RD). Para un plan de España la
telemetría «esta base no tiene NINGUNA plantilla» mentía, y el día que alguien baje `MEALFIT_LOW_TEMPLATE_COVERAGE_PENALTY`
de 1.0, el ×0.5 penalizaría sistemáticamente las bases de cada país. Ahora el país viaja hasta la ruta
(`dish_library._templates_path_for_country`, SSOT) y ENTRA en la clave de `_TEMPLATE_COVERAGE_CACHE`: sin eso el
arreglo nacía inerte (la primera lectura, la de RD, se servía a todos).

tooltip-anchor: P1-PLAN-LOTE-704
"""
import ai_helpers as ah
import dish_library as dl


def _tpl(nombre, proteina):
    return {"name": nombre, "slots": [ah._TEMPLATE_COVERAGE_MAIN_SLOT], "protein": proteina, "base": "arroz"}


_RD = [_tpl("Chivo guisado", "chivo"), _tpl("Pollo guisado", "pollo")]
_ES = [_tpl("Merluza a la plancha", "merluza"), _tpl("Pollo al ajillo", "pollo")]


def _bibliotecas(monkeypatch):
    rutas = {None: dl._TEMPLATES_PATH, "ES": "/fake/dish_templates_es.json"}
    monkeypatch.setattr(dl, "_templates_path_for_country", lambda c=None: rutas.get(c, dl._TEMPLATES_PATH))
    monkeypatch.setattr(dl, "load_dish_templates", lambda path=None: _ES if path == rutas["ES"] else _RD)
    ah._TEMPLATE_COVERAGE_CACHE.clear()


def test_el_pais_elige_la_biblioteca(monkeypatch):
    _bibliotecas(monkeypatch)
    assert ah._template_coverage("Merluza", "protein", country="ES") > 0
    assert ah._template_coverage("Merluza", "protein") == 0
    assert ah._template_coverage("Chivo", "protein", country="ES") == 0
    assert ah._template_coverage("Chivo", "protein") > 0


def test_el_pais_entra_en_la_clave_de_la_cache(monkeypatch):
    _bibliotecas(monkeypatch)
    rd = ah._dish_template_class_counts(ah._TEMPLATE_COVERAGE_MAIN_SLOT, "protein")
    es = ah._dish_template_class_counts(ah._TEMPLATE_COVERAGE_MAIN_SLOT, "protein", country="ES")
    assert "chivo" in rd and "chivo" not in es
    assert "merluza" in es


def test_el_penalty_mide_con_el_pais(monkeypatch):
    _bibliotecas(monkeypatch)
    pesos, n = ah._apply_low_template_coverage_penalty(["Merluza", "Chivo"], [1.0, 1.0], "protein", 0.5, country="ES")
    assert pesos == [1.0, 0.5] and n == 1


def test_el_seeder_pasa_el_pais_del_form():
    src = open(ah.__file__, encoding="utf-8").read()
    assert src.count('"protein", _tpl_factor, country=_tpl_country)') == 1
    assert src.count('"base", _tpl_factor, country=_tpl_country)') == 1
    assert "_template_coverage(_tpl_b, _tpl_field, country=_tpl_country)" in src
