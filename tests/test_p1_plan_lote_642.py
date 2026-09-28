# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-642 · 2026-09-27] El coach cambia el país aunque el usuario lo diga en otro idioma (G63 + G87).

`update_form_field` acepta `country` (está en la whitelist desde P2-COUNTRY-HOUSEKEEPING) pero:
  - la docstring —el contrato que el modelo LEE— no lo nombraba entre los campos válidos;
  - el resolvedor solo reconocía el nombre en español: con la app en inglés, «I moved to Spain» llegaba como
    `Spain`, se RECHAZABA, y el usuario seguía recibiendo planes de RD;
  - antes de buscar por nombre pasaba por `canonicalize_country`, que registra «valor de país no canónico
    descartado → DO» para «España» aunque la tool lo resuelva un paso después (un falso positivo de corrupción).

tooltip-anchor: P1-PLAN-LOTE-642
"""
import inspect
import logging

import pytest


@pytest.mark.parametrize("dicho,codigo", [
    ("ES", "ES"), ("es", "ES"), ("España", "ES"), ("espana", "ES"), ("Spain", "ES"), ("Espagne", "ES"),
    ("Espanha", "ES"), ("Spagna", "ES"), ("Mexico", "MX"), ("Mexique", "MX"), ("Messico", "MX"),
    ("United States", "US"), ("USA", "US"), ("Estados Unidos", "US"), ("États-Unis", "US"), ("Stati Uniti", "US"),
    ("Puerto Rico", "PR"), ("Porto Rico", "PR"), ("Colombia", "CO"), ("Colombie", "CO"), ("Colômbia", "CO"),
    ("Dominican Republic", "DO"), ("República Dominicana", "DO"), ("République dominicaine", "DO"),
])
def test_el_pais_se_entiende_en_los_cinco_idiomas(dicho, codigo):
    from tools import _valor_de_campo_para_perfil
    assert _valor_de_campo_para_perfil("country", dicho) == (True, codigo)


@pytest.mark.parametrize("dicho", ["Marte", "Argentina", "", "Spainland"])
def test_lo_que_no_es_uno_de_los_seis_se_rechaza(dicho):
    from tools import _valor_de_campo_para_perfil
    assert _valor_de_campo_para_perfil("country", dicho) == (False, None)


def test_un_nombre_bien_escrito_no_deja_rastro_de_corrupcion(caplog):
    from tools import _valor_de_campo_para_perfil
    with caplog.at_level(logging.WARNING):
        assert _valor_de_campo_para_perfil("country", "España") == (True, "ES")
    assert "no canónico descartado" not in caplog.text


def test_la_docstring_que_lee_el_modelo_nombra_el_pais():
    import tools
    doc = inspect.getdoc(tools.update_form_field.func if hasattr(tools.update_form_field, "func")
                         else tools.update_form_field) or getattr(tools.update_form_field, "description", "")
    assert "'country'" in doc


def test_cada_pais_del_ssot_tiene_sus_nombres_en_otros_idiomas():
    from constants import COUNTRY_PROFILES
    from tools import _NOMBRES_DE_PAIS
    assert set(_NOMBRES_DE_PAIS) == set(COUNTRY_PROFILES)
