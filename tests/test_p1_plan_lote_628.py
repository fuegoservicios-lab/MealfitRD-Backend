# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-628 · 2026-09-27] El escáner y los estimadores por texto saben en qué país vive el usuario.

Auditoría de países: los cinco prompts que estiman comida —la foto (`vision_agent`), el texto libre
(`/consumed/estimate-macros`, `plato_descrito`), la duda de la foto (`ajuste_de_duda`) y el ingrediente corregido
(`ingrediente_corregido`)— empiezan «Eres un nutricionista dominicano» y ninguno recibe el país. Un español que escribe
«plátano» se estimaba como plátano de cocinar (en España es la banana), con sus porciones dominicanas.

`pais_del_estimador.contexto(país)` le dice al modelo dónde vive el usuario y cómo leer sus palabras ALLÍ, sin tocar
la frontera del motor: los nombres que devuelve siguen siendo los del catálogo (español dominicano). En RD, cadena
vacía: los cinco prompts quedan byte a byte como estaban. Si el perfil no se puede leer, también vacía.

tooltip-anchor: P1-PLAN-LOTE-628
"""
import asyncio
import inspect

import pytest


def test_en_rd_y_sin_pais_no_cambia_nada():
    from pais_del_estimador import contexto
    for pais in ("DO", None, "", "XX"):
        assert contexto(pais) == ""


@pytest.mark.parametrize("pais,nombre", [("ES", "España"), ("MX", "México"), ("US", "Estados Unidos"),
                                         ("PR", "Puerto Rico"), ("CO", "Colombia")])
def test_en_un_pais_beta_nombra_el_pais_y_la_frontera_del_catalogo(pais, nombre):
    from pais_del_estimador import contexto
    c = contexto(pais)
    assert nombre in c
    assert "catálogo" in c and "español dominicano" in c


def test_el_platano_es_banana_solo_donde_lo_es():
    from pais_del_estimador import contexto
    for pais in ("ES", "MX"):
        assert "Guineo" in contexto(pais)
    for pais in ("CO", "PR"):
        assert "«plátano» es la banana" not in contexto(pais)


def test_los_ejemplos_salen_del_mismo_lexico_que_las_alergias():
    # los nombres de otro país que da como ejemplo son los de `food_names_i18n` (lote 623), no una tabla nueva
    from pais_del_estimador import contexto
    from food_names_i18n import _variantes
    reg = _variantes("variantes_regionales")
    c = contexto("ES")
    assert "Duraznos" in reg and "melocotón" in c.lower() and "Duraznos" in c


def test_sin_perfil_no_rompe_y_no_anade_nada(monkeypatch):
    import pais_del_estimador as p
    def roto(_uid):
        raise RuntimeError("sin base")
    monkeypatch.setattr(p, "pais_de", roto)
    assert asyncio.run(p.contexto_del_usuario("u1")) == ""
    assert asyncio.run(p.contexto_del_usuario(None)) == ""


def test_la_foto_lleva_el_pais_en_su_prompt():
    import vision_agent as va
    assert va._prompt_de_escaneo(None, base="BASE", pais="DO") == "BASE"
    assert va._prompt_de_escaneo(None, base="BASE", pais="ES").startswith("BASE")
    assert "España" in va._prompt_de_escaneo(None, base="BASE", pais="ES")
    assert "pais" in inspect.signature(va.process_image_with_vision).parameters


@pytest.mark.parametrize("modulo", ["plato_descrito", "ajuste_de_duda", "ingrediente_corregido"])
def test_cada_estimador_anade_el_contexto_del_usuario(modulo):
    import importlib
    src = inspect.getsource(importlib.import_module(modulo).estimar_con_ia)
    assert "contexto_del_usuario(user_id)" in src


def test_el_router_pasa_el_pais_a_la_foto_y_al_texto_libre():
    from routers import diary
    src = inspect.getsource(diary)
    assert "process_image_with_vision(file_bytes, aclaracion=aclaracion, pais=" in src
    assert "_ESTIMATE_SYSTEM_PROMPT + await contexto_del_usuario(" in src
