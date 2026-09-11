"""[P1-PLAN-LOTE-7 · 2026-09-11] Séptimo lote del plan de pendientes: dos decisiones del dueño.

  G53/G74 · El prompt clínico neutralizaba el léxico es-DO para países beta con una SEGUNDA tabla de 10 filas
            (`condition_rules._BETA_CLINICAL_FOOD_SWAPS`), nacida el mismo día que el SSOT `constants._DO_LEXICON_NEUTRAL`
            y divergida desde entonces: el bariátrico español seguía leyendo «víveres hervidos» y «lechosa». Ahora hay UN
            léxico: el SSOT, que heredó las tres frases largas de la tabla (primero, orden largo→corto).
  Knob    · `MEALFIT_HARDEN_SAMEDAY_PROTEIN` estaba `true` en producción y no gobernaba NADA (sin rama desde julio). Fuera
            del god-file y de `prod_profile`; si la clase 1 se implementa, que nazca con rama y OFF.

Cada test expresa el comportamiento ESPERADO. Ninguno codifica el defecto como especificación.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


# ─────────────────────────── G53/G74 · un solo léxico ───────────────────────────

def test_g53_la_segunda_tabla_clinica_desaparecio_y_el_prompt_usa_el_ssot():
    src = _src("condition_rules.py")
    assert "_BETA_CLINICAL_FOOD_SWAPS = (" not in src, "no resucites la 2.ª tabla: el léxico es constants._DO_LEXICON_NEUTRAL"
    assert "neutralize_do_lexicon," in src.split("BARIATRIC_CONDITION_TERMS,")[1][:200], "importado del SSOT"
    i = src.find("def build_condition_prompt(")
    j = src.find("\ndef ", i + 10)
    cuerpo = src[i:j]
    assert "if _country_is_beta(form_data):" in cuerpo
    assert "_rendered = neutralize_do_lexicon(_rendered)" in cuerpo
    assert "_BETA_CLINICAL_FOOD_SWAPS" not in cuerpo


def test_g53_las_tres_frases_clinicas_viven_en_el_ssot_antes_que_los_tokens_sueltos():
    from constants import _DO_LEXICON_NEUTRAL
    fuentes = [s for s, _ in _DO_LEXICON_NEUTRAL]
    for frase in ("Revoltillo de Huevo con Casabe", "Atún con Casabe", "Pescado al Horno con Auyama"):
        assert frase in fuentes, frase
        assert fuentes.index(frase) < fuentes.index("casabe") and fuentes.index(frase) < fuentes.index("auyama"), (
            "orden largo→corto: la frase debe ganar a la palabra suelta")
    destinos = dict(_DO_LEXICON_NEUTRAL)
    assert destinos["Revoltillo de Huevo con Casabe"] == "Revoltillo de Huevo con Tostada integral"
    assert destinos["Pescado al Horno con Auyama"] == "Pescado al Horno con Calabaza"


def test_g74_el_bariatrico_beta_ya_no_lee_viveres_ni_lechosa_y_do_sigue_intacto(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    import condition_rules as cr
    es = cr.build_condition_prompt({"medicalConditions": ["bariatric"], "country": "ES"})
    do = cr.build_condition_prompt({"medicalConditions": ["bariatric"], "country": "DO"})
    assert es and do
    low = es.lower()
    for residuo in ("casabe", "auyama", "tayota", "vainitas", "lechosa", "víveres", "viveres"):
        assert residuo not in low, f"ES sigue leyendo «{residuo}»"
    assert "Tostada integral" in es and "Calabaza" in es
    assert "papaya" in low and "tubérculos" in low, "lo que la 2.ª tabla no sabía y el SSOT sí"
    # DO: byte-identidad con el knob (el país nativo no se neutraliza)
    assert "Casabe" in do and "Auyama" in do and "víveres" in do.lower()
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "false")
    assert cr.build_condition_prompt({"medicalConditions": ["bariatric"], "country": "ES"}) == do, (
        "con el knob apagado, beta cae a la conducta dominicana byte a byte")


def test_g53_guineo_no_se_mapea_a_platano():
    """Guineo (banana) y Plátano son DOS filas del catálogo con macros distintas: un nombre de alimento es un
    identificador. El gap lo advirtió; este test lo fija."""
    from constants import _DO_LEXICON_NEUTRAL
    for fuente, destino in _DO_LEXICON_NEUTRAL:
        assert not (fuente.lower().startswith("guineo") and "plátano" in destino.lower()), (fuente, destino)


# ─────────────────────────── el knob sin rama, fuera ───────────────────────────

def test_knob_harden_sameday_protein_retirado_en_codigo_y_en_el_perfil_de_prod():
    import graph_orchestrator as go
    import prod_profile as pp
    assert not hasattr(go, "HARDEN_SAMEDAY_PROTEIN")
    assert '_env_bool("MEALFIT_HARDEN_SAMEDAY_PROTEIN"' not in _src("graph_orchestrator.py")
    assert "MEALFIT_HARDEN_SAMEDAY_PROTEIN" not in pp.perfil_completo(), "prod_profile ya no declara un knob que no existe"
    # los hermanos con rama siguen
    for k in ("HARDEN_POOLS_ENABLED", "HARDEN_CONDITION_CATALOG", "HARDEN_SALTCURED_MAIN", "HARDEN_CROSSDAY_QUOTA", "HARDEN_MAIN_ARITY"):
        assert hasattr(go, k), k
    # el contrato del dict de conteos no cambia (dos tests lo anclan)
    assert '"sameday_bound": 0' in _src("graph_orchestrator.py")


def test_marker_bumpeado():
    import app
    assert "[P1-PLAN-LOTE-7 · 2026-09-11]" in _src("app.py")
    assert app._LAST_KNOWN_PFIX.startswith("P1-PLAN-") and "2026-09-11" in app._LAST_KNOWN_PFIX
