"""[P1-DAYGEN-PROMPT-NO-NEUTRALIZE · 2026-08-23]

El day-generator es el último prompt estático antes de generar una comida. Su render beta
debe pasar por el neutralizador SSOT *después* de las sustituciones por país y *antes* de
entrar al caché. La única excepción es la regla técnica que explica cómo no arruinar Casabe:
es una defensa de cocción, no una recomendación de servirlo.
"""

from __future__ import annotations

import re

import pytest


_BETA_COUNTRIES = ("ES", "US", "MX", "PR", "CO")
_DIETS = ("balanced", "vegetarian", "vegan")

_CASABE_TECHNIQUE = (
    'TÉCNICA CORRECTA POR ALIMENTO [P1-CASABE-NO-BOIL · 2026-07-30]: el CASABE es una '
    'torta seca de yuca YA COCIDA — se sirve tal cual, se tuesta o se calienta en '
    'sartén/horno 1-2 min; JAMÁS se hierve, se cocina en agua ni "se deja reposar tapado" '
    'como si fuera arroz (un plan real instruyó "Cocina Casabe en 1½ tazas de agua con sal, '
    'tapa y hierve 15 minutos" — eso arruina el plato). Lo mismo aplica a pan, tostadas, '
    'galletas y tortillas ya horneadas: NUNCA les apliques la plantilla de cocción de granos '
    '(proporción agua:grano, hervir, reposar). Esa plantilla es SOLO para arroz, bulgur, '
    'quinoa, avena y granos crudos.'
)


def _ssot_source_terms() -> tuple[str, ...]:
    """Deriva el vocabulario en cada llamada: añadir una fila al SSOT amplía el guard solo."""
    from constants import _DO_LEXICON_NEUTRAL

    return tuple(
        sorted(
            {source.casefold() for source, _replacement in _DO_LEXICON_NEUTRAL},
            key=lambda value: (-len(value), value),
        )
    )


def _unexpected_ssot_hits(text: str) -> list[str]:
    scoped = text.replace(_CASABE_TECHNIQUE, "", 1).casefold()
    return [term for term in _ssot_source_terms() if term in scoped]


@pytest.fixture(autouse=True)
def _clean_country_prompt_cache():
    import prompts.day_generator as daygen

    daygen._COUNTRY_PROMPT_RENDER_CACHE.clear()
    yield
    daygen._COUNTRY_PROMPT_RENDER_CACHE.clear()


@pytest.mark.parametrize("country", _BETA_COUNTRIES)
@pytest.mark.parametrize("diet", _DIETS)
def test_beta_neutraliza_todo_el_ssot_salvo_la_regla_tecnica(diet, country):
    from prompts.day_generator import build_day_generator_system_prompt

    rendered = build_day_generator_system_prompt(diet, country)
    assert _CASABE_TECHNIQUE in rendered
    assert not _unexpected_ssot_hits(rendered), (
        f"{diet}/{country}: términos del SSOT sobrevivieron fuera de la única whitelist: "
        f"{_unexpected_ssot_hits(rendered)}"
    )


def test_el_neutralizador_corre_despues_de_la_tabla_beta(monkeypatch):
    """Mutación de orden: una fila beta nueva que introduce Casabe también queda neutralizada."""
    import prompts.day_generator as daygen

    target = "Tu misión es crear las comidas detalladas para UN SOLO DÍA del plan alimenticio."
    replacement = "Tu misión es crear Casabe para UN SOLO DÍA del plan alimenticio."
    extra_row = (daygen._diet_invariant(target), daygen._diet_invariant(replacement))
    monkeypatch.setattr(daygen, "_BETA_FRAGMENT_TABLE", [*daygen._BETA_FRAGMENT_TABLE, extra_row])

    rendered = daygen.build_day_generator_system_prompt("balanced", "ES")
    assert replacement not in rendered
    assert "Tu misión es crear Pan tostado integral" in rendered


def test_el_neutralizador_corre_una_vez_antes_del_cache(monkeypatch):
    """La segunda llamada idéntica debe reutilizar el render ya neutralizado."""
    import constants
    import prompts.day_generator as daygen

    calls = []
    original = constants.neutralize_do_lexicon

    def tracked(text):
        calls.append(text)
        return original(text)

    monkeypatch.setattr(constants, "neutralize_do_lexicon", tracked)
    first = daygen.build_day_generator_system_prompt("balanced", "ES")
    second = daygen.build_day_generator_system_prompt("balanced", "ES")

    assert first is second
    assert len(calls) == 1


def test_una_fila_nueva_del_ssot_entra_sin_editar_el_guard(monkeypatch):
    """Segunda mutación: el vocabulario del guard y el render nacen de la misma tabla."""
    import constants
    from prompts.day_generator import build_day_generator_system_prompt

    monkeypatch.setattr(
        constants,
        "_DO_LEXICON_NEUTRAL",
        (*constants._DO_LEXICON_NEUTRAL, ("Tu misión", "La misión")),
    )
    rendered = build_day_generator_system_prompt("balanced", "ES")

    assert "Tu misión" not in rendered
    assert "tu misión" in _ssot_source_terms()
    assert not _unexpected_ssot_hits(rendered)


def test_la_whitelist_es_exactamente_la_regla_tecnica():
    import prompts.day_generator as daygen

    assert daygen._BETA_NEUTRALIZATION_SURVIVORS == (_CASABE_TECHNIQUE,)


def test_guias_positivas_beta_no_fuerzan_comida_dominicana_fuera_del_pool():
    from prompts.day_generator import build_day_generator_system_prompt

    rendered = build_day_generator_system_prompt("balanced", "ES")
    for stale in (
        "AJÍ MORRÓN ≠ AJÍ CUBANELA",
        "Batido proteico con frutas (mamey, lechosa, guineo, fresas)",
        "Fruta + mantequilla de maní/almendras (manzana con pb, guineo con pb)",
        "la merienda usa OTRA fruta (lechosa, guineo, fresa, piña, manzana…)",
        "Rota a otro carbo de cena: batata, yuca, ñame o pan integral",
    ):
        assert stale not in rendered, f"guía positiva beta aún fija vocabulario DO: {stale!r}"

    for expected in (
        "Para platos rellenos usa el pimiento morrón verificado del catálogo",
        "Batido proteico con una fruta del pool asignado",
        "Fruta del pool asignado + mantequilla de maní/almendras",
        "la merienda usa OTRA fruta DEL POOL ASIGNADO",
        "Rota a otro carbohidrato del pool asignado distinto del arroz",
    ):
        assert expected in rendered

    assert "caldo concentrados" not in rendered
    assert "tortitas, panqueques, tortitas" not in rendered
    assert "bolitas al horno/croquetas al horno" not in rendered
    assert "caldos concentrados" in rendered
    assert "tortitas, panqueques, preparaciones al horno" in rendered
    assert "bolitas/croquetas al\n      horno" in rendered


def test_do_conserva_objeto_y_bytes_originales():
    from prompts.day_generator import DAY_GENERATOR_SYSTEM_PROMPT, build_day_generator_system_prompt

    assert build_day_generator_system_prompt("balanced", "DO") is DAY_GENERATOR_SYSTEM_PROMPT
    assert build_day_generator_system_prompt("balanced", None) is DAY_GENERATOR_SYSTEM_PROMPT
    assert _CASABE_TECHNIQUE in DAY_GENERATOR_SYSTEM_PROMPT


def test_guard_derivado_detecta_un_termino_ssot_nuevo_en_un_render_mutado(monkeypatch):
    """Prueba explícita de que el guard no es otra lista manual disfrazada."""
    import constants

    monkeypatch.setattr(
        constants,
        "_DO_LEXICON_NEUTRAL",
        (*constants._DO_LEXICON_NEUTRAL, ("VOCABULARIO NUEVO", "equivalente neutro")),
    )
    assert "vocabulario nuevo" in _unexpected_ssot_hits("texto con VOCABULARIO NUEVO")


def test_marker_del_pfix():
    from pathlib import Path

    implementation = (
        Path(__file__).resolve().parents[1] / "prompts" / "day_generator.py"
    ).read_text(encoding="utf-8")
    assert "P1-DAYGEN-PROMPT-NO-NEUTRALIZE · 2026-08-23" in implementation
