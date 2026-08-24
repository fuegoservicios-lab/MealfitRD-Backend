"""[P3-SEMOLA-MAIZ-GLUTEN-FP · 2026-08-23] A un celíaco se le quitaba la «Sémola de maíz», que no
tiene gluten.

LO MEDIDO. Barrido de las 347 filas del catálogo contra `clinical_backstop_for_meal(
allergies=['Gluten'])`: 30 filas marcadas, y entre ellas «Sémola de maíz» (grits/polenta). Las
vecinas se comportan bien («Tortilla de maíz» → 0), o sea que era un caso aislado y no un fallo
del criterio. Y no era sólo el backstop: la fila desaparecía también del bloque «USA
EXCLUSIVAMENTE ESTOS ALIMENTOS» de un perfil US con alergia a gluten y reaparecía sin ella.

LA FORMA DEL ARREGLO YA ESTABA VALIDADA POR EL PROPIO CÓDIGO: «Sémola de arroz» ya quedaba absuelta
porque 'arroz' vive en `_PLANT_ADJ_EXCUSE_RX` (la excusa que resolvió «mantequilla de maní»). Lo
que faltaban eran las bases que ese regex no lista.

POR QUÉ NO SE AÑADE 'maiz' A LA PLANT-ADJ. Esa excusa es UNIVERSAL: absuelve cualquier término de
cualquier categoría seguido de la base. Con 'maiz' dentro, «Pan de maíz» —que SÍ lleva trigo—
dejaría de marcarse. La excusa nueva está acotada AL TÉRMINO que casó ('semola'), así que no puede
alcanzar a 'pan' ni a 'harina'. La dirección peligrosa (servir el alérgeno) no se abre: la sémola
de trigo sigue marcada, y también la desnuda, que es la que puede ser de trigo.

Corrección al audit: la fila es US-only, no de Puerto Rico.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


def _viola(go, texto, alergia="Gluten"):
    return go.clinical_backstop_for_meal({"name": texto, "ingredients": [texto]},
                                         allergies=[alergia])


# ── A. El falso positivo se cierra ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("texto", [
    "Sémola de maíz", "semola de maiz", "Sopa de sémola de maíz",
    "Sémola de yuca", "Sémola de arroz",
])
def test_la_semola_sin_gluten_deja_de_ser_violacion(go, texto):
    assert not _viola(go, texto), f"{texto!r} se marcó como gluten y no lo lleva"


# ── B. La dirección peligrosa NO se abre ────────────────────────────────────────────────────────

@pytest.mark.parametrize("texto", [
    "Sémola de trigo",      # sémola de verdad
    "Sémola",               # desnuda: puede ser de trigo → sesgo a sobre-detectar
    "Pan de maíz",          # LLEVA trigo: la razón por la que la excusa no puede ser genérica
    "Harina de trigo",
    "Tortilla de harina",
])
def test_lo_que_si_lleva_gluten_sigue_marcado(go, texto):
    assert _viola(go, texto), f"{texto!r} dejó de marcarse para un celíaco"


def test_la_excusa_esta_acotada_al_termino_que_caso(go):
    """La propiedad que separa este arreglo de «añadir maíz a la plant-adj»: la absolución la
    concede el TÉRMINO, no la base."""
    assert go._allergen_term_base_excused("semola", " de maiz") is True
    assert go._allergen_term_base_excused("pan", " de maiz") is False
    assert go._allergen_term_base_excused("harina", " de maiz") is False
    assert go._allergen_term_base_excused("trigo", " de maiz") is False
    # sin entrada no hay excusa genérica, y una cola vacía no absuelve a nadie
    assert go._allergen_term_base_excused("semola", "") is False
    assert go._allergen_term_base_excused(None, " de maiz") is False


def test_la_excusa_exige_frontera_de_palabra_en_la_base(go):
    """'maiz' no puede absolver por prefijo de otra palabra."""
    assert go._allergen_term_base_excused("semola", " de maizena") is False


def test_ninguna_entrada_del_mapa_absuelve_a_un_termino_que_no_le_toca(go):
    for termino, bases in go._ALLERGEN_TERM_BASE_EXCUSES.items():
        assert termino == termino.lower() and " " not in termino.strip()
        assert bases, f"{termino}: lista de bases vacía"


# ── C. La otra superficie: el catálogo que se le OFRECE al modelo ───────────────────────────────

@pytest.mark.e2e
def test_el_catalogo_del_celiaco_conserva_la_semola_de_maiz_y_pierde_el_pan_de_maiz(
        go, monkeypatch):
    """Si sólo se arregla el scanner, el celíaco sigue sin ver la fila en «USA EXCLUSIVAMENTE»:
    el bloque tiene su propio filtro por alérgeno. Misma excusa en los dos sitios, no una segunda
    tabla."""
    # `tests/conftest.py` apaga VERIFIED-ONLY para preservar el baseline histórico de la suite;
    # el bloque sólo existe con el knob encendido (default ON en producción desde
    # P1-VERIFIED-ONLY-DEFAULT-ON). La caché es de módulo y está keyed por (excluidos, país):
    # sin vaciarla, una entrada de otro test decidiría este.
    monkeypatch.setenv("MEALFIT_VERIFIED_INGREDIENTS_ONLY", "true")
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    go._VERIFIED_CATALOG_INSTRUCTION_CACHE.clear()
    bloque = go._get_verified_catalog_instruction({"country": "US", "allergies": ["Gluten"]})
    if not bloque:
        pytest.skip("catálogo verificado no disponible (sin DB o knob apagado)")
    assert "émola de maíz" in bloque, "la fila sin gluten sigue amputada del catálogo del celíaco"
    assert "Pan de maíz" not in bloque, "el catálogo del celíaco le ofrece pan de trigo"
    go._VERIFIED_CATALOG_INSTRUCTION_CACHE.clear()
