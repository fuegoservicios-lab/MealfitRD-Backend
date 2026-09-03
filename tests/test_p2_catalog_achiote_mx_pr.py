"""[P2-CATALOG-ACHIOTE-MX-PR · 2026-08-23] El prompt le decía al mexicano y al puertorriqueño que
OMITIERA el achiote y en la misma pantalla se lo ofrecía.

Medido antes del arreglo, cruzando el render del catálogo verificado contra su propia prosa
(knob de países encendido, catálogo vivo de 347 filas):

    DO ['salsa de soya'] · ES ['salsa de soya'] · US ['salsa de soya'] · CO ['salsa de soya']
    MX ['achiote', 'salsa de soya']   ·   PR ['achiote', 'salsa de soya']
    Filas en el render de MX: 'Aceite de achiote', 'Achiote', 'Sazón con culantro y achiote'
    Filas en el render de PR: 'Aceite de achiote', 'Sazón con culantro y achiote'

Al derivar los ejemplos de la propia lista apareció un CUARTO contradicho que la lista escrita a
mano no sabía: 'mostaza', fila viva en los seis países.

Es la misma auto-contradicción que `P1-SPICES-CATALOG-SYNC` arregló a mano el 2026-07-01, con su
modo de fallo ya medido: «el LLM omitía sazones legítimas y los guisos salían desabridos». El
achiote es la base del pernil y del sofrito puertorriqueño. Aquella vez se arregló escribiendo otra
lista a mano, y volvió a driftar en cuanto Fase 2 dio de alta 141 filas — por eso ahora los
ejemplos no se afirman, se DERIVAN del catálogo que ese mismo prompt ofrece.

SON DOS SUPERFICIES: la prosa del bloque de catálogo (`graph_orchestrator`) y la regla 5 del
day-gen (`prompts/day_generator`), que además gritaba «SALSA DE SOYA» en mayúsculas. Las dos se
alimentan de la MISMA tupla de ejemplos: dos listas serían dos tablas.

Los tests no tocan Neon: el catálogo va mockeado (`monkeypatch`) para que la propiedad se pruebe
contra un catálogo CONOCIDO en vez de contra el estado de la base ese día.
"""
from __future__ import annotations

import pytest

import graph_orchestrator as go
from prompts.day_generator import (
    DAY_GENERATOR_SYSTEM_PROMPT,
    PROHIBITED_EXAMPLE_FOODS,
    RULE5_PROHIBITED_EXAMPLES_LITERAL,
    prohibited_examples_not_offered,
    strip_offered_prohibited_examples,
)

# Catálogo mínimo: ofrece achiote (en TRES grafías, como MX) y mostaza; NO ofrece clavo dulce.
_CATALOGO_CON_ACHIOTE = [
    {"name": "Achiote", "price_per_lb": 0},
    {"name": "Aceite de achiote", "price_per_lb": 0},
    {"name": "Sazón con culantro y achiote", "price_per_lb": 0},
    {"name": "Mostaza", "price_per_lb": 1.5},
    {"name": "Pechuga de pollo", "price_per_lb": 2.0},
    {"name": "Arroz blanco", "price_per_lb": 0.5},
]
_CATALOGO_SIN_ACHIOTE = [r for r in _CATALOGO_CON_ACHIOTE if "achiote" not in r["name"].lower()]


@pytest.fixture
def catalogo(monkeypatch):
    """Sustituye el catálogo vivo y vacía las dos cachés (bloque + nombres) en cada caso."""
    import shopping_calculator as sc

    def _instalar(rows, country="DO"):
        monkeypatch.setattr(sc, "get_master_ingredients", lambda *a, **k: list(rows))
        monkeypatch.setattr(sc, "_verified_ingredients_only_enabled", lambda *a, **k: True)
        monkeypatch.setattr(sc, "is_country_catalog_unpriced_item", lambda *a, **k: True,
                            raising=False)
        monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
        go._VERIFIED_CATALOG_INSTRUCTION_CACHE.clear()
        go._VERIFIED_CATALOG_NAMES_CACHE.clear()
        return {"country": country}

    yield _instalar
    go._VERIFIED_CATALOG_INSTRUCTION_CACHE.clear()
    go._VERIFIED_CATALOG_NAMES_CACHE.clear()


# ── A · el literal de la regla 5 sigue siendo el que el prompt lleva dentro ───────────────────
def test_el_literal_de_ejemplos_vive_en_la_regla_5():
    """Ancla del guard: si alguien reescribe la regla 5, el filtro se vuelve un no-op silencioso."""
    assert RULE5_PROHIBITED_EXAMPLES_LITERAL in DAY_GENERATOR_SYSTEM_PROMPT, (
        "el literal de ejemplos de la regla 5 ya no coincide con `PROHIBITED_EXAMPLE_FOODS`: el "
        "strip dejaría de podar nada y el prompt volvería a contradecir al catálogo en silencio."
    )


# ── B · la propiedad: un ejemplo ofrecido no puede seguir siendo ejemplo de prohibido ─────────
def test_lo_que_el_catalogo_ofrece_deja_de_ser_ejemplo_de_prohibido():
    ofrecidos = ["Aceite de achiote", "Salsa de soya", "Mostaza"]
    kept = prohibited_examples_not_offered(PROHIBITED_EXAMPLE_FOODS, ofrecidos)
    bajos = [k.lower() for k in kept]
    assert "achiote" not in bajos, "'achiote' sobrevive aunque el catálogo ofrezca 'Aceite de achiote'"
    assert "salsa de soya" not in bajos
    assert "mostaza" not in bajos


def test_lo_que_el_catalogo_no_ofrece_sobrevive():
    """PERDER LA ADVERTENCIA ES EL FALLO CARO: el filtro poda, no vacía."""
    kept = prohibited_examples_not_offered(PROHIBITED_EXAMPLE_FOODS, ["Aceite de achiote"])
    bajos = [k.lower() for k in kept]
    assert "clavo dulce" in bajos and "teriyaki" in bajos, (
        "el filtro se comió ejemplos que el catálogo NO ofrece: el prompt pierde la advertencia."
    )
    assert len(kept) == len(PROHIBITED_EXAMPLE_FOODS) - 1


def test_sin_catalogo_los_ejemplos_quedan_intactos():
    """Fail-open: catálogo vacío/no disponible ⇒ el prompt no cambia."""
    assert strip_offered_prohibited_examples(DAY_GENERATOR_SYSTEM_PROMPT, []) == DAY_GENERATOR_SYSTEM_PROMPT
    assert strip_offered_prohibited_examples(DAY_GENERATOR_SYSTEM_PROMPT, None) == DAY_GENERATOR_SYSTEM_PROMPT


def test_el_strip_es_idempotente():
    ofrecidos = ["Achiote", "Salsa de soya", "Mostaza"]
    una = strip_offered_prohibited_examples(DAY_GENERATOR_SYSTEM_PROMPT, ofrecidos)
    dos = strip_offered_prohibited_examples(una, ofrecidos)
    assert una == dos


# ── C · superficie 1: la prosa del bloque de catálogo ─────────────────────────────────────────
def _frase_de_prohibidos(bloque: str) -> str:
    i = bloque.find("Si una receta tradicional pide algo que no está aquí")
    assert i >= 0, "la frase de «prohibido» ya no está en el bloque de catálogo"
    return bloque[i:bloque.find("OMÍTELO", i)]


def test_el_bloque_de_catalogo_no_prohibe_lo_que_ofrece(catalogo):
    fd = catalogo(_CATALOGO_CON_ACHIOTE, country="MX")
    bloque = go._get_verified_catalog_instruction(fd)
    assert "Achiote" in bloque, "el catálogo mockeado debería estar ofreciendo achiote"
    frase = _frase_de_prohibidos(bloque)
    assert "achiote" not in frase.lower(), (
        "el bloque sigue poniendo el achiote como ejemplo de prohibido mientras lo ofrece en la "
        f"lista de abajo: {frase!r}"
    )
    assert "mostaza" not in frase.lower()


def test_el_bloque_si_prohibe_lo_que_no_ofrece(catalogo):
    fd = catalogo(_CATALOGO_SIN_ACHIOTE, country="MX")
    frase = _frase_de_prohibidos(go._get_verified_catalog_instruction(fd))
    assert "achiote" in frase.lower(), (
        "sin achiote en el catálogo el ejemplo debe seguir ahí — si no, el arreglo borró la "
        "advertencia en vez de sincronizarla."
    )


# ── D · superficie 2: la regla 5 del day-gen, ya ensamblada ───────────────────────────────────
def test_la_regla_5_del_daygen_pierde_el_ejemplo_ofrecido(catalogo):
    fd = catalogo(_CATALOGO_CON_ACHIOTE, country="MX")
    ensamblado = go._strip_offered_prohibited_examples_for(DAY_GENERATOR_SYSTEM_PROMPT, fd)
    i = ensamblado.find("Si una receta tradicional pide algo que no está en el catálogo")
    assert i >= 0, "la regla 5 perdió su frase — mueve el guard con ella"
    frase = ensamblado[i:ensamblado.find("OMÍTELO", i)]
    assert "achiote" not in frase.lower(), f"la regla 5 sigue contradiciendo al catálogo: {frase!r}"
    assert "mostaza" not in frase.lower()
    assert "clavo dulce" in frase.lower(), "se perdió un ejemplo que el catálogo NO ofrece"


def test_la_regla_dura_nunca_se_toca(catalogo):
    """Se podan los EJEMPLOS, jamás la prohibición."""
    fd = catalogo(_CATALOGO_CON_ACHIOTE, country="MX")
    ensamblado = go._strip_offered_prohibited_examples_for(DAY_GENERATOR_SYSTEM_PROMPT, fd)
    assert "PROHIBIDO ABSOLUTO inventar o usar cualquier alimento fuera del catálogo" in ensamblado


def test_el_wrapper_es_fail_open(monkeypatch):
    """Un fallo del catálogo devuelve el prompt intacto: sin prompt no hay plan."""
    monkeypatch.setattr(go, "_verified_catalog_names",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert go._strip_offered_prohibited_examples_for(DAY_GENERATOR_SYSTEM_PROMPT, {}) == \
        DAY_GENERATOR_SYSTEM_PROMPT


# ── E · las dos superficies comparten tupla (no hay segunda tabla) ────────────────────────────
def test_el_bloque_de_catalogo_usa_la_tupla_ssot(catalogo):
    """Si alguien reescribe los ejemplos a mano en graph_orchestrator, las dos vuelven a driftar."""
    fd = catalogo(_CATALOGO_SIN_ACHIOTE, country="DO")
    frase = _frase_de_prohibidos(go._get_verified_catalog_instruction(fd))
    esperados = [e.lower() for e in prohibited_examples_not_offered(
        PROHIBITED_EXAMPLE_FOODS, go._verified_catalog_names(fd))]
    for e in esperados:
        assert e in frase.lower(), (
            f"'{e}' sale de la tupla SSOT y no aparece en la prosa del bloque: hay una segunda "
            "lista escrita a mano."
        )
