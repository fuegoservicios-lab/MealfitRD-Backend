"""[P1-PREG-MERCURY-COUNTRY · 2026-08-23] El guard de mercurio en embarazo no conocía los nombres
con los que el pez espada y el tiburón se VENDEN fuera de RD.

LO MEDIDO (`clinical_backstop_for_meal`, country='ES', Embarazo), ANTES de tocar nada:

    Pez espada a la plancha → 1        Emperador a la plancha → 0
    Tiburón guisado         → 2        Cazón en adobo         → 0
    Blanquillo al horno     → 2

«Emperador» es EL nombre comercial del pez espada en toda pescadería española y «cazón en adobo»
es un plato andaluz clásico. La lista del motor era la de la FDA traducida al español dominicano:
nombraba la ESPECIE, no el producto. AESAN nombra literalmente «pez espada/emperador, atún rojo,
tiburón (cazón, marrajo, mielgas, pintarroja y tintorera) y lucio».

POR QUÉ ES P1 AUNQUE EL AUDIT LO LISTE COMO P2. Swap, regenerate-day y chat-modify NO pasan por el
grafo: ni reviewer clínico ni capa determinista. Su ÚNICA defensa determinista es este backstop, y
el metilmercurio es teratógeno.

LAS DOS TRAMPAS, ancladas abajo como tests:
  · El 'atún' enlatado sigue PERMITIDO a propósito (FDA Best/Good Choice en moderación). El token
    nuevo es 'atun rojo', de dos palabras: si alguien lo acorta a 'atun', el test de la ensalada
    de atún se pone rojo.
  · El matcher es `token in texto` (substring, no word-boundary), así que un token corto puede
    casar dentro de otra palabra. 'peto' (wahoo) se quedó FUERA por eso: «espeto de sardinas» lo
    contiene, y las sardinas son pescado recomendado en embarazo. El test lo vigila.

Byte-identidad DO: barrido de los 20 candidatos contra 7.458 cadenas vivas (catálogo con alias y
`name_en`, los JSON de platos de los 6 países, los 24 pools) ⇒ cero coincidencias. Ese barrido es
el test `test_ningun_token_de_mercurio_colisiona_...` de abajo, ahora automático.
"""
from __future__ import annotations

import pytest

from condition_rules import _PREGNANCY_MERCURY_SUBS
from constants import (COUNTRY_POOLS, COUNTRY_PROFILES, DOMINICAN_CARBS, DOMINICAN_FRUITS,
                       DOMINICAN_PROTEINS, DOMINICAN_VEGGIES_FATS, strip_accents)


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


def _backstop(go, texto, *, embarazo=True, country="ES"):
    fd = {"country": country}
    if embarazo:
        fd["medicalConditions"] = ["Embarazo"]
    return go.clinical_backstop_for_meal({"name": texto, "ingredients": [texto]}, form_data=fd)


_TOKENS = tuple(_PREGNANCY_MERCURY_SUBS[0][0])

#: Nombres comerciales que un usuario beta ve en su pescadería. Ninguno nombra una especie NUEVA:
#: son los nombres de las que ya estaban vetadas (emperador = pez espada; los cinco escualos =
#: tiburón; carite/sierra = king mackerel, ya presente como «caballa gigante»/«macarela rey»).
_DEBE_BLOQUEAR = [
    "Emperador a la plancha", "Cazón en adobo", "Marrajo al horno", "Tintorera guisada",
    "Pintarroja frita", "Mielga en salsa", "Lucio al horno", "Atún rojo en tataki",
    "Swordfish steak", "Shark bites", "Carite guisado", "Sierra frita",
    # los que YA bloqueaban: no pueden dejar de hacerlo
    "Pez espada a la plancha", "Tiburón guisado", "Blanquillo al horno",
]

#: La dirección opuesta: lo que NO puede empezar a bloquearse.
_NO_DEBE_BLOQUEAR = [
    "Ensalada de atún",           # atún light/enlatado: FDA Best/Good Choice, exclusión declarada
    "Sándwich de atún claro",
    "Espeto de sardinas",         # 'espeto' contiene 'peto': la colisión que dejó wahoo fuera
    "Filete de tilapia",
    "Pescado blanco al vapor",
    "Salmón a la parrilla",
]


@pytest.mark.parametrize("texto", _DEBE_BLOQUEAR)
def test_el_backstop_reconoce_el_nombre_con_el_que_se_vende(go, texto):
    assert _backstop(go, texto), f"{texto!r} atravesó el backstop de mercurio en embarazo"


@pytest.mark.parametrize("texto", _NO_DEBE_BLOQUEAR)
def test_el_backstop_no_se_lleva_por_delante_lo_permitido(go, texto):
    assert not _backstop(go, texto), f"{texto!r} se marcó como pescado alto en mercurio"


@pytest.mark.parametrize("texto", ["Emperador a la plancha", "Cazón en adobo", "Carite guisado"])
def test_sin_embarazo_ni_lactancia_el_guard_no_existe(go, texto):
    assert not _backstop(go, texto, embarazo=False)


@pytest.mark.parametrize("cc", sorted(COUNTRY_PROFILES))
def test_la_cobertura_es_de_TODOS_los_paises_incluida_rd(go, cc):
    """El nombre comercial no es una excepción por país: un dominicano puede escribir «cazón» y un
    español «pez espada». La tabla es única (SSOT) y no se filtra por país a propósito."""
    assert _backstop(go, "Emperador a la plancha", country=cc)


# ── El barrido de colisiones, ahora automático ──────────────────────────────────────────────────

def _cadenas_offline():
    fuera = []
    for pool in (DOMINICAN_PROTEINS, DOMINICAN_CARBS, DOMINICAN_VEGGIES_FATS, DOMINICAN_FRUITS):
        fuera.extend(pool)
    for pools in COUNTRY_POOLS.values():
        for lst in pools.values():
            fuera.extend(lst)
    return fuera


def test_ningun_token_de_mercurio_colisiona_con_un_nombre_de_pool():
    """El matcher es substring: un token que cae dentro de otro nombre convierte comida sana en
    violación. Es la 17ª colisión documentada del proyecto (sal⊂salsa, pollo⊂repollo)."""
    choques = []
    for tok in _TOKENS:
        t = strip_accents(str(tok).lower())
        if not t:
            continue
        for nombre in _cadenas_offline():
            if t in strip_accents(str(nombre).lower()):
                choques.append((tok, nombre))
    assert not choques, f"tokens de mercurio que casan dentro de un nombre de pool: {choques}"


@pytest.mark.e2e
def test_ningun_token_de_mercurio_colisiona_con_el_catalogo_vivo():
    try:
        from shopping_calculator import get_master_ingredients
        filas = get_master_ingredients() or []
    except Exception as e:  # pragma: no cover
        pytest.skip(f"catálogo no disponible: {e}")
    if not filas:
        pytest.skip("catálogo vacío (¿pool de Neon sin abrir?)")
    heno = []
    for r in filas:
        heno.append(str(r.get("name") or ""))
        heno.extend(str(a) for a in (r.get("aliases") or []))
        if r.get("name_en"):
            heno.append(str(r.get("name_en")))
    choques = []
    for tok in _TOKENS:
        t = strip_accents(str(tok).lower())
        if not t:
            continue
        choques.extend((tok, h) for h in heno if t in strip_accents(h.lower()))
    assert not choques, f"tokens de mercurio que casan dentro del catálogo vivo: {choques}"


# ── El SSOT no se puede reorganizar en silencio ─────────────────────────────────────────────────

def test_la_fila_indexada_como_ssot_sigue_siendo_la_del_mercurio():
    """`_scan_mercury_pregnancy_violations` indexa `_PREGNANCY_MERCURY_SUBS[0][0]`. Si alguien
    antepone otra fila (como ya pasó al nacer las tuplas de uterotónico y fruta), el scanner
    empezaría a vigilar otra cosa sin que nadie lo note."""
    fila = _PREGNANCY_MERCURY_SUBS[0]
    assert fila[1] == "Filete de pescado blanco"
    assert "mercurio" in fila[2]
    assert isinstance(fila[0], tuple) and len(fila[0]) >= 20
