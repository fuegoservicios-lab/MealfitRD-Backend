"""[P1-COUNTRY-CATALOG-BY-COUNTRY · 2026-08-21] El catálogo beta era el mismo para los 5 países.

`P1-VERIFIED-CATALOG-COUNTRY` (esta misma ola, hace unas horas) arregló la sub-inclusión: el bloque
«USA EXCLUSIVAMENTE ESTOS ALIMENTOS» filtraba por precio RD, las 141 filas beta nacieron sin precio
a propósito, y por eso el render para España era byte-idéntico al dominicano. Lo cerré reusando
`is_country_catalog_unpriced_item` — el MISMO predicado del agregador, para no crear un segundo
espejo que driftara.

Y ahí declaré un precio aceptado que resultó ser evitable. Medido hoy sobre el render vivo:

    DO: 3824 chars ·  ES: 5777 ·  MX: 5777 ·  US: 5777   ← los tres beta, IDÉNTICOS

O sea que al español se le ofrecen huitlacoche, nopales, xoconostle, chontaduro, bagels y pretzels;
al mexicano, percebes y turrón. El catálogo pasó de «sólo dominicano» a «no-dominicano», que es
exactamente la queja de P1-BETA-FRAGMENT-DEPTH aplicada a los datos en vez de al prompt.

POR QUÉ LO DI POR IRRESOLUBLE Y POR QUÉ NO LO ERA. Escribí: «acotar por país es tarea de DATOS: no
existe membresía por país en `master_ingredients`». La primera mitad es cierta —la tabla no tiene
columna de país— y por eso no busqué más. Pero la membresía SÍ existe: vive en la propia tupla de
tokens, agrupada en bloques `T5 (ES)`, `T6 (MX/CO)`, `T7 (PR/US)` y `Task 8 (RD)`… **en un
comentario**. Estructura real, escrita a mano, que ningún programa puede leer. Es la clase que este
repo lleva pagando todo el día: el catálogo de F2 también era inerte, y por la misma razón —el dato
estaba, pero nada lo consultaba.

Este P-fix no inventa una clasificación: PROMUEVE la que ya estaba escrita. Los bloques son la
autoridad; yo no reasigno ningún token de país.

EL CAMBIO ES ADITIVO A PROPÓSITO. Los 4 call sites del agregador siguen preguntando sin país, y ahí
está bien: si un alimento español acaba en la lista de la compra, se conserva venga de donde venga
—el fallo caro ahí es perder comida en silencio—. El único que pregunta por país es el catálogo del
generador, que es una decisión de QUÉ OFRECER, no de qué conservar.

Cubre:
  A. La partición existe como dato y no pierde ni inventa ningún token.
  B. Ningún alimento se queda sin país (nadie pierde su keep).
  C. El predicado por país discrimina de verdad.
  D. Sin país ⇒ conducta de hoy, byte-idéntica (los 4 call sites del agregador).
  E. El catálogo del generador deja de ser el mismo para los 5 países.
  F. Byte-identidad dominicana.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def sc():
    import shopping_calculator as _sc
    return _sc


@pytest.fixture
def knob_on(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")


# El set histórico: 32 (T5·ES) + 46 (T6·MX/CO) + 62 (T7·PR/US) + 1 (Task 8·RD) = 141.
_TOTAL_HISTORICO = 141

_PAISES = ("ES", "MX", "CO", "PR", "US", "DO")


# ── A. La partición es dato, y es fiel ──────────────────────────────────────────────────────────

def test_la_particion_por_pais_existe_como_dato(sc):
    """Antes vivía en un comentario. Un comentario no lo puede leer un programa — y este repo ya
    tiene registrada esa clase de fallo varias veces en el mismo día."""
    m = sc._COUNTRY_CATALOG_UNPRICED_BY_COUNTRY
    assert isinstance(m, dict)
    assert set(m) == set(_PAISES), f"faltan/sobran países: {sorted(m)}"
    for cc, toks in m.items():
        assert isinstance(toks, tuple) and toks, f"{cc}: vacío o no es tupla"


def test_la_tupla_plana_se_deriva_y_no_pierde_ni_inventa(sc):
    """El ancla de seguridad de todo el cambio. La tupla plana es la que usan los 4 call sites del
    agregador; si al partirla se cae un token, un alimento desaparece de la lista de la compra sin
    aviso — el fallo que más teme el dueño. Se compara el CONJUNTO, no el orden."""
    plano = set(sc._COUNTRY_CATALOG_UNPRICED_TOKENS)
    union = set()
    for toks in sc._COUNTRY_CATALOG_UNPRICED_BY_COUNTRY.values():
        union |= set(toks)
    assert union == plano, (
        f"la unión por país no reproduce la tupla plana. "
        f"perdidos={sorted(plano - union)} inventados={sorted(union - plano)}"
    )
    assert len(plano) == _TOTAL_HISTORICO, (
        f"la tupla plana tiene {len(plano)} tokens, el histórico son {_TOTAL_HISTORICO}"
    )


@pytest.mark.parametrize("cc,n", [("ES", 32), ("MX", 28), ("CO", 18), ("PR", 19), ("US", 43),
                                  ("DO", 1)])
def test_cada_pais_conserva_el_tamano_de_su_bloque(sc, cc, n):
    """Los tamaños salen de los bloques del fuente, no de mi criterio: T5 declara 32 altas de ES,
    T6 declara 46 (28 MX + 18 CO), T7 declara 62 (19 PR + 43 US) y Task 8 una sola para RD."""
    assert len(sc._COUNTRY_CATALOG_UNPRICED_BY_COUNTRY[cc]) == n


# ── B. Nadie pierde su keep ─────────────────────────────────────────────────────────────────────

def test_ninguna_fila_del_catalogo_se_queda_sin_pais(sc):
    """La verificación que NO es circular: para cada fila viva que el predicado plano conserva,
    tiene que existir AL MENOS un país que también la conserve. Si una fila sólo la reclamaba la
    tupla plana, al partir se volvería inalcanzable para todos los países a la vez."""
    filas = sc.get_master_ingredients() or []
    if not filas:
        pytest.skip("catálogo no disponible (sin DB)")
    huerfanas = []
    for r in filas:
        nombre = str(r.get("name") or "")
        if not sc.is_country_catalog_unpriced_item(nombre):
            continue
        if not any(sc.is_country_catalog_unpriced_item(nombre, country=cc) for cc in _PAISES):
            huerfanas.append(nombre)
    assert not huerfanas, f"filas que ningún país reclama: {huerfanas}"


# ── C. El predicado por país discrimina ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("nombre,suyo,ajeno", [
    ("Jamón serrano", "ES", "MX"),
    ("Huitlacoche", "MX", "ES"),
    ("Chontaduro", "CO", "ES"),
    ("Pernil", "PR", "MX"),
    ("Pretzels", "US", "ES"),
])
def test_el_predicado_por_pais_discrimina(sc, nombre, suyo, ajeno):
    assert sc.is_country_catalog_unpriced_item(nombre, country=suyo) is True
    assert sc.is_country_catalog_unpriced_item(nombre, country=ajeno) is False


# ── D. Sin país ⇒ conducta de hoy ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("nombre", ["Jamón serrano", "Huitlacoche", "Chontaduro", "Pernil",
                                    "Pretzels", "Hummus"])
def test_sin_pais_conserva_la_conducta_historica(sc, nombre):
    """Los 4 call sites del agregador NO pasan país y no deben cambiar: ahí conservar de más es
    correcto (perder comida de la lista en silencio es el fallo caro), y este P-fix no los toca."""
    assert sc.is_country_catalog_unpriced_item(nombre) is True


@pytest.mark.parametrize("basura", [None, "", "  ", "ZZ", "basura", 42])
def test_un_pais_no_canonico_cae_a_la_conducta_historica(sc, basura):
    """Fail-open deliberado: el predicado corre en el camino caliente del agregador y una excepción
    ahí rompe la lista entera. Un país que no reconozco NO puede estrechar el filtro."""
    assert sc.is_country_catalog_unpriced_item("Jamón serrano", country=basura) is True


def test_el_pais_se_canonicaliza_por_el_ssot(sc):
    """`constants.canonicalize_country` es el ÚNICO SSOT (lección P1-DIET-CANON-SSOT). No se
    escribe aquí una segunda tabla de países."""
    for variante in ("es", "ES", "  es  "):
        assert sc.is_country_catalog_unpriced_item("Jamón serrano", country=variante) is True


# ── E. El catálogo del generador deja de ser el mismo para todos ────────────────────────────────

# Catálogo sintético: una fila sin precio por país + una fila PRECIADA (la dominicana, que es la
# que tiene mercado RD que cotizar). Se inyecta a propósito en vez de leer el catálogo vivo — la
# primera versión de estos tres tests hacía `pytest.skip` sin DB, o sea que las tres anclas del
# EFECTO del P-fix no corrían nunca en CI. Un guard que siempre salta es una coartada, no una
# defensa; ya me pasó hoy con tres guards que medían el entorno en vez del contrato.
_CATALOGO_FALSO = [
    {"name": "Arroz blanco", "price_per_lb": 35.0, "price_per_unit": 0},   # PRECIADA (RD)
    {"name": "Jamón serrano", "price_per_lb": 0, "price_per_unit": 0},     # ES
    {"name": "Huitlacoche", "price_per_lb": 0, "price_per_unit": 0},       # MX
    {"name": "Chontaduro", "price_per_lb": 0, "price_per_unit": 0},        # CO
    {"name": "Pernil", "price_per_lb": 0, "price_per_unit": 0},            # PR
    {"name": "Pretzels", "price_per_lb": 0, "price_per_unit": 0},          # US
    {"name": "Percebes", "price_per_lb": 0, "price_per_unit": 0},          # ES
]


@pytest.fixture
def catalogo(monkeypatch, knob_on):
    """Inyecta el catálogo sintético y vacía la caché de módulo (que está keyed por país desde
    P1-VERIFIED-CATALOG-COUNTRY, así que sin limpiarla el primer país fijaría el bloque)."""
    import graph_orchestrator as go
    import shopping_calculator as _sc
    monkeypatch.setattr(_sc, "get_master_ingredients", lambda *a, **k: list(_CATALOGO_FALSO))
    monkeypatch.setattr(_sc, "_verified_ingredients_only_enabled", lambda *a, **k: True)
    go._VERIFIED_CATALOG_INSTRUCTION_CACHE.clear()
    yield go
    go._VERIFIED_CATALOG_INSTRUCTION_CACHE.clear()


def test_el_catalogo_verificado_ya_no_es_identico_entre_paises(catalogo):
    """El síntoma medido que abre este P-fix: ES, MX y US devolvían el MISMO string de 5777 chars."""
    go = catalogo
    es = go._get_verified_catalog_instruction({"country": "ES"})
    mx = go._get_verified_catalog_instruction({"country": "MX"})
    assert es and mx
    assert es != mx, "el catálogo sigue siendo idéntico entre España y México"
    assert "Huitlacoche" not in es, "a un español se le sigue ofreciendo huitlacoche"
    assert "Percebes" not in mx, "a un mexicano se le siguen ofreciendo percebes"


def test_cada_pais_sigue_viendo_su_propia_comida(catalogo):
    """El error opuesto —y peor— sería estrechar tanto que volviera la sub-inclusión que
    P1-VERIFIED-CATALOG-COUNTRY acababa de cerrar: el español SIN su jamón serrano."""
    go = catalogo
    es = go._get_verified_catalog_instruction({"country": "ES"})
    mx = go._get_verified_catalog_instruction({"country": "MX"})
    assert "Jamón serrano" in es and "Percebes" in es
    assert "Huitlacoche" in mx


def test_lo_preciado_lo_ve_todo_el_mundo(catalogo):
    """Acotar por país sólo toca la rama SIN precio. El arroz —que sí tiene precio RD— sigue
    ofreciéndose a todos: estrechar eso dejaría a los países beta sin alimentos básicos."""
    go = catalogo
    for cc in _PAISES:
        assert "Arroz blanco" in go._get_verified_catalog_instruction({"country": cc})


# ── F. Byte-identidad dominicana ────────────────────────────────────────────────────────────────

def test_el_catalogo_dominicano_no_cambia(catalogo):
    """DO nunca pasó por la rama beta y sigue sin pasar."""
    go = catalogo
    do = go._get_verified_catalog_instruction({"country": "DO"})
    assert do
    for ajeno in ("Huitlacoche", "Percebes", "Pretzels", "Chontaduro", "Pernil", "Jamón serrano"):
        assert ajeno not in do, f"{ajeno} se coló en el catálogo dominicano"
