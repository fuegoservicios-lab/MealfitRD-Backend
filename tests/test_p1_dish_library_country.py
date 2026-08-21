"""[P1-DISH-LIBRARY-COUNTRY · 2026-08-21] Los `dish_templates_<cc>.json` de Fase 2 existían y
sólo los leía el JUEZ. El generador seguía recibiendo la biblioteca dominicana.

Fase 2 creó cinco archivos de plantillas por país —55 (ES), 49 (MX), 51 (CO), 48 (PR), 48 (US)—
y un resolvedor de ruta, `_dish_templates_path_for_country`, cuyo ÚNICO lector es
`_culinary_judge_rubric_for_country`. `dish_library` nunca se enteró: `_TEMPLATES_PATH` es una
constante de módulo apuntando a `data/dish_templates.json` y `build_dish_library_context` no tenía
parámetro de país. Resultado medido:

    inspect.signature(build_dish_library_context) -> (skeleton_day, day_num) -> str

Un usuario de México recibía, en el tramo DINÁMICO del prompt —el más concreto y el más cercano a
la generación—, un bloque encabezado «🍽️ INSPIRACIÓN DOMINICANA» con ocho platos dominicanos por
día: Mangú, Yaniqueques, Revoltillo con casabe, Chivo guisado estilo liniero, Tipile, Majarete.
Sus 49 plantillas mexicanas estaban en disco y ningún nodo generador las abría.

POR QUÉ IMPORTA LA POSICIÓN. Es el modo de fallo que este repo ya midió en
P1-DIET-BLIND-DIRECTIVES: una directiva de cabecera SOLA pierde contra órdenes específicas. La
cabecera beta de Fase 1 dice «los platos dominicanos NO son requisito ni default»; veinte mil
caracteres después, este bloque le ponía ocho platos dominicanos concretos en la mano. Entre una
declaración general y un ejemplo concreto, el modelo obedece al ejemplo.

TRES SUPERFICIES, UN MÓDULO. `build_dish_library_context` (day-gen),
`build_swap_inspiration_context` (swap y chat-modify) y `_dish_template_class_counts`
(`ai_helpers`, cobertura del seeder) leen la MISMA biblioteca. Las tres eran país-ciegas; el fix
las cubre a las tres, porque arreglar sólo la primera dejaría el swap de un plan mexicano
devolviendo mangú — la asimetría que P1-COUNTRY-SYSTEM-F1 ya pagó una vez con los callers no
gateados de `slot_coherence_backstop_for_meal`.

LA CACHÉ. `load_dish_templates` cacheaba en UN global. Con rutas por país, ese global sirve el
primer archivo que se cargue a todos los países que vengan después — exactamente la trampa que
`_VERIFIED_CATALOG_INSTRUCTION_CACHE` tenía en P1-VERIFIED-CATALOG-COUNTRY. Pasa a ser un dict
por ruta.

Cubre:
  A. Byte-identidad DO (el contrato de Fase 1/2) y con el knob apagado.
  B. Cada país beta recibe SUS plantillas y no las dominicanas.
  C. El encabezado deja de decir «DOMINICANA» para un país beta.
  D. La caché por ruta no mezcla países.
  E. El swap recibe la biblioteca de su país.
  F. Parser-based: el call site del day-gen threadea el país que ya recibe.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DATA = _BACKEND_ROOT / "data"
_DAYGEN_PATH = _BACKEND_ROOT / "prompts" / "day_generator.py"

# Platos-firma por país, leídos de los propios archivos de Fase 2 (no inventados aquí: si alguien
# reescribe un JSON, el test se apoya en su contenido real).
_ARCHIVOS = {
    "DO": "dish_templates.json",
    "ES": "dish_templates_es.json",
    "MX": "dish_templates_mx.json",
    "CO": "dish_templates_co.json",
    "PR": "dish_templates_pr.json",
    "US": "dish_templates_us.json",
}


def _nombres(cc):
    data = json.loads((_DATA / _ARCHIVOS[cc]).read_text(encoding="utf-8"))
    return {str(t.get("name") or "") for t in (data.get("templates") or []) if t.get("name")}


@pytest.fixture(scope="module")
def dl():
    import dish_library as _dl
    return _dl


@pytest.fixture(autouse=True)
def cache_limpia(dl):
    """La biblioteca cachea por módulo: sin limpiar, el primer país que cargue fija el resto."""
    for attr in ("_CACHE", "_CACHE_BY_PATH"):
        if hasattr(dl, attr):
            val = getattr(dl, attr)
            setattr(dl, attr, {} if isinstance(val, dict) else None)
    yield


@pytest.fixture
def knob_on(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")


_ESQUELETO = {
    "protein_pool": ["Pollo", "Huevos", "Habichuelas rojas"],
    "meal_types": ["Desayuno", "Almuerzo", "Merienda", "Cena"],
}


def _ctx(dl, country=None):
    if country is None:
        return dl.build_dish_library_context(dict(_ESQUELETO), 1)
    return dl.build_dish_library_context(dict(_ESQUELETO), 1, country=country)


# ── A. Byte-identidad dominicana ────────────────────────────────────────────────────────────────

def test_do_es_identico_a_no_declarar_pais(dl, knob_on):
    """El contrato de toda Fase 1/2: un dominicano no puede notar que el sistema existe. Se
    compara declarar 'DO' contra no declarar nada, que es como llamaban los call sites de antes."""
    assert _ctx(dl, "DO") == _ctx(dl)


def test_el_pais_beta_cae_a_dominicana_con_el_knob_apagado(dl, monkeypatch):
    """Rollback de emergencia: quitar `MEALFIT_COUNTRY_SYSTEM` devuelve el motor a conducta
    dominicana aunque el frontend siga mostrando el selector. El gate vive en la única puerta."""
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    assert _ctx(dl, "ES") == _ctx(dl, "DO")


# ── B. Cada país recibe SUS plantillas ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("cc", ["ES", "MX", "CO", "PR", "US"])
def test_el_pais_beta_recibe_platos_de_su_propia_biblioteca(dl, knob_on, cc):
    """RED pre-fix: los 5 recibían el bloque dominicano. Se comprueba contra los nombres REALES
    del JSON del país, no contra una lista escrita a mano en el test."""
    bloque = _ctx(dl, cc)
    propios = _nombres(cc)
    assert bloque, f"{cc} no recibió ningún bloque de inspiración"
    assert any(n in bloque for n in propios), (
        f"{cc} no recibió NINGUNA de sus {len(propios)} plantillas propias"
    )


@pytest.mark.parametrize("cc", ["ES", "MX", "CO", "PR", "US"])
def test_el_pais_beta_no_recibe_los_platos_exclusivamente_dominicanos(dl, knob_on, cc):
    """El otro lado: no basta con añadir los propios si siguen llegando mangú y chivo liniero.
    Se miden los nombres que SÓLO están en el archivo dominicano."""
    solo_do = _nombres("DO") - _nombres(cc)
    bloque = _ctx(dl, cc)
    intrusos = sorted(n for n in solo_do if n in bloque)
    assert not intrusos, f"{cc} sigue recibiendo platos exclusivamente dominicanos: {intrusos}"


# ── C. El encabezado ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cc,nombre", [("ES", "España"), ("MX", "México"), ("CO", "Colombia")])
def test_el_encabezado_nombra_el_pais_del_usuario(dl, knob_on, cc, nombre):
    """«INSPIRACIÓN DOMINICANA» era el título del bloque para todo el mundo. El nombre sale de
    `COUNTRY_PROFILES[cc]['name_es']`, que es el mismo SSOT que usa el juez culinario — no una
    segunda tabla de gentilicios."""
    bloque = _ctx(dl, cc)
    assert "DOMINICANA" not in bloque.upper(), f"{cc} sigue leyendo «INSPIRACIÓN DOMINICANA»"
    assert nombre.upper() in bloque.upper(), f"el encabezado de {cc} no nombra a {nombre}"


def test_el_encabezado_dominicano_no_cambia(dl, knob_on):
    """Control del anterior."""
    assert "INSPIRACIÓN DOMINICANA" in _ctx(dl, "DO")


# ── D. La caché por ruta ────────────────────────────────────────────────────────────────────────

def test_la_cache_no_sirve_la_biblioteca_de_un_pais_a_otro(dl, knob_on):
    """`load_dish_templates` cacheaba en UN global. Se pide DO primero a propósito: es el orden
    que enmascara el bug, y en un backend real el primer usuario del proceso decide el resto."""
    do = _ctx(dl, "DO")
    es = _ctx(dl, "ES")
    assert es != do, "la caché sirvió la biblioteca dominicana a un usuario español"
    assert any(n in es for n in _nombres("ES"))


def test_load_dish_templates_devuelve_listas_distintas_por_pais(dl):
    """Directo sobre el cargador, sin pasar por el render: dos rutas, dos listas."""
    p_do = str(_DATA / _ARCHIVOS["DO"])
    p_es = str(_DATA / _ARCHIVOS["ES"])
    assert dl.load_dish_templates(p_do) != dl.load_dish_templates(p_es)
    assert len(dl.load_dish_templates(p_es)) > 0


# ── E. El swap ──────────────────────────────────────────────────────────────────────────────────

def test_el_swap_recibe_la_biblioteca_de_su_pais(dl, knob_on):
    """`build_swap_inspiration_context` alimenta swap individual y chat-modify. Arreglar sólo el
    day-gen dejaría el swap de un plan español devolviendo mangú — la misma asimetría de callers
    no gateados que Fase 1 tuvo que barrer dos veces."""
    es = dl.build_swap_inspiration_context("Almuerzo", seed=3, country="ES")
    do = dl.build_swap_inspiration_context("Almuerzo", seed=3, country="DO")
    assert es and do
    assert es != do
    solo_do = _nombres("DO") - _nombres("ES")
    assert not [n for n in solo_do if n in es]


# ── F. Parser-based ─────────────────────────────────────────────────────────────────────────────

def test_el_call_site_del_daygen_threadea_el_pais():
    """`build_day_assignment_context` YA recibía `country=` desde `graph_orchestrator`: el bloque
    de inspiración era el único de su cuerpo que no lo pasaba. El threading es de una línea, y
    este guard impide que un refactor lo suelte."""
    src = _DAYGEN_PATH.read_text(encoding="utf-8", errors="replace")
    i = src.find("build_dish_library_context(")
    assert i > 0, "el call site desapareció"
    assert "country" in src[i:i + 220], (
        "el call site del day-gen volvió a llamar a la biblioteca sin país"
    )


def test_el_resolvedor_de_ruta_sigue_siendo_el_unico_ssot():
    """La ruta por país se resuelve en `_dish_templates_path_for_country` (Fase 2). `dish_library`
    debe LLAMARLO, no escribir una segunda tabla de rutas — la lección de P1-DIET-CANON-SSOT."""
    src = (_BACKEND_ROOT / "dish_library.py").read_text(encoding="utf-8", errors="replace")
    assert "_dish_templates_path_for_country" in src
    assert "P1-DISH-LIBRARY-COUNTRY" in src
    assert src.count("dish_templates_es.json") == 0, (
        "dish_library escribió su propia tabla de rutas por país"
    )
