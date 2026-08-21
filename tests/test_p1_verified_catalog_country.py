"""[P1-VERIFIED-CATALOG-COUNTRY · 2026-08-21] El bloque que MANDA sobre los ingredientes era
país-ciego, y por eso el catálogo entero de Fase 2 era INERTE para la generación.

`_get_verified_catalog_instruction` cierra el system prompt del day-generator con:

    «=== CATÁLOGO VERIFICADO — USA EXCLUSIVAMENTE ESTOS ALIMENTOS ===
      … PROHIBIDO ABSOLUTO inventar o agregar cualquier alimento fuera de esta lista …»

y construía la lista con un solo predicado: `price_per_lb > 0 OR price_per_unit > 0`. Las 141
filas que Fase 2 dio de alta para los 5 países beta nacieron **sin precio a propósito** (son
`beta_no_prices`: el país no tiene precios nativos), así que quedaban fuera. Y las 206 filas
dominicanas —«Orégano dominicano», Casabe, Yautía, Auyama, Salami— quedaban dentro.

Medido contra Neon con `MEALFIT_COUNTRY_SYSTEM=true` antes del fix:

    _get_verified_catalog_instruction({'country':'ES'})
      == _get_verified_catalog_instruction({'country':'DO'})   ->  True  (byte-idéntico)
    'Jamón serrano' / 'Boquerones' / 'Acelgas' / 'Almejas'     ->  ausentes
    'Orégano dominicano' / 'Casabe' / 'Yautía'                 ->  presentes

y en los 2 planes beta vivos, 44 de 48 ítems (ES) y 22 de 25 (US) de la lista de compras eran
filas del catálogo dominicano. Los cuatro alimentos españoles que sí aparecieron entraron por
DESOBEDIENCIA del modelo, no por el sistema.

POR QUÉ ESTE VA PRIMERO. Fase 1 quitó la imposición criolla de los prompts NARRATIVOS y Fase 2
construyó el catálogo; este bloque es el que decide qué ingredientes existen, y no se enteró de
ninguna de las dos fases. Además gobierna TRES superficies con el mismo código —day-gen, swap
individual (`agent.py`) y chat-modify (`tools.py`)—, así que el defecto se repetía en cada
interacción, no sólo al crear el plan.

EL PREDICADO ELEGIDO, Y POR QUÉ NO OTRO. Se midió contra el catálogo vivo antes de decidir:

    pool por país (`COUNTRY_POOLS[cc]`)  ->  cubre sólo 22 de las filas sin precio de ES;
                                            55 filas sin precio no las reclama NINGÚN pool
                                            (Azafrán, Alioli, Aceitunas rellenas, Nata…)
    set global (`is_country_catalog_unpriced_item`) -> las 140, incluidas las mexicanas

Ninguno es perfecto. Se elige el segundo porque es EXACTAMENTE el predicado con el que el
agregador de la lista de compras ya decide qué es «comprable sin precio»: usar aquí un predicado
distinto crearía un segundo espejo que driftaría — la forma precisa del defecto que la costura (a)
del coherence guard costó. La sobre-inclusión que acepta (a un español se le ofrece chipotle) es
un problema de VARIEDAD que el fragmento de país del prompt ya combate; la sub-inclusión que
cierra (a un español se le PROHÍBE el jamón serrano) es el P1. Acotar por país es una tarea de
DATOS —no existe membresía por país en `master_ingredients`— y queda registrada aparte.

LA TRAMPA QUE SE OLVIDA. `_VERIFIED_CATALOG_INSTRUCTION_CACHE` estaba keyed SÓLO por el frozenset
de tokens excluidos por alergia/dieta. Arreglar el predicado sin meter el país en la clave deja el
fix INERTE: la caché sirve el bloque del primer país que llegó al proceso. Hay un test para eso.

Cubre:
  A. Byte-identidad DO (con el knob encendido y apagado) — el contrato de toda Fase 1/2.
  B. El país beta recibe SU comida y el bloque deja de ser byte-idéntico al dominicano.
  C. La caché distingue países.
  D. El filtro clínico de alergia/dieta sigue corriendo sobre la lista ampliada.
  E. Knob maestro apagado ⇒ conducta previa exacta.
  F. La prosa criolla del cierre no se le sirve a un país beta.
  G. Parser-based anchor.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_GO_PATH = _BACKEND_ROOT / "graph_orchestrator.py"


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


# Catálogo-sonda: filas reales por NOMBRE (los tokens de `_COUNTRY_CATALOG_UNPRICED_TOKENS` los
# reconocen), con la misma forma de precio que la DB viva — DO con precio, beta a cero.
_CATALOGO = [
    # Dominicanas, CON precio (las que hoy se sirven a todo el mundo)
    {"name": "Pollo", "price_per_lb": 95, "price_per_unit": 0},
    {"name": "Arroz blanco", "price_per_lb": 30, "price_per_unit": 0},
    {"name": "Habichuelas rojas", "price_per_lb": 60, "price_per_unit": 0},
    {"name": "Orégano dominicano", "price_per_lb": 0, "price_per_unit": 45},
    {"name": "Casabe", "price_per_lb": 0, "price_per_unit": 70},
    {"name": "Camarones", "price_per_lb": 320, "price_per_unit": 0},
    # Beta, SIN precio a propósito (las 141 altas de Fase 2)
    {"name": "Jamón serrano", "price_per_lb": 0, "price_per_unit": 0},
    {"name": "Boquerones", "price_per_lb": 0, "price_per_unit": 0},
    {"name": "Acelgas", "price_per_lb": 0, "price_per_unit": 0},
    {"name": "Almejas", "price_per_lb": 0, "price_per_unit": 0},
    {"name": "Chorizo español", "price_per_lb": 0, "price_per_unit": 0},
]

_BETA = ["Jamón serrano", "Boquerones", "Acelgas", "Chorizo español"]


@pytest.fixture
def catalogo(monkeypatch, go):
    """Catálogo determinista + caché limpia. La caché es de módulo: sin limpiarla, el primer test
    que corra fija el bloque para todos los demás — que es exactamente el modo de fallo que el
    test de la clave de caché persigue en producción."""
    import shopping_calculator as sc
    monkeypatch.setattr(sc, "get_master_ingredients", lambda *a, **k: list(_CATALOGO))
    monkeypatch.setattr(sc, "_verified_ingredients_only_enabled", lambda: True)
    go._VERIFIED_CATALOG_INSTRUCTION_CACHE.clear()
    yield
    go._VERIFIED_CATALOG_INSTRUCTION_CACHE.clear()


@pytest.fixture
def knob_on(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")


@pytest.fixture
def knob_off(monkeypatch):
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)


def _bloque(go, country, **extra):
    fd = {"country": country}
    fd.update(extra)
    return go._get_verified_catalog_instruction(fd)


# ── A. Byte-identidad dominicana ────────────────────────────────────────────────────────────────

def test_do_es_byte_identico_con_el_knob_encendido_y_apagado(go, catalogo, monkeypatch):
    """El contrato que sostiene TODA Fase 1 y 2: un usuario dominicano no puede notar que el
    sistema de países existe. Se compara el MISMO país por las dos ramas del knob."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    con_knob = _bloque(go, "DO")
    go._VERIFIED_CATALOG_INSTRUCTION_CACHE.clear()
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    sin_knob = _bloque(go, "DO")
    assert con_knob == sin_knob, "el bloque dominicano cambió al encender el sistema de países"


def test_do_sigue_sin_ver_las_filas_sin_precio(go, catalogo, knob_on):
    """Control negativo del fix: ampliar el predicado para beta NO debe ampliar el dominicano —
    en RD esas filas no tienen precio porque no se venden ahí, y el bloque promete «alimentos con
    precio verificado en el supermercado»."""
    do = _bloque(go, "DO")
    for nombre in _BETA:
        assert nombre not in do, f"'{nombre}' se coló en el catálogo dominicano"


# ── B. El país beta recibe SU comida ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("nombre", _BETA)
def test_el_pais_beta_recibe_los_alimentos_de_su_catalogo(go, catalogo, knob_on, nombre):
    """RED pre-fix: los 4 ausentes. Un plan español cuyo modelo tiene PROHIBIDO el jamón serrano
    no es un plan español."""
    assert nombre in _bloque(go, "ES"), (
        f"'{nombre}' sigue fuera del bloque que el modelo recibe como lista cerrada"
    )


def test_el_bloque_beta_deja_de_ser_byte_identico_al_dominicano(go, catalogo, knob_on):
    """RED pre-fix: `es == do` -> True, medido contra Neon (len 3824 en ambos). Era la prueba de
    que la mitad de Fase 2 no llegaba al motor."""
    assert _bloque(go, "ES") != _bloque(go, "DO")


def test_el_beta_conserva_las_filas_universales_con_precio(go, catalogo, knob_on):
    """La ampliación es UNIÓN, no sustitución: pollo y arroz siguen siendo comprables en España.
    Si el fix hubiera cambiado el predicado en vez de ampliarlo, el español se quedaría sin base."""
    es = _bloque(go, "ES")
    for nombre in ("Pollo", "Arroz blanco"):
        assert nombre in es


# ── C. La caché distingue países (la trampa que deja el fix inerte) ─────────────────────────────

def test_la_cache_no_sirve_el_bloque_de_un_pais_a_otro(go, catalogo, knob_on):
    """`_VERIFIED_CATALOG_INSTRUCTION_CACHE` estaba keyed sólo por los tokens excluidos por
    alergia/dieta. Con el predicado arreglado pero la clave intacta, el primer país que llegara al
    proceso fijaría el bloque para todos: en un backend de producción, el primer usuario decide lo
    que ven los demás. Se pide DO primero a propósito, que es el orden que enmascara el bug."""
    do_primero = _bloque(go, "DO")
    es_despues = _bloque(go, "ES")
    assert es_despues != do_primero, "la caché sirvió el bloque dominicano a un usuario español"
    assert "Jamón serrano" in es_despues


def test_la_cache_sigue_sirviendo_al_mismo_pais(go, catalogo, knob_on):
    """Control: la clave gana una dimensión, no deja de cachear."""
    primero = _bloque(go, "ES")
    assert _bloque(go, "ES") == primero


# ── D. El filtro clínico corre sobre la lista ampliada ──────────────────────────────────────────

def test_la_alergia_excluye_tambien_los_alimentos_beta(go, catalogo, knob_on):
    """Lo más importante del fix después de que funcione: las filas nuevas entran POR el mismo
    camino que las viejas, así que el filtro de alérgenos (`_verified_catalog_excluded_tokens`)
    las ve. Un español alérgico a mariscos no puede recibir «Almejas» en su lista cerrada — y
    'Almejas' es una de las altas de Fase 2, es decir una fila que antes de este P-fix ni siquiera
    llegaba aquí."""
    es = _bloque(go, "ES", allergies=["mariscos"])
    assert "Almejas" not in es, "una fila beta esquivó el filtro de alérgenos"
    assert "Camarones" not in es, "el filtro de alérgenos dejó de correr sobre las filas con precio"
    assert "Jamón serrano" in es, "el filtro se llevó por delante alimentos que no son mariscos"


def test_la_dieta_excluye_tambien_los_alimentos_beta(go, catalogo, knob_on):
    """Espejo del anterior para el eje de dieta: un vegano español no puede recibir jamón serrano
    ni chorizo en su lista cerrada."""
    es = _bloque(go, "ES", dietType="vegana")
    assert "Jamón serrano" not in es and "Chorizo español" not in es
    assert "Arroz blanco" in es


# ── E. Knob maestro apagado ⇒ conducta previa exacta ────────────────────────────────────────────

def test_con_el_knob_apagado_el_pais_declarado_se_ignora(go, catalogo, knob_off):
    """El rollback de emergencia documentado en el runbook: quitar `MEALFIT_COUNTRY_SYSTEM` del
    `.env` y reiniciar devuelve el motor a byte-identidad dominicana en segundos, AUNQUE el
    frontend siga mostrando el selector. Ese contrato pasa por la única puerta
    (`country_for_form_data`), no por un `if` nuevo aquí."""
    es_sin_knob = _bloque(go, "ES")
    for nombre in _BETA:
        assert nombre not in es_sin_knob, f"'{nombre}' apareció con el sistema de países APAGADO"


# ── F. La prosa criolla no viaja a un país beta ─────────────────────────────────────────────────

def test_el_bloque_beta_no_pide_sabor_criollo(go, catalogo, knob_on):
    """El cierre del bloque decía «úsalos para dar sabor criollo real» — a un usuario de Madrid,
    en el último bloque del prompt, que es el que más pesa. El resto del stack beta le está
    pidiendo cocina española al mismo tiempo."""
    es = _bloque(go, "ES").lower()
    assert "criollo" not in es, "el bloque le sigue pidiendo sabor criollo a un país beta"


def test_el_bloque_dominicano_conserva_su_prosa(go, catalogo, knob_on):
    """Control del anterior: en RD esa frase es correcta y se queda."""
    assert "criollo" in _bloque(go, "DO").lower()


# ── G. Parser-based anchor ──────────────────────────────────────────────────────────────────────

def test_el_fuente_declara_el_marker_y_la_puerta_unica():
    """La derivación de país tiene que pasar por `country_for_form_data` (la espina T1). Un
    segundo canonicalizador aquí sería la tabla que P1-DIET-CANON-SSOT ya pagó una vez."""
    src = _GO_PATH.read_text(encoding="utf-8", errors="replace")
    assert "P1-VERIFIED-CATALOG-COUNTRY" in src
    i = src.find("def _get_verified_catalog_instruction")
    assert i > 0
    cuerpo = src[i:i + 5000]
    assert "country_for_form_data" in cuerpo, (
        "la función no deriva el país por la única puerta de lectura del motor"
    )
    assert "is_country_catalog_unpriced_item" in cuerpo, (
        "el predicado de comprabilidad beta no reusa el SSOT del agregador"
    )
