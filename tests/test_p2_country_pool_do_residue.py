"""[P2-COUNTRY-POOL-DO-RESIDUE · 2026-08-23] Los pools deterministas de dos países beta traían
dentro el nombre de un alimento DOMINICANO, con gentilicio y todo.

LO MEDIDO (parseando `constants.py` y contrastando contra el catálogo VIVO de Neon):

    COUNTRY_POOLS['MX']['proteins'] = [... 'Carne de res molida', 'Longaniza dominicana', ...]
    COUNTRY_POOLS['CO']['proteins'] = [... 'Morcilla',            'Longaniza dominicana', ...]

NO ES COPY. Ese string es el IDENTIFICADOR de punta a punta: por él resuelven la lista de la
compra, la Nevera (`pantry_names_match`) y el backstop de alérgenos. Los pools alimentan tres
caminos —el seeder de la generación principal, el camino DEGRADADO sin LLM ni review, y el closer
del piso de proteína, donde `_propias` entra sin filtrar y va PRIMERO—, así que al mexicano se le
asignaba como proteína del día la fila dominicana de verdad.

POR QUÉ ESTE FICHERO NO BUSCA LA CADENA «Longaniza dominicana». Anclar el literal de lo que se
arregla es un guard que se rompe al arreglarlo y que no ve la siguiente alta con el mismo defecto.
Lo que se ancla es la PROPIEDAD —«ningún pool de un país beta lleva un gentilicio dominicano»— y
el DATO —«todo nombre de todo pool existe como fila viva del catálogo»—, que crece solo.

LOS DOS CONTRAPESOS, porque el arreglo obvio tiene dos maneras de salir caro:
  · Borrar el residuo y ya: perder comida en silencio es el fallo más caro de la doctrina de este
    repo. Por eso hay un SUELO de tamaño por lista.
  · Sustituirlo por un nombre que no existe: el pool quedaría apuntando a nada y el seeder
    asignaría un alimento que la lista de compras dropea sin aviso. Por eso el guard de dato
    exige fila viva (e2e, contra Neon).

LO QUE EL AUDIT PEDÍA Y **NO** SE HIZO, medido antes de decidir. El mismo gap listaba «quitar
'Salami' de PR» y revisar 'Auyama'/'Habichuelas *' (US) y 'Ají morrón'/'Vainitas' (ES). Ninguno
lleva gentilicio, y el catálogo VIVO enseña por qué no hay nada que mover: sus filas ya traen el
alias del mercado beta —Auyama←`calabaza`/`zapallo`, Ají morrón←`pimiento`/`pimiento rojo`,
Vainitas←`judías verdes`/`ejotes`, Salami←`salami` (el nombre desnudo, sin gentilicio)—, o sea que
resuelven para un usuario beta. Y NO existe fila destino con el nombre local (no hay «Judías
verdes» ni «Pimiento» como fila propia), así que renombrar el item del pool lo dejaría apuntando a
nada y borrar el item sería perder comida sin destino. Regla 4 del repo: un nombre de alimento es
un IDENTIFICADOR — no se traduce ni se mueve.

Byte-identidad DO: los cuatro pools `DOMINICAN_*` no se tocan — 'Longaniza' sigue siendo suya.
"""
from __future__ import annotations

import pytest

from constants import (COUNTRY_POOLS, COUNTRY_PROFILES, DOMINICAN_CARBS, DOMINICAN_FRUITS,
                       DOMINICAN_PROTEINS, DOMINICAN_VEGGIES_FATS, strip_accents)

#: Gentilicios del país nativo. No es «palabras dominicanas» (eso sería un léxico infinito y
#: discutible): es la marca EXPLÍCITA de pertenencia a otro país dentro del nombre de un alimento.
_GENTILICIOS_DO = frozenset({
    "dominicano", "dominicana", "dominicanos", "dominicanas",
    "quisqueyano", "quisqueyana", "quisqueyanos", "quisqueyanas",
})

_LISTAS = ("proteins", "carbs", "veggies_fats", "fruits")

#: Suelo medido el 2026-08-23, DESPUÉS de curar el residuo. Existe para que la próxima curación no
#: se resuelva borrando: un pool sólo puede crecer.
_SUELO = {
    "ES": {"proteins": 23, "carbs": 10, "veggies_fats": 16, "fruits": 7},
    "MX": {"proteins": 20, "carbs": 10, "veggies_fats": 15, "fruits": 7},
    "CO": {"proteins": 20, "carbs": 10, "veggies_fats": 14, "fruits": 8},
    "PR": {"proteins": 20, "carbs": 10, "veggies_fats": 16, "fruits": 7},
    "US": {"proteins": 20, "carbs": 10, "veggies_fats": 15, "fruits": 7},
}


def _tokens(nombre: str) -> set:
    return set(strip_accents(str(nombre or "").lower()).replace(",", " ").split())


def _paises_beta() -> list:
    return sorted(cc for cc, p in COUNTRY_PROFILES.items() if p.get("is_beta"))


def _gentilicios_en_pool(pools: dict) -> list:
    malos = []
    for clave in _LISTAS:
        for nombre in (pools.get(clave) or []):
            if _tokens(nombre) & _GENTILICIOS_DO:
                malos.append((clave, nombre))
    return malos


# ── A. La propiedad ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cc", _paises_beta())
def test_ningun_pool_beta_lleva_un_gentilicio_dominicano(cc):
    pools = COUNTRY_POOLS.get(cc)
    if pools is None:
        pytest.skip(f"{cc} no tiene pool propio (lo cubre test_p2_country_septimo_pais_fallback_mudo)")
    malos = _gentilicios_en_pool(pools)
    assert not malos, f"{cc}: nombres con gentilicio dominicano en su propio pool: {malos}"


def test_la_regla_sabe_reconocer_el_defecto_que_cerro():
    """Un guard que no puede fallar es peor que no tener guard: se le enseña el caso real."""
    assert _gentilicios_en_pool({"proteins": ["Longaniza dominicana"]})
    # …y no confunde 'criolla' (que Colombia usa igual: 'Gallina criolla' es fila suya) con un
    # gentilicio dominicano.
    assert not _gentilicios_en_pool({"proteins": ["Gallina criolla", "Chorizo santarrosano"]})


# ── B. Curar no puede ser borrar ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cc", sorted(_SUELO))
def test_curar_el_residuo_no_encoge_el_pool(cc):
    for clave, minimo in _SUELO[cc].items():
        assert len(COUNTRY_POOLS[cc][clave]) >= minimo, (
            f"{cc}.{clave} tiene {len(COUNTRY_POOLS[cc][clave])} items y el suelo son {minimo}: "
            "un residuo se sustituye, no se borra (perder comida en silencio es el fallo caro)")


@pytest.mark.parametrize("cc", sorted(COUNTRY_POOLS))
def test_ningun_pool_repite_un_nombre_dentro_de_la_misma_lista(cc):
    """Sustituir el residuo por un nombre que YA estaba sería perder el hueco igual, sólo que sin
    que se note."""
    for clave in _LISTAS:
        nombres = [strip_accents(str(n).lower()) for n in COUNTRY_POOLS[cc][clave]]
        repes = sorted({n for n in nombres if nombres.count(n) > 1})
        assert not repes, f"{cc}.{clave} repite: {repes}"


# ── C. Byte-identidad dominicana ────────────────────────────────────────────────────────────────

def test_los_pools_dominicanos_no_pierden_su_longaniza():
    assert "Longaniza" in DOMINICAN_PROTEINS
    for pool in (DOMINICAN_PROTEINS, DOMINICAN_CARBS, DOMINICAN_VEGGIES_FATS, DOMINICAN_FRUITS):
        assert pool, "un pool dominicano vacío es el otro modo de romper la byte-identidad"


# ── D. El guard de DATO (crece solo) ────────────────────────────────────────────────────────────

@pytest.mark.e2e
def test_todo_nombre_de_todo_pool_de_pais_existe_como_fila_viva_del_catalogo():
    """El contrapeso del arreglo: un pool que apunta a una fila inexistente hace que el seeder
    asigne un alimento que `_is_verified_for_shopping` dropea de la lista EN SILENCIO (el modo de
    fallo de `check_pool_prices.py`, aquí para los pools de país).

    Match por NOMBRE EXACTO a propósito: los `COUNTRY_POOLS` se curaron con el nombre canónico de
    la fila (medido: 5/5 países al 100% hoy), así que exigir la cascada de alias sería un contrato
    más flojo que el que los datos ya cumplen. Los `DOMINICAN_*` sí usan display names que
    resuelven por sinónimos — ésos los cubre `scripts/check_pool_prices.py`."""
    try:
        from shopping_calculator import get_master_ingredients
        filas = get_master_ingredients() or []
    except Exception as e:  # pragma: no cover - entorno sin DB
        pytest.skip(f"catálogo no disponible: {e}")
    if not filas:
        pytest.skip("catálogo vacío (¿pool de Neon sin abrir?)")
    vivos = {str(r.get("name") or "").strip() for r in filas}
    huerfanos = []
    for cc, pools in COUNTRY_POOLS.items():
        for clave in _LISTAS:
            for nombre in pools[clave]:
                if nombre not in vivos:
                    huerfanos.append(f"{cc}.{clave}:{nombre}")
    assert not huerfanos, f"nombres de pool sin fila viva en master_ingredients: {huerfanos}"
