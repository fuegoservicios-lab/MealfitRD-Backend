"""[P3-TZ-COUNTRY-COVERAGE · 2026-08-22] La tabla que preselecciona el país por la zona horaria del
navegador no cubre todas las zonas de los seis países que ofrece.

`frontend/src/config/countries.js::TZ_COUNTRY_EXACT` traduce el NOMBRE de una zona IANA al país.
Lo que no está en la tabla cae al fail-safe 'DO', así que a un español en Ceuta o a un mexicano en
Ciudad Juárez el asistente les sugiere República Dominicana.

EL HALLAZGO DEL MÉTODO, que vale más que las zonas que faltaban. La auditoría pedía añadir
**`Europe/Ceuta`**. Esa zona **NO EXISTE**: la base IANA la llama `Africa/Ceuta` (Ceuta está en el
continente africano). Añadir lo que la auditoría decía habría metido una fila muerta que no casa
con ningún navegador del mundo — y habría dejado el gap CERRADO en la lista de tareas, con el
usuario de Ceuta recibiendo 'DO' igual que antes. Un arreglo indistinguible de no haber hecho
nada, y encima con el marcador en verde.

Por eso el guard central de este fichero no es «están estas quince zonas»: es **toda entrada de la
tabla tiene que ser una zona IANA real**, comprobado contra `zoneinfo.available_timezones()`. Ese
caso habría cazado el error sin que nadie tuviera que sospecharlo, y sigue cazando el próximo
dedazo (`America/Mexico-City`, `Atlantic/Canarias`, una zona renombrada por tzdata).

LO QUE SE AÑADE, todas verificadas contra la base IANA antes de escribirlas:

  ES  Africa/Ceuta
  MX  America/Ciudad_Juarez            (zona real desde tzdata 2022g)
  US  America/Juneau, Sitka, Metlakatla, Yakutat, Nome, Adak   (sureste y oeste de Alaska)
  US  America/Menominee                (Michigan, en horario del Centro)
  US  America/North_Dakota/…           (Center, New_Salem, Beulah — vía prefijo)

LO QUE NO SE TOCA, y es la otra mitad de lo que la auditoría pedía. El fail-safe de la
preselección escribe 'DO' ante una zona desconocida, y la auditoría lo llama «otra vez el default
sembrado». Es una **decisión declarada del dueño**, citada literalmente en el código de `QCountry`
(Addendum §4: «simplest honest approach»): el paso queda VISIBLE con las seis tarjetas y sugerir
no es decidir. Además el argumento del «default sembrado» no aplica igual aquí — quien vive en
Argentina no tiene tarjeta correcta que elegir, porque su país no está entre los seis. Revertir
eso es una decisión de producto, no un arreglo, y meterla en este P-fix sería decidir por él.
"""
from __future__ import annotations

import re
import zoneinfo
from pathlib import Path

import pytest

_COUNTRIES_JS = (Path(__file__).resolve().parent.parent.parent
                 / "frontend" / "src" / "config" / "countries.js")

#: Zonas que DEBEN resolver a su país. Curadas y verificadas contra `available_timezones()`.
_ESPERADAS = {
    "America/Santo_Domingo": "DO",
    "America/Puerto_Rico": "PR",
    "Europe/Madrid": "ES",
    "Atlantic/Canary": "ES",
    "Africa/Ceuta": "ES",
    "America/Bogota": "CO",
    "America/Mexico_City": "MX",
    "America/Tijuana": "MX",
    "America/Ciudad_Juarez": "MX",
    "America/New_York": "US",
    "America/Los_Angeles": "US",
    "America/Anchorage": "US",
    "America/Juneau": "US",
    "America/Sitka": "US",
    "America/Metlakatla": "US",
    "America/Yakutat": "US",
    "America/Nome": "US",
    "America/Adak": "US",
    "America/Menominee": "US",
    "Pacific/Honolulu": "US",
}


@pytest.fixture(scope="module")
def js() -> str:
    if not _COUNTRIES_JS.is_file():
        pytest.skip("countries.js no está en este árbol")
    return _COUNTRIES_JS.read_text(encoding="utf-8", errors="replace")


@pytest.fixture(scope="module")
def tabla(js) -> dict:
    i = js.index("const TZ_COUNTRY_EXACT = {")
    bloque = js[i:js.index("\n};", i)]
    return dict(re.findall(r"'([\w/+\-]+)':\s*'([A-Z]{2})'", bloque))


@pytest.fixture(scope="module")
def prefijos(js) -> list:
    i = js.index("const TZ_COUNTRY_PREFIXES = [")
    bloque = js[i:js.index("\n];", i)]
    return re.findall(r"\['([\w/+\-]+)',\s*'([A-Z]{2})'\]", bloque)


def _resuelve(zona: str, tabla: dict, prefijos: list) -> str | None:
    if zona in tabla:
        return tabla[zona]
    for pref, pais in prefijos:
        if zona.startswith(pref):
            return pais
    return None


# ── El guard que habría cazado el error de la auditoría ─────────────────────────────────────────

def test_toda_zona_de_la_tabla_existe_de_verdad(tabla):
    """EL CASO CENTRAL. Una zona inventada no casa con ningún navegador: el usuario sigue cayendo
    al fail-safe, pero la tarea queda marcada como hecha. Un arreglo indistinguible de no haber
    hecho nada es peor que el hueco, porque nadie vuelve a mirarlo.

    Es lo que habría pasado añadiendo `Europe/Ceuta`, que es lo que la auditoría pedía y no
    existe."""
    reales = zoneinfo.available_timezones()
    inventadas = sorted(z for z in tabla if z not in reales)
    assert not inventadas, (
        f"estas entradas de TZ_COUNTRY_EXACT no son zonas IANA reales: {inventadas}. Nunca van a "
        f"casar con `Intl.DateTimeFormat().resolvedOptions().timeZone`"
    )


def test_todo_prefijo_cubre_zonas_que_existen(prefijos):
    """Lo mismo para los prefijos: `America/Indiana/` sirve porque hay zonas debajo. Un prefijo sin
    zonas es la misma fila muerta, sólo que más difícil de ver."""
    reales = zoneinfo.available_timezones()
    for pref, pais in prefijos:
        assert any(z.startswith(pref) for z in reales), (
            f"el prefijo {pref!r} → {pais} no cubre ninguna zona IANA real"
        )


def test_la_tabla_no_esta_vacia(tabla):
    """Sanity: sin filas, la cobertura de abajo pasaría por vacuidad."""
    assert len(tabla) >= 20, f"TZ_COUNTRY_EXACT bajó a {len(tabla)} filas"


# ── La cobertura ────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("zona,pais", sorted(_ESPERADAS.items()))
def test_cada_zona_conocida_resuelve_a_su_pais(zona, pais, tabla, prefijos):
    """Un navegador en esa zona tiene que ver preseleccionado SU país, no el fail-safe."""
    obtenido = _resuelve(zona, tabla, prefijos)
    assert obtenido == pais, (
        f"{zona} resuelve a {obtenido!r} y debería ser {pais!r}. Sin entrada, el asistente le "
        f"sugiere República Dominicana a alguien que no vive ahí"
    )


def test_north_dakota_entra_por_prefijo(tabla, prefijos):
    """Las tres de Dakota del Norte (Center, New_Salem, Beulah) comparten forma, así que un
    prefijo las cubre y no hay que acordarse de las tres."""
    for ciudad in ("Center", "New_Salem", "Beulah"):
        zona = f"America/North_Dakota/{ciudad}"
        assert zona in zoneinfo.available_timezones(), f"{zona} no es una zona real"
        assert _resuelve(zona, tabla, prefijos) == "US", f"{zona} no resuelve a US"


def test_una_zona_fuera_de_los_seis_paises_no_inventa_un_match(tabla, prefijos):
    """El límite, anclado en negativo: Argentina no está entre los seis, así que su zona NO debe
    aparecer en la tabla. Lo que hace el fail-safe con ella es decisión declarada del dueño y vive
    en `QCountry`, no aquí."""
    assert _resuelve("America/Argentina/Buenos_Aires", tabla, prefijos) is None, (
        "se añadió una zona de un país que el selector no ofrece"
    )
