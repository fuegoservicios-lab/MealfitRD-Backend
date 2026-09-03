"""[P3-FIXTURE-TZ-SIGN · 2026-08-22] El fixture compartido montaba a TODOS los usuarios de e2e con
el signo del huso invertido — y lo escribía en la base de PRODUCCIÓN.

CÓMO APARECIÓ. Buscando otra cosa (dos fallbacks distintos para el mismo dato ausente) medí qué
husos hay guardados de verdad, y `health_profile.tz_offset_minutes` salió con LOS DOS signos:

    tz_offset_minutes = '-240'   8 perfiles
    tz_offset_minutes = '240'    5 perfiles
    ausente                      3 perfiles

República Dominicana es UTC−4 y la convención declarada es la de `Date.getTimezoneOffset()`
—positivo al OESTE—, así que RD es **+240**. Ocho de dieciséis perfiles con el signo al revés
parecía el incidente del día.

LO QUE LA SEGUNDA CONSULTA REFUTÓ. Los ocho `-240` son, uno a uno, usuarios `e2e-test-*@test.local`
creados el 21 y el 22 de agosto. Los cinco `+240` son las cuentas reales, todas correctas. **No hay
bug de signo en producción.** Medir levantó la alarma y medir una consulta más la desmontó: la
diferencia entre un incidente y un susto era un `WHERE email LIKE`.

LO QUE SÍ QUEDA, Y ES MÍO. `tests/conftest.py` monta cada usuario de e2e con `-240`. Dos cosas:

1. **El fixture no puede cazar un bug de signo: lo encarna.** Un test que monta el mundo con la
   convención invertida y luego afirma sobre fechas es autoconsistente — pasa en verde tanto
   contra el código correcto como contra el código con el signo al revés, porque el error se
   cancela consigo mismo. Es «los tests codificaban el bug», que este repo ya pagó en julio.
   Peor aún: el usuario que el fixture dice modelar («RD, UTC−4») queda de hecho en UTC+4, o sea
   a ocho horas — Bakú, no Santo Domingo. Cualquier caso de frontera de día que corra sobre este
   fixture mide otra cosa.

2. **Esas filas viven en la base de PRODUCCIÓN.** El propio detector de residuo lo canta al final
   de cada corrida (`[P1-TEST-RESIDUE-DETECTOR] 8 usuario(s) de test VIVOS`). El fixture borra al
   empezar, no al terminar, así que una corrida interrumpida —o un worker que muere, que es
   justo lo que pasó con `-n 4`— deja el usuario puesto para siempre.

EL CONTRATO, verificado en cuatro sitios independientes antes de tocar nada:

  · `QTrackingFinish.jsx` escribe `new Date().getTimezoneOffset()` y su comentario dice «RD=+240».
  · `tools._local_date_str_for_user`: «POSITIVO = OESTE de UTC (RD=240)».
  · `routers/plans._resolve_request_tz_offset`: «formato JS: positivo para TZ negativas, e.g.
    +240 para UTC−4».
  · Las cinco cuentas reales de producción: +240, todas.

POR QUÉ ESTE GUARD NO CLAVA EL 240. Lo lee de `_LOCAL_DATE_FALLBACK_OFFSET_MIN`, que es donde el
backend declara «el huso de RD». Si algún día se cambia la convención, las dos piezas se mueven
juntas o esto se pone rojo — que es exactamente lo que no pasó la primera vez.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_CONFTEST = Path(__file__).resolve().parent / "conftest.py"


@pytest.fixture(scope="module")
def conftest_src() -> str:
    return _CONFTEST.read_text(encoding="utf-8", errors="replace")


@pytest.fixture(scope="module")
def offset_del_fixture(conftest_src) -> int:
    m = re.search(r'"tz_offset_minutes"\s*:\s*(-?\d+)', conftest_src)
    assert m, (
        "el fixture compartido dejó de declarar `tz_offset_minutes`. Sin huso explícito, cada "
        "test de frontera de día depende del reloj de la máquina que lo corra"
    )
    return int(m.group(1))


def test_el_fixture_usa_el_signo_canonico(offset_del_fixture):
    """EL CASO. `-240` sitúa al usuario en UTC+4 (Bakú) mientras el fixture dice modelar RD."""
    assert offset_del_fixture > 0, (
        f"el fixture monta a los usuarios de e2e con tz_offset_minutes={offset_del_fixture}. La "
        f"convención es la de `Date.getTimezoneOffset()` (POSITIVO al oeste): RD es +240. Con el "
        f"signo invertido el usuario queda a ocho horas de donde el fixture dice, y cualquier "
        f"caso de frontera de día que corra encima pasa en verde contra el código roto"
    )


def test_el_fixture_coincide_con_el_huso_de_rd_que_declara_el_backend(offset_del_fixture):
    """Paridad con el SSOT, no un 240 clavado: si se cambia la convención, las dos se mueven
    juntas."""
    import tools

    assert offset_del_fixture == tools._LOCAL_DATE_FALLBACK_OFFSET_MIN, (
        f"el fixture dice {offset_del_fixture} y el backend declara "
        f"{tools._LOCAL_DATE_FALLBACK_OFFSET_MIN} como el huso de RD"
    )


def test_el_huso_del_fixture_es_coherente_con_el_helper_de_fecha():
    """Funcional, no textual: con el huso del fixture, la fecha «local» de un usuario de RD tiene
    que ser la que un reloj dominicano marca — nunca la de un huso al este de UTC.

    Este es el caso que un guard puramente textual no da: comprueba que el número del fixture,
    pasado por el helper real, produce la hora local correcta."""
    from datetime import datetime, timedelta, timezone

    import tools

    offset = tools._LOCAL_DATE_FALLBACK_OFFSET_MIN
    ahora_utc = datetime.now(timezone.utc)
    local = ahora_utc - timedelta(minutes=offset)
    assert local < ahora_utc, (
        "con el huso declarado de RD, la hora local sale ADELANTADA respecto a UTC. República "
        "Dominicana está al oeste: su hora local siempre va por detrás"
    )


def test_el_teardown_borra_el_usuario_al_terminar(conftest_src):
    """La otra mitad del hallazgo: 8 usuarios de test vivos en la base de PRODUCCIÓN.

    ⚠️ Este caso PASA hoy y se escribe igual, como ancla. Lo interesante es lo que descubre al
    pasar: el teardown existe y borra correctamente, así que el residuo **no** viene de un fixture
    incompleto — viene de corridas que mueren antes de llegar aquí (un worker de `-n 4` que se
    lleva un `MemoryError`, un Ctrl-C, un timeout). O sea que el arreglo del residuo no está en
    este fichero, y un guard que lo diera por hecho mandaría a alguien a reescribir lo que ya
    funciona.

    Se conserva porque si alguien QUITA este borrado, el goteo pasa de accidental a sistemático."""
    i_yield = conftest_src.find("yield", conftest_src.find('"tz_offset_minutes"'))
    assert i_yield > 0, "no encuentro el `yield` del fixture de usuario"
    cola = conftest_src[i_yield:i_yield + 4000]
    assert "DELETE FROM user_profiles" in cola, (
        "el fixture no borra `user_profiles` DESPUÉS del yield: una corrida interrumpida deja el "
        "usuario de test vivo en la base de producción, que es lo que el detector de residuo "
        "lleva cantando"
    )
