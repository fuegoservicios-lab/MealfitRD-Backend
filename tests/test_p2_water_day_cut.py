"""[P2-WATER-DAY-CUT · 2026-08-21] El contador de agua lo escriben dos superficies, y cortaban el
día en husos distintos.

La fila de `water_intake_log` está keyed por `(user_id, log_date)`, y la escriben dos caminos:

  · el **card del Dashboard** (`WaterTracker.jsx`), que arma la fecha con `d.getFullYear()` sobre
    un `Date` local — o sea el huso del NAVEGADOR — y la manda como parámetro `date`;
  · las **tools del coach** (`check_hydration_today`, `log_water_glass`), que la derivan
    server-side con `tools._local_date_str_for_user`.

Mientras ese helper devolvía UTC−4 para todo el mundo, las dos superficies discrepaban para
**cualquier usuario fuera de República Dominicana**: el usuario marcaba un vaso en el card y el
coach, preguntado acto seguido, contestaba sobre otro día. Split-brain silencioso — ninguna de las
dos vistas está «mal», simplemente hablan de filas distintas.

**Ya está cerrado**, y no por este fichero: lo cerró `P2-LOCAL-DATE-STR-UTC4`, que hizo que el
helper leyera el huso persistido del usuario. Desde entonces las dos superficies derivan el día del
MISMO offset y coinciden siempre que el perfil esté al día.

Este test existe porque un arreglo del que nadie escribió el contrato se deshace solo. Ancla la
invariante —«las dos superficies cortan el día por el mismo sitio»— para que reintroducir un offset
fijo en cualquiera de las dos falle aquí en vez de en el vaso de agua de alguien.

RESIDUO CONOCIDO, dicho para no venderlo como cerrado del todo: si el `tzOffset` persistido está
**rancio** (el usuario viajó, o cambió el horario de verano y nadie reescribió el perfil), vuelve la
discrepancia. Eso no es este gap sino la frescura del huso, y su captura la cierra
`P1-TRACKING-TZ-CAPTURE`.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_WATER_TRACKER = _BACKEND.parent / "frontend" / "src" / "components" / "dashboard" / "WaterTracker.jsx"


@pytest.fixture(scope="module")
def tools():
    import tools as _t
    return _t


@pytest.mark.parametrize("offset", [240, -60, -120, 360, 300, 0, 720])
def test_las_dos_superficies_cortan_el_dia_por_el_mismo_sitio(tools, monkeypatch, offset):
    """El card manda la fecha del navegador; las tools la derivan del huso persistido. Con el
    mismo offset tienen que dar el MISMO día — si no, el vaso que marcas y el que el coach cuenta
    viven en filas distintas de `water_intake_log`."""
    monkeypatch.setattr(tools, "user_tz_offset_min", lambda uid: offset)
    dia_backend = tools._local_date_str_for_user("u1")
    # Lo que `WaterTracker.jsx` produce: la fecha civil del usuario en su propio huso.
    dia_navegador = (datetime.now(timezone.utc) - timedelta(minutes=offset)).date().isoformat()
    assert dia_backend == dia_navegador, (
        f"offset {offset}: el backend dice {dia_backend} y el navegador {dia_navegador} — el card "
        f"y el coach escribirían filas distintas del contador de agua"
    )


def test_el_backend_no_vuelve_a_cablear_un_huso_fijo(tools):
    """El defecto original en una línea: el helper se llamaba «for_user» y hacía `now(utc) - 4h`.
    Se mide sobre la CONDUCTA (dos husos, dos días) y no sobre la grafía, para que un renombre de
    la constante no esquive el guard."""
    import unittest.mock as _m
    with _m.patch.object(tools, "user_tz_offset_min", lambda uid: 720):
        oeste = tools._local_date_str_for_user("oeste")
    with _m.patch.object(tools, "user_tz_offset_min", lambda uid: -600):
        este = tools._local_date_str_for_user("este")
    assert oeste != este, (
        "el helper devuelve el mismo día para husos separados 22 horas: volvió a haber un offset "
        "fijo cableado"
    )


def test_el_card_arma_la_fecha_en_local_no_en_utc():
    """La otra mitad del par. `toISOString()` daría la fecha UTC y reintroduciría el corte
    divergente desde el lado del navegador — el mismo bug, la otra superficie."""
    if not _WATER_TRACKER.is_file():
        pytest.skip("WaterTracker.jsx no está en este árbol")
    src = _WATER_TRACKER.read_text(encoding="utf-8", errors="replace")
    i = src.find("getFullYear")
    assert i > 0, "el card ya no arma la fecha con componentes locales (¿pasó a UTC?)"
    ventana = src[max(0, i - 400):i + 400]
    assert "toISOString" not in ventana, (
        "el card arma la fecha del contador con `toISOString()` (UTC): para un usuario al oeste "
        "de Greenwich de noche, eso ya es mañana y el vaso cae en el día equivocado"
    )
