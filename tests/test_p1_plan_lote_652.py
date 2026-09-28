# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-652 · 2026-09-27] (P0) El avance del plan ya no retrocede el ancla 4 horas en cada shift.

`/shift-plan` y el cron de refill calculan los días transcurridos con el ancla pasada a reloj LOCAL (`start_dt =
ancla_utc - tz_offset`) y luego PERSISTÍAN ese mismo `start_dt + N días` como nuevo `grocery_start_date`: el reloj local
etiquetado como UTC. En RD (UTC-4) el ancla bajaba 4 h en cada shift; cuando cruzaba las 04:00 UTC, la siguiente llamada
del MISMO día creía que había pasado otro día y archivaba otro. En producción: dea00a2f (17-sep, 1 → 0 días vivos) y
6594aae1 (26-sep, el 26 archivado dos veces, `_shift_days_accumulated`=5 con 4 días reales). Las renovaciones tenían el
mismo sesgo (`today.isoformat()` con `today` ya en reloj local).

Ahora las dos ramas usan el SSOT `constants.fecha_local_del_ancla` para contar y `constants.ancla_tras_shift` para
persistir (el MISMO instante UTC + N días), y la renovación guarda el instante real.

tooltip-anchor: P1-PLAN-LOTE-652
"""
import inspect
from datetime import datetime, timedelta, timezone

import pytest


def _dias(ancla, tz, ahora_utc):
    from constants import fecha_local_del_ancla
    hoy_local = (ahora_utc - timedelta(minutes=tz)).date()
    return (hoy_local - fecha_local_del_ancla(ancla, tz)).days


def test_el_ancla_conserva_su_instante_utc():
    from constants import ancla_tras_shift
    assert ancla_tras_shift("2026-09-22T16:48:07.348599+00:00", 1) == "2026-09-23T16:48:07.348599+00:00"
    assert ancla_tras_shift("2026-09-22T12:00:00-04:00", 2) == "2026-09-24T16:00:00+00:00"
    assert ancla_tras_shift("2026-09-22", 3) == "2026-09-25"          # fecha sola → fecha sola


@pytest.mark.parametrize("tz", [240, 300, 360, -60, 0])
@pytest.mark.parametrize("hora_ancla", [0, 2, 3, 4, 5, 12, 20, 23])
def test_dos_llamadas_seguidas_no_archivan_dos_veces(tz, hora_ancla):
    """La invariante que se rompía: shift → nueva ancla → otra llamada con el MISMO reloj ⇒ 0 días."""
    from constants import ancla_tras_shift
    ancla = datetime(2026, 9, 22, hora_ancla, 30, tzinfo=timezone.utc).isoformat()
    for horas in (27, 51, 75, 99):   # varias llamadas a lo largo de los días
        ahora = datetime(2026, 9, 22, hora_ancla, 30, tzinfo=timezone.utc) + timedelta(hours=horas)
        d = _dias(ancla, tz, ahora)
        if d > 0:
            ancla = ancla_tras_shift(ancla, d)
        assert _dias(ancla, tz, ahora) == 0, (tz, hora_ancla, horas, ancla)


def test_la_secuencia_de_6594aae1_ya_no_duplica():
    """Un shift diario en RD durante cinco días y una segunda llamada tres segundos después (como la del 26-sep). Con
    la fórmula vieja el ancla baja 4 h por shift (20:48 → 16:48 → … → 00:48 UTC) y la segunda llamada archiva OTRO
    día: acumulado 6 para 5 días (medido con la fórmula vieja antes de escribir el arreglo). Ahora, 5."""
    from constants import ancla_tras_shift
    ancla, acumulado, tz = "2026-09-22T20:48:07.348599+00:00", 0, 240
    llamadas = ["2026-09-23T13:10:00", "2026-09-24T13:02:00", "2026-09-25T12:40:00",
                "2026-09-26T13:23:16", "2026-09-27T13:23:16", "2026-09-27T13:23:19"]
    for ts in llamadas:
        ahora = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
        d = _dias(ancla, tz, ahora)
        if d > 0:
            ancla, acumulado = ancla_tras_shift(ancla, d), acumulado + d
    assert acumulado == 5


def test_las_dos_ramas_usan_el_ssot():
    import cron_tasks
    from routers import plans
    for mod in (plans, cron_tasks):
        src = inspect.getsource(mod)
        assert "ancla_tras_shift(start_date_str, days_since_creation)" in src, mod.__name__
        assert "new_start = start_dt + timedelta(days=days_since_creation)" not in src, mod.__name__
        assert "grocery_start_date'] = today.isoformat()" not in src and \
            'grocery_start_date"] = today.isoformat()' not in src, mod.__name__


def test_el_snapshot_de_una_fecha_sola_es_su_medianoche_local():
    from constants import ancla_tras_shift, chunk_anchor_local_midnight_utc
    from datetime import datetime as _dt
    inst = ancla_tras_shift("2026-09-22", 1, tz_offset_min=240)
    assert inst == "2026-09-23T04:00:00+00:00"            # 00:00 en RD
    assert chunk_anchor_local_midnight_utc(_dt.fromisoformat(inst), 240).isoformat() == inst
