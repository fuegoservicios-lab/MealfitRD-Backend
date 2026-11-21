"""[P1-PLAN-LOTE-1075 · 2026-11-17] Comida de una franja ya pasada, anotada tarde → se guarda en su franja.

Caso real del dueño (17-nov, RD UTC-4): a las 22:23 «No desayuné, almorcé esto» y a las 22:25 «Merendé esto»; el
almuerzo y la merienda quedaron a las 22:23/22:25 y el informe habría dicho «primera comida a las 22:23»."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from hora_de_la_franja import consumed_at_de_franja_pasada
import informe_medico as im

RD = 240   # UTC − local, en minutos


def _utc(y, mo, d, h, mi):
    """Hora LOCAL de RD → datetime UTC."""
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc) + timedelta(minutes=RD)


def _local(iso):
    return datetime.fromisoformat(iso) - timedelta(minutes=RD)


@pytest.mark.parametrize("franja,hora_local,esperada", [
    ("almuerzo", (22, 23), (13, 0)),
    ("merienda", (22, 25), (16, 30)),
    ("desayuno", (14, 0), (8, 0)),
    ("Almuerzo", (17, 0), (13, 0)),
])
def test_la_franja_pasada_se_guarda_en_su_hora(franja, hora_local, esperada):
    iso = consumed_at_de_franja_pasada(franja, _utc(2026, 11, 17, *hora_local), RD)
    loc = _local(iso)
    assert (loc.date().isoformat(), loc.hour, loc.minute) == ("2026-11-17", *esperada)


@pytest.mark.parametrize("franja,hora_local", [
    ("almuerzo", (15, 30)),   # aún es hora razonable de almorzar: la hora real vale
    ("merienda", (19, 0)),
    ("desayuno", (10, 0)),
    ("cena", (23, 30)),       # la cena nunca se mueve
    ("snack", (23, 0)),
    ("suplemento", (22, 0)),
    (None, (22, 0)),
])
def test_si_no_paso_o_no_es_franja_movible_no_toca_la_hora(franja, hora_local):
    assert consumed_at_de_franja_pasada(franja, _utc(2026, 11, 17, *hora_local), RD) is None


def test_de_madrugada_es_el_dia_biologico_de_ayer():
    # 01:30 del 18 = día biológico 17: el almuerzo va al 17 a las 13:00
    loc = _local(consumed_at_de_franja_pasada("almuerzo", _utc(2026, 11, 18, 1, 30), RD))
    assert (loc.date().isoformat(), loc.hour) == ("2026-11-17", 13)


def test_la_merienda_tras_la_cena_conserva_su_hora_real():
    # 18-nov, 00:25: «Merendé esto» con la cena ya anotada → picoteo de noche, NO a las 16:30
    assert consumed_at_de_franja_pasada("merienda", _utc(2026, 11, 18, 0, 25), RD, ya_ceno=True) is None
    assert consumed_at_de_franja_pasada("merienda", _utc(2026, 11, 17, 22, 25), RD, ya_ceno=False) is not None
    # el almuerzo atrasado sí se mueve aunque ya haya cenado (se está poniendo al día)
    assert consumed_at_de_franja_pasada("almuerzo", _utc(2026, 11, 17, 22, 40), RD, ya_ceno=True) is not None


def test_sin_huso_no_inventa():
    assert consumed_at_de_franja_pasada("almuerzo", _utc(2026, 11, 17, 22, 0), None) is None


def test_la_herramienta_solo_mueve_la_franja_que_nombro_el_usuario_y_hoy():
    src = (Path(__file__).resolve().parents[1] / "tools.py").read_text(encoding="utf-8")
    i = src.index("[P1-PLAN-LOTE-1075]")
    bloque = src[i:i + 5000]
    # [P1-PLAN-LOTE-1145] + `not _sigue_en_ayer`: sin dormir, «almorcé» cuenta en el día que sigue viviendo
    assert "if _days_ago == 0 and explicit_slot and not _es_merienda_nocturna and not _sigue_en_ayer:" in bloque
    assert "consumed_at_override=_hora_guardada" in bloque
    assert "ya_ceno=_ya_ceno" in bloque


def test_el_informe_no_usa_horas_estimadas_para_los_horarios():
    t0 = datetime(2026, 11, 17, 2, 23, tzinfo=timezone.utc)
    assert im.hora_fiable(t0, t0 + timedelta(seconds=1)) is True
    assert im.hora_fiable(datetime(2026, 11, 17, 17, 0, tzinfo=timezone.utc), t0 + timedelta(hours=1)) is False
    assert im.hora_fiable(None, t0) is True   # filas antiguas sin dato: como antes
    src = (Path(__file__).resolve().parents[1] / "informe_medico.py").read_text(encoding="utf-8")
    assert '"hora_fiable": hora_fiable(ca, r.get("created_at"))' in src
    assert 'if c.get("hora_fiable", True):' in src
