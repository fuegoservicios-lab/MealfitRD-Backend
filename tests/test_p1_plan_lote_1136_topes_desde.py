"""[P1-PLAN-LOTE-1136 · 2026-11-21] El mes de los topes no cuenta lo gastado ANTES de que existieran (20-nov).

Caso real: una cuenta gratis quedó frenada («Llegaste al uso de IA incluido en tu plan este mes») por US$ 0,43 de un
día anómalo del 13-nov, antes de que hubiera topes. El mes de los topes empieza en lo más tardío de (día 1,
nacimiento de los topes); desde diciembre manda el día 1."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import topes_por_nivel as tp  # noqa: E402


def _con_mes(monkeypatch, y, m):
    monkeypatch.setattr(tp, "_inicio_de_mes_natural", lambda: datetime(y, m, 1, tzinfo=timezone.utc))


def test_noviembre_cuenta_desde_el_nacimiento_de_los_topes(monkeypatch):
    monkeypatch.delenv("MEALFIT_TOPES_CUENTAN_DESDE", raising=False)
    _con_mes(monkeypatch, 2026, 11)
    assert tp._inicio_de_mes() == datetime(2026, 11, 20, tzinfo=timezone.utc)


def test_desde_diciembre_manda_el_dia_1(monkeypatch):
    monkeypatch.delenv("MEALFIT_TOPES_CUENTAN_DESDE", raising=False)
    _con_mes(monkeypatch, 2026, 12)
    assert tp._inicio_de_mes() == datetime(2026, 12, 1, tzinfo=timezone.utc)


def test_knob_movible_y_basura_cae_al_dia_1(monkeypatch):
    _con_mes(monkeypatch, 2026, 11)
    monkeypatch.setenv("MEALFIT_TOPES_CUENTAN_DESDE", "2026-11-25")
    assert tp._inicio_de_mes() == datetime(2026, 11, 25, tzinfo=timezone.utc)
    monkeypatch.setenv("MEALFIT_TOPES_CUENTAN_DESDE", "basura")
    assert tp._inicio_de_mes() == datetime(2026, 11, 1, tzinfo=timezone.utc)


def test_el_gasto_y_la_voz_usan_ese_inicio():
    src = Path(tp.__file__).read_text(encoding="utf-8")
    for fn in ("def gasto_mes_usd", "def voz_segundos_mes"):
        cuerpo = src.split(fn, 1)[1].split("\ndef ", 1)[0]
        assert "_inicio_de_mes()" in cuerpo, fn
