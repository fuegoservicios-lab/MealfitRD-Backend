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


# [P1-PLAN-LOTE-1139] El dueño (admin) recibió «Llegaste al uso de IA incluido en tu plan» en el escáner: una lectura
# fallida del perfil caía a «gratis» y frenaba con el tope gratis. Ahora: la lista de admins del entorno manda sin
# tocar la base, y un perfil ilegible NO frena (fail-open, como promete `frenado`).
def _gasto_alto(monkeypatch):
    monkeypatch.delenv("MEALFIT_TOPES_POR_NIVEL", raising=False)
    monkeypatch.setattr(tp, "gasto_mes_usd", lambda uid: 99.0)
    monkeypatch.setattr(tp, "fotos_24h", lambda uid: 999)


def test_admin_por_entorno_nunca_se_frena(monkeypatch):
    _gasto_alto(monkeypatch)
    uid = "61a13831-2a70-4437-a084-0d3e09b653e4"
    monkeypatch.setenv("MEALFIT_ADMIN_USER_IDS", f"otro, {uid}")
    import db
    monkeypatch.setattr(db, "get_user_profile", lambda u: (_ for _ in ()).throw(RuntimeError("no debería leerse")))
    assert tp._tier(uid) == "admin"
    assert tp.frenado(uid, "foto") is None


def test_perfil_ilegible_no_frena(monkeypatch):
    _gasto_alto(monkeypatch)
    monkeypatch.delenv("MEALFIT_ADMIN_USER_IDS", raising=False)
    import db
    uid = "11111111-1111-4111-8111-111111111111"
    monkeypatch.setattr(db, "get_user_profile", lambda u: (_ for _ in ()).throw(RuntimeError("pool")))
    assert tp.frenado(uid, "foto") is None
    monkeypatch.setattr(db, "get_user_profile", lambda u: None)
    assert tp.frenado(uid, "foto") is None
    monkeypatch.setattr(db, "get_user_profile", lambda u: {"plan_tier": "gratis"})
    assert tp.frenado(uid, "foto")["codigo"] == "tope_mes_usd"
