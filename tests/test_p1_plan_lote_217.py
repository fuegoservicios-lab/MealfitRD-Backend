# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-217 · 2026-09-24] La Nevera opcional TAMBIÉN en modo plan.

En modo plan la Nevera era obligatoria y, vacía 48 h, congelaba el plan: c7b90ca3 creó su plan el 17-sep, nunca abrió
la Nevera y quedó congelado el 19-sep con 12 días por generar. Ahora se apaga en Configuración en los dos modos; en
automático, a un usuario ACTIVO se le apaga sola en vez de congelarle el plan (y si ya estaba congelado por eso, se
reanuda). Quien la encendió a mano conserva el congelado. Sin congelado, el freno de una cuenta abandonada con la Nevera
apagada es la inactividad.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import nevera_opcional as no  # noqa: E402


@pytest.mark.parametrize("perfil, esperado", [
    ({"plan_mode": "plan", "nevera_enabled": False}, False),
    ({"plan_mode": "plan", "nevera_enabled": None}, True),
    ({"plan_mode": "plan", "nevera_enabled": True}, True),
    ({"plan_mode": "tracking", "nevera_enabled": False}, False),
    ({"nevera_enabled": False}, False),
    ({}, True),
])
def test_la_regla_en_los_dos_modos(perfil, esperado):
    assert no.nevera_activa_de(perfil) is esperado


def test_el_knob_devuelve_la_regla_del_23_sep(monkeypatch):
    monkeypatch.setenv("MEALFIT_NEVERA_OFF_IN_PLAN_MODE", "false")
    assert no.nevera_activa_de({"plan_mode": "plan", "nevera_enabled": False}) is True
    assert no.nevera_activa_de({"plan_mode": "tracking", "nevera_enabled": False}) is False
    assert no.apagar_por_plan_vacio("u1") is False


def test_activo_reciente(monkeypatch):
    ahora = datetime.now(timezone.utc)
    monkeypatch.setattr(no, "execute_sql_query", lambda *a, **k: {"m": ahora - timedelta(days=3)})
    assert no.activo_reciente("u1") is True
    monkeypatch.setattr(no, "execute_sql_query", lambda *a, **k: {"m": ahora - timedelta(days=40)})
    assert no.activo_reciente("u1") is False
    monkeypatch.setattr(no, "execute_sql_query", lambda *a, **k: {"m": None})
    assert no.activo_reciente("u1") is False, "sin evidencia de uso no se decide por él"

    def _revienta(*a, **k):
        raise RuntimeError("db")
    monkeypatch.setattr(no, "execute_sql_query", _revienta)
    assert no.activo_reciente("u1") is False
    assert no.activo_reciente("guest") is False and no.activo_reciente(None) is False


def test_el_apagado_en_modo_plan_solo_toca_a_quien_nunca_eligio(monkeypatch):
    llamadas = []
    monkeypatch.setattr(no, "execute_sql_write", lambda sql, params, **k: llamadas.append((sql, params)) or [{"id": "u1"}])
    assert no.apagar_por_plan_vacio("u1") is True
    sql, params = llamadas[0]
    assert "nevera_enabled IS NULL" in sql and "WHERE id = %s" in sql and "<> 'tracking'" in sql
    assert "nevera_auto_off_at = now()" in sql and params == ("u1",)


def test_el_bloque_no_se_cocina_con_una_nevera_apagada(monkeypatch):
    import cron_tasks as ct
    monkeypatch.setattr(no, "nevera_activa", lambda uid: False)

    def _no_debe_leerse(*a, **k):
        raise AssertionError("con la Nevera apagada no se lee el inventario")
    monkeypatch.setattr(ct, "_refresh_chunk_pantry_inner", _no_debe_leerse)
    fd = ct._refresh_chunk_pantry("u1", {"current_pantry_ingredients": ["Huevo"]}, {}, task_id=1, week_number=2)
    assert fd["current_pantry_ingredients"] == [] and fd["_pantry_advisory_only"] is True
    assert fd["_fresh_pantry_source"] == "nevera_off" and fd["_nevera_apagada"] is True
    # y ninguna guarda de Nevera puede pausarlo
    assert ct._pantry_gate_waiver_reason(chunk_kind="rolling_refill", form_data=fd) == "advisory_only"
    assert ct._should_pause_for_empty_pantry("nevera_off", [], {}, fd) is False


def _barrido(monkeypatch, *, nevera_enabled, frozen_at, activo, horas_vacia=60):
    import cron_tasks as ct
    ahora = datetime.now(timezone.utc)
    fila = {"plan_id": "p1", "user_id": "u1", "created_at": ahora - timedelta(hours=horas_vacia), "gstatus": "partial",
            "frozen_at": frozen_at, "reminder_at": None, "archived_at": None, "unfrozen_at": None,
            "plan_mode": "plan", "nevera_enabled": nevera_enabled}

    def _q(sql, params=(), **k):
        if "DISTINCT ON (mp.user_id)" in sql:
            return [fila]
        if "SELECT ingredient_name FROM user_inventory" in sql:
            return []
        if "MAX(updated_at)" in sql:
            return {"m": None}
        return None
    escrituras, reanudados, apagados = [], [], []
    monkeypatch.setattr(ct, "execute_sql_query", _q)
    monkeypatch.setattr(ct, "execute_sql_write", lambda sql, params=(), **k: escrituras.append(sql))
    monkeypatch.setattr(ct, "_resume_frozen_plan", lambda pid, uid, fz: reanudados.append(pid) or True)
    monkeypatch.setattr(ct, "_dispatch_pantry_nudge", lambda *a, **k: None)
    monkeypatch.setattr(no, "activo_reciente", lambda uid, dias=None: activo)
    monkeypatch.setattr(no, "apagar_por_plan_vacio", lambda uid: apagados.append(uid) or True)
    stats = ct._plan_freeze_sweep()
    congelo = any("'{_frozen_at}'" in s for s in escrituras)
    return stats, congelo, reanudados, apagados


def test_automatica_y_activo_se_apaga_en_vez_de_congelar(monkeypatch):
    stats, congelo, reanudados, apagados = _barrido(monkeypatch, nevera_enabled=None, frozen_at=None, activo=True)
    assert apagados == ["u1"] and not congelo and stats["nevera_off"] == 1 and stats["frozen"] == 0


def test_automatica_pero_inactivo_se_congela_como_siempre(monkeypatch):
    stats, congelo, reanudados, apagados = _barrido(monkeypatch, nevera_enabled=None, frozen_at=None, activo=False)
    assert apagados == [] and congelo and stats["frozen"] == 1


def test_encendida_a_mano_conserva_el_congelado(monkeypatch):
    stats, congelo, reanudados, apagados = _barrido(monkeypatch, nevera_enabled=True, frozen_at=None, activo=True)
    assert apagados == [] and congelo


def test_el_congelado_por_nevera_vacia_se_reanuda_al_apagarla(monkeypatch):
    """El caso de c7b90ca3: congelado, automática, vacía y con uso reciente."""
    fz = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    stats, congelo, reanudados, apagados = _barrido(monkeypatch, nevera_enabled=None, frozen_at=fz, activo=True)
    assert apagados == ["u1"] and reanudados == ["p1"] and stats["resumed"] == 1


def test_apagada_no_se_congela_y_si_estaba_congelado_se_reanuda(monkeypatch):
    stats, congelo, reanudados, apagados = _barrido(monkeypatch, nevera_enabled=False, frozen_at=None, activo=False)
    assert not congelo and reanudados == [] and stats["reminded"] == 0
    fz = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    stats, congelo, reanudados, apagados = _barrido(monkeypatch, nevera_enabled=False, frozen_at=fz, activo=False)
    assert reanudados == ["p1"]


def test_apagarla_descongela_al_instante(monkeypatch):
    import cron_tasks as ct
    monkeypatch.setattr(ct, "execute_sql_query", lambda sql, params=(), **k: {
        "plan_id": "p1", "frozen_at": "2026-09-19T16:35:13+00:00", "gstatus": "partial"})
    monkeypatch.setattr(no, "nevera_activa", lambda uid: False)
    monkeypatch.setattr(ct, "_resume_frozen_plan", lambda pid, uid, fz: pid == "p1")
    assert ct.try_unfreeze_plan_for_user("u1") is True
    src = (_BACKEND / "routers" / "preferences.py").read_text(encoding="utf-8")
    i = src.index("async def api_set_nevera(")
    assert "try_unfreeze_plan_for_user" in src[i:i + 2500] and "if not body.enabled:" in src[i:i + 2500]


def test_el_relleno_frena_a_la_cuenta_abandonada_sin_nevera():
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    i = src.index("def trigger_background_rolling_refill(")
    cuerpo = src[i:src.index("\ndef ", i + 10)]
    assert "if not _nev_bg.nevera_activa(uid) and not _nev_bg.activo_reciente(uid):" in cuerpo
    assert "tooltip-anchor: MEALFIT_NEVERA_OFF_IN_PLAN_MODE" in (_BACKEND / "nevera_opcional.py").read_text(encoding="utf-8")
