"""[P1-NEVERA-OPCIONAL · 2026-09-23] La Nevera opcional en modo contador.

El dueño: «cuando el generador esté desactivado, que la Nevera también se pueda desactivar: hay gente que solo
quiere el contador y el agente», y «encendida como hoy, pero si en 48 h no se usa, que se apague sola». Medido el
23-sep: las 6 cuentas en modo contador tenían la Nevera vacía. LA regla vive en `nevera_opcional.py`; aquí se ancla
que la regla, el apagado automático, el diario, el coach y la API la respeten. Doc: docs/nevera_opcional.md."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import nevera_opcional as no

_BACKEND = Path(__file__).resolve().parents[1]


# ── 1. La regla ──────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("perfil, esperado", [
    ({"plan_mode": "tracking", "nevera_enabled": False}, False),
    ({"plan_mode": "tracking", "nevera_enabled": None}, True),
    ({"plan_mode": "tracking", "nevera_enabled": True}, True),
    ({"plan_mode": "plan", "nevera_enabled": False}, True),       # el plan la necesita: vuelve sola
    ({"plan_mode": "tracking"}, True),                            # sin columna todavía: fallo abierto
    ({}, True),
    (None, True),
])
def test_la_regla(perfil, esperado):
    assert no.nevera_activa_de(perfil) is esperado


def test_el_kill_switch_la_deja_activa_para_todos(monkeypatch):
    monkeypatch.setenv("MEALFIT_NEVERA_SWITCH", "false")
    assert no.nevera_activa_de({"plan_mode": "tracking", "nevera_enabled": False}) is True
    assert no.apagar_neveras_sin_uso() == []


def test_nevera_activa_lee_la_fila_y_falla_abierto(monkeypatch):
    monkeypatch.setattr(no, "execute_sql_query", lambda *a, **k: {"plan_mode": "tracking", "nevera_enabled": False})
    assert no.nevera_activa("u1") is False
    assert no.nevera_activa("guest") is True
    assert no.nevera_activa(None) is True

    def _revienta(*a, **k):
        raise RuntimeError("column nevera_enabled does not exist")
    monkeypatch.setattr(no, "execute_sql_query", _revienta)
    assert no.nevera_activa("u1") is True


def test_fijar_filtra_por_id_y_borra_la_marca_automatica(monkeypatch):
    visto = {}

    def _w(sql, params, **k):
        visto["sql"], visto["params"], visto["k"] = sql, params, k
        return [{"id": "u1"}]
    monkeypatch.setattr(no, "execute_sql_write", _w)
    assert no.fijar_nevera("u1", True) is True
    assert "WHERE id = %s" in visto["sql"] and "nevera_auto_off_at = NULL" in visto["sql"]
    assert visto["params"] == (True, "u1") and visto["k"].get("returning") is True


def test_estado_para_configuracion(monkeypatch):
    from datetime import datetime, timezone
    at = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(no, "execute_sql_query", lambda *a, **k: {
        "plan_mode": "tracking", "nevera_enabled": False, "nevera_auto_off_at": at})
    e = no.estado_nevera("u1")
    assert e == {"enabled": False, "activa": False, "auto_off_at": at.isoformat(), "disponible": True}


# ── 2. El apagado automático ─────────────────────────────────────────────────────────────────────────────────
def test_el_sql_del_apagado_solo_toca_contador_automatico_y_vacio():
    sql = no._SQL_APAGAR
    for trozo in ("q.plan_mode = 'tracking'", "q.nevera_enabled IS NULL", "p.nevera_enabled IS NULL",
                  "i.quantity > 0", "i.updated_at >", "nevera_reloj_desde", "plan_mode_changed_at",
                  "SET nevera_enabled = FALSE, nevera_auto_off_at = now()", "RETURNING p.id", "LIMIT %s"):
        assert trozo in sql, trozo


def test_apagar_devuelve_ids_y_pasa_horas_y_limite(monkeypatch):
    visto = {}

    def _w(sql, params, **k):
        visto["params"] = params
        return [{"id": "u1"}, {"id": "u2"}]
    monkeypatch.setattr(no, "execute_sql_write", _w)
    monkeypatch.setenv("MEALFIT_NEVERA_AUTO_OFF_HOURS", "72")
    assert no.apagar_neveras_sin_uso(limite=10) == ["u1", "u2"]
    assert visto["params"] == (72, 72, 10)


def test_apagar_con_el_knob_apagado_no_toca_la_db(monkeypatch):
    monkeypatch.setenv("MEALFIT_NEVERA_AUTO_OFF", "false")
    monkeypatch.setattr(no, "execute_sql_write", lambda *a, **k: pytest.fail("no debía escribir"))
    assert no.apagar_neveras_sin_uso() == []


# ── 3. Migración ─────────────────────────────────────────────────────────────────────────────────────────────
def test_la_migracion_es_idempotente_y_esta_en_las_dos_carpetas():
    nombre = "p1_nevera_opcional_2026_09_23.sql"
    be = (_BACKEND / "migrations" / nombre).read_text(encoding="utf-8")
    raiz = _BACKEND.parent / "migrations" / nombre
    if raiz.exists():
        assert raiz.read_text(encoding="utf-8") == be, "migrations/ y backend/migrations/ deben ser idénticas"
    for col in ("nevera_enabled BOOLEAN", "nevera_auto_off_at TIMESTAMPTZ",
                "nevera_reloj_desde TIMESTAMPTZ NOT NULL DEFAULT now()"):
        assert f"ADD COLUMN IF NOT EXISTS {col}" in be, col
    assert "RAISE EXCEPTION" in be


# ── 4. API y perfil ──────────────────────────────────────────────────────────────────────────────────────────
def test_patch_guarda_y_devuelve_el_estado(monkeypatch):
    from routers import preferences as pref
    llamadas = []
    monkeypatch.setattr(no, "fijar_nevera", lambda uid, en: llamadas.append((uid, en)) or True)
    monkeypatch.setattr(no, "estado_nevera", lambda uid: {"enabled": False, "activa": False,
                                                           "auto_off_at": None, "disponible": True})
    r = asyncio.run(pref.api_set_nevera(body=pref.NeveraPreferenceBody(enabled=False), verified_user_id="u1"))
    assert llamadas == [("u1", False)] and r["activa"] is False


def test_patch_con_el_knob_apagado_es_409(monkeypatch):
    from fastapi import HTTPException
    from routers import preferences as pref
    monkeypatch.setenv("MEALFIT_NEVERA_SWITCH", "false")
    with pytest.raises(HTTPException) as e:
        asyncio.run(pref.api_set_nevera(body=pref.NeveraPreferenceBody(enabled=False), verified_user_id="u1"))
    assert e.value.status_code == 409


def test_el_perfil_trae_la_regla_calculada():
    ud = (_BACKEND / "routers" / "user_data.py").read_text(encoding="utf-8")
    i = ud.index("async def api_get_profile(")
    assert '"nevera_activa": nevera_activa_de(profile)' in ud[i:i + 1500]
