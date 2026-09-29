"""[P1-PLAN-LOTE-841 · 2026-09-29] El rastro del equipo deja de identificar una cuenta borrada y se purga con plazo.

LO QUE PASABA («rastro-del-equipo-sobrevive-al-borrado», abierta en `contenido-legal.json` del landing)
    `admin_access_log` no tiene FK a propósito: el rastro de lo que el personal consulta y cambia sobrevive a la
    cuenta. Pero guardaba, SIN plazo, el id de la cuenta ya borrada y el motivo que escribe el personal (texto libre
    de 3 a 300 caracteres), contra la limitación del plazo de conservación (RGPD art. 5.1.e) y contra la promesa de la
    Política de Privacidad de borrar todo al eliminar la cuenta.

LA DECISIÓN (objetivo del dueño «lo legal al 100 % para producción», 29-sep)
    Al CERRAR la cuenta, sus filas pierden el id (`target`) y el texto libre (`motivo`, `error`): queda qué hizo el
    personal y cuándo. Y todo el rastro se purga a los `MEALFIT_ADMIN_LOG_RETENTION_DAYS` días (730 por defecto) con
    un cron diario. La purga administrativa de datos (`include_profile=False`) no lo toca: la cuenta sigue viva.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import admin_acceso
import db_profiles

_UID = "11111111-2222-3333-4444-555555555555"


class _FakeDB:
    def __init__(self):
        self.writes: list[tuple[str, tuple]] = []

    def execute_sql_write(self, query, params=None, returning=False, lock_timeout_ms=None):
        q = " ".join(str(query).split())
        self.writes.append((q, params))
        return [{"id": params[0]}] if returning else True

    def execute_sql_query(self, *a, **k):
        return [] if k.get("fetch_all") else None


@pytest.fixture
def db(monkeypatch):
    fake = _FakeDB()
    monkeypatch.setattr(db_profiles, "execute_sql_write", fake.execute_sql_write)
    monkeypatch.setattr(db_profiles, "execute_sql_query", fake.execute_sql_query, raising=False)
    monkeypatch.setattr(db_profiles, "connection_pool", object(), raising=False)
    monkeypatch.setattr(db_profiles, "_purge_visual_diary_storage", lambda uid: 0, raising=False)
    return fake


def _rastro(db):
    return [(q, p) for q, p in db.writes if "admin_access_log" in q]


def test_cerrar_la_cuenta_quita_su_id_y_el_texto_libre_del_rastro(db):
    r = db_profiles.delete_account_data(_UID, include_profile=True)
    filas = _rastro(db)
    assert len(filas) == 1, filas
    q, p = filas[0]
    assert q.startswith("UPDATE public.admin_access_log SET target = NULL,"), q
    assert "- 'motivo' - 'error'" in q, "el texto libre que escribe el personal tiene que salir"
    assert "jsonb_build_object('cuenta_eliminada', true)" in q
    assert q.endswith("WHERE target = %s RETURNING id"), q
    assert p == (_UID,)
    assert r["anonymized"]["admin_access_log"] == 1
    assert "admin_access_log" not in r["failed_steps"]


def test_la_purga_administrativa_no_toca_el_rastro(db):
    """`include_profile=False` vacía datos y deja la cuenta viva: el rastro sigue refiriéndose a una cuenta que existe."""
    r = db_profiles.delete_account_data(_UID, include_profile=False)
    assert _rastro(db) == []
    assert "admin_access_log" not in r["anonymized"]


def test_un_fallo_del_paso_queda_en_failed_steps_y_no_revienta(db, monkeypatch):
    original = db.execute_sql_write

    def falla_en_el_rastro(query, params=None, returning=False, lock_timeout_ms=None):
        if "admin_access_log" in str(query):
            raise RuntimeError("base caída")
        return original(query, params, returning, lock_timeout_ms)

    monkeypatch.setattr(db_profiles, "execute_sql_write", falla_en_el_rastro)
    r = db_profiles.delete_account_data(_UID, include_profile=True)
    assert "admin_access_log" in r["failed_steps"]


def test_la_purga_borra_lo_que_pasa_del_plazo(monkeypatch):
    capturado = []

    def fake(query, params=None, returning=False, lock_timeout_ms=None):
        capturado.append((" ".join(str(query).split()), params))
        return [{"id": 1}, {"id": 2}]

    monkeypatch.setattr(admin_acceso, "execute_sql_write", fake)
    monkeypatch.delenv("MEALFIT_ADMIN_LOG_RETENTION_DAYS", raising=False)
    assert admin_acceso.purgar_rastro_antiguo() == 2
    q, p = capturado[0]
    assert q == "DELETE FROM public.admin_access_log WHERE at < now() - make_interval(days => %s) RETURNING id"
    assert p == (730,)


@pytest.mark.parametrize("crudo, esperado", [("365", 365), ("90", 90), ("3650", 3650), ("30", 730), ("x", 730)])
def test_el_plazo_es_un_knob_acotado(monkeypatch, crudo, esperado):
    monkeypatch.setenv("MEALFIT_ADMIN_LOG_RETENTION_DAYS", crudo)
    assert admin_acceso.dias_de_rastro() == esperado


def test_la_purga_nunca_revienta_el_cron(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("base caída")

    monkeypatch.setattr(admin_acceso, "execute_sql_write", boom)
    assert admin_acceso.purgar_rastro_antiguo() == 0


def test_el_cron_diario_esta_registrado():
    from cron_tasks import register_plan_chunk_scheduler

    fake_scheduler = MagicMock()
    fake_scheduler.get_job.return_value = None
    register_plan_chunk_scheduler(fake_scheduler)
    llamadas = [c for c in fake_scheduler.add_job.call_args_list if c.kwargs.get("id") == "purge_admin_access_log"]
    assert len(llamadas) == 1
    llamada = llamadas[0]
    registrada = getattr(llamada.args[0], "__wrapped__", llamada.args[0])
    assert registrada is admin_acceso.purgar_rastro_antiguo
    assert llamada.args[1] == "interval" and llamada.kwargs.get("hours") == 24
    assert llamada.kwargs.get("max_instances") == 1 and llamada.kwargs.get("coalesce") is True
    assert "jitter" in llamada.kwargs  # pasó por `_add_job_jittered`
