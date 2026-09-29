# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-798 · 2026-09-29] Las fotos del chat subidas y NO enviadas se borran a las 24 horas, aunque nadie
vuelva a subir otra.

La política de privacidad lo promete. Hasta este lote la única purga era `db_chat._cleanup_orphan_chat_attachments`, y
solo la llamaba `create_chat_attachment` — es decir, cuando alguien subía OTRA foto —, como mucho una vez por hora y por
proceso y 100 filas por pasada. Si nadie volvía a subir, la foto huérfana se quedaba en la base para siempre.

Ahora un cron horario (`purge_orphan_chat_attachments`, registrado en `register_plan_chunk_scheduler`) borra TODAS las
filas con `message_id IS NULL` y más de 24 h, por lotes acotados hasta vaciar o hasta el tope por pasada. El SQL de la
purga es uno solo (`_delete_orphan_chat_attachments_batch`) y lo comparten el cron —sin throttle— y la limpieza
oportunista de `create_chat_attachment`, que conserva su throttle de una vez por hora y proceso.
"""
from __future__ import annotations

import logging
import re
import time
from unittest.mock import MagicMock

import pytest


_JOB_ID = "purge_orphan_chat_attachments"


def _norm(sql: str) -> str:
    return re.sub(r"\s+", " ", sql or "").strip()


class _FakeWrite:
    """Doble de `execute_sql_write`: registra cada llamada y responde por orden."""

    def __init__(self, responses=None, insert_row=None, fail=None):
        self.calls: list[tuple[str, tuple, dict]] = []
        self._responses = list(responses or [])
        self._insert_row = insert_row
        self._fail = fail

    def __call__(self, query, params=None, returning=False, **kwargs):
        self.calls.append((query, params, {"returning": returning, **kwargs}))
        if self._fail is not None:
            raise self._fail
        if "INSERT INTO public.chat_attachments" in query:
            return [self._insert_row] if self._insert_row else []
        if self._responses:
            return self._responses.pop(0)
        return [] if returning else True

    def deletes(self):
        return [c for c in self.calls if "DELETE FROM public.chat_attachments" in c[0]]

    def inserts(self):
        return [c for c in self.calls if "INSERT INTO public.chat_attachments" in c[0]]


@pytest.fixture
def db_chat():
    import db_chat as module
    return module


# ---------------------------------------------------------------------------
# (a) El job está registrado en register_plan_chunk_scheduler
# ---------------------------------------------------------------------------

def test_el_cron_esta_registrado_en_register_plan_chunk_scheduler(db_chat):
    from cron_tasks import register_plan_chunk_scheduler

    fake_scheduler = MagicMock()
    fake_scheduler.get_job.return_value = None
    register_plan_chunk_scheduler(fake_scheduler)

    calls = [c for c in fake_scheduler.add_job.call_args_list if c.kwargs.get("id") == _JOB_ID]
    seen = [c.kwargs.get("id") for c in fake_scheduler.add_job.call_args_list]
    assert len(calls) == 1, f"`{_JOB_ID}` no está registrado (o lo está dos veces). Jobs vistos: {seen}"
    call = calls[0]

    registered = call.args[0]
    registered = getattr(registered, "__wrapped__", registered)
    assert registered is db_chat.purge_orphan_chat_attachments, (
        "El cron debe correr la purga SIN throttle de db_chat, no la limpieza oportunista de create_chat_attachment."
    )
    assert call.args[1] == "interval"
    # Cadencia horaria (o más frecuente): con más de 60 min la foto viviría bastante más de las 24 h prometidas.
    assert 0 < int(call.kwargs.get("minutes", 0)) <= 60
    assert call.kwargs.get("max_instances") == 1
    assert call.kwargs.get("coalesce") is True
    # Pasó por `_add_job_jittered` (P0-NEW-2): el wrapper siempre pone `jitter`.
    assert "jitter" in call.kwargs


def test_el_registro_vive_en_cron_tasks_y_la_purga_en_db_chat():
    """cron_tasks.py solo registra; el SQL de la purga nace en db_chat.py (código nuevo en su módulo)."""
    from pathlib import Path

    backend = Path(__file__).resolve().parents[1]
    cron_src = (backend / "cron_tasks.py").read_text(encoding="utf-8")
    m = re.search(r"^def register_plan_chunk_scheduler\(.*?(?=^def )", cron_src, re.DOTALL | re.MULTILINE)
    assert m, "register_plan_chunk_scheduler no encontrado"
    body = m.group(0)
    assert "P1-PLAN-LOTE-798" in body
    assert f'id="{_JOB_ID}"' in body
    # Ni un DELETE sobre chat_attachments en cron_tasks.py: el SQL es de db_chat.
    assert "DELETE FROM public.chat_attachments" not in cron_src
    db_chat_src = (backend / "db_chat.py").read_text(encoding="utf-8")
    assert db_chat_src.count("DELETE FROM public.chat_attachments") == 1, (
        "Un solo SQL de purga, compartido por el cron y por la limpieza oportunista."
    )


# ---------------------------------------------------------------------------
# (b) La purga ejecuta el DELETE correcto, sin throttle, en bucle hasta vaciar
# ---------------------------------------------------------------------------

def test_la_purga_borra_por_lotes_hasta_que_un_lote_devuelve_cero(db_chat, monkeypatch):
    fake = _FakeWrite(responses=[
        [{"id": "a"}, {"id": "b"}],
        [{"id": "c"}, {"id": "d"}],
        [{"id": "e"}],
        [],
    ])
    monkeypatch.setattr(db_chat, "execute_sql_write", fake)
    # El throttle de la limpieza oportunista acaba de disparar: el cron NO debe verse frenado por él.
    monkeypatch.setattr(db_chat, "_last_chat_attachment_cleanup", time.monotonic())

    purged = db_chat.purge_orphan_chat_attachments(batch_size=2, max_batches=10)

    assert purged == 5
    assert len(fake.calls) == 4, "debe seguir hasta que un lote borre 0 filas"
    for query, params, kwargs in fake.calls:
        sql = _norm(query)
        assert "DELETE FROM public.chat_attachments" in sql
        assert "message_id IS NULL" in sql
        assert "created_at < now() - interval '24 hours'" in sql
        assert "LIMIT %s" in sql
        assert "RETURNING" in sql
        assert params == (2,)
        assert kwargs["returning"] is True


def test_la_purga_no_tiene_throttle_por_proceso(db_chat, monkeypatch):
    monkeypatch.setattr(db_chat, "_last_chat_attachment_cleanup", time.monotonic())
    first = _FakeWrite(responses=[[{"id": "a"}], []])
    monkeypatch.setattr(db_chat, "execute_sql_write", first)
    assert db_chat.purge_orphan_chat_attachments(batch_size=5, max_batches=10) == 1

    second = _FakeWrite(responses=[[{"id": "b"}], []])
    monkeypatch.setattr(db_chat, "execute_sql_write", second)
    assert db_chat.purge_orphan_chat_attachments(batch_size=5, max_batches=10) == 1
    assert len(second.deletes()) == 2, "dos pasadas seguidas del cron borran las dos: no hay throttle"


def test_el_tope_por_pasada_corta_el_bucle_y_lo_avisa(db_chat, monkeypatch, caplog):
    class _SiempreLleno(_FakeWrite):
        def __call__(self, query, params=None, returning=False, **kwargs):
            self.calls.append((query, params, {"returning": returning, **kwargs}))
            return [{"id": str(i)} for i in range(params[0])]

    fake = _SiempreLleno()
    monkeypatch.setattr(db_chat, "execute_sql_write", fake)
    with caplog.at_level(logging.WARNING, logger=db_chat.logger.name):
        purged = db_chat.purge_orphan_chat_attachments(batch_size=3, max_batches=4)
    assert purged == 12
    assert len(fake.calls) == 4
    assert any("P1-PLAN-LOTE-798" in r.getMessage() and "tope" in r.getMessage() for r in caplog.records)


def test_un_fallo_de_la_base_corta_la_pasada_y_llega_al_listener(db_chat, monkeypatch, caplog):
    """La promesa de las 24 h es pública: un fallo NO se traga. Se registra, corta la pasada y se relanza para que
    APScheduler emita EVENT_JOB_ERROR y `_scheduler_alert_listener` lo escale a `system_alerts` (el scheduler sigue
    vivo: APScheduler captura la excepción del job)."""
    fake = _FakeWrite(fail=RuntimeError("db caída"))
    monkeypatch.setattr(db_chat, "execute_sql_write", fake)
    with caplog.at_level(logging.WARNING, logger=db_chat.logger.name):
        with pytest.raises(RuntimeError, match="db caída"):
            db_chat.purge_orphan_chat_attachments(batch_size=2, max_batches=3)
    assert len(fake.calls) == 1, "tras un fallo no insiste en la misma pasada"
    assert any("P1-PLAN-LOTE-798" in r.getMessage() for r in caplog.records)


def test_el_interruptor_apaga_el_cron(db_chat, monkeypatch):
    monkeypatch.setenv("MEALFIT_CHAT_ATTACHMENT_PURGE_ENABLED", "false")
    fake = _FakeWrite(responses=[[{"id": "a"}], []])
    monkeypatch.setattr(db_chat, "execute_sql_write", fake)
    assert db_chat.purge_orphan_chat_attachments() == 0
    assert fake.calls == []


def test_los_knobs_leen_sus_defaults(db_chat, monkeypatch):
    for name in (
        "MEALFIT_CHAT_ATTACHMENT_PURGE_ENABLED",
        "MEALFIT_CHAT_ATTACHMENT_PURGE_BATCH",
        "MEALFIT_CHAT_ATTACHMENT_PURGE_MAX_BATCHES",
    ):
        monkeypatch.delenv(name, raising=False)
    fake = _FakeWrite(responses=[[]])
    monkeypatch.setattr(db_chat, "execute_sql_write", fake)
    assert db_chat.purge_orphan_chat_attachments() == 0
    assert len(fake.calls) == 1
    assert fake.calls[0][1] == (100,), "lote por defecto = 100 filas, el mismo que la limpieza oportunista"


# ---------------------------------------------------------------------------
# (c) create_chat_attachment sigue llamando a la limpieza throttled
# ---------------------------------------------------------------------------

def test_create_chat_attachment_sigue_limpiando_con_throttle(db_chat, monkeypatch):
    monkeypatch.setattr(db_chat, "connection_pool", object())
    monkeypatch.setattr(db_chat, "_last_chat_attachment_cleanup", time.monotonic() - 7200)
    fake = _FakeWrite(insert_row={"id": "11111111-1111-4111-8111-111111111111"})
    monkeypatch.setattr(db_chat, "execute_sql_write", fake)

    args = ("sess-1", "user-1", b"\xff\xd8\xff", "image/jpeg")
    assert db_chat.create_chat_attachment(*args) == "11111111-1111-4111-8111-111111111111"
    assert db_chat.create_chat_attachment(*args) == "11111111-1111-4111-8111-111111111111"

    assert len(fake.inserts()) == 2
    deletes = fake.deletes()
    assert len(deletes) == 1, "la limpieza oportunista sigue limitada a una vez por hora y proceso"
    # Mismo acotado de siempre: un solo lote de 100 filas, y ANTES del INSERT.
    assert deletes[0][1] == (100,)
    assert "DELETE FROM public.chat_attachments" in fake.calls[0][0]

    # Y es el MISMO SQL que corre el cron (SSOT).
    cron_fake = _FakeWrite(responses=[[]])
    monkeypatch.setattr(db_chat, "execute_sql_write", cron_fake)
    db_chat.purge_orphan_chat_attachments(batch_size=100, max_batches=1)
    assert _norm(cron_fake.calls[0][0]) == _norm(deletes[0][0])


def test_la_limpieza_oportunista_no_rompe_la_subida_si_la_base_falla(db_chat, monkeypatch):
    monkeypatch.setattr(db_chat, "connection_pool", object())
    monkeypatch.setattr(db_chat, "_last_chat_attachment_cleanup", time.monotonic() - 7200)

    class _DeleteFalla(_FakeWrite):
        def __call__(self, query, params=None, returning=False, **kwargs):
            if "DELETE FROM public.chat_attachments" in query:
                self.calls.append((query, params, {"returning": returning, **kwargs}))
                raise RuntimeError("lock timeout")
            return super().__call__(query, params, returning, **kwargs)

    fake = _DeleteFalla(insert_row={"id": "22222222-2222-4222-8222-222222222222"})
    monkeypatch.setattr(db_chat, "execute_sql_write", fake)
    assert db_chat.create_chat_attachment("s", "u", b"x", "image/png") == "22222222-2222-4222-8222-222222222222"
