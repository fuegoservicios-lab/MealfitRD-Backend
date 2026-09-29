"""[P1-PLAN-LOTE-716 · 2026-09-28] Privacidad de la cuenta (Configuración → Privacidad) lista para producción.

tooltip-anchor: P1-PLAN-LOTE-716

Lo que cubre cada bloque (y por qué cada test falla contra el código previo):

  1. P0 · `summary_archive` (resúmenes archivados de las charlas de salud, sin user_id ni FK) no lo borraba
     ninguna vía: ni borrar un chat / todos / el TTL (`delete_agent_sessions_with_checkpoints`), ni «Eliminar
     cuenta» (`delete_account_data`). Ahora cae con el MISMO conjunto de hilos que los checkpoints, y el barrido
     diario de huérfanos lo cubre también.
  2. P1 · «Olvidar» un recuerdo era un soft delete que nadie purgaba y el export lo devolvía. Ahora
     `forget_user_fact` lo BORRA (con la síntesis del Dreaming que lo cite) y el endpoint distingue
     200 / 404 / 403 / 503 en vez de responder «éxito» siempre.
  3. P1 · El export omitía la memoria del agente y 10 tablas personales, cortaba con LIMIT sin ORDER BY,
     arrastraba vectores, contaba los hechos desactivados, se «saltaba» tablas en silencio y serializaba en el
     event loop. Ahora: tablas nuevas, orden, vectores fuera en la base, `omitted`/`complete`, bytes en el hilo.
  4. P2 · Borrado de cuenta: con el perfil sin borrar ya NO se borra la identidad (flags `profile_deleted` /
     `identity_deleted`), al cliente van códigos y no el texto de la base, un borrado a medias deja alerta
     `account_delete_partial:<uid>`, y se purgan las cachés por usuario (rag_/reflection_/pending_pipeline...).
  5. P1 · «Empezar desde cero» borra también la memoria del coach (cola de hechos, síntesis y bitácora del
     Dreaming, gustos aprendidos, cachés) y CONSERVA el chat; si la transacción hace ROLLBACK, 500 `reset_failed`.

Todo con la base falsa: el `.env` de desarrollo apunta a producción y ningún test escribe en ella.
"""
from __future__ import annotations

import asyncio
import json
import re
import threading
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent

UID = "11111111-2222-3333-4444-555555555555"
OTRO = "99999999-8888-7777-6666-555555555555"
FID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
_S1 = "aaaaaaaa-0000-4000-8000-000000000001"
_S2 = "bbbbbbbb-0000-4000-8000-000000000002"


def _norm(sql) -> str:
    return " ".join(str(sql).split())


# ─────────────────────────────────────────────────────────── base falsa (pool / conexión / cursor)
class _FakeCursor:
    def __init__(self, pool):
        self.pool = pool
        self.rowcount = 0
        self._rows: list = []

    def execute(self, sql, params=None):
        q = _norm(sql)
        self.pool.log.append(("exec", q, params))
        for fragment in self.pool.fail_on:
            if fragment in q:
                raise RuntimeError(f"fallo simulado de la base en: {fragment}")
        rows, rowcount = self.pool.respond(q, params)
        self._rows = list(rows)
        self.rowcount = rowcount

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeTx:
    def __init__(self, pool):
        self.pool = pool

    def __enter__(self):
        self.pool.log.append(("begin",))
        return self

    def __exit__(self, exc_type, *a):
        self.pool.log.append(("rollback",) if exc_type else ("commit",))
        return False


class _FakeConn:
    def __init__(self, pool):
        self.pool = pool

    def transaction(self):
        return _FakeTx(self.pool)

    def cursor(self, row_factory=None):
        return _FakeCursor(self.pool)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakePool:
    """Registra cada sentencia; `answers` es una lista de (fragmento_sql, filas, rowcount)."""

    def __init__(self, answers=None, fail_on=()):
        self.log: list = []
        self.answers = list(answers or [])
        self.fail_on = tuple(fail_on)

    def connection(self):
        return _FakeConn(self)

    def respond(self, q, params):
        for fragment, rows, rowcount in self.answers:
            if fragment in q:
                return rows, rowcount
        return [], 0

    def execs(self):
        return [(e[1], e[2]) for e in self.log if e[0] == "exec"]

    def sqls(self):
        return [e[1] for e in self.log if e[0] == "exec"]


# ═════════════════════════════════════════════ 1. summary_archive en TODAS las vías de borrado
@pytest.fixture
def chat_pool(monkeypatch):
    import db_chat
    import db_core

    pool = _FakePool(answers=[("RETURNING id", [{"id": _S2}, {"id": _S1}], 2)])
    monkeypatch.setattr(db_core, "_guard_test_write_to_prod", lambda q: None)
    monkeypatch.setattr(db_chat, "connection_pool", pool)
    return pool


@pytest.mark.parametrize("via", ["un_chat_legacy", "todos_los_chats", "un_chat_con_idor_guard", "ttl"])
def test_borrar_sesiones_arrastra_su_archivo_de_resumenes_en_la_misma_transaccion(chat_pool, monkeypatch, via):
    import db_chat

    monkeypatch.setattr(db_chat, "get_session_owner", lambda sid: "user-1")
    monkeypatch.setattr(db_chat, "execute_sql_write", lambda *a, **k: True)
    if via == "un_chat_legacy":
        assert db_chat.delete_single_agent_session(_S1) is True
    elif via == "todos_los_chats":
        assert db_chat.delete_user_agent_sessions("user-1") is True
    elif via == "un_chat_con_idor_guard":
        assert db_chat.delete_chat_session(_S1, "user-1") == (True, "")
    else:  # la forma del SQL del TTL de cron_tasks (`_sweep_stale_chat_sessions`)
        db_chat.delete_agent_sessions_with_checkpoints(
            "DELETE FROM agent_sessions WHERE id IN (SELECT id FROM agent_sessions "
            "WHERE created_at < NOW() - (%s::int * INTERVAL '1 day') ORDER BY created_at ASC LIMIT %s) RETURNING id",
            (90, 500),
        )

    arch = [(q, p) for q, p in chat_pool.execs() if "summary_archive" in q]
    assert arch == [("DELETE FROM public.summary_archive WHERE session_id = ANY(%s::text[])", ([_S1, _S2],))], (
        f"el archivo frío de resúmenes debe caer con los ids de las sesiones borradas: {chat_pool.execs()}"
    )
    kinds = [e[0] for e in chat_pool.log]
    assert kinds[0] == "begin" and kinds[-1] == "commit" and kinds.count("begin") == 1, (
        "sesión, checkpoints y archivo en UNA transacción"
    )


def test_el_barrido_diario_limpia_el_archivo_huerfano_en_su_propia_transaccion(chat_pool):
    import db_chat

    chat_pool.answers = [
        ("SELECT ck.thread_id", [{"thread_id": _S1, "last_ts": None}], 1),
        ("summary_archive", [{"id": "a1"}, {"id": "a2"}, {"id": "a3"}], 3),
    ]
    res = db_chat.sweep_orphan_chat_checkpoints(7, 100)
    assert res["threads"] == [_S1]
    assert res["deleted"].get("summary_archive") == 3
    arch = [(q, p) for q, p in chat_pool.execs() if "summary_archive" in q]
    assert len(arch) == 1
    q, p = arch[0]
    assert q.startswith("DELETE FROM public.summary_archive")
    assert "NOT EXISTS ( SELECT 1 FROM public.agent_sessions s WHERE s.id::text = sa.session_id )" in q
    assert "LIMIT %s" in q and p == (100,)
    assert [e[0] for e in chat_pool.log].count("begin") == 2, "transacción propia: no comparte la de checkpoints"


def test_si_el_barrido_del_archivo_falla_el_de_checkpoints_sigue_en_pie(chat_pool):
    import db_chat

    chat_pool.answers = [("SELECT ck.thread_id", [{"thread_id": _S1, "last_ts": None}], 1)]
    chat_pool.fail_on = ("summary_archive",)
    res = db_chat.sweep_orphan_chat_checkpoints(7, 100)
    assert res["threads"] == [_S1] and "summary_archive" not in res["deleted"]
    ck_tx_end = [e for e in chat_pool.log if e[0] in ("commit", "rollback")]
    assert ck_tx_end[0] == ("commit",) and ck_tx_end[-1] == ("rollback",)
    assert any(q.startswith("DELETE FROM public.summary_archive") for q in chat_pool.sqls()), (
        "el barrido del archivo debe intentarse (y fallar aquí)"
    )


# ═════════════════════════════════════════════ 2. «Olvidar» = borrado de verdad
@pytest.fixture
def facts_pool(monkeypatch):
    import db_facts

    pool = _FakePool()
    rag: list = []
    monkeypatch.setattr(db_facts, "connection_pool", pool)
    monkeypatch.setattr(db_facts, "_invalidate_rag_cache", lambda uid: rag.append(uid))
    monkeypatch.setattr(db_facts, "execute_sql_write",
                        lambda *a, **k: pytest.fail("«olvidar» no debe ir por el soft delete suelto"))
    pool.rag = rag
    return pool


def test_olvidar_borra_la_fila_y_la_sintesis_del_dreaming_que_la_cita(facts_pool):
    import db_facts

    facts_pool.answers = [
        ("SELECT user_id::text AS user_id FROM public.user_facts", [{"user_id": UID}], 1),
        ("DELETE FROM public.user_facts", [], 1),
        ("DELETE FROM public.user_memory_profile", [], 1),
    ]
    assert db_facts.forget_user_fact(FID, UID) == "deleted"

    execs = facts_pool.execs()
    assert execs[0] == ("SELECT user_id::text AS user_id FROM public.user_facts WHERE id = %s FOR UPDATE", (FID,))
    assert ("DELETE FROM public.user_facts WHERE id = %s AND user_id = %s", (FID, UID)) in execs
    assert ("DELETE FROM public.user_memory_profile WHERE user_id = %s AND %s::uuid = ANY(evidence_fact_ids)",
            (UID, FID)) in execs
    assert not any("is_active" in q for q, _ in execs), "nada de soft delete: la fila deja de existir"
    kinds = [e[0] for e in facts_pool.log]
    assert kinds[0] == "begin" and kinds[-1] == "commit" and kinds.count("begin") == 1
    assert facts_pool.rag == [UID], "el contexto RAG cacheado se invalida tras el borrado"


def test_olvidar_un_hecho_ajeno_no_borra_nada(facts_pool):
    import db_facts

    facts_pool.answers = [("SELECT user_id::text AS user_id FROM public.user_facts", [{"user_id": OTRO}], 1)]
    assert db_facts.forget_user_fact(FID, UID) == "forbidden"
    assert not any(q.startswith("DELETE") for q in facts_pool.sqls())
    assert facts_pool.rag == []


def test_olvidar_un_hecho_que_no_existe(facts_pool):
    import db_facts

    assert db_facts.forget_user_fact(FID, UID) == "not_found"
    assert not any(q.startswith("DELETE") for q in facts_pool.sqls())


def test_olvidar_un_id_que_no_es_uuid_no_toca_la_base(facts_pool):
    import db_facts

    assert db_facts.forget_user_fact("no-es-un-uuid", UID) == "not_found"
    assert facts_pool.log == []


def test_olvidar_con_la_base_caida_levanta_y_hace_rollback(facts_pool):
    import db_facts

    facts_pool.fail_on = ("FOR UPDATE",)
    with pytest.raises(RuntimeError):
        db_facts.forget_user_fact(FID, UID)
    assert facts_pool.log[-1] == ("rollback",)
    assert facts_pool.rag == []


def test_olvidar_sin_usuario_se_rechaza(facts_pool):
    import db_facts

    with pytest.raises(ValueError):
        db_facts.forget_user_fact(FID, "")
    assert facts_pool.log == []


# ═════════════════════════════════════════════ app.py: endpoints (sin levantar el servidor)
@pytest.fixture(scope="module")
def app_module():
    import app as _app

    return _app


def _run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize("desenlace,status", [
    ("deleted", 200), ("not_found", 404), ("forbidden", 403), ("boom", 503),
])
def test_endpoint_olvidar_da_un_codigo_por_desenlace(app_module, monkeypatch, desenlace, status):
    import db
    from fastapi import HTTPException

    llamadas: list = []

    def fake_forget(fact_id, user_id):
        llamadas.append((fact_id, user_id))
        if desenlace == "boom":
            raise RuntimeError("couldn't get a connection after 8.00 sec")
        return desenlace

    monkeypatch.setattr(db, "forget_user_fact", fake_forget)
    if status == 200:
        r = app_module.api_delete_user_fact(FID, verified_user_id=UID)
        assert r["success"] is True
    else:
        with pytest.raises(HTTPException) as exc:
            app_module.api_delete_user_fact(FID, verified_user_id=UID)
        assert exc.value.status_code == status
        if status == 503:
            assert exc.value.detail["code"] == "fact_delete_unavailable"
            assert "connection" not in json.dumps(exc.value.detail), "el texto de la base no viaja al cliente"
        if status == 404:
            assert exc.value.detail["code"] == "fact_not_found"
    assert llamadas == [(FID, UID)], "el endpoint delega en forget_user_fact con el uid VERIFICADO"


# ═════════════════════════════════════════════ 3. export
_EXPORT_NUEVAS = (
    "user_memory_profile", "conversation_summaries", "summary_archive", "weight_log", "water_intake_log",
    "meal_likes", "meal_rejections", "abandoned_meal_reasons", "custom_shopping_items",
    "user_brand_preferences", "inventory_consumption_events", "visual_diary", "chat_attachments",
)


class _ExportDB:
    def __init__(self, fail_on=()):
        self.calls: list = []
        self.fail_on = tuple(fail_on)
        self.threads: list = []

    def __call__(self, sql, params=None, fetch_one=False, fetch_all=False):
        q = _norm(sql)
        self.calls.append((q, params))
        self.threads.append(threading.current_thread() is threading.main_thread())
        tabla = re.search(r"FROM public\.(\w+)\)", q).group(1)
        if tabla in self.fail_on or "*" in self.fail_on:
            raise RuntimeError(f"relation {tabla} fallo simulado")
        return [{"r": {"tabla": tabla, "id": 1}}]

    def sql_de(self, tabla):
        return next((q, p) for q, p in self.calls if f"FROM public.{tabla})" in q)


def test_export_trae_la_memoria_y_las_tablas_personales_ordenadas_y_sin_vectores(app_module, monkeypatch):
    from starlette.responses import Response

    fake = _ExportDB()
    monkeypatch.setattr(app_module, "execute_sql_query", fake)
    resp = _run(app_module.api_export_my_account(verified_user_id=UID))

    assert isinstance(resp, Response) and resp.media_type == "application/json"
    assert isinstance(resp.body, (bytes, bytearray)), "el JSON viaja ya serializado"
    payload = json.loads(resp.body)
    for tabla in _EXPORT_NUEVAS + ("user_profiles", "meal_plans", "user_inventory", "user_facts", "agent_messages"):
        assert tabla in payload["data"], f"falta {tabla} en el export"
    assert payload["complete"] is True and payload["omitted"] == []
    assert "skipped" not in payload

    for q, p in fake.calls:
        assert " ORDER BY " in q and " LIMIT " in q, f"cap sin orden: {q}"
        assert q.startswith("SELECT to_jsonb(t) - %s::text[] AS r FROM (")
        assert {"embedding", "profile_embedding"} <= set(p[0]), "los vectores se quitan EN la base"
        assert q.count("%s") == len(p), f"placeholders ≠ parámetros en {q}"
        assert all(x == UID for x in p[1:]), "todo parámetro del WHERE es el uid verificado"

    q_facts, _ = fake.sql_de("user_facts")
    assert "is_active IS TRUE" in q_facts, "los hechos desactivados no salen en el export"
    q_att, _ = fake.sql_de("chat_attachments")
    assert "SELECT * FROM public.chat_attachments" not in q_att and "content_type" in q_att, (
        "las fotos del chat: metadatos, jamás el bytea"
    )
    hilos = "SELECT id::text FROM agent_sessions WHERE user_id = %s UNION SELECT session_id::text FROM agent_messages"
    for tabla in ("summary_archive", "agent_messages", "conversation_summaries", "agent_sessions"):
        q, _ = fake.sql_de(tabla)
        assert hilos in q, f"{tabla} debe acotarse por los hilos del usuario (SSOT del borrado)"
    assert "ORDER BY archived_at DESC NULLS LAST" in fake.sql_de("summary_archive")[0]
    assert "ORDER BY rejected_at DESC NULLS LAST" in fake.sql_de("meal_rejections")[0]
    assert "ORDER BY log_date DESC" in fake.sql_de("water_intake_log")[0]
    assert not any(fake.threads), "la lectura corre fuera del event loop"


def test_export_una_tabla_que_falla_queda_en_omitted_y_la_copia_no_se_da_por_completa(app_module, monkeypatch, caplog):
    fake = _ExportDB(fail_on=("weight_log",))
    monkeypatch.setattr(app_module, "execute_sql_query", fake)
    with caplog.at_level("WARNING"):
        resp = _run(app_module.api_export_my_account(verified_user_id=UID))
    payload = json.loads(resp.body)
    assert payload["omitted"] == ["weight_log"] and payload["complete"] is False
    assert "weight_log" not in payload["data"]
    assert any("weight_log" in r.getMessage() and r.levelname == "WARNING" for r in caplog.records)


def test_export_sin_ninguna_tabla_legible_es_503_y_no_un_archivo_vacio(app_module, monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(app_module, "execute_sql_query", _ExportDB(fail_on=("*",)))
    with pytest.raises(HTTPException) as exc:
        _run(app_module.api_export_my_account(verified_user_id=UID))
    assert exc.value.status_code == 503 and exc.value.detail["code"] == "export_unavailable"


def test_export_serializa_en_el_hilo_y_no_en_el_event_loop(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "execute_sql_query", _ExportDB())
    hilos: list = []
    original = app_module._account_export_json_bytes

    def espia(payload):
        hilos.append(threading.current_thread() is threading.main_thread())
        return original(payload)

    monkeypatch.setattr(app_module, "_account_export_json_bytes", espia)
    _run(app_module.api_export_my_account(verified_user_id=UID))
    assert hilos == [False], "json.dumps del export debe correr en el hilo de la lectura"


# ═════════════════════════════════════════════ 4. borrado de cuenta (motor)
class _WriteDB:
    def __init__(self, fail_on=()):
        self.writes: list = []
        self.fail_on = tuple(fail_on)

    def __call__(self, query, params=None, returning=False, lock_timeout_ms=None):
        q = _norm(query)
        self.writes.append((q, params))
        for fragment in self.fail_on:
            if fragment in q:
                raise RuntimeError(f'update or delete on table violates foreign key constraint ({fragment})')
        return [{"id": "x"}] if returning else True

    def sqls(self):
        return [q for q, _ in self.writes]


@pytest.fixture
def purga(monkeypatch):
    import db_profiles

    def _montar(fail_on=()):
        fake = _WriteDB(fail_on)
        monkeypatch.setattr(db_profiles, "execute_sql_write", fake)
        monkeypatch.setattr(db_profiles, "execute_sql_query",
                            lambda *a, **k: [] if k.get("fetch_all") else None, raising=False)
        monkeypatch.setattr(db_profiles, "connection_pool", object(), raising=False)
        monkeypatch.setattr(db_profiles, "_purge_visual_diary_storage", lambda uid: 0, raising=False)
        monkeypatch.setattr(db_profiles, "_purge_user_redis_caches", lambda *a, **k: 0, raising=False)
        return fake

    return _montar


def test_si_el_perfil_no_se_borra_la_identidad_se_conserva(purga):
    import db_profiles

    fake = purga(fail_on=("DELETE FROM user_profiles",))
    r = db_profiles.delete_account_data(UID, include_profile=True)
    assert not any('neon_auth."user"' in q for q in fake.sqls()), (
        "con el perfil vivo, borrar la identidad deja la salud huérfana y al usuario sin poder reintentar"
    )
    assert r["profile_deleted"] is False and r["identity_deleted"] is False
    assert "user_profiles" in r["failed_steps"]
    assert any("foreign key" in e for e in r["errors"]), "el texto crudo se queda en `errors` (logs/alerta)"


def test_borrado_completo_marca_perfil_e_identidad(purga):
    import db_profiles

    purga()
    r = db_profiles.delete_account_data(UID, include_profile=True)
    assert r["profile_deleted"] is True and r["identity_deleted"] is True
    assert r["failed_steps"] == [] and r["errors"] == []


def test_fallo_de_la_identidad_queda_como_paso_fallido(purga):
    import db_profiles

    purga(fail_on=('neon_auth."user"',))
    r = db_profiles.delete_account_data(UID, include_profile=True)
    assert r["profile_deleted"] is True and r["identity_deleted"] is False
    assert r["failed_steps"] == ["neon_auth_user"]


def test_la_purga_borra_el_archivo_de_resumenes_con_los_hilos_del_usuario_antes_que_sesiones_y_mensajes(purga):
    import db_profiles

    fake = purga()
    db_profiles.delete_account_data(UID, include_profile=True)
    qs = fake.sqls()
    i_arch = next(i for i, q in enumerate(qs) if q.startswith("DELETE FROM summary_archive"))
    q, p = fake.writes[i_arch]
    assert q == (
        "DELETE FROM summary_archive WHERE session_id IN (SELECT id::text FROM agent_sessions WHERE user_id = %s "
        "UNION SELECT session_id::text FROM agent_messages WHERE user_id = %s UNION SELECT %s::text) RETURNING id"
    )
    assert p == (UID, UID, UID)
    i_ses = next(i for i, q in enumerate(qs) if q.startswith("DELETE FROM agent_sessions"))
    i_msg = next(i for i, q in enumerate(qs) if q.startswith("DELETE FROM agent_messages"))
    assert i_arch < i_ses and i_arch < i_msg, "el conjunto de hilos sale de sesiones y mensajes: antes de borrarlos"


def test_la_purga_borra_las_sesiones_sin_dueno_que_son_inequivocamente_suyas(purga):
    import db_profiles

    fake = purga()
    db_profiles.delete_account_data(UID, include_profile=True)
    qs = fake.sqls()
    i_nul = next(i for i, q in enumerate(qs) if "s.user_id IS NULL" in q)
    q, p = fake.writes[i_nul]
    assert q.startswith("DELETE FROM agent_sessions s WHERE s.user_id IS NULL AND (s.id = %s::uuid OR (")
    assert "o.user_id IS NOT NULL AND o.user_id <> %s" in q, "si otra cuenta escribió en ella, no es suya"
    assert p == (UID, UID, UID)
    i_msg = next(i for i, q in enumerate(qs) if q.startswith("DELETE FROM agent_messages WHERE user_id"))
    assert i_nul < i_msg, "se identifican por los mensajes: antes de borrar esos mensajes"


def test_la_purga_limpia_las_caches_por_usuario_sin_esperar_al_ttl(purga):
    import db_profiles

    fake = purga()
    db_profiles.delete_account_data(UID, include_profile=True)
    exactas = next(p[0] for q, p in fake.writes if q == "DELETE FROM app_kv_store WHERE key = ANY(%s) RETURNING key")
    for clave in (f"pending_pipeline:{UID}", f"pantry_nudge_last:{UID}", f"hydration_state:{UID}"):
        assert clave in exactas
    patrones = next(p[0] for q, p in fake.writes if q == "DELETE FROM app_kv_store WHERE key LIKE ANY(%s) RETURNING key")
    assert any(_like(pt, f"rag_{UID}_0123abcd") for pt in patrones)
    assert any(_like(pt, f"reflection_{UID}_7_abcd1234") for pt in patrones)
    assert any(_like(pt, f"regen_day_done:{UID}:plan-1:3") for pt in patrones)


def _like(pattern: str, value: str) -> bool:
    """LIKE de Postgres con `\\` como escape por defecto (lo justo para comprobar los patrones)."""
    rx, i = [], 0
    while i < len(pattern):
        c = pattern[i]
        if c == "\\" and i + 1 < len(pattern):
            rx.append(re.escape(pattern[i + 1]))
            i += 2
            continue
        rx.append(".*" if c == "%" else "." if c == "_" else re.escape(c))
        i += 1
    return re.fullmatch("".join(rx), value, re.DOTALL) is not None


def test_los_patrones_like_escapan_los_comodines():
    import db_profiles

    patrones = db_profiles._kv_like_patterns(UID, ("rag_{uid}_",))
    assert _like(patrones[0], f"rag_{UID}_x")
    assert not _like(patrones[0], f"ragX{UID}Yx"), "`_` sin escapar casaría cualquier carácter"
    assert not _like(patrones[0], f"rag_{OTRO}_x")


def test_la_lista_de_purga_cubre_las_tablas_que_solo_caian_por_el_perfil():
    import db_profiles

    for t in ("chat_attachments", "plan_generation_runs", "user_taste_events", "inventory_consumption_events",
              "user_brand_preferences", "device_push_tokens", "plan_jobs", "ai_training_corpus"):
        assert t in db_profiles._USER_SCOPED_TABLES_USERID, f"{t} fuera de la purga"
    assert db_profiles._USER_SCOPED_TABLES_USERID[-1] == "meal_plans"


# ═════════════════════════════════════════════ 4-bis. borrado de cuenta (endpoint)
def _resultado(**kw):
    base = {"user_id": UID, "deleted": {"user_profiles": 1}, "anonymized": {}, "errors": [], "failed_steps": [],
            "profile_deleted": True, "identity_deleted": True, "storage_objects_removed": 0}
    base.update(kw)
    return base


@pytest.fixture
def borrar(app_module, monkeypatch):
    import db_profiles
    from starlette.responses import Response

    alertas: list = []
    monkeypatch.setattr(app_module, "execute_sql_query", lambda *a, **k: None)  # sin suscripción PayPal
    monkeypatch.setattr(app_module, "execute_sql_write", lambda q, p=None, **k: alertas.append((_norm(q), p)) or True)

    def _llamar(resultado):
        monkeypatch.setattr(db_profiles, "delete_account_data", lambda uid, include_profile=True: resultado)
        resp = Response()
        try:
            out = _run(app_module.api_delete_my_account(
                response=resp, data={"confirm": "ELIMINAR"}, verified_user_id=UID))
        except Exception as e:  # noqa: BLE001 — el test inspecciona la HTTPException
            return resp, e, alertas
        return resp, out, alertas

    return _llamar


def test_borrado_a_medias_con_la_cuenta_ya_cerrada_devuelve_codigos_y_deja_alerta(borrar):
    crudo = 'consumed_meals: update or delete on table "consumed_meals" violates foreign key Key (user_id)=(1111)'
    resp, out, alertas = borrar(_resultado(errors=[crudo], failed_steps=["consumed_meals"]))
    assert isinstance(out, dict) and out["success"] is False
    assert out["errors"] == ["delete_failed:consumed_meals"], "al cliente van códigos, no el texto de la base"
    assert "violates" not in json.dumps(out)
    assert out["profile_deleted"] is True and out["identity_deleted"] is True
    assert resp.headers.getlist("set-cookie"), "la cuenta ya no existe: la sesión se invalida"
    ins = [(q, p) for q, p in alertas if q.startswith("INSERT INTO system_alerts")]
    assert len(ins) == 1 and ins[0][1][0] == f"account_delete_partial:{UID}"
    meta = json.loads(ins[0][1][4])
    assert meta["failed_steps"] == ["consumed_meals"] and crudo in meta["errors"], "el SRE sí ve el error crudo"


def test_sin_identidad_borrada_es_503_y_la_sesion_sigue_viva_para_reintentar(borrar):
    from fastapi import HTTPException

    resp, out, alertas = borrar(_resultado(
        errors=["user_profiles: boom"], failed_steps=["user_profiles"], profile_deleted=False,
        identity_deleted=False, deleted={}))
    assert isinstance(out, HTTPException) and out.status_code == 503
    assert out.detail["code"] == "account_delete_incomplete"
    assert out.detail["profile_deleted"] is False and out.detail["identity_deleted"] is False
    assert out.detail["errors"] == ["delete_failed:user_profiles"]
    assert resp.headers.getlist("set-cookie") == [], "la cuenta sigue viva: no se cierra la sesión"
    assert any(q.startswith("INSERT INTO system_alerts") for q, _ in alertas)


def test_borrado_completo_cierra_una_alerta_previa_de_borrado_a_medias(borrar):
    resp, out, alertas = borrar(_resultado())
    assert out["success"] is True and out["errors"] == []
    assert out["profile_deleted"] is True and out["identity_deleted"] is True
    upd = [(q, p) for q, p in alertas if q.startswith("UPDATE system_alerts SET resolved_at")]
    assert upd == [("UPDATE system_alerts SET resolved_at = NOW() WHERE alert_key = %s AND resolved_at IS NULL",
                    (f"account_delete_partial:{UID}",))]
    assert not any(q.startswith("INSERT INTO system_alerts") for q, _ in alertas)


def test_la_alerta_nueva_esta_documentada_y_la_emite_app_py():
    doc = (_BACKEND / "docs" / "system_alerts_resolution_table.md").read_text(encoding="utf-8")
    assert re.search(r"^\| `account_delete_partial:<user_id>` \|", doc, re.MULTILINE), (
        "cada alert_key nuevo necesita su fila en la tabla canónica (test_p2_audit_4)"
    )
    app_src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    assert 'alert_key = f"account_delete_partial:{user_id}"' in app_src


# ═════════════════════════════════════════════ 5. «Empezar desde cero»
@pytest.fixture
def reset_pool(monkeypatch):
    import db_profiles

    pool = _FakePool()
    redis: list = []
    monkeypatch.setattr(db_profiles, "connection_pool", pool)
    monkeypatch.setattr(db_profiles, "_purge_visual_diary_storage", lambda uid: 0)
    monkeypatch.setattr(db_profiles, "_purge_user_redis_caches", lambda *a, **k: redis.append(a) or 0, raising=False)
    pool.redis = redis
    return pool


def test_empezar_desde_cero_borra_la_memoria_del_coach_y_conserva_el_chat(reset_pool):
    import db_profiles

    assert db_profiles.reset_user_account_preferences(UID) is True
    sqls = reset_pool.sqls()
    for stmt in ("DELETE FROM pending_facts_queue WHERE user_id = %s",
                 "DELETE FROM user_memory_profile WHERE user_id = %s",
                 "DELETE FROM dream_consolidation_log WHERE user_id = %s",
                 "DELETE FROM user_taste_events WHERE user_id = %s",
                 "DELETE FROM user_facts WHERE user_id = %s"):
        assert stmt in sqls, f"el reset no borra: {stmt}"
    patrones = next(p[0] for q, p in reset_pool.execs() if q == "DELETE FROM app_kv_store WHERE key LIKE ANY(%s)")
    assert any(_like(pt, f"rag_{UID}_abc") for pt in patrones)
    assert any(_like(pt, f"reflection_{UID}_5_abc") for pt in patrones)
    for chat in ("agent_sessions", "agent_messages", "conversation_summaries", "summary_archive", "checkpoint"):
        assert not any(chat in q for q in sqls), f"el reset CONSERVA el chat y tocó {chat}"
    kinds = [e[0] for e in reset_pool.log]
    assert kinds[0] == "begin" and kinds[-1] == "commit" and kinds.count("begin") == 1, "una sola transacción"
    assert reset_pool.redis, "la copia en Redis de las cachés se purga tras el COMMIT"


def test_si_el_reset_hace_rollback_devuelve_false_y_no_purga_nada_mas(reset_pool):
    import db_profiles

    reset_pool.fail_on = ("DELETE FROM user_memory_profile",)
    assert db_profiles.reset_user_account_preferences(UID) is False
    assert reset_pool.log[-1] == ("rollback",)
    assert reset_pool.redis == []


def test_endpoint_reset_con_rollback_es_500_con_codigo(app_module, monkeypatch):
    import db_profiles
    from fastapi import HTTPException

    monkeypatch.setattr(db_profiles, "reset_user_account_preferences", lambda uid: False)
    with pytest.raises(HTTPException) as exc:
        app_module.api_reset_user_preferences(verified_user_id=UID)
    assert exc.value.status_code == 500 and exc.value.detail["code"] == "reset_failed"

    monkeypatch.setattr(db_profiles, "reset_user_account_preferences", lambda uid: True)
    assert app_module.api_reset_user_preferences(verified_user_id=UID)["success"] is True
