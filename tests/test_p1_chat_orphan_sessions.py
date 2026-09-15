"""[P1-CHAT-ORPHAN-SESSIONS · 2026-09-14] Sesiones de chat sin dueño, semilla repetida y
checkpoints de LangGraph huérfanos.

Forense de producción (2026-09-14):
  1. Las 175 `agent_sessions` y los 303 `agent_messages` tenían `user_id` NULL: los call sites
     de `routers/plans.py` creaban la sesión con `get_or_create_session(session_id)` sin usuario
     y la semilla de «plan generado» se guardaba sin usuario.
  2. La semilla se repetía en la misma sesión (hasta 24 veces; 57 repeticiones, TODAS
     consecutivas).
  3. 54 de 55 hilos de checkpoint no tenían sesión (cuentas borradas, perfil de salud dentro):
     ninguna vía de borrado de sesión tocaba `checkpoints`/`checkpoint_blobs`/`checkpoint_writes`.

Ningún test escribe en la base ni llama al LLM: pool, cursor y guardas van falsos.
"""
from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_PLANS = _BACKEND / "routers" / "plans.py"
_CRON = _BACKEND / "cron_tasks.py"
_SCRIPT = _BACKEND / "scripts" / "purge_orphan_chat_checkpoints.py"

_CK_TABLES = ("checkpoint_writes", "checkpoint_blobs", "checkpoints")
_S1 = "aaaaaaaa-0000-4000-8000-000000000001"
_S2 = "bbbbbbbb-0000-4000-8000-000000000002"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ─────────────────────────────────────────── pool / cursor falsos (registran el orden)
class _FakeCursor:
    def __init__(self, log, rows):
        self.log = log
        self.rows = rows
        self.rowcount = 0
        self._last = []

    def execute(self, sql, params=None):
        q = " ".join(str(sql).split())
        self.log.append(("exec", q, params))
        if q.upper().startswith("SELECT"):
            self._last = list(self.rows.get("select", []))
        elif "RETURNING" in q.upper():
            self._last = list(self.rows.get("returning", []))
        else:
            self._last = []
        self.rowcount = 2 if q.startswith("DELETE FROM public.checkpoint") else 0

    def fetchall(self):
        return self._last

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeTx:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        self.log.append(("begin",))
        return self

    def __exit__(self, exc_type, *a):
        self.log.append(("rollback",) if exc_type else ("commit",))
        return False


class _FakeConn:
    def __init__(self, log, rows):
        self.log = log
        self.rows = rows

    def transaction(self):
        return _FakeTx(self.log)

    def cursor(self, row_factory=None):
        return _FakeCursor(self.log, self.rows)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakePool:
    def __init__(self, rows=None):
        self.log: list = []
        self.rows = rows or {}
        self.guard_calls: list = []

    def connection(self):
        return _FakeConn(self.log, self.rows)

    def sqls(self):
        return [e[1] for e in self.log if e[0] == "exec"]


@pytest.fixture
def fake_pool(monkeypatch):
    import db_chat
    import db_core

    pool = _FakePool(rows={"returning": [{"id": _S2}, {"id": _S1}]})
    # La guarda de tests contra producción se invoca (y aquí se registra en vez de bloquear).
    monkeypatch.setattr(db_core, "_guard_test_write_to_prod", lambda q: pool.guard_calls.append(q))
    monkeypatch.setattr(db_chat, "connection_pool", pool)
    return pool


# ─────────────────────────────────────────── 3. checkpoints: helper único
def test_delete_checkpoints_for_threads_borra_las_tres_tablas_sin_tocar_hilos_con_sesion():
    import db_chat

    log: list = []
    cur = _FakeCursor(log, {})
    counts = db_chat.delete_checkpoints_for_threads(cur, [_S2, _S1, _S1, None, ""])
    sqls = [e[1] for e in log]
    assert len(sqls) == 3
    for tbl, sql in zip(_CK_TABLES, sqls):
        assert sql.startswith(f"DELETE FROM public.{tbl} ")
        assert "ANY(%s::text[])" in sql
        assert "NOT EXISTS (SELECT 1 FROM public.agent_sessions" in sql
    assert all(e[2] == ([_S1, _S2],) for e in log), "ids deduplicados, sin vacíos, ordenados"
    assert counts == {t: 2 for t in _CK_TABLES}


def test_delete_checkpoints_for_threads_sin_hilos_no_toca_la_base():
    import db_chat

    log: list = []
    assert db_chat.delete_checkpoints_for_threads(_FakeCursor(log, {}), []) == {t: 0 for t in _CK_TABLES}
    assert log == []


@pytest.mark.parametrize("via", ["un_chat_legacy", "todos_los_chats", "un_chat_con_idor_guard"])
def test_borrar_sesiones_arrastra_sus_checkpoints_en_la_misma_transaccion(fake_pool, monkeypatch, via):
    import db_chat

    monkeypatch.setattr(db_chat, "get_session_owner", lambda sid: "user-1")
    loose_writes: list = []
    monkeypatch.setattr(db_chat, "execute_sql_write", lambda *a, **k: loose_writes.append(a) or True)

    if via == "un_chat_legacy":
        assert db_chat.delete_single_agent_session(_S1) is True
    elif via == "todos_los_chats":
        assert db_chat.delete_user_agent_sessions("user-1") is True
    else:
        assert db_chat.delete_chat_session(_S1, "user-1") == (True, "")

    kinds = [e[0] for e in fake_pool.log]
    assert kinds and kinds[0] == "begin" and kinds[-1] == "commit", (
        f"sesión y checkpoints deben ir en UNA transacción; log={fake_pool.log}"
    )
    sqls = fake_pool.sqls()
    assert "agent_sessions" in sqls[0] and sqls[0].startswith("DELETE FROM") and "RETURNING id" in sqls[0]
    ck = [e for e in fake_pool.log if e[0] == "exec" and "checkpoint" in e[1]]
    assert [e[1].split()[2] for e in ck] == [f"public.{t}" for t in _CK_TABLES]
    assert all(e[2] == ([_S1, _S2],) for e in ck), "los hilos borrados son los ids que devolvió el DELETE"
    assert fake_pool.guard_calls, "la guarda de tests contra Neon producción no se consultó"
    assert not any("agent_sessions" in str(a[0]) for a in loose_writes), (
        "el DELETE de agent_sessions no puede ir suelto (fuera de la transacción de checkpoints)"
    )
    if via == "un_chat_con_idor_guard":
        first = next(e for e in fake_pool.log if e[0] == "exec")
        assert "AND user_id = %s" in first[1] and first[2] == (_S1, "user-1")


def test_barrido_de_huerfanos_selecciona_sin_sesion_y_viejos_y_borra_en_la_misma_transaccion(fake_pool):
    import db_chat

    fake_pool.rows["select"] = [{"thread_id": _S1, "last_ts": None}]
    res = db_chat.sweep_orphan_chat_checkpoints(7, 100)
    assert res["threads"] == [_S1]
    sel = next(e for e in fake_pool.log if e[0] == "exec" and e[1].startswith("SELECT"))
    assert "NOT EXISTS" in sel[1] and "public.agent_sessions" in sel[1]
    assert "checkpoint->>'ts'" in sel[1] and "make_interval(days => %s::int)" in sel[1]
    assert sel[2] == (7, 100)
    ck = [e for e in fake_pool.log if e[0] == "exec" and e[1].startswith("DELETE FROM public.checkpoint")]
    assert len(ck) == 3 and all("NOT EXISTS" in e[1] for e in ck), "el borrado re-verifica que siga sin sesión"
    assert [e[0] for e in fake_pool.log][0] == "begin" and fake_pool.log[-1] == ("commit",)


# ─────────────────────────────────────────── 3c. cron: dentro del job ya registrado
@pytest.fixture(scope="module")
def cron():
    import cron_tasks

    return cron_tasks


@pytest.mark.parametrize("raw_days,expected_days", [("0", 1), ("7", 7), ("500", 90)])
def test_barrido_cron_clampa_dias_y_batch(cron, monkeypatch, raw_days, expected_days):
    import db

    got: dict = {}

    def fake_sweep(days, batch):
        got.update(days=days, batch=batch)
        return {"threads": [_S1], "deleted": {}}

    monkeypatch.setattr(db, "sweep_orphan_chat_checkpoints", fake_sweep)
    monkeypatch.delenv("MEALFIT_CHECKPOINT_ORPHAN_SWEEP_ENABLED", raising=False)
    monkeypatch.setenv("MEALFIT_CHECKPOINT_ORPHAN_SWEEP_DAYS", raw_days)
    monkeypatch.setenv("MEALFIT_CHECKPOINT_ORPHAN_SWEEP_BATCH", "999999")
    assert cron._sweep_orphan_chat_checkpoints() == 1
    assert got == {"days": expected_days, "batch": 1000}


def test_barrido_cron_tiene_interruptor(cron, monkeypatch):
    import db

    monkeypatch.setattr(db, "sweep_orphan_chat_checkpoints", lambda *a: pytest.fail("no debía correr"))
    monkeypatch.setenv("MEALFIT_CHECKPOINT_ORPHAN_SWEEP_ENABLED", "false")
    assert cron._sweep_orphan_chat_checkpoints() == 0


def test_ttl_borra_sesiones_con_sus_checkpoints_y_corre_el_barrido(cron, monkeypatch):
    import db

    seen: dict = {}

    def fake_delete(sql, params):
        seen["sql"] = " ".join(sql.split())
        seen["params"] = params
        return [_S1, _S2]

    monkeypatch.setattr(db, "delete_agent_sessions_with_checkpoints", fake_delete)
    monkeypatch.setattr(cron, "_sweep_orphan_chat_checkpoints", lambda: 3)
    monkeypatch.delenv("MEALFIT_CHAT_SESSION_TTL_ENABLED", raising=False)
    writes: list = []
    monkeypatch.setattr(cron, "execute_sql_write", lambda *a, **k: writes.append(a) or True)

    assert cron._sweep_stale_chat_sessions() == 2
    assert "DELETE FROM agent_sessions" in seen["sql"] and "created_at <" in seen["sql"]
    assert "RETURNING id" in seen["sql"]
    assert not any("DELETE FROM agent_sessions" in " ".join(str(a[0]).split()) for a in writes), (
        "el TTL no puede borrar sesiones fuera de la transacción que arrastra sus checkpoints"
    )
    tick = next(a for a in writes if "pipeline_metrics" in str(a[0]))
    assert '"orphan_checkpoint_threads": 3' in tick[1][1]


def test_el_barrido_vive_en_el_job_registrado(cron):
    src = _read(_CRON)
    sched = src[src.index("def register_plan_chunk_scheduler("):]
    sched = sched[: sched.index("\ndef ", 1)]
    assert "_sweep_stale_chat_sessions" in sched
    ttl = src[src.index("def _sweep_stale_chat_sessions("):]
    ttl = ttl[: ttl.index("\ndef ", 1)]
    assert "_sweep_orphan_chat_checkpoints()" in ttl


# ─────────────────────────────────────────── 3b. purga de cuenta
def test_purga_de_cuenta_borra_checkpoints_de_todas_las_fuentes_antes_que_las_sesiones(monkeypatch):
    import db_profiles

    writes: list = []

    def fake_write(query, params=None, returning=False, lock_timeout_ms=None):
        writes.append((" ".join(str(query).split()), params))
        return [] if returning else True

    uid = "11111111-2222-3333-4444-555555555555"
    monkeypatch.setattr(db_profiles, "execute_sql_write", fake_write)
    monkeypatch.setattr(db_profiles, "execute_sql_query", lambda *a, **k: [] if k.get("fetch_all") else None, raising=False)
    monkeypatch.setattr(db_profiles, "connection_pool", object(), raising=False)
    monkeypatch.setattr(db_profiles, "_purge_visual_diary_storage", lambda u: 0, raising=False)
    db_profiles.delete_account_data(uid, include_profile=False)

    qs = [q for q, _ in writes]
    i_ses = next(i for i, q in enumerate(qs) if q.startswith("DELETE FROM agent_sessions "))
    ck = [(i, q, p) for i, (q, p) in enumerate(writes) if q.startswith("DELETE FROM checkpoint")]
    assert len(ck) == 3 and all(i < i_ses for i, _, _ in ck)
    for _, q, p in ck:
        assert "FROM agent_sessions WHERE user_id = %s" in q
        assert "FROM agent_messages WHERE user_id = %s" in q, "sesión con user_id NULL pero mensajes del usuario"
        assert "SELECT %s::text" in q, "hilo cuyo id ES el user_id"
        assert p == (uid, uid, uid)


# ─────────────────────────────────────────── 1 + 2. dueño y semilla
def test_semilla_no_se_repite_si_el_ultimo_mensaje_de_usuario_es_identico(monkeypatch):
    import db_chat

    monkeypatch.setattr(db_chat, "get_or_create_session", lambda sid, user_id=None: {"id": sid, "user_id": user_id})
    monkeypatch.setattr(db_chat, "execute_sql_query", lambda *a, **k: {"content": "SEMILLA"})
    assert db_chat.should_seed_plan_messages("s1", "SEMILLA", "u1") is False
    monkeypatch.setattr(db_chat, "execute_sql_query", lambda *a, **k: {"content": "otra cosa"})
    assert db_chat.should_seed_plan_messages("s1", "SEMILLA", "u1") is True
    monkeypatch.setattr(db_chat, "execute_sql_query", lambda *a, **k: None)
    assert db_chat.should_seed_plan_messages("s1", "SEMILLA", "u1") is True


def test_semilla_mira_el_ultimo_mensaje_del_USUARIO_de_esa_sesion(monkeypatch):
    import db_chat

    seen: dict = {}
    monkeypatch.setattr(db_chat, "get_or_create_session", lambda sid, user_id=None: {"id": sid, "user_id": None})

    def fake_query(sql, params=None, **k):
        seen["sql"] = " ".join(sql.split())
        seen["params"] = params
        return None

    monkeypatch.setattr(db_chat, "execute_sql_query", fake_query)
    db_chat.should_seed_plan_messages("s1", "SEMILLA", None)
    assert "role = 'user'" in seen["sql"] and "ORDER BY created_at DESC" in seen["sql"] and "LIMIT 1" in seen["sql"]
    assert seen["params"] == ("s1",)


def test_semilla_crea_o_rellena_el_dueno_de_la_sesion(monkeypatch):
    import db_chat

    got: dict = {}

    def fake_goc(sid, user_id=None):
        got["user_id"] = user_id
        return {"id": sid, "user_id": user_id}

    monkeypatch.setattr(db_chat, "get_or_create_session", fake_goc)
    monkeypatch.setattr(db_chat, "execute_sql_query", lambda *a, **k: None)
    assert db_chat.should_seed_plan_messages("s1", "SEMILLA", "u1") is True
    assert got["user_id"] == "u1"


def test_semilla_no_se_escribe_en_la_sesion_de_otro_usuario(monkeypatch):
    import db_chat

    monkeypatch.setattr(db_chat, "get_or_create_session", lambda sid, user_id=None: {"id": sid, "user_id": "otro"})
    monkeypatch.setattr(db_chat, "execute_sql_query", lambda *a, **k: None)
    assert db_chat.should_seed_plan_messages("s1", "SEMILLA", "u1") is False
    assert db_chat.should_seed_plan_messages("s1", "SEMILLA", None) is False, "invitado sobre sesión con dueño"


def test_la_semilla_no_se_toma_como_respuesta_a_un_nudge(monkeypatch):
    """Con la sesión ya con dueño, `save_message(role='user')` dispara `handle_nudge_response`
    (clasifica el texto y marca el nudge pendiente como respondido). La semilla no la escribió
    el usuario: `process_nudge=False`."""
    import db_chat

    called: list = []
    fake_pa = types.ModuleType("proactive_agent")
    fake_pa.handle_nudge_response = lambda u, c: called.append((u, c))
    monkeypatch.setitem(sys.modules, "proactive_agent", fake_pa)
    inserts: list = []
    monkeypatch.setattr(db_chat, "_save_message_insert_with_retry", lambda *a: inserts.append(a))
    monkeypatch.setattr(db_chat, "connection_pool", object())

    db_chat.save_message("s1", "user", "hola", "u1", process_nudge=False)
    assert called == [] and inserts == [("s1", "user", "hola", "u1")]
    db_chat.save_message("s1", "user", "hola", "u1")
    assert called == [("u1", "hola")], "el default sigue procesando nudges (mensajes reales del chat)"


def _calls(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if (fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)) == name:
                yield node


def test_los_call_sites_de_plans_crean_la_sesion_con_el_usuario_verificado():
    src = _read(_PLANS)
    tree = ast.parse(src)
    calls = list(_calls(tree, "get_or_create_session"))
    assert len(calls) >= 2
    for c in calls:
        kw = {k.arg: k for k in c.keywords}
        assert "user_id" in kw, f"routers/plans.py:{c.lineno} crea la sesión sin dueño"
        seg = ast.get_source_segment(src, kw["user_id"].value) or ""
        assert "verified_user_id" in seg, f"routers/plans.py:{c.lineno}: el dueño debe salir de la auth"
        assert "data" not in seg, f"routers/plans.py:{c.lineno}: jamás el user_id del body"


def test_la_semilla_de_plans_lleva_dueno_dedupe_y_no_procesa_nudges():
    src = _read(_PLANS)
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_postprocess_pipeline_result"
    )
    body = ast.get_source_segment(src, fn) or ""
    assert "should_seed_plan_messages(" in body
    seeds = [
        c for c in _calls(fn, "save_message")
        if len(c.args) >= 3 and isinstance(c.args[1], ast.Constant) and c.args[1].value in ("user", "model")
    ]
    assert {c.args[1].value for c in seeds} == {"user", "model"}
    for c in seeds:
        kw = {k.arg: k for k in c.keywords}
        assert "user_id" in kw, f"routers/plans.py:{c.lineno}: semilla sin dueño"
        if c.args[1].value == "user":
            pn = kw.get("process_nudge")
            assert pn is not None and isinstance(pn.value, ast.Constant) and pn.value.value is False


# ─────────────────────────────────────────── 4. script one-shot
def test_script_one_shot_es_dry_run_por_defecto_y_reusa_el_ssot():
    src = _read(_SCRIPT)
    assert '"--apply", action="store_true"' in src, "--apply opt-in; sin él, dry-run"
    assert "default_transaction_read_only = on" in src
    assert "from db_chat import ORPHAN_CHECKPOINT_THREADS_SQL, delete_checkpoints_for_threads" in src
    assert "DELETE FROM" not in src, "el script no tiene SQL de borrado propio (SSOT en db_chat)"
