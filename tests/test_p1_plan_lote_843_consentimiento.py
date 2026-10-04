# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-843 · 2026-09-29] El permiso para la IA de terceros: el SSOT (`consentimientos.py`), sus endpoints
(`/api/consents`), el 428, el invitado, la adopción, la migración, la exportación y el CORS.

Apple 5.1.2(i) + RGPD 9(2)(a)/49(1)(a): sin permiso explícito y registrado no sale nada hacia la IA. Todo con la base
FALSA: el `.env` de desarrollo apunta a producción y ningún test escribe en ella.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

import consentimientos as cs
from auth import get_verified_user_id

_BACKEND = Path(__file__).resolve().parent.parent
_MIG = "p1_plan_lote_843_user_consents_2026_09_29.sql"
UID = "11111111-2222-4333-8444-555555555555"
SESION = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
T0 = datetime(2026, 9, 29, 10, 0, tzinfo=timezone.utc)


def _norm(sql) -> str:
    return " ".join(str(sql).split())


def _fila(version=cs.AI_CONSENT_VERSION, at=True, cn=True, revocado=False, analytics=None):
    return {"ai_consent_version": version, "ai_consent_at": T0 if at else None,
            "ai_cn_transfer_at": T0 if cn else None, "ai_consent_revoked_at": T0 if revocado else None,
            "analytics_consent": analytics}


@pytest.fixture
def modo(monkeypatch):
    def _poner(valor):
        monkeypatch.setenv("MEALFIT_AI_CONSENT_GATE", valor)
    return _poner


@pytest.fixture
def lectura(monkeypatch):
    """`execute_sql_query` de consentimientos devuelve `estado["fila"]` (o lanza si es una excepción)."""
    estado = {"fila": None, "lecturas": 0}

    def _q(query, params=None, fetch_one=False, fetch_all=False):
        estado["lecturas"] += 1
        if isinstance(estado["fila"], Exception):
            raise estado["fila"]
        return estado["fila"]

    monkeypatch.setattr(cs, "execute_sql_query", _q)
    return estado


# ─────────────────────────────────────────── base falsa (pool / conexión / cursor), como la del lote 716
class _Cursor:
    def __init__(self, pool):
        self.pool = pool
        self.rowcount = 0
        self._rows: list = []

    def execute(self, sql, params=None):
        q = _norm(sql)
        self.pool.log.append(("exec", q, params))
        if any(f in q for f in self.pool.fail_on):
            raise RuntimeError("fallo simulado")
        rows, rowcount = self.pool.respond(q, params)
        self._rows, self.rowcount = list(rows), rowcount

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Tx:
    def __init__(self, pool):
        self.pool = pool

    def __enter__(self):
        self.pool.log.append(("begin",))
        return self

    def __exit__(self, exc_type, *a):
        self.pool.log.append(("rollback",) if exc_type else ("commit",))
        return False


class _Conn:
    def __init__(self, pool):
        self.pool = pool

    def transaction(self):
        return _Tx(self.pool)

    def cursor(self, row_factory=None):
        return _Cursor(self.pool)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Pool:
    """`answers`: lista de (fragmento_sql, filas, rowcount); gana el primero que casa."""

    def __init__(self, answers=None, fail_on=()):
        self.log: list = []
        self.answers = list(answers or [])
        self.fail_on = tuple(fail_on)

    def connection(self):
        return _Conn(self)

    def respond(self, q, params):
        for fragmento, filas, rowcount in self.answers:
            if fragmento in q:
                return filas, rowcount
        return [], 1

    def execs(self):
        return [(e[1], e[2]) for e in self.log if e[0] == "exec"]


@pytest.fixture
def pool(monkeypatch):
    import db_core

    def _crear(answers=None, fail_on=()):
        p = _Pool(answers, fail_on)
        monkeypatch.setattr(db_core, "connection_pool", p)
        return p
    return _crear


# ═════════════════════════════════════════════ 1. el knob y el estado
def test_el_default_de_codigo_es_block_y_queda_en_el_registro(monkeypatch):
    monkeypatch.delenv("MEALFIT_AI_CONSENT_GATE", raising=False)
    assert cs.modo() == "block"
    monkeypatch.setenv("MEALFIT_AI_CONSENT_GATE", "quizas")
    assert cs.modo() == "block", "un valor raro cae al default seguro"
    from knobs import get_knobs_registry_snapshot
    assert get_knobs_registry_snapshot()["MEALFIT_AI_CONSENT_GATE"]["default"] == "block"
    for valor in ("off", "log", "block"):
        monkeypatch.setenv("MEALFIT_AI_CONSENT_GATE", valor)
        assert cs.modo() == valor


def test_la_suite_corre_con_el_gate_apagado_por_conftest():
    src = (_BACKEND / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert '_os_conftest.environ.setdefault("MEALFIT_AI_CONSENT_GATE", "off")' in src


@pytest.mark.parametrize("fila, esperado", [
    (_fila(), True),
    (_fila(version="ia-2026-01"), False),       # versión vieja: se vuelve a pedir
    (_fila(cn=False), False),                    # sin la transferencia a China
    (_fila(at=False), False),                    # sin el tratamiento
    (_fila(revocado=True), False),               # retirado
    (None, False),                               # sin fila
])
def test_vigente_exige_version_actual_las_dos_claves_y_sin_retirar(lectura, fila, esperado):
    lectura["fila"] = fila
    assert cs.vigente(UID) is esperado


def test_vigente_ilegible_o_sin_uuid_es_no(lectura):
    lectura["fila"] = RuntimeError("base caída")
    assert cs.vigente(UID) is False
    assert cs.vigente("guest") is False and cs.vigente(None) is False


def test_estado_de_una_cuenta_sin_permiso(lectura):
    lectura["fila"] = {"ai_consent_version": None, "ai_consent_at": None, "ai_cn_transfer_at": None,
                       "ai_consent_revoked_at": None, "analytics_consent": None}
    assert cs.estado(UID) == {
        "version": "ia-2026-10-voz", "vigente": False, "ai_consent_version": None, "ai_consent_at": None,
        "ai_cn_transfer_at": None, "ai_consent_revoked_at": None, "analytics": None,
    }


def test_estado_de_fila_vale_para_el_perfil_ya_leido():
    """`GET /api/profile` lo calcula de la fila de `get_user_profile`, que trae las fechas ya en texto ISO."""
    perfil = {"ai_consent_version": "ia-2026-10-voz", "ai_consent_at": "2026-09-29T10:00:00+00:00",
              "ai_cn_transfer_at": "2026-09-29T10:00:00+00:00", "ai_consent_revoked_at": None,
              "analytics_consent": False, "health_profile": {"x": 1}}
    e = cs.estado_de_fila(perfil)
    assert e["vigente"] is True and e["ai_consent_at"] == "2026-09-29T10:00:00+00:00" and e["analytics"] is False
    assert cs.estado_de_fila({**perfil, "ai_consent_at": T0})["ai_consent_at"] == T0.isoformat()


# ═════════════════════════════════════════════ 2. validar lo que manda el cliente
_OK = {"version": "ia-2026-10-voz", "ai_processing": True, "ai_transfer_cn": True}


def test_validar_la_concesion_completa():
    p = cs.validar_peticion({**_OK, "analytics": False, "locale": "es-DO", "platform": "ios", "app_build": 106,
                             "text_sha256": "A" * 64})
    assert p == {"ia": True, "analytics": False, "text_sha256": "a" * 64, "locale": "es-DO", "platform": "ios",
                 "app_build": "106"}


def test_la_analitica_sola_es_independiente():
    p = cs.validar_peticion({"version": "ia-2026-10-voz", "analytics": True})
    assert p["ia"] is False and p["analytics"] is True


@pytest.mark.parametrize("cuerpo, status, codigo", [
    ({**_OK, "version": "ia-2026-01"}, 409, "ai_consent_version_outdated"),
    ({"ai_processing": True, "ai_transfer_cn": True}, 409, "ai_consent_version_outdated"),
    ({**_OK, "ai_transfer_cn": False}, 422, "ai_consent_incomplete"),
    ({"version": "ia-2026-10-voz", "ai_processing": True}, 422, "ai_consent_incomplete"),
    ({"version": "ia-2026-10-voz"}, 422, "ai_consent_nothing_to_record"),
    ({**_OK, "ai_processing": "sí"}, 422, "ai_consent_invalid_field"),
    ({**_OK, "platform": "windows"}, 422, "ai_consent_invalid_field"),
    ({**_OK, "text_sha256": "no-es-hex"}, 422, "ai_consent_invalid_field"),
    ({**_OK, "locale": "español de verdad"}, 422, "ai_consent_invalid_field"),
    ({**_OK, "app_build": "x" * 65}, 422, "ai_consent_invalid_field"),
])
def test_validar_rechaza_con_su_codigo(cuerpo, status, codigo):
    with pytest.raises(cs.ErrorDeConsentimiento) as exc:
        cs.validar_peticion(cuerpo)
    assert exc.value.status_code == status and exc.value.error_code == codigo


# ═════════════════════════════════════════════ 3. la dependencia: 428, modos, invitado
@pytest.fixture
def app_ia(monkeypatch):
    quien = {"uid": UID}
    app = FastAPI()
    cs.instalar(app)

    @app.post("/ia")
    def _ia(_c: None = Depends(cs.requiere_consentimiento_ia)):
        return {"ok": True}

    @app.post("/suave")
    def _suave(permiso: bool = Depends(cs.hay_permiso_ia)):
        return {"permiso": permiso}

    app.dependency_overrides[get_verified_user_id] = lambda: quien["uid"]
    return TestClient(app), quien


def test_sin_permiso_428_con_el_cuerpo_plano(app_ia, modo, lectura):
    cliente, _ = app_ia
    modo("block")
    lectura["fila"] = _fila(at=False, cn=False, version=None)
    r = cliente.post("/ia")
    assert r.status_code == 428
    assert r.json() == {"error_code": "ai_consent_required", "version": "ia-2026-10-voz", "detail": cs.MENSAJE_REQUERIDO}


def test_con_permiso_200(app_ia, modo, lectura):
    cliente, _ = app_ia
    modo("block")
    lectura["fila"] = _fila()
    assert cliente.post("/ia").status_code == 200
    assert cliente.post("/suave").json() == {"permiso": True}


@pytest.mark.parametrize("fila", [_fila(version="ia-2026-01"), _fila(revocado=True)])
def test_version_vieja_o_retirado_428(app_ia, modo, lectura, fila):
    cliente, _ = app_ia
    modo("block")
    lectura["fila"] = fila
    assert cliente.post("/ia").status_code == 428
    assert cliente.post("/suave").json() == {"permiso": False}


def test_modo_log_solo_anota_la_falta_de_permiso(app_ia, modo, lectura, caplog):
    """Anotado en `info`, no en `warning`: en `log` saldría en CADA petición durante el despliegue (ronda de arreglo 1)."""
    cliente, _ = app_ia
    modo("log")
    lectura["fila"] = _fila(at=False, cn=False, version=None)
    with caplog.at_level("INFO"):
        assert cliente.post("/ia").status_code == 200
        assert cliente.post("/suave").json() == {"permiso": True}
    nuestros = [r for r in caplog.records if "P1-PLAN-LOTE-843" in r.getMessage() and "modo log" in r.getMessage()]
    assert len(nuestros) == 2 and all(r.levelname == "INFO" for r in nuestros)


def test_modo_log_respeta_la_retirada_explicita(app_ia, modo, lectura):
    cliente, _ = app_ia
    modo("log")
    lectura["fila"] = _fila(revocado=True)
    assert cliente.post("/ia").status_code == 428


def test_modo_off_no_hace_nada_ni_lee_la_base(app_ia, modo, lectura):
    cliente, _ = app_ia
    modo("off")
    lectura["fila"] = RuntimeError("no debería leerse")
    assert cliente.post("/ia").status_code == 200
    assert cliente.post("/suave").json() == {"permiso": True}
    assert lectura["lecturas"] == 0


def test_base_ilegible_503_en_block_y_pasa_en_log(app_ia, modo, lectura):
    cliente, _ = app_ia
    lectura["fila"] = RuntimeError("base caída")
    modo("block")
    r = cliente.post("/ia")
    assert r.status_code == 503 and r.json()["error_code"] == "ai_consent_unavailable"
    modo("log")
    assert cliente.post("/ia").status_code == 200


def test_invitado_sin_cabecera_428_con_ella_200(app_ia, modo, lectura):
    cliente, quien = app_ia
    modo("block")
    quien["uid"] = None
    r = cliente.post("/ia")
    assert r.status_code == 428 and r.json()["error_code"] == "ai_consent_required"
    assert cliente.post("/ia", headers={"X-Bioboros-AI-Consent": "ia-2026-10-voz"}).status_code == 200
    assert cliente.post("/ia", headers={"X-Bioboros-AI-Consent": "ia-2026-01"}).status_code == 428
    assert cliente.post("/suave").json() == {"permiso": False}
    assert cliente.post("/suave", headers={"X-Bioboros-AI-Consent": "ia-2026-10-voz"}).json() == {"permiso": True}
    assert lectura["lecturas"] == 0, "el invitado no tiene fila: se decide por la cabecera"


def test_con_cuenta_la_cabecera_no_sustituye_al_permiso(app_ia, modo, lectura):
    cliente, _ = app_ia
    modo("block")
    lectura["fila"] = _fila(at=False, cn=False, version=None)
    assert cliente.post("/ia", headers={"X-Bioboros-AI-Consent": "ia-2026-10-voz"}).status_code == 428


def test_sin_el_manejador_el_428_sigue_siendo_428():
    """Una app que no llamó a `instalar` (un test, un router suelto) da el 428 igual, con el cuerpo dentro de `detail`."""
    exc = cs.ErrorDeConsentimiento(428, "ai_consent_required", "x")
    assert exc.status_code == 428 and exc.detail == {"error_code": "ai_consent_required", "version": "ia-2026-10-voz",
                                                     "detail": "x"}


# ═════════════════════════════════════════════ 4. segundo plano: permite_ia
@pytest.mark.parametrize("m, fila, esperado", [
    ("block", _fila(), True), ("block", None, False), ("block", _fila(revocado=True), False),
    ("log", None, True), ("log", _fila(revocado=True), False),
    ("off", _fila(revocado=True), True),
])
def test_permite_ia_por_modo(modo, lectura, m, fila, esperado):
    modo(m)
    lectura["fila"] = fila
    assert cs.permite_ia(UID, "test") is esperado


def test_permite_ia_es_fail_closed_en_block(modo, lectura):
    lectura["fila"] = RuntimeError("base caída")
    modo("block")
    assert cs.permite_ia(UID, "test") is False
    assert cs.permite_ia("no-es-uuid", "test") is False
    modo("log")
    assert cs.permite_ia(UID, "test") is True


def test_el_fragmento_sql_es_constante_y_depende_del_modo(modo):
    modo("block")
    f = cs.fragmento_sql_permiso("q1.user_id")
    assert "upc.id = q1.user_id" in f and "upc.ai_consent_version = 'ia-2026-10-voz'" in f
    assert "upc.ai_consent_revoked_at IS NULL" in f and "upc.ai_cn_transfer_at IS NOT NULL" in f
    assert "%" not in f and "{" not in f
    modo("log")
    f = cs.fragmento_sql_permiso("q1.user_id")
    assert "NOT EXISTS" in f and "upc.ai_consent_revoked_at IS NOT NULL" in f
    modo("off")
    assert cs.fragmento_sql_permiso("q1.user_id") == ""
    with pytest.raises(ValueError):
        cs.condicion_sql_permiso("q1.user_id; DROP TABLE x")


# ═════════════════════════════════════════════ 5. registrar, retirar y reanudar
@pytest.fixture(autouse=True)
def _interruptor_del_plan(monkeypatch):
    """El interruptor de P1-PLAN-MODE encendido (su default), fijado para que ningún entorno lo cambie."""
    import plan_mode
    monkeypatch.setattr(plan_mode, "PLAN_MODE_SWITCH_ENABLED", True)


def test_registrar_escribe_registro_y_estado_en_una_transaccion(pool, monkeypatch):
    p = pool([("FOR UPDATE", [{"plan_mode": "plan", "plan_mode_changed_at": None, "ai_consent_paused_at": None}], 1),
              ("RETURNING ai_consent_version", [_fila(analytics=True)], 1)])
    monkeypatch.setattr("plan_mode.resume_plan_generation", lambda uid: pytest.fail("no había pausa que deshacer"))
    out = cs.registrar(UID, ia=True, analytics=True, locale="es-DO", platform="ios", app_build="106",
                       text_sha256="a" * 64)
    assert [e[0] for e in p.log][0] == "begin" and p.log[-1] == ("commit",)
    assert "SELECT plan_mode, plan_mode_changed_at, ai_consent_paused_at FROM user_profiles" in p.execs()[0][0]
    inserts = [(q, par) for q, par in p.execs() if q.startswith("INSERT INTO public.user_consents")]
    assert [(par[2], par[4]) for _, par in inserts] == [("ai_processing", True), ("ai_transfer_cn", True),
                                                        ("analytics", True)]
    assert all(par[0] == UID and par[1] is None and par[3] == "ia-2026-10-voz" and par[5] == "a" * 64
               and par[9] == "cuenta" for _, par in inserts)
    assert all(", origen) VALUES " in q for q, _ in inserts)
    update = next(q for q, _ in p.execs() if q.startswith("UPDATE user_profiles SET"))
    for trozo in ("ai_consent_version = %s", "ai_consent_at = now()", "ai_cn_transfer_at = now()",
                  "ai_consent_revoked_at = NULL", "ai_consent_paused_at = NULL", "analytics_consent = %s"):
        assert trozo in update
    assert out["vigente"] is True and out["analytics"] is True
    assert out["plan_reanudado"] is False and out["plan_expired"] is False


def test_registrar_solo_la_analitica_no_toca_la_ia(pool):
    p = pool([("FOR UPDATE", [{"plan_mode": "plan"}], 1), ("RETURNING ai_consent_version", [_fila()], 1)])
    cs.registrar(UID, ia=False, analytics=False)
    inserts = [par for q, par in p.execs() if q.startswith("INSERT INTO public.user_consents")]
    assert [(par[2], par[4]) for par in inserts] == [("analytics", False)]
    update = next(q for q, _ in p.execs() if q.startswith("UPDATE user_profiles SET"))
    assert "ai_consent_version" not in update.split("RETURNING")[0] and "analytics_consent = %s" in update
    assert "ai_consent_paused_at" not in update, "solo conceder la IA cierra la pausa de la retirada"


def test_registrar_sin_perfil_hace_rollback(pool):
    p = pool([("FOR UPDATE", [], 0)])
    with pytest.raises(cs.PerfilInexistente):
        cs.registrar(UID, ia=True)
    assert p.log[-1] == ("rollback",)


def test_retirar_apaga_el_generador_en_la_misma_transaccion_y_luego_la_cola(pool, monkeypatch):
    p = pool([("SELECT ai_consent_revoked_at, plan_mode", [{"ai_consent_revoked_at": None, "plan_mode": "plan"}], 1),
              ("SELECT ai_consent_version", [_fila(revocado=True)], 1)])
    llamadas = []

    def _pausa(uid):
        llamadas.append(("pausa", list(p.log)))
        return {"plan_mode": "tracking", "chunks_cancelled": 2}

    monkeypatch.setattr("plan_mode.pause_plan_generation", _pausa)
    out = cs.retirar(UID, platform="web")
    assert len(llamadas) == 1
    log_al_pausar = llamadas[0][1]
    assert log_al_pausar[-1] == ("commit",), "la bandera y el modo se confirman ANTES de tocar la cola"
    sqls = [e[1] for e in log_al_pausar if e[0] == "exec"]
    assert "UPDATE user_profiles SET ai_consent_revoked_at = now() WHERE id = %s" in sqls
    assert ("UPDATE user_profiles SET plan_mode = 'tracking', plan_mode_changed_at = now(), "
            "ai_consent_paused_at = now() WHERE id = %s") in sqls, "el mismo now() en las dos columnas"
    inserts = [e[2] for e in log_al_pausar if e[0] == "exec" and e[1].startswith("INSERT INTO public.user_consents")]
    assert [(par[2], par[4], par[9]) for par in inserts] == [("ai_processing", False, "cuenta"),
                                                             ("ai_transfer_cn", False, "cuenta")]
    assert out["vigente"] is False and out["ai_consent_revoked_at"] and out["plan_pausado"] is True


def test_retirar_dos_veces_conserva_la_primera_fecha_y_no_escribe(pool, monkeypatch):
    p = pool([("SELECT ai_consent_revoked_at, plan_mode", [{"ai_consent_revoked_at": T0, "plan_mode": "tracking"}], 1),
              ("SELECT ai_consent_version", [_fila(revocado=True)], 1)])
    monkeypatch.setattr("plan_mode.pause_plan_generation", lambda uid: {"plan_mode": "tracking", "chunks_cancelled": 0})
    out = cs.retirar(UID)
    sqls = [q for q, _ in p.execs()]
    assert not any(q.startswith("UPDATE") or q.startswith("INSERT") for q in sqls)
    assert out["plan_pausado"] is False


def test_quien_ya_estaba_en_seguimiento_no_se_re_estampa_ni_cuenta_como_pausado(pool, monkeypatch):
    p = pool([("SELECT ai_consent_revoked_at, plan_mode", [{"ai_consent_revoked_at": None, "plan_mode": "tracking"}], 1),
              ("SELECT ai_consent_version", [_fila(revocado=True)], 1)])
    cancelada = []
    monkeypatch.setattr("plan_mode.pause_plan_generation",
                        lambda uid: cancelada.append(uid) or {"plan_mode": "tracking", "chunks_cancelled": 0})
    out = cs.retirar(UID)
    assert not any("plan_mode = 'tracking'" in q for q, _ in p.execs())
    assert out["plan_pausado"] is False, "plan_pausado sale del modo ANTERIOR a la retirada"
    assert cancelada == [UID], "la cola se cancela igual (restos de una carrera)"


def test_si_la_cola_falla_la_retirada_y_la_pausa_quedan(pool, monkeypatch):
    pool([("SELECT ai_consent_revoked_at, plan_mode", [{"ai_consent_revoked_at": None, "plan_mode": "plan"}], 1),
          ("SELECT ai_consent_version", [_fila(revocado=True)], 1)])

    def _boom(uid):
        raise RuntimeError("cola caída")

    monkeypatch.setattr("plan_mode.pause_plan_generation", _boom)
    out = cs.retirar(UID)
    assert out["vigente"] is False and out["plan_pausado"] is True, "el modo ya cambió en la transacción"


def test_con_el_interruptor_del_plan_apagado_la_retirada_no_toca_el_modo(pool, monkeypatch):
    import plan_mode
    monkeypatch.setattr(plan_mode, "PLAN_MODE_SWITCH_ENABLED", False)
    p = pool([("SELECT ai_consent_revoked_at, plan_mode", [{"ai_consent_revoked_at": None, "plan_mode": "plan"}], 1),
              ("SELECT ai_consent_version", [_fila(revocado=True)], 1)])
    monkeypatch.setattr("plan_mode.pause_plan_generation", lambda uid: pytest.fail("sin interruptor no hay pausa"))
    out = cs.retirar(UID)
    assert not any("plan_mode = 'tracking'" in q for q, _ in p.execs()) and out["plan_pausado"] is False


@pytest.mark.parametrize("plan_mode, cambio, pausa, reanuda", [
    ("tracking", T0, T0, True),                                  # la pausa vigente es la de la retirada
    ("tracking", T0 + timedelta(hours=2), T0, False),            # apagó a mano después: su pausa manda
    ("tracking", T0 - timedelta(days=3), None, False),           # ya estaba en seguimiento: sin marca
    ("plan", T0, T0, False),                                     # el generador ya estaba encendido
])
def test_volver_a_conceder_reanuda_solo_la_pausa_de_la_retirada(pool, monkeypatch, plan_mode, cambio, pausa, reanuda):
    pool([("FOR UPDATE", [{"plan_mode": plan_mode, "plan_mode_changed_at": cambio, "ai_consent_paused_at": pausa}], 1),
          ("RETURNING ai_consent_version", [_fila()], 1)])
    llamadas = []
    monkeypatch.setattr("plan_mode.resume_plan_generation",
                        lambda uid: llamadas.append(uid) or {"plan_mode": "plan", "chunks_revived": 3})
    out = cs.registrar(UID, ia=True)
    assert (llamadas == [UID]) is reanuda and out["plan_reanudado"] is reanuda


@pytest.mark.parametrize("vencido", [True, False])
def test_conceder_propaga_plan_expired(pool, monkeypatch, vencido):
    pool([("FOR UPDATE", [{"plan_mode": "tracking", "plan_mode_changed_at": T0, "ai_consent_paused_at": T0}], 1),
          ("RETURNING ai_consent_version", [_fila()], 1)])
    monkeypatch.setattr("plan_mode.resume_plan_generation",
                        lambda uid: {"plan_mode": "plan", "paused_days": 40 if vencido else 2, "plan_expired": vencido})
    out = cs.registrar(UID, ia=True)
    assert out["plan_reanudado"] is True and out["plan_expired"] is vencido


# ─────────── las tres secuencias de la revisión, con un perfil que cambia de verdad entre pasos
class _Perfil:
    """Una fila de `user_profiles` y un reloj: cada transacción ve su propio `now()`. Interpreta SOLO las sentencias de
    `registrar`/`retirar` (la base de verdad la validó el script de la ronda de arreglo 1 sobre tablas temporales) y la
    bandera de `plan_mode.pause/resume_plan_generation` (su CASE: solo cambia la hora si cambia el modo)."""

    def __init__(self, plan_mode="plan"):
        self.t = T0
        self.f = {"plan_mode": plan_mode, "plan_mode_changed_at": T0 - timedelta(days=10),
                  "ai_consent_paused_at": None, "ai_consent_revoked_at": None, "ai_consent_version": None,
                  "ai_consent_at": None, "ai_cn_transfer_at": None, "analytics_consent": None}
        self.reanudadas = 0

    def tic(self):
        self.t += timedelta(minutes=7)
        return self.t

    def respond(self, q, params):
        f, now = self.f, self.t
        if q.startswith("SELECT") and "FOR UPDATE" in q or q.startswith("SELECT ai_consent_version"):
            return [dict(f)], 1
        if q == "UPDATE user_profiles SET ai_consent_revoked_at = now() WHERE id = %s":
            f["ai_consent_revoked_at"] = now
        elif q.startswith("UPDATE user_profiles SET plan_mode = 'tracking'"):
            f.update(plan_mode="tracking", plan_mode_changed_at=now, ai_consent_paused_at=now)
        elif q.startswith("UPDATE user_profiles SET ai_consent_version"):
            f.update(ai_consent_version=cs.AI_CONSENT_VERSION, ai_consent_at=now, ai_cn_transfer_at=now,
                     ai_consent_revoked_at=None, ai_consent_paused_at=None)
            return [dict(f)], 1
        return [], 1

    def a_modo(self, modo):
        self.tic()
        if self.f["plan_mode"] != modo:
            self.f.update(plan_mode=modo, plan_mode_changed_at=self.t)
        return {"plan_mode": modo}


@pytest.fixture
def perfil(pool, monkeypatch):
    def _crear(plan_mode="plan"):
        pf = _Perfil(plan_mode)
        p = pool()
        p.respond = pf.respond
        monkeypatch.setattr("plan_mode.pause_plan_generation", lambda uid: pf.a_modo("tracking"))

        def _reanuda(uid):
            pf.reanudadas += 1
            return pf.a_modo("plan")

        monkeypatch.setattr("plan_mode.resume_plan_generation", _reanuda)
        return pf
    return _crear


def _paso(pf, fn, *a, **k):
    pf.tic()
    return fn(*a, **k)


def test_secuencia_retirar_encender_apagar_conceder_no_reanuda(perfil):
    import plan_mode
    pf = perfil()
    assert _paso(pf, cs.retirar, UID)["plan_pausado"] is True
    plan_mode.resume_plan_generation(UID)      # encender a mano (Configuración)
    plan_mode.pause_plan_generation(UID)       # apagar a mano
    pf.reanudadas = 0
    out = _paso(pf, cs.registrar, UID, ia=True)
    assert out["plan_reanudado"] is False and pf.reanudadas == 0 and pf.f["plan_mode"] == "tracking"
    assert pf.f["ai_consent_paused_at"] is None, "conceder limpia la marca"


def test_secuencia_retirar_encender_retirar_conceder_si_reanuda(perfil):
    import plan_mode
    pf = perfil()
    _paso(pf, cs.retirar, UID)
    plan_mode.resume_plan_generation(UID)      # encender a mano
    assert _paso(pf, cs.retirar, UID)["plan_pausado"] is True, "la segunda retirada vuelve a pausar"
    pf.reanudadas = 0
    out = _paso(pf, cs.registrar, UID, ia=True)
    assert out["plan_reanudado"] is True and pf.reanudadas == 1 and pf.f["plan_mode"] == "plan"


def test_secuencia_ya_en_seguimiento_retirar_conceder_no_reanuda(perfil):
    pf = perfil("tracking")
    antes = pf.f["plan_mode_changed_at"]
    assert _paso(pf, cs.retirar, UID)["plan_pausado"] is False
    assert pf.f["plan_mode_changed_at"] == antes and pf.f["ai_consent_paused_at"] is None
    out = _paso(pf, cs.registrar, UID, ia=True)
    assert out["plan_reanudado"] is False and pf.reanudadas == 0 and pf.f["plan_mode"] == "tracking"


# ═════════════════════════════════════════════ 6. el invitado y la adopción
def test_el_invitado_se_guarda_con_el_hash_y_nunca_con_el_id(monkeypatch):
    capturas = []
    monkeypatch.setattr(cs, "execute_sql_write", lambda q, params=None, **k: capturas.append((q, params)) or True)
    out = cs.registrar_invitado(SESION, ia=True, analytics=True, platform="web", locale="en-US")
    h = hashlib.sha256(SESION.encode("utf-8")).hexdigest()
    assert out == {"ok": True, "version": "ia-2026-10-voz", "header": "X-Bioboros-AI-Consent", "ai": True, "analytics": True}
    q, params = capturas[0]
    assert _norm(q).startswith("INSERT INTO public.user_consents (user_id, guest_hash,") and q.count("(%s") == 3
    assert SESION not in [str(x) for x in params], "el session_id crudo no se guarda"
    assert params[0] is None and params[1] == h and params[2] == "ai_processing"
    assert ", origen) VALUES " in q and params[9::10] == ("invitado", "invitado", "invitado"), "origen = invitado"
    with pytest.raises(ValueError):
        cs.registrar_invitado("corto", ia=True)


def test_adoptar_copia_con_la_fecha_original_y_sin_duplicar(pool):
    h = hashlib.sha256(SESION.encode("utf-8")).hexdigest()
    filas = [{"consent_key": "ai_processing", "version": "ia-2026-10-voz", "granted": True, "text_sha256": None,
              "locale": "es-DO", "platform": "ios", "app_build": "106", "created_at": T0},
             {"consent_key": "ai_transfer_cn", "version": "ia-2026-10-voz", "granted": True, "text_sha256": None,
              "locale": "es-DO", "platform": "ios", "app_build": "106", "created_at": T0},
             {"consent_key": "analytics", "version": "ia-2026-10-voz", "granted": False, "text_sha256": None,
              "locale": "es-DO", "platform": "ios", "app_build": "106", "created_at": T0}]
    p = pool([("WHERE guest_hash = %s", filas, 3), ("INSERT INTO public.user_consents", [], 1),
              ("UPDATE user_profiles SET ai_consent_version", [], 1), ("analytics_consent IS NULL", [], 1)])
    out = cs.adoptar_de_invitado(SESION, UID)
    assert out == {"adoptadas": 3, "estado_actualizado": True}
    execs = p.execs()
    assert execs[0][1] == (h,)
    for q, par in [e for e in execs if e[0].startswith("INSERT INTO public.user_consents")]:
        assert par[0] == UID and par[8] == T0, "la FECHA del invitado, no la de hoy"
        assert "WHERE NOT EXISTS" in q
        assert "created_at, origen) SELECT" in q and "%s::timestamptz, 'adopcion'" in q, "origen = adopcion"
    upd = next((q, par) for q, par in execs if q.startswith("UPDATE user_profiles SET ai_consent_version"))
    assert upd[1][:3] == ("ia-2026-10-voz", T0, T0)
    assert "(ai_consent_at IS NULL OR ai_consent_at < %s)" in upd[0], "no pisa una decisión propia más reciente"
    assert "(ai_consent_revoked_at IS NULL OR ai_consent_revoked_at < %s)" in upd[0]


def test_adoptar_una_version_vieja_no_da_permiso(pool):
    viejas = [{"consent_key": k, "version": "ia-2026-01", "granted": True, "created_at": T0}
              for k in ("ai_processing", "ai_transfer_cn")]
    p = pool([("WHERE guest_hash = %s", viejas, 2), ("INSERT INTO public.user_consents", [], 1)])
    out = cs.adoptar_de_invitado(SESION, UID)
    assert out["estado_actualizado"] is False
    assert not any(q.startswith("UPDATE user_profiles") for q, _ in p.execs())


def test_adoptar_nunca_revienta(pool):
    pool(fail_on=("guest_hash",))
    assert cs.adoptar_de_invitado(SESION, UID)["adoptadas"] == 0
    assert cs.adoptar_de_invitado(None, UID) == {"adoptadas": 0, "estado_actualizado": False}


def test_la_adopcion_del_plan_llama_a_adoptar_antes_de_guardar():
    src = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    i = src.index("def api_adopt_guest_plan(")
    cuerpo = src[i:src.index("\n@router.", i)]
    a = cuerpo.index('adoptar_de_invitado((data or {}).get("session_id"), verified_user_id)')
    assert cuerpo.index("account_already_has_plan") < a < cuerpo.index("_save_plan_and_track_background(")


# ═════════════════════════════════════════════ 7. los endpoints /api/consents
@pytest.fixture
def api(monkeypatch):
    from routers import consents as rc
    quien = {"uid": UID}
    app = FastAPI()
    cs.instalar(app)
    app.include_router(rc.router)
    app.dependency_overrides[rc._CONSENTS_READ_LIMITER] = lambda: quien["uid"]
    app.dependency_overrides[rc._CONSENTS_WRITE_LIMITER] = lambda: quien["uid"]
    return TestClient(app), quien


def test_get_sin_permiso_y_sin_sesion(api, lectura):
    cliente, quien = api
    lectura["fila"] = None
    r = cliente.get("/api/consents")
    assert r.status_code == 200 and r.json()["vigente"] is False and r.json()["ai_consent_at"] is None
    quien["uid"] = None
    assert cliente.get("/api/consents").status_code == 401


def test_post_concede_con_lo_validado(api, monkeypatch):
    cliente, _ = api
    vistos = {}
    monkeypatch.setattr(cs, "registrar", lambda uid, **k: vistos.update(uid=uid, **k) or {"vigente": True})
    r = cliente.post("/api/consents", json={**_OK, "platform": "android"})
    assert r.status_code == 200 and vistos["uid"] == UID and vistos["ia"] is True and vistos["platform"] == "android"


def test_post_con_version_vieja_409_plano(api):
    cliente, _ = api
    r = cliente.post("/api/consents", json={**_OK, "version": "ia-2026-01"})
    assert r.status_code == 409
    assert r.json()["error_code"] == "ai_consent_version_outdated" and r.json()["version"] == "ia-2026-10-voz"


def test_withdraw_y_guest(api, monkeypatch):
    cliente, quien = api
    monkeypatch.setattr(cs, "retirar", lambda uid, **k: {"uid": uid, **k})
    assert cliente.post("/api/consents/withdraw").json()["uid"] == UID
    vistos = {}
    monkeypatch.setattr(cs, "registrar_invitado", lambda sid, **k: vistos.update(sid=sid, **k) or {"ok": True})
    quien["uid"] = None
    r = cliente.post("/api/consents/guest", json={**_OK})
    assert r.status_code == 422 and r.json()["error_code"] == "ai_consent_invalid_session"
    assert cliente.post("/api/consents/guest", json={**_OK, "session_id": SESION}).json() == {"ok": True}
    assert vistos["sid"] == SESION and vistos["ia"] is True


def test_el_router_esta_exento_de_cuota_y_con_sus_cupos():
    src = (_BACKEND / "routers" / "consents.py").read_text(encoding="utf-8")
    assert "_CONSENTS_READ_LIMITER = RateLimiter(max_calls=30, period_seconds=60)" in src
    assert "_CONSENTS_WRITE_LIMITER = RateLimiter(max_calls=10, period_seconds=60)" in src
    assert "verify_api_quota" not in src and "log_api_usage" not in src
    app_src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    assert "app.include_router(consents_router)" in app_src and "_consentimientos.instalar(app)" in app_src


# ═════════════════════════════════════════════ 8. la migración
def test_migracion_forma_e_idempotencia():
    sql = (_BACKEND / "migrations" / _MIG).read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS public.user_consents" in sql
    assert "user_id UUID REFERENCES public.user_profiles(id) ON DELETE CASCADE" in sql
    assert "CHECK (num_nonnulls(user_id, guest_hash) = 1)" in sql
    assert "CHECK (consent_key IN ('ai_processing', 'ai_transfer_cn', 'analytics'))" in sql
    assert "CHECK (platform IS NULL OR platform IN ('ios', 'android', 'web'))" in sql
    assert "text_sha256 IS NULL OR text_sha256 ~ '^[0-9a-f]{64}$'" in sql
    assert "version TEXT NOT NULL" in sql and "granted BOOLEAN NOT NULL" in sql
    assert "created_at TIMESTAMPTZ NOT NULL DEFAULT now()" in sql
    for c in re.findall(r"ADD CONSTRAINT (\w+)", sql):
        assert f"DROP CONSTRAINT IF EXISTS {c};" in sql, f"{c} sin DROP previo: no sería idempotente"
    assert "ON public.user_consents (user_id, consent_key, created_at DESC)" in sql
    assert "ON public.user_consents (guest_hash, created_at DESC)" in sql
    assert sql.count("CREATE INDEX IF NOT EXISTS") == 2
    assert "REVOKE ALL ON public.user_consents FROM PUBLIC;" in sql
    for col in ("ai_consent_version TEXT", "ai_consent_at TIMESTAMPTZ", "ai_consent_revoked_at TIMESTAMPTZ",
                "ai_cn_transfer_at TIMESTAMPTZ", "analytics_consent BOOLEAN", "ai_consent_paused_at TIMESTAMPTZ"):
        assert f"ALTER TABLE public.user_profiles ADD COLUMN IF NOT EXISTS {col};" in sql
    # [ronda de arreglo 1] origen de cada fila, en la MISMA migración (aún sin aplicar)
    assert "origen TEXT NOT NULL DEFAULT 'cuenta'," in sql
    assert "ALTER TABLE public.user_consents ADD COLUMN IF NOT EXISTS origen TEXT NOT NULL DEFAULT 'cuenta';" in sql
    assert "CHECK (origen IN ('cuenta', 'invitado', 'adopcion'))" in sql
    assert "CHECK ((origen = 'invitado') = (guest_hash IS NOT NULL))" in sql
    assert set(cs.ORIGENES) == {"cuenta", "invitado", "adopcion"}
    assert "DO $$" in sql and "RAISE EXCEPTION" in sql
    assert "'ai_cn_transfer_at', 'analytics_consent', 'ai_consent_paused_at')) <> 6" in sql
    assert "user_consents_origen_titular_chk') THEN" in sql
    assert (_BACKEND.parent / "migrations" / _MIG).read_text(encoding="utf-8") == sql, "copia SSOT de la raíz"


def test_nadie_edita_ni_borra_el_registro():
    """Solo inserción: ni un UPDATE ni un DELETE sobre `user_consents` en el código (el borrado es el CASCADE)."""
    malos = []
    for f in list(_BACKEND.glob("*.py")) + list((_BACKEND / "routers").glob("*.py")):
        t = f.read_text(encoding="utf-8")
        if re.search(r"(UPDATE|DELETE\s+FROM)\s+(public\.)?user_consents\b", t, re.I):
            malos.append(f.name)
    assert not malos, malos


def test_las_columnas_del_permiso_no_se_escriben_por_patch_profile():
    from routers.user_data import _PROFILE_SCALAR_WHITELIST
    assert not (_PROFILE_SCALAR_WHITELIST & {"ai_consent_version", "ai_consent_at", "ai_consent_revoked_at",
                                             "ai_cn_transfer_at", "analytics_consent", "ai_consent_paused_at"})


# ═════════════════════════════════════════════ 9. exportación, CORS, perfil y paridad con el frontend
def test_la_exportacion_trae_el_registro_sin_guest_hash():
    src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    tablas = re.search(r"_ACCOUNT_EXPORT_TABLES\s*=\s*\((.*?)\n\)", src, re.DOTALL).group(1)
    assert re.search(r'\("user_consents",\s*"user_id",\s*\d+\)', tablas)
    quitadas = re.search(r"_ACCOUNT_EXPORT_STRIPPED_KEYS\s*=\s*\((.*?)\)", src).group(1)
    assert '"guest_hash"' in quitadas and quitadas.lstrip().startswith('"embedding"')


def test_el_cors_deja_pasar_la_cabecera_del_invitado():
    import app as app_module
    r = TestClient(app_module.app).options("/api/plans/analyze/stream", headers={
        "Origin": "capacitor://localhost", "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "x-bioboros-ai-consent,content-type"})
    assert r.status_code == 200
    assert "x-bioboros-ai-consent" in r.headers.get("access-control-allow-headers", "").lower()
    assert cs.ErrorDeConsentimiento in app_module.app.exception_handlers


def test_el_perfil_de_arranque_trae_el_permiso(monkeypatch):
    import db
    from routers import user_data as ud
    perfil = {"id": UID, "health_profile": {}, **{k: None for k in ("ai_consent_version", "ai_consent_at",
                                                                   "ai_cn_transfer_at", "ai_consent_revoked_at",
                                                                   "analytics_consent")}}
    monkeypatch.setattr(db, "get_user_profile", lambda uid: dict(perfil))
    out = asyncio.run(ud.api_get_profile(verified_user_id=UID))
    assert out["profile"]["ai_consent"] == cs.estado_de_fila(None)


def test_patch_profile_sin_permiso_guarda_y_no_traduce(monkeypatch):
    import db
    import plan_display_i18n
    from routers import user_data as ud
    monkeypatch.setattr(db, "execute_sql_write", lambda q, params=None, returning=False, **k: [{"id": UID}])
    monkeypatch.setattr(ud, "permite_ia", lambda uid, donde="": False)
    monkeypatch.setattr(plan_display_i18n, "schedule_plan_display_enrichment",
                        lambda *a, **k: pytest.fail("sin permiso no se traduce el plan"))
    monkeypatch.setattr(plan_display_i18n, "active_plan_missing_locale", lambda *a, **k: "plan-1")
    out = asyncio.run(ud.api_patch_profile(body=ud.ProfilePatchBody(fields={"locale": "en-US"}), verified_user_id=UID))
    assert out == {"success": True, "translation_skipped": "ai_consent_required"}


def test_la_version_del_frontend_es_la_misma():
    for base in (_BACKEND.parent,):
        f = base / "frontend" / "src" / "consent" / "version.js"
        if f.exists():
            assert f"'{cs.AI_CONSENT_VERSION}'" in f.read_text(encoding="utf-8") or \
                f'"{cs.AI_CONSENT_VERSION}"' in f.read_text(encoding="utf-8")
            return
    pytest.skip("frontend/src/consent/version.js aún no existe (Task 844)")
