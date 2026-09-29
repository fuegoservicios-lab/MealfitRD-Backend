"""[P1-PLAN-LOTE-845 · 2026-09-29] Acceso de App Review (auditoría App Store, fila 8.1, §A.2).

El revisor de Apple no puede leer el buzón de la cuenta de demostración, y Bioboros solo se entra con un código de un
solo uso que llega a ese buzón: rechazo seguro por la guideline 2.1. `review_login.py` acepta UN código fijo para UN
correo, configurados solo en el `.env` del VPS. Lo que este fichero fija:

  - inerte sin las dos variables (o con un hash que no es sha256 en hexadecimal);
  - las dos variables son SECRETOS, no knobs: el registro de knobs se publica sin auth y el hash de 6 cifras se
    revierte en menos de un segundo;
  - correo y código correctos → la sesión first-party de siempre (`set_session_cookie`), sin pasar por Neon;
  - código equivocado u otro correo → el camino de siempre hacia Neon, con la misma respuesta;
  - la cuenta tiene que existir: si no, no se crea y la respuesta es el 401 de un código inválido;
  - la comparación es en tiempo constante (`hmac.compare_digest`);
  - el límite de ritmo del endpoint va antes que todo;
  - cada uso deja rastro (`logger.warning` + `review_login_used:<uid>` en `system_alerts`).
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import logging
import re
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import review_login
import routers.auth_session as auth_session
from auth import get_neon_bearer_user_id, get_verified_user_id

_BACKEND = Path(__file__).resolve().parent.parent

CORREO = "Demo.Review@Example.com"
CODIGO = "482913"
HASH = hashlib.sha256(CODIGO.encode("utf-8")).hexdigest()
UID = "0a1b2c3d-0000-4000-8000-000000000845"


#: la de verdad, antes de que el fixture autouse la sustituya
_ALERTA_FALLOS_REAL = review_login._alerta_fallos


@pytest.fixture(autouse=True)
def contador_limpio(monkeypatch):
    """[ronda 1] Cada test empieza con la ventana de fallos vacía y sin Redis (memoria del proceso)."""
    monkeypatch.setattr(review_login, "_redis", lambda: None)
    monkeypatch.setattr(review_login, "_fallos_locales", [])
    monkeypatch.setattr(review_login, "_bloqueo_avisado_hasta", 0.0)
    alertas = []
    monkeypatch.setattr(review_login, "_alerta_fallos", lambda total, bloqueado: alertas.append((total, bloqueado)))
    return alertas


@pytest.fixture
def con_acceso(monkeypatch):
    monkeypatch.setenv(review_login.ENV_CORREO, CORREO)
    monkeypatch.setenv(review_login.ENV_HASH, HASH)


@pytest.fixture
def sin_acceso(monkeypatch):
    monkeypatch.delenv(review_login.ENV_CORREO, raising=False)
    monkeypatch.delenv(review_login.ENV_HASH, raising=False)


# ─────────────────────────── 1. configuración: inerte y secreta ───────────────────────────

def test_inerte_sin_las_dos_variables(monkeypatch, sin_acceso):
    assert review_login.configuracion() is None
    assert review_login.coincide(CORREO, CODIGO) is False
    monkeypatch.setenv(review_login.ENV_CORREO, CORREO)
    assert review_login.coincide(CORREO, CODIGO) is False, "solo el correo no basta"
    monkeypatch.delenv(review_login.ENV_CORREO)
    monkeypatch.setenv(review_login.ENV_HASH, HASH)
    assert review_login.coincide(CORREO, CODIGO) is False, "solo el hash no basta"


@pytest.mark.parametrize("malo", ["123456", HASH[:-1], HASH + "0", "z" * 64, CODIGO])
def test_un_hash_que_no_es_sha256_hex_lo_deja_apagado(monkeypatch, malo):
    monkeypatch.setenv(review_login.ENV_CORREO, CORREO)
    monkeypatch.setenv(review_login.ENV_HASH, malo)
    assert review_login.configuracion() is None
    assert review_login.coincide(CORREO, CODIGO) is False


def test_un_correo_sin_arroba_lo_deja_apagado(monkeypatch):
    monkeypatch.setenv(review_login.ENV_CORREO, "sin-arroba")
    monkeypatch.setenv(review_login.ENV_HASH, HASH)
    assert review_login.configuracion() is None


def test_el_hash_en_mayusculas_y_con_espacios_vale(monkeypatch):
    monkeypatch.setenv(review_login.ENV_CORREO, f"  {CORREO}  ")
    monkeypatch.setenv(review_login.ENV_HASH, f" {HASH.upper()} ")
    assert review_login.configuracion() == (CORREO.lower(), HASH)
    assert review_login.coincide(CORREO, CODIGO) is True


def test_las_variables_son_secretos_y_no_entran_en_el_registro_de_knobs(con_acceso):
    """`/admin/knobs` (entero) y `/health/version` (`knobs_diff`) publican el registro SIN auth. El sha256 de un código
    de 6 cifras se revierte probando el millón de combinaciones: publicar el hash es publicar el código."""
    import knobs

    assert review_login.coincide(CORREO, CODIGO) is True
    snap = knobs.get_knobs_registry_snapshot()
    assert review_login.ENV_CORREO not in snap and review_login.ENV_HASH not in snap
    src = (_BACKEND / "review_login.py").read_text(encoding="utf-8")
    codigo = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert not re.search(r"_env_(str|int|bool|float)\(", codigo), "las variables del revisor no pueden ir por knobs"
    assert "from knobs import" not in codigo and "import knobs" not in codigo


# ─────────────────────────── 2. la comparación ───────────────────────────

def test_coincide_correo_sin_mayusculas_y_codigo_exacto(con_acceso):
    assert review_login.coincide(CORREO, CODIGO)
    assert review_login.coincide(CORREO.upper(), CODIGO)
    assert review_login.coincide(f"  {CORREO.lower()} ", CODIGO)
    assert not review_login.coincide(CORREO, "482914")
    assert not review_login.coincide(CORREO, CODIGO + "0")
    assert not review_login.coincide(CORREO, "")
    assert not review_login.coincide("otra@example.com", CODIGO)
    assert not review_login.coincide("demo.review@example.co", CODIGO), "el correo casa EXACTO, no por prefijo"


def test_la_comparacion_del_hash_es_en_tiempo_constante():
    """Ancla sobre el código (no sobre la prosa): `coincide` compara con `hmac.compare_digest` y en su cuerpo no hay
    ningún `==`/`!=` que involucre el hash calculado o el esperado."""
    src = inspect.getsource(review_login.coincide)
    assert "tooltip-anchor: P1-PLAN-LOTE-845-COMPARE" in src
    arbol = ast.parse(src.lstrip())
    llamadas = [n for n in ast.walk(arbol) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "compare_digest"
                and isinstance(n.func.value, ast.Name) and n.func.value.id == "hmac"]
    assert len(llamadas) == 1, "coincide() debe comparar con hmac.compare_digest"
    args = {a.id for a in llamadas[0].args if isinstance(a, ast.Name)}
    assert args == {"digest", "esperado"}
    for comp in (n for n in ast.walk(arbol) if isinstance(n, ast.Compare)):
        nombres = {x.id for x in ast.walk(comp) if isinstance(x, ast.Name)}
        assert not ({"digest", "esperado"} & nombres), f"comparación NO constante sobre el hash: {ast.unparse(comp)}"


# ─────────────────────────── 3. el endpoint ───────────────────────────

class _FakeResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _FakeNeon:
    """httpx.AsyncClient sin red: cuenta las llamadas a Neon (el camino de siempre)."""
    resp = _FakeResp(401, {})
    llamadas: list = []

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, **k):
        _FakeNeon.llamadas.append((url, json))
        return _FakeNeon.resp


@pytest.fixture
def cliente(monkeypatch):
    """TestClient del router real con Neon, la cookie, la DB y el rastro sustituidos. El limitador se sustituye por
    un no-op (con el real, 10/60 s por IP, la batería entera chocaría con él); el test del límite lo reemplaza."""
    import neon_auth

    monkeypatch.setattr(neon_auth, "NEON_AUTH_BASE_URL", "https://fake.neonauth.test/neondb/auth")
    _FakeNeon.resp = _FakeResp(401, {"message": "invalid otp"})
    _FakeNeon.llamadas = []
    monkeypatch.setattr(auth_session.httpx, "AsyncClient", _FakeNeon)
    monkeypatch.setattr(auth_session, "session_cookies_enabled", lambda: True)
    monkeypatch.setattr(auth_session, "set_session_cookie", lambda resp, uid, iat=None: f"mf-token-{uid}")
    monkeypatch.setattr(auth_session, "derive_form_key", lambda uid: f"fk-{uid}")
    estado = {"cuentas": [], "perfiles": [], "usos": [], "ausentes": 0, "admin": False, "privilegiadas": [],
              "cuenta": {"id": UID, "email": CORREO.lower(), "name": "Demo", "banned": None}}

    def _cuenta(correo):
        estado["cuentas"].append(correo)
        return estado["cuenta"]

    def _ausente():
        estado["ausentes"] += 1

    monkeypatch.setattr(review_login, "cuenta_existente", _cuenta)
    monkeypatch.setattr(review_login, "cuenta_privilegiada", lambda uid: estado["admin"])
    monkeypatch.setattr(review_login, "anotar_cuenta_privilegiada", lambda uid: estado["privilegiadas"].append(uid))
    monkeypatch.setattr(review_login, "anotar_uso", lambda uid: estado["usos"].append(uid))
    monkeypatch.setattr(review_login, "anotar_cuenta_ausente", _ausente)
    monkeypatch.setattr(auth_session, "ensure_user_profile_exists",
                        lambda uid, email=None, name=None: estado["perfiles"].append((uid, email)))
    app = FastAPI()
    app.include_router(auth_session.router)
    app.dependency_overrides[get_neon_bearer_user_id] = lambda: None
    app.dependency_overrides[get_verified_user_id] = lambda: None
    app.dependency_overrides[auth_session._OTP_VERIFY_LIMITER] = lambda: None
    return TestClient(app), estado, app


def _verify(c, email, otp):
    return c.post("/api/auth/email-otp/verify", json={"email": email, "otp": otp})


def test_codigo_de_revision_emite_la_sesion_sin_ir_a_neon(con_acceso, cliente):
    c, estado, _ = cliente
    r = _verify(c, CORREO, CODIGO)
    assert r.status_code == 200
    body = r.json()
    # La misma forma que el OTP de siempre: es la que lee `verifyEmailOtpFirstParty`.
    assert body == {"ok": True, "user_id": UID, "email": CORREO.lower(), "token": f"mf-token-{UID}",
                    "form_key": f"fk-{UID}", "session_cookie": True}
    assert _FakeNeon.llamadas == [], "con el código de revisión no se consulta a Neon"
    assert estado["cuentas"] == [CORREO.lower()]
    assert estado["perfiles"] == [(UID, CORREO.lower())]
    assert estado["usos"] == [UID], "cada uso deja rastro"


def test_codigo_equivocado_sigue_el_camino_de_siempre(con_acceso, cliente):
    c, estado, _ = cliente
    r = _verify(c, CORREO, "482914")
    assert r.status_code == 401
    assert r.content == b"", "la misma respuesta vacía de un código inválido"
    assert _FakeNeon.llamadas == [("https://fake.neonauth.test/neondb/auth/sign-in/email-otp",
                                   {"email": CORREO, "otp": "482914"})]
    assert estado["cuentas"] == [] and estado["usos"] == []


def test_el_codigo_real_de_neon_del_correo_de_demo_sigue_valiendo(con_acceso, cliente):
    """El dueño prepara la cuenta de demostración entrando con el código de Neon: eso no puede romperse."""
    c, estado, _ = cliente
    _FakeNeon.resp = _FakeResp(200, {"user": {"id": UID, "email": CORREO.lower()}})
    r = _verify(c, CORREO, "111222")
    assert r.status_code == 200 and r.json()["user_id"] == UID
    assert len(_FakeNeon.llamadas) == 1
    assert estado["usos"] == [], "un login normal no es un uso del acceso de revisión"


def test_otro_correo_con_el_codigo_de_revision_va_a_neon(con_acceso, cliente):
    c, estado, _ = cliente
    r = _verify(c, "otra.persona@example.com", CODIGO)
    assert r.status_code == 401
    assert _FakeNeon.llamadas == [("https://fake.neonauth.test/neondb/auth/sign-in/email-otp",
                                   {"email": "otra.persona@example.com", "otp": CODIGO})]
    assert estado["cuentas"] == [] and estado["usos"] == []


def test_sin_variables_el_codigo_de_revision_es_un_codigo_mas(sin_acceso, cliente):
    c, estado, _ = cliente
    r = _verify(c, CORREO, CODIGO)
    assert r.status_code == 401
    assert len(_FakeNeon.llamadas) == 1
    assert estado["cuentas"] == [] and estado["usos"] == []


def test_la_cuenta_inexistente_no_se_crea_y_responde_como_un_codigo_invalido(con_acceso, cliente):
    c, estado, _ = cliente
    estado["cuenta"] = None
    r = _verify(c, CORREO, CODIGO)
    assert r.status_code == 401 and r.content == b""
    assert _FakeNeon.llamadas == [], "Neon crearía la cuenta con un OTP válido: aquí ni se le pregunta"
    assert estado["usos"] == [] and estado["perfiles"] == []
    assert estado["ausentes"] == 1, "el operador tiene que enterarse de que el revisor no podrá entrar"


def test_un_error_leyendo_la_cuenta_no_emite_sesion(con_acceso, cliente, monkeypatch):
    c, estado, _ = cliente

    def _revienta(correo):
        raise RuntimeError("neon caído")

    monkeypatch.setattr(review_login, "cuenta_existente", _revienta)
    r = _verify(c, CORREO, CODIGO)
    assert r.status_code == 401 and r.content == b""
    assert estado["usos"] == [] and _FakeNeon.llamadas == []


def test_sin_la_cookie_first_party_no_hay_sesion(con_acceso, cliente, monkeypatch):
    c, estado, _ = cliente
    monkeypatch.setattr(auth_session, "session_cookies_enabled", lambda: False)
    r = _verify(c, CORREO, CODIGO)
    assert r.status_code == 503
    assert estado["usos"] == [] and estado["cuentas"] == []


def test_el_limite_de_ritmo_va_antes_que_el_codigo_de_revision(con_acceso, cliente):
    """El `_OTP_VERIFY_LIMITER` del endpoint se evalúa antes del cuerpo: con el cupo agotado ni el código de revisión
    correcto pasa, ni se lee la cuenta."""
    c, estado, app = cliente

    def _agotado():
        raise HTTPException(status_code=429, detail="Demasiadas solicitudes.")

    app.dependency_overrides[auth_session._OTP_VERIFY_LIMITER] = _agotado
    r = _verify(c, CORREO, CODIGO)
    assert r.status_code == 429
    assert estado["cuentas"] == [] and estado["usos"] == []
    firma = inspect.signature(auth_session.email_otp_verify)
    assert firma.parameters["_rl"].default.dependency is auth_session._OTP_VERIFY_LIMITER


def test_el_endpoint_decide_antes_de_ir_a_neon_y_solo_con_coincide():
    src = inspect.getsource(auth_session.email_otp_verify)
    i_coincide = src.index("asyncio.to_thread(review_login.coincide, email, otp)")
    assert i_coincide < src.index("httpx.AsyncClient"), "el acceso de revisión se decide antes de llamar a Neon"
    assert i_coincide > src.index("4 <= len(otp) <= 12"), "y después de validar la forma del cuerpo, como siempre"
    sesion = inspect.getsource(auth_session._sesion_de_revision)
    assert "set_session_cookie(response, uid)" in sesion, "la MISMA sesión first-party que el OTP y Apple"
    assert "tooltip-anchor: P1-PLAN-LOTE-845-SESION" in sesion


# ─────────────────────────── 4. la cuenta y el rastro ───────────────────────────

def test_cuenta_existente_es_solo_lectura_y_no_devuelve_vetados(monkeypatch):
    vistas = []

    def _query(sql, params=None, fetch_all=False):
        vistas.append((sql, params, fetch_all))
        return filas

    monkeypatch.setattr(review_login, "execute_sql_query", _query)
    filas = [{"id": UID, "email": "demo.review@example.com", "name": "Demo", "banned": None}]
    assert review_login.cuenta_existente("  Demo.Review@Example.com ")["id"] == UID
    sql, params, fetch_all = vistas[0]
    assert sql.lstrip().upper().startswith("SELECT") and 'neon_auth."user"' in sql and "lower(email) = %s" in sql
    assert params == ("demo.review@example.com",) and fetch_all is True
    filas = [{"id": UID, "email": "demo.review@example.com", "name": "Demo", "banned": True}]
    assert review_login.cuenta_existente(CORREO) is None, "una cuenta vetada no entra ni con el código fijo"
    filas = []
    assert review_login.cuenta_existente(CORREO) is None
    src = (_BACKEND / "review_login.py").read_text(encoding="utf-8")
    assert not re.search(r"INSERT\s+INTO\s+neon_auth", src, re.I), "este camino JAMÁS da de alta una identidad"


def test_anotar_uso_deja_warning_y_alerta_con_contador(monkeypatch, caplog):
    escritas = []
    monkeypatch.setattr(review_login, "execute_sql_write", lambda sql, params=None, **k: escritas.append((sql, params)))
    with caplog.at_level(logging.WARNING, logger="review_login"):
        review_login.anotar_uso(UID)
    assert any("[P1-PLAN-LOTE-845]" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)
    sql, params = escritas[0]
    assert "INSERT INTO system_alerts" in sql and "ON CONFLICT (alert_key) DO UPDATE" in sql
    assert "'uses', COALESCE((system_alerts.metadata->>'uses')::int, 0) + 1" in sql, "cada uso suma"
    assert "resolved_at = NULL" in sql, "cada uso reabre la alerta"
    assert params[0] == f"review_login_used:{UID}" and params[1] == "review_login_used"
    meta = json.loads(params[4])
    assert meta["uses"] == 1 and meta["user_id"] == UID and meta["first_used_at"] == meta["last_used_at"]
    assert json.loads(params[5]) == [UID]


def test_el_rastro_no_lanza_si_la_db_falla(monkeypatch, caplog):
    def _revienta(*a, **k):
        raise RuntimeError("db caída")

    monkeypatch.setattr(review_login, "execute_sql_write", _revienta)
    with caplog.at_level(logging.WARNING, logger="review_login"):
        review_login.anotar_uso(UID)
        review_login.anotar_cuenta_ausente()
    niveles = [r.levelno for r in caplog.records if "[P1-PLAN-LOTE-845]" in r.getMessage()]
    assert logging.WARNING in niveles and logging.ERROR in niveles


def test_anotar_cuenta_ausente_escribe_su_alerta(monkeypatch):
    escritas = []
    monkeypatch.setattr(review_login, "execute_sql_write", lambda sql, params=None, **k: escritas.append((sql, params)))
    review_login.anotar_cuenta_ausente()
    sql, params = escritas[0]
    assert "INSERT INTO system_alerts" in sql and "'critical'" in sql
    assert params[0] == "review_login_account_missing" and params[1] == "review_login_account_missing"


def test_las_dos_alertas_estan_documentadas():
    tabla = (_BACKEND / "docs" / "system_alerts_resolution_table.md").read_text(encoding="utf-8")
    assert "| `review_login_used:<user_id>` |" in tabla
    assert "| `review_login_account_missing` |" in tabla
    escaner = (_BACKEND / "tests" / "test_p2_audit_4_alert_keys_documented.py").read_text(encoding="utf-8")
    assert '_BACKEND / "review_login.py"' in escaner, "el escáner de alert_keys tiene que ver al emisor"


def test_env_example_documenta_las_variables_sin_valor():
    env = (_BACKEND / ".env.example").read_text(encoding="utf-8")
    assert re.search(r"^# MEALFIT_REVIEW_LOGIN_EMAIL=$", env, re.M)
    assert re.search(r"^# MEALFIT_REVIEW_LOGIN_CODE_SHA256=$", env, re.M)


# ─────────────────────────── 5. [ronda 1] cuenta admin, fallos y bloqueo ───────────────────────────

def test_una_cuenta_admin_no_recibe_sesion(con_acceso, cliente):
    c, estado, _ = cliente
    estado["admin"] = True
    r = _verify(c, CORREO, CODIGO)
    assert r.status_code == 401 and r.content == b"", "la misma respuesta que un código inválido"
    assert estado["privilegiadas"] == [UID] and estado["usos"] == [] and estado["perfiles"] == []
    assert _FakeNeon.llamadas == []
    sesion = inspect.getsource(auth_session._sesion_de_revision)
    assert sesion.index("review_login.cuenta_privilegiada") < sesion.index("set_session_cookie(response, uid)")


def test_cuenta_privilegiada_mira_la_lista_aunque_el_panel_este_apagado(monkeypatch):
    import admin_acceso
    monkeypatch.setenv("MEALFIT_ADMIN_USER_IDS", f"otro-id,{UID.upper()}")
    monkeypatch.setenv("MEALFIT_ADMIN_PANEL", "false")
    assert admin_acceso.es_admin(UID) is False
    assert review_login.cuenta_privilegiada(UID) is True
    monkeypatch.setenv("MEALFIT_ADMIN_USER_IDS", "otro-id")
    assert review_login.cuenta_privilegiada(UID) is False

    def _revienta():
        raise RuntimeError("x")

    monkeypatch.setattr(admin_acceso, "admin_ids", _revienta)
    assert review_login.cuenta_privilegiada(UID) is True, "sin poder comprobarlo, fail-secure"


def test_anotar_cuenta_privilegiada_escribe_su_alerta(monkeypatch):
    escritas = []
    monkeypatch.setattr(review_login, "execute_sql_write", lambda sql, params=None, **k: escritas.append((sql, params)))
    review_login.anotar_cuenta_privilegiada(UID)
    sql, params = escritas[0]
    assert "INSERT INTO system_alerts" in sql and "'critical'" in sql
    assert params[0] == params[1] == "review_login_account_privileged"


def test_la_alerta_de_fallos_escribe_severidad_y_bloqueo(monkeypatch):
    escritas = []
    monkeypatch.setattr(review_login, "execute_sql_write", lambda sql, params=None, **k: escritas.append((sql, params)))
    _ALERTA_FALLOS_REAL(30, bloqueado=True)
    sql, params = escritas[0]
    assert params[0] == params[1] == "review_login_failed_burst" and params[2] == "critical"
    assert json.loads(params[5]) == {"failures_last_hour": 30, "locked": True}


def test_el_sha256_se_calcula_antes_de_mirar_el_correo():
    src = inspect.getsource(review_login.coincide)
    assert src.index("hashlib.sha256(") < src.index("configuracion()") < src.index("!= correo")


def test_los_fallos_con_el_correo_de_demo_se_cuentan_y_los_demas_no(con_acceso, contador_limpio):
    assert review_login.coincide(CORREO, "000000") is False
    assert review_login.coincide("otra@example.com", "000000") is False, "otro correo no cuenta"
    assert review_login.coincide(CORREO, CODIGO) is True, "un acierto no cuenta"
    assert review_login.fallos_en_ventana() == 1


def test_al_pasar_de_10_fallos_salta_la_alerta_una_vez(con_acceso, contador_limpio):
    for _ in range(review_login.UMBRAL_ALERTA):
        review_login.coincide(CORREO, "000000")
    assert contador_limpio == []
    review_login.coincide(CORREO, "000000")
    review_login.coincide(CORREO, "000000")
    assert contador_limpio == [(review_login.UMBRAL_ALERTA + 1, False)], "una vez al cruzar, no en cada fallo"


def test_a_los_30_la_rama_queda_inerte_y_la_respuesta_no_cambia(con_acceso, cliente, contador_limpio):
    c, estado, _ = cliente
    respuestas = set()
    for _ in range(review_login.UMBRAL_BLOQUEO):
        r = _verify(c, CORREO, "000000")
        respuestas.add((r.status_code, r.content))
    assert contador_limpio[-1] == (review_login.UMBRAL_BLOQUEO, True)
    llamadas_antes = len(_FakeNeon.llamadas)
    r = _verify(c, CORREO, CODIGO)
    respuestas.add((r.status_code, r.content))
    assert respuestas == {(401, b"")}, "bloqueado, el código correcto responde EXACTAMENTE como uno equivocado"
    assert len(_FakeNeon.llamadas) == llamadas_antes + 1, "y sigue el camino normal hacia Neon"
    assert estado["cuentas"] == [] and estado["usos"] == []
    assert review_login.fallos_en_ventana() == review_login.UMBRAL_BLOQUEO, "bloqueado, el intento no cuenta"


def test_el_bloqueo_se_levanta_al_vaciarse_la_ventana(con_acceso, contador_limpio, monkeypatch):
    ahora = [1_000_000.0]
    monkeypatch.setattr(review_login.time, "time", lambda: ahora[0])
    for _ in range(review_login.UMBRAL_BLOQUEO):
        review_login.coincide(CORREO, "000000")
    assert review_login.coincide(CORREO, CODIGO) is False
    ahora[0] += review_login.VENTANA_S + 1
    assert review_login.coincide(CORREO, CODIGO) is True


class _FakeRedis:
    def __init__(self):
        self.z = {}

    def zremrangebyscore(self, k, lo, hi):
        self.z = {m: v for m, v in self.z.items() if not (lo <= v <= hi)}

    def zcard(self, k):
        return len(self.z)

    def pipeline(self):
        return _FakePipe(self)


class _FakePipe:
    def __init__(self, r):
        self.r, self.ops = r, []

    def zadd(self, k, mapping):
        self.ops.append(lambda: self.r.z.update(mapping) or len(mapping))

    def zremrangebyscore(self, k, lo, hi):
        self.ops.append(lambda: self.r.zremrangebyscore(k, lo, hi))

    def zcard(self, k):
        self.ops.append(lambda: self.r.zcard(k))

    def expire(self, k, secs):
        self.ops.append(lambda: True)

    def execute(self):
        return [op() for op in self.ops]


def test_con_redis_el_contador_es_compartido(con_acceso, contador_limpio, monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(review_login, "_redis", lambda: fake)
    for _ in range(3):
        review_login.coincide(CORREO, "000000")
    assert fake.zcard("x") == 3 and review_login._fallos_locales == [], "con Redis no se usa la memoria local"
    assert review_login.fallos_en_ventana() == 3


def test_las_dos_alertas_de_la_ronda_1_estan_documentadas():
    tabla = (_BACKEND / "docs" / "system_alerts_resolution_table.md").read_text(encoding="utf-8")
    assert "| `review_login_account_privileged` |" in tabla and "| `review_login_failed_burst` |" in tabla
    fila = next(linea for linea in tabla.splitlines() if linea.startswith("| `review_login_account_missing` |"))
    assert "si no fue Apple, rota el hash antes de recrear la cuenta" in fila
