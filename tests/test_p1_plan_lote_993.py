# backend/tests/test_p1_plan_lote_993.py
"""[P1-PLAN-LOTE-993 · 2026-11-12] «Requiere atención» lee cada aviso contra el estado ACTUAL.

Medido en producción el 12-nov: de 9 filas ámbar/rojas, los 2 «críticos» eran bloques descartados A PROPÓSITO (caían
después del final del plan; 0 intentos) y el panel decía «Falló todos sus reintentos»; los 4 atrasados y 4 de los 7
planes sin aprobar eran de cuentas sin actividad desde septiembre u octubre. Tras el cambio: 0 críticos, 7 avisos que
sí piden revisar, 9 en espera de cuentas inactivas, y lo resuelto con su fecha.
"""
import datetime as dt
import inspect
import re

import pytest

import admin_avisos as av
import admin_metricas as am
from tests.test_p1_plan_lote_576_admin_metricas import UUID, _bloque
from tests.test_p1_plan_lote_637 import ADMIN_ENV, _con

U_VIVA = "11111111-1111-1111-1111-111111111111"
U_DORMIDA = "22222222-2222-2222-2222-222222222222"
P_ACTUAL = "33333333-3333-3333-3333-333333333333"
P_VIEJO = "44444444-4444-4444-4444-444444444444"
P_BORRADO = "55555555-5555-5555-5555-555555555555"
AHORA = dt.datetime.now(dt.timezone.utc)

_ABIERTAS = "resolved_at IS NULL ORDER BY"
_RESUELTAS = "resolved_at >= now()"
_SENALES = "MAX(x.t) AS ultima"
_PLANES = "AS actual FROM public.meal_plans"
_COLA_POR_CUENTA = "AS uid, COUNT(*) AS n FROM public.plan_chunk_queue"
_COLA = "FROM public.plan_chunk_queue"


@pytest.fixture(autouse=True)
def _admins(monkeypatch):
    monkeypatch.setattr(am, "admin_ids", lambda: frozenset({ADMIN_ENV}))


def _senales(q, p):
    return [{"u": U_VIVA, "ultima": AHORA - dt.timedelta(days=1)},
            {"u": U_DORMIDA, "ultima": AHORA - dt.timedelta(days=40)},
            {"u": ADMIN_ENV, "ultima": AHORA - dt.timedelta(days=90)}]


def _alerta(clave, severidad="warning", dias=3, uid=None, pid=None, razon=None):
    return {"alert_key": clave, "severity": severidad, "triggered_at": AHORA - dt.timedelta(days=dias),
            "uid": uid, "pid": pid, "razon": razon}


def _atencion(monkeypatch, abiertas=(), resueltas=(), cola=None, por_cuenta=()):
    cola = cola or {"programados": 0, "listos": 0, "en_curso": 0, "atrasados": 0, "esperan_usuario": 0}
    f, capturadas = _con(**{
        _ABIERTAS: list(abiertas), _RESUELTAS: list(resueltas), _SENALES: _senales,
        _PLANES: [{"id": P_ACTUAL, "actual": True}, {"id": P_VIEJO, "actual": False}],
        _COLA_POR_CUENTA: list(por_cuenta), _COLA: cola})
    monkeypatch.setattr(am, "execute_sql_query", f)
    r = am.metricas(7)
    return _bloque(r, "atencion")["items"], r, capturadas


def _fila(items, titulo, grupo=None):
    return next(i for i in items if i["titulo"] == titulo and (grupo is None or i["grupo"] == grupo))


def test_una_cuenta_inactiva_espera_y_no_alarma(monkeypatch):
    items, r, _ = _atencion(monkeypatch, [
        _alerta(f"plan_quality_degraded:{U_DORMIDA}:{P_ACTUAL}", uid=U_DORMIDA, pid=P_ACTUAL)])
    f = _fila(items, "Plan entregado sin aprobar la revisión")
    assert (f["grupo"], f["nivel"]) == ("espera", "info")
    assert "No hace falta actuar: su última actividad fue el " in f["detalle"] and "se retoma si vuelve" in f["detalle"]
    assert "accion" not in f
    estado = next(t for t in _bloque(r, "resumen")["tarjetas"] if t["etiqueta"] == "Estado del sistema")
    assert estado["valor"] == "Todo bien" and "1 aviso espera a que vuelva una cuenta inactiva" in estado["ayuda"]


def test_una_cuenta_viva_pide_revision_con_accion(monkeypatch):
    items, _, _ = _atencion(monkeypatch, [
        _alerta(f"plan_quality_degraded:{U_VIVA}:{P_ACTUAL}", uid=U_VIVA, pid=P_ACTUAL)])
    f = _fila(items, "Plan entregado sin aprobar la revisión")
    assert (f["grupo"], f["nivel"]) == ("revision", "aviso") and f["accion"].startswith("Ábrelo en «Cuentas»")


@pytest.mark.parametrize("pid", [P_VIEJO, P_BORRADO])
def test_un_plan_reemplazado_o_borrado_ya_no_aplica(monkeypatch, pid):
    items, r, _ = _atencion(monkeypatch, [_alerta(f"dead_lettered_chunk:{pid}:7", "critical", uid=U_VIVA, pid=pid)])
    f = items[0]
    assert (f["grupo"], f["estado"], f["nivel"]) == ("historial", "obsoleto", "info")
    assert "Ya no aplica" in f["detalle"]
    estado = next(t for t in _bloque(r, "resumen")["tarjetas"] if t["etiqueta"] == "Estado del sistema")
    assert estado["valor"] == "Todo bien" and "nota" not in estado["ayuda"]          # lo obsoleto no se cuenta


def test_el_descarte_por_diseno_no_dice_que_fallo(monkeypatch):
    items, _, _ = _atencion(monkeypatch, [_alerta(f"dead_lettered_chunk:{P_ACTUAL}:8", "critical", uid=U_VIVA,
                                                  pid=P_ACTUAL, razon="execute_after_beyond_plan_window")])
    f = items[0]
    assert f["titulo"] == "Bloques descartados al final del plan" and f["grupo"] == "historial"
    assert "reintentos" not in f["detalle"] and "no fallaron" in f["detalle"]


def test_un_critico_de_una_cuenta_viva_es_activo(monkeypatch):
    items, _, _ = _atencion(monkeypatch, [_alerta(f"dead_lettered_chunk:{P_ACTUAL}:3", "critical", uid=U_VIVA,
                                                  pid=P_ACTUAL)])
    assert (items[0]["grupo"], items[0]["nivel"]) == ("activo", "critico")


def test_lo_del_sistema_no_espera_a_nadie(monkeypatch):
    items, _, _ = _atencion(monkeypatch, [_alerta("llm_provider_balance_exhausted:zai", "warning", uid=U_DORMIDA)])
    assert items[0]["grupo"] == "activo" and items[0]["accion"] == "Recarga la cuenta del proveedor."


def test_tus_cuentas_no_esperan_y_se_dice(monkeypatch):
    items, _, _ = _atencion(monkeypatch, [_alerta(f"dream_contradiction:{ADMIN_ENV}", uid=ADMIN_ENV)])
    assert items[0]["grupo"] == "revision" and items[0]["detalle"].endswith("Es de una de tus cuentas.")


def test_las_filas_en_espera_se_juntan_por_tipo(monkeypatch):
    items, _, _ = _atencion(monkeypatch, [
        _alerta(f"plan_quality_degraded:{U_DORMIDA}:{P_ACTUAL}", dias=50, uid=U_DORMIDA, pid=P_ACTUAL),
        _alerta(f"plan_quality_degraded:{U_DORMIDA}:x", dias=45, uid=U_DORMIDA)])
    espera = [i for i in items if i["grupo"] == "espera"]
    assert len(espera) == 1 and espera[0]["valor"] == "2" and "se retoma si vuelven" in espera[0]["detalle"]


def test_lo_resuelto_sale_con_su_fecha(monkeypatch):
    items, _, _ = _atencion(monkeypatch, resueltas=[
        {"alert_key": "chunk_pantry_snapshots_stale", "severity": "warning",
         "triggered_at": AHORA - dt.timedelta(days=5), "resolved_at": AHORA - dt.timedelta(days=2)}])
    f = items[0]
    assert (f["grupo"], f["estado"]) == ("historial", "resuelto")
    assert f["titulo"] == "Nevera desactualizada en la cola"
    assert re.match(r"^Resuelto el \d{1,2} [a-z]{3}( \d{4})? · hace 2 días$", f["momento"]), f["momento"]
    assert "resolver" not in f


def test_los_atrasados_se_parten_por_cuenta(monkeypatch):
    cola = {"programados": 0, "listos": 0, "en_curso": 0, "atrasados": 3, "esperan_usuario": 0}
    items, r, _ = _atencion(monkeypatch, cola=cola, por_cuenta=[{"uid": U_DORMIDA, "n": 2}, {"uid": U_VIVA, "n": 1}])
    assert _fila(items, "Bloques de plan atrasados")["valor"] == "1"
    assert _fila(items, "Bloques de plan atrasados")["grupo"] == "activo"
    assert _fila(items, "Bloques de plan en pausa")["valor"] == "2"
    estado = next(t for t in _bloque(r, "resumen")["tarjetas"] if t["etiqueta"] == "Estado del sistema")
    assert estado["valor"] == "1 aviso"


def test_app_review_viejo_es_historial_y_reciente_pide_revision(monkeypatch):
    items, _, _ = _atencion(monkeypatch, [_alerta(f"review_login_used:{U_DORMIDA}", dias=30, uid=U_DORMIDA)])
    assert items[0]["titulo"] == "Se usó el acceso de App Review" and items[0]["grupo"] == "historial"
    items, _, _ = _atencion(monkeypatch, [_alerta(f"review_login_used:{U_DORMIDA}", dias=1, uid=U_DORMIDA)])
    assert items[0]["grupo"] == "revision" and "MEALFIT_REVIEW_LOGIN_CODE" in items[0]["accion"]


def test_sin_ids_y_con_huellas_que_casan(monkeypatch):
    clave = f"plan_quality_degraded:{U_VIVA}:{P_ACTUAL}"
    _, r, _ = _atencion(monkeypatch, [_alerta(clave, uid=U_VIVA, pid=P_ACTUAL)])
    item = _bloque(r, "atencion")["items"][0]
    assert item["resolver"] == [av._huella(clave)] and re.fullmatch(r"[0-9a-f]{16}", item["resolver"][0])
    import json
    assert not UUID.search(json.dumps(r, ensure_ascii=False))


def test_las_fechas_en_llano():
    ahora = dt.datetime(2026, 11, 12, 18, 0, tzinfo=dt.timezone.utc)              # 2:00 p. m. en RD
    assert av.fecha(dt.datetime(2026, 11, 12, 17, 26, tzinfo=dt.timezone.utc), ahora) == "hoy, 1:26 p. m."
    assert av.fecha(dt.datetime(2026, 11, 11, 13, 5, tzinfo=dt.timezone.utc), ahora) == "ayer, 9:05 a. m."
    assert av.fecha(dt.datetime(2026, 10, 27, 15, 0, tzinfo=dt.timezone.utc), ahora) == "27 oct"
    assert av.fecha(dt.datetime(2025, 10, 27, 15, 0, tzinfo=dt.timezone.utc), ahora) == "27 oct 2025"
    # medianoche en UTC es la noche ANTERIOR en RD: el día lo pone el reloj del dueño
    assert av.fecha(dt.datetime(2026, 10, 28, 2, 0, tzinfo=dt.timezone.utc), ahora) == "27 oct"
    uno = dt.datetime(2026, 10, 27, 15, 0, tzinfo=dt.timezone.utc)
    assert av._momento(uno, uno, ahora) == "Detectado el 27 oct · hace 16 días"
    assert av._momento(uno, dt.datetime(2026, 11, 10, 15, 0, tzinfo=dt.timezone.utc), ahora) == \
        "Detectados del 27 oct al 10 nov · el último hace 2 días"


def test_marcar_revisados_solo_cierra_abiertas_y_filtra_basura(monkeypatch):
    escritas = []

    def w(q, params=None, returning=False, **kw):
        escritas.append((" ".join(q.split()), params))
        return [{"ok": 1}, {"ok": 1}]
    monkeypatch.setattr(am, "execute_sql_write", w)
    assert av.marcar_revisados([]) == 0 and av.marcar_revisados(["'; DROP TABLE x;--", "zz"]) == 0
    assert escritas == []
    assert av.marcar_revisados(["ABCDEF0123456789", "abcdef0123456789", "no-vale"]) == 2
    q, p = escritas[0]
    assert "WHERE resolved_at IS NULL" in q and "sha256(convert_to(alert_key, 'UTF8'))" in q
    assert p == (["abcdef0123456789"],)


def test_la_huella_del_sql_es_la_de_python():
    # el endpoint casa la huella en SQL: left(encode(sha256(...), 'hex'), 16) debe dar lo mismo que `_huella`
    import hashlib
    assert av._huella("dream_contradiction:x") == hashlib.sha256(b"dream_contradiction:x").hexdigest()[:16]
    assert "left(encode(sha256(convert_to(alert_key, 'UTF8')), 'hex'), 16)" in inspect.getsource(av.marcar_revisados)


def test_el_endpoint_exige_la_cabecera_y_lo_anota(monkeypatch):
    from routers import admin as ra
    ruta = next(r for r in ra.router.routes if getattr(r, "path", "") == "/api/admin/avisos/revisado")
    deps = [d.dependency for d in ruta.dependencies]
    assert ra._exigir_cabecera in deps and ra._AVISOS_LIMITER in deps
    anotadas = []
    monkeypatch.setattr(ra, "_anotar_vista", lambda *a: anotadas.append(a))
    monkeypatch.setattr(av, "marcar_revisados", lambda h: 1)
    assert ra.api_admin_avisos_revisado(ra._Revisados(huellas=["abcdef0123456789"]), admin_id="a") == \
        {"ok": True, "cerradas": 1}
    assert anotadas and anotadas[0][1] == "marcar_avisos_revisados"
