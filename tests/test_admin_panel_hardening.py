# backend/tests/test_admin_panel_hardening.py
"""Tests de regresión para el endurecimiento y corrección de bugs del panel de administración."""
from __future__ import annotations

import datetime as dt
import pytest
from fastapi import HTTPException

import admin_acceso
import admin_avisos
import admin_cuentas
import admin_cuentas_lista as acl
import admin_metricas as am
import admin_prueba_detalle as apd
import ajustes_cuenta
import ios_free
from routers import admin as ra


def test_csv_formula_injection_blocks_leading_whitespace_and_newlines():
    for peligroso in ("=SUM(A1)", " =HYPERLINK(\"http://x\")", "\t+1+1", "\n-2+3", "  \r@cmd"):
        assert acl._celda(peligroso).startswith("'"), f"falló en {peligroso!r}"
    assert acl._celda("Nombre normal") == "Nombre normal"
    assert acl._celda("  hola") == "  hola"

    fila = ajustes_cuenta.fila_csv([{"clave": ajustes_cuenta.REGISTRO[0].clave, "estado": "valor", "valor": " =1+1"}])
    assert fila[0].startswith("'")


def test_admin_avisos_ids_distingue_plan_y_usuario():
    uid = "11111111-1111-1111-1111-111111111111"
    pid = "22222222-2222-2222-2222-222222222222"

    # Clave con plan primero: no debe confundir pid con uid
    assert admin_avisos._ids_de({"alert_key": f"dead_lettered_chunk:{pid}:7"}) == (None, pid)
    assert admin_avisos._ids_de({"alert_key": f"plan_stranded_partial:{pid}", "uid": uid}) == (uid, pid)

    # Clave con usuario primero y plan segundo: extrae ambos sin metadata
    assert admin_avisos._ids_de({"alert_key": f"plan_quality_degraded:{uid}:{pid}"}) == (uid, pid)


def test_admin_avisos_clasificar_y_rango_fechas_naive_aware():
    ahora = dt.datetime(2026, 11, 19, 15, 0, tzinfo=dt.timezone.utc)
    viejo_naive = dt.datetime(2026, 10, 1, 12, 0)  # naive
    tipo_sin_espera = next(iter(admin_avisos._TIPOS_SIN_ESPERA))
    grupo, nivel, estado, _ = admin_avisos._clasificar(
        tipo_sin_espera, "aviso", None, None, None, cuando=viejo_naive, ahora=ahora
    )
    assert (grupo, nivel, estado) == ("historial", "info", "abierto")

    # _rango_fechas con hoy no debe producir "el hoy"
    hoy_mismo = dt.datetime(2026, 11, 19, 14, 0, tzinfo=dt.timezone.utc)
    rango = admin_avisos._rango_fechas(hoy_mismo, hoy_mismo, ahora)
    assert not rango.startswith("el hoy")


def test_admin_cuentas_validaciones_y_scope_web(monkeypatch):
    monkeypatch.setattr(admin_cuentas.rc, "activo", lambda: True)
    monkeypatch.setattr(admin_cuentas, "_regalable", lambda uid: {"id": uid, "plan_tier": "gratis"})
    monkeypatch.setattr(admin_cuentas, "ficha", lambda uid: {
        "plan_efectivo": "gratis", "cortesia": None,
        "creditos": {"usados": 0, "regalo": 0, "tope": 3},
        "coach": {"usados": 0, "regalo": 0, "tope": 10},
    })

    # Rechaza bool como cantidad
    with pytest.raises(admin_cuentas.ErrorRegalo) as exc_bool:
        admin_cuentas.regalar_creditos("admin-1", "u1", "generacion", "sumar", True, "mes", "motivo valido")
    assert exc_bool.value.status == 422

    # Rechaza fecha con basura al final
    with pytest.raises(admin_cuentas.ErrorRegalo) as exc_fecha:
        admin_cuentas.dar_cortesia("admin-1", "u1", "plus", "2026-12-01xyz", "motivo valido")
    assert exc_fecha.value.status == 422

    # _estado soporta ends_at naive sin TypeError
    ahora = dt.datetime(2026, 11, 19, 15, 0, tzinfo=dt.timezone.utc)
    assert admin_cuentas._estado({"ends_at": dt.datetime(2026, 11, 1, 0, 0)}, ahora) == "caducado"

    # _fallo_al_guardar distingue UniqueViolation de cortesía vs otras acciones
    monkeypatch.setattr(admin_cuentas, "_anotar_fallo", lambda *a, **kw: None)
    UniqueViolation = type("UniqueViolation", (Exception,), {})
    err_c = admin_cuentas._fallo_al_guardar("a", "dar_cortesia", "u", "g", UniqueViolation())
    err_r = admin_cuentas._fallo_al_guardar("a", "regalar_creditos", "u", "g", UniqueViolation())
    assert "cortesía" in err_c.detalle
    assert "cortesía" not in err_r.detalle


def test_admin_cuentas_ficha_aisla_scope_web(monkeypatch):
    scopes_vistos = []
    monkeypatch.setattr(admin_cuentas, "_perfil", lambda uid: {
        "id": uid, "email": "a@b.com", "full_name": "Ana", "created_at": None,
        "plan_tier": "gratis", "subscription_status": None, "subscription_end_date": None, "tiene_paypal": False,
    })
    monkeypatch.setattr(admin_cuentas.rc, "regalos_vigentes", lambda uid: [])
    monkeypatch.setattr(admin_cuentas, "execute_sql_query", lambda *a, **kw: [])
    monkeypatch.setattr(admin_cuentas, "get_monthly_api_usage", lambda uid, kind="plan": (scopes_vistos.append(ios_free.scope()) or 2))
    monkeypatch.setattr(ios_free, "enabled", lambda: False)

    with ios_free.using("ios_free"):
        admin_cuentas.ficha("11111111-1111-1111-1111-111111111111")
    assert scopes_vistos == ["web", "web"]


def test_router_avisos_fail_closed_y_hecho_amplia_ficha(monkeypatch):
    orden = []
    monkeypatch.setattr(ra, "_anotar_vista", lambda *a: orden.append("rastro"))
    monkeypatch.setattr(admin_avisos, "marcar_revisados", lambda h: (orden.append("write") or 1))
    res = ra.api_admin_avisos_revisado(ra._Revisados(huellas=["abcdef0123456789"]), admin_id="adm")
    assert res == {"ok": True, "cerradas": 1}
    assert orden == ["rastro", "write"]

    # Si falla la escritura en BD, devuelve 503 en vez de 500
    monkeypatch.setattr(admin_avisos, "marcar_revisados", lambda h: (_ for _ in ()).throw(RuntimeError("db down")))
    with pytest.raises(HTTPException) as exc:
        ra.api_admin_avisos_revisado(ra._Revisados(huellas=["abcdef0123456789"]), admin_id="adm")
    assert exc.value.status_code == 503

    # _hecho conserva la ficha ampliada cuando el interruptor de pruebas está activo
    monkeypatch.setattr(ra.cp, "activo", lambda: True)
    monkeypatch.setattr(ra.ac, "ficha", lambda uid: {"user_id": uid, "email": "a@b.com"})
    monkeypatch.setattr(ra.acl, "ampliar_ficha", lambda c: {**c, "actividad": {"comidas_total": 3}})
    hecho = ra._hecho("u1", {"grant_id": "g1"})
    assert hecho["cuenta"]["actividad"] == {"comidas_total": 3}


def test_metricas_analizador_resiliente_y_sql_sesiones_filtra_system_title(monkeypatch):
    monkeypatch.setattr(am, "_todos", lambda *a, **kw: [
        {"ran_at": None, "model": "m1", "notes": "corrida", "metrics": '{"valida": true, "kcal": {"mediana": 0.1}}'},
        {"ran_at": "2026-11-19 10:00:00", "model": "m2", "notes": None, "metrics": "basura-json"},
    ])
    bloque = am.bloque_analizador(am._Ctx(dias=7, fuera=[]))
    assert len(bloque["filas"]) == 2
    assert bloque["filas"][0][0] == "—"
    assert "no válida" in bloque["filas"][1][1]

    assert "NOT m.content LIKE '[SYSTEM_TITLE]%%'" in apd._SQL_SESIONES
    assert "NOT m.content LIKE '[SYSTEM_TITLE]%%'" in apd._SQL_MENSAJES


def test_registrar_acceso_serializa_uuid_sin_romper(monkeypatch):
    import uuid
    escritos = []
    monkeypatch.setattr(admin_acceso, "execute_sql_write", lambda q, p: escritos.append(p))
    admin_acceso.registrar_acceso("adm", "test", "u1", {"id": uuid.uuid4(), "at": dt.datetime(2026, 11, 19, tzinfo=dt.timezone.utc)})
    assert len(escritos) == 1


def test_avisos_catalogo_y_gramatica_fechas_hoy():
    for clave in ("model_effective_changed", "chunk_lesson_synth_ratio_high", "pipeline_emergency_fallback_p1_5"):
        titulo, detalle = admin_avisos._textos(clave)
        assert not titulo.startswith("Aviso técnico:")
        assert detalle != "Sin descripción todavía."
        assert clave in admin_avisos._ACCION

    ahora = dt.datetime(2026, 11, 19, 17, 5, tzinfo=dt.timezone.utc)
    pasado = dt.datetime(2026, 10, 19, 14, 0, tzinfo=dt.timezone.utc)
    momento = admin_avisos._momento(pasado, ahora, ahora)
    assert " al hoy" not in momento
    assert "a hoy," in momento

    rango = admin_avisos._rango_fechas(pasado, ahora, ahora)
    assert "y el hoy" not in rango
    assert "y hoy," in rango


def test_auto_resolve_model_drift_y_avisos_obsoletos(monkeypatch):
    import model_visibility as mv
    import cron_tasks

    # 1) model_effective_changed se auto-resuelve cuando modelos son estables y hay alerta abierta
    writes = []
    snap = {"coach_model": "gpt-6-luna", "coach_fallback_model": "gemini-4-hydrogen", "vision_model": "gemini-4-hydrogen", "vision_provider": "gemini"}

    def fake_query(sql, params=None, fetch_one=False):
        if "app_kv_store" in sql:
            return {"value": snap}
        if "system_alerts" in sql:
            return {"open": 1}
        return None

    monkeypatch.setattr(mv, "get_effective_models_snapshot", lambda: snap)
    monkeypatch.setattr("db_core.connection_pool", object())
    monkeypatch.setattr("db_core.execute_sql_query", fake_query)
    monkeypatch.setattr("db_core.execute_sql_write", lambda sql, params=None: writes.append((sql, params)) or 1)
    res = mv.detect_and_record_model_drift_on_startup()
    assert res["drift_detected"] is False
    assert any("UPDATE system_alerts" in s and "models_stable_on_restart" in s for s, _ in writes)

    # 2) chunk_lesson_synth_ratio_high se auto-resuelve en ventana tranquila (0 sintetizados)
    cron_writes = []
    monkeypatch.setattr(cron_tasks, "execute_sql_query", lambda *a, **kw: {"synthesized_events": 0, "total_chunks": 1})
    monkeypatch.setattr(cron_tasks, "execute_sql_write", lambda sql, params=None: cron_writes.append((sql, params)))
    cron_tasks._alert_high_synthesized_lesson_ratio()
    assert any("quiet_window_zero_synthesized" in s for s, _ in cron_writes)

    # 3) auto_resolver_obsoletas cierra planes reemplazados y descartes, pero NO planes actuales
    pid_viejo = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    pid_actual = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    monkeypatch.setattr(am, "_todos", lambda *a, **kw: [
        {"alert_key": f"plan_quality_degraded:11111111-1111-1111-1111-111111111111:{pid_viejo}", "uid": None, "pid": pid_viejo, "razon": None},
        {"alert_key": f"pipeline_emergency_fallback_p1_5:11111111-1111-1111-1111-111111111111:{pid_actual}", "uid": None, "pid": pid_actual, "razon": None},
        {"alert_key": "dead_lettered_chunk:cccccccc-cccc-cccc-cccc-cccccccccccc", "uid": None, "pid": None, "razon": "execute_after_beyond_plan_window"},
    ])
    monkeypatch.setattr(admin_avisos, "_planes", lambda pids: {pid_viejo: False, pid_actual: True})
    cerradas_params = []
    monkeypatch.setattr(am, "execute_sql_write", lambda sql, params=None, returning=False: cerradas_params.append(params) or [{"ok": 1}, {"ok": 1}])
    n = admin_avisos.auto_resolver_obsoletas()
    assert n == 2
    assert f"plan_quality_degraded:11111111-1111-1111-1111-111111111111:{pid_viejo}" in cerradas_params[0][0]
    assert "dead_lettered_chunk:cccccccc-cccc-cccc-cccc-cccccccccccc" in cerradas_params[0][0]
    assert f"pipeline_emergency_fallback_p1_5:11111111-1111-1111-1111-111111111111:{pid_actual}" not in cerradas_params[0][0]
