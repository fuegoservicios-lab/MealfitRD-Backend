# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-135 · 2026-09-20] Tres encargos del dueño con la captura del contador en su iPhone.

  1. «¿Quieres que la IA te arme el plan?» aparecía «tan seguido»: el descarte vivía en el localStorage de CADA
     dispositivo (cada binario de TestFlight, cada navegador). Ahora es una vez por SEMANA y por USUARIO (`plan_invite.py`,
     `GET/PATCH /api/user/preferences/plan-invite`): vista → 24 h a la vista → escondida hasta los 7 días; «Ahora no» →
     escondida en el acto hasta 7 días después.
  2. Hidratación: avisos a las 11/15/19 h locales si va por debajo de lo esperado, por Web Push (cron
     `run_hydration_checks`) o locales en la app nativa (`water` dentro de `GET /api/notifications/meal-reminders`).
  3. …y si pasan 48 h de avisos sin UN vaso anotado, la hidratación se apaga sola — solo a quien los avisos le LLEGAN.

Contrato del cliente: `frontend/src/__tests__/lote135.test.js`."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 9, 20, 15, 0, tzinfo=timezone.utc)


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8").replace("\r\n", "\n")


# ── 1 · la invitación semanal ──────────────────────────────────────────────────────────────────────────────────────────
def test_la_invitacion_se_ve_una_vez_por_semana():
    import plan_invite as pi
    assert pi.estado_de_la_invitacion(None, T0)["visible"] is True, "nunca vista: se muestra"
    vista = {"shown_at": T0.isoformat()}
    assert pi.estado_de_la_invitacion(vista, T0 + timedelta(hours=6))["visible"] is True, "el mismo día sigue a la vista"
    assert pi.estado_de_la_invitacion(vista, T0 + timedelta(hours=25))["visible"] is False, "al día siguiente se esconde sola"
    assert pi.estado_de_la_invitacion(vista, T0 + timedelta(days=6, hours=23))["visible"] is False
    assert pi.estado_de_la_invitacion(vista, T0 + timedelta(days=7))["visible"] is True, "a la semana vuelve"


def test_ahora_no_la_esconde_en_el_acto_y_por_una_semana():
    import plan_invite as pi
    doc = {"shown_at": T0.isoformat(), "dismissed_at": (T0 + timedelta(minutes=1)).isoformat()}
    assert pi.estado_de_la_invitacion(doc, T0 + timedelta(minutes=2))["visible"] is False
    assert pi.estado_de_la_invitacion(doc, T0 + timedelta(days=6))["visible"] is False
    assert pi.estado_de_la_invitacion(doc, T0 + timedelta(days=7, minutes=2))["visible"] is True


def test_repintarla_no_alarga_su_dia_y_la_vista_de_la_semana_siguiente_abre_otra(monkeypatch):
    import plan_invite as pi
    almacen = {}
    monkeypatch.setattr(pi, "_leer", lambda _u: dict(almacen))
    monkeypatch.setattr(pi, "_guardar", lambda _u, doc: (almacen.clear(), almacen.update(doc)))
    pi.anotar("u", "seen", T0)
    pi.anotar("u", "seen", T0 + timedelta(hours=5))
    assert almacen["shown_at"] == T0.isoformat(), "pintarla otra vez dentro de sus 24 h no mueve el reloj"
    pi.anotar("u", "seen", T0 + timedelta(days=3))
    assert almacen["shown_at"] == T0.isoformat(), "escondida: una vista fantasma no abre semana"
    pi.anotar("u", "dismiss", T0 + timedelta(days=7, hours=1))
    r = pi.anotar("u", "seen", T0 + timedelta(days=8))
    assert r["visible"] is False, "descartada hace un día: sigue escondida"
    r = pi.anotar("u", "seen", T0 + timedelta(days=14, hours=2))
    assert r["visible"] is True and almacen["shown_at"] == (T0 + timedelta(days=14, hours=2)).isoformat()


def test_los_endpoints_de_la_invitacion_no_pasan_por_el_paywall():
    src = _src("routers/preferences.py")
    assert '@router.get("/plan-invite")' in src and '@router.patch("/plan-invite")' in src
    i = src.index('@router.patch("/plan-invite")')
    assert "Depends(get_verified_user_id)" in src[i:i + 300]
    assert "Depends(verify_api_quota)" not in src
    assert "if body.action not in (\"seen\", \"dismiss\"):" in src


# ── 2 · los avisos de hidratación ──────────────────────────────────────────────────────────────────────────────────────
def test_los_puntos_de_control_y_lo_esperado_en_cada_uno():
    import hydration_reminders as hr
    assert hr.PUNTOS_DE_CONTROL == ((11, 0.25), (15, 0.55), (19, 0.80))
    assert [hr.vasos_esperados(9, f) for _h, f in hr.PUNTOS_DE_CONTROL] == [2, 4, 7]
    assert hr.vasos_esperados(6, 0.25) == 1, "nunca menos de 1: a las 11 con meta 6 también toca"
    assert hr.punto_de_control(10.9) is None
    assert hr.punto_de_control(11.5) == (11, 0.25)
    assert hr.punto_de_control(12.6) == (11, 0.25), "margen de 2 h: un tick perdido no pierde el aviso"
    assert hr.punto_de_control(13.1) is None
    assert hr.punto_de_control(19.55) == (19, 0.80) and hr.punto_de_control(21.2) is None


def test_los_textos_existen_en_los_cinco_idiomas_y_caben_en_la_pantalla_de_bloqueo():
    import hydration_reminders as hr
    locs = ("es-DO", "en-US", "pt-BR", "fr-FR", "it-IT")
    for vasos in (0, 2.5):
        cuerpos = {hr.texto_del_aviso_de_agua(loc, vasos, 9)[1] for loc in locs}
        assert len(cuerpos) == 5 and all(len(c) <= 110 for c in cuerpos)
    assert hr.texto_del_aviso_de_agua("es-DO", 2.5, 9)[1] == "Llevas 2.5 de 9 vasos hoy. ¿Un vaso de agua ahora?"
    assert hr.texto_del_aviso_de_agua("xx", 0, 9) == hr.texto_del_aviso_de_agua("es-DO", 0, 9)
    assert len({hr.texto_del_apagado(loc)[1] for loc in locs}) == 5


def test_el_horario_del_telefono_marca_lo_que_hoy_ya_va_al_dia():
    import hydration_reminders as hr
    h = hr.horario_de_avisos_de_agua("es-DO", 4, 9)
    assert [(r["hour"], r["minute"], r["met_today"]) for r in h] == [(11, 35, True), (15, 35, True), (19, 35, False)]
    assert all(r["kind"] == "water" and r["tag"] == "agua" and r["body_generic"] for r in h)
    assert hr.DIAS_EN_EL_TELEFONO == 3, "hoy + 2: lo que tarda en apagarse sola; después el teléfono calla"


# ── 3 · el apagado automático ──────────────────────────────────────────────────────────────────────────────────────────
def test_apagar_exige_48_horas_tres_avisos_y_ni_un_vaso():
    import hydration_reminders as hr
    estado = {"ignored_since": T0.isoformat(), "nudges": 3}
    assert hr.evaluar_apagado(estado, T0 + timedelta(hours=47), hubo_agua=False) is False
    assert hr.evaluar_apagado(estado, T0 + timedelta(hours=48), hubo_agua=False) is True
    assert hr.evaluar_apagado(estado, T0 + timedelta(hours=72), hubo_agua=True) is False, "un vaso basta"
    assert hr.evaluar_apagado({**estado, "nudges": 2}, T0 + timedelta(hours=72), hubo_agua=False) is False
    assert hr.evaluar_apagado({}, T0, hubo_agua=False) is False and hr.evaluar_apagado(None, T0, hubo_agua=False) is False


def _tick(monkeypatch, estado, vasos, ahora, con_push=True, agua_desde=False):
    """Un tick de `revisar_usuario` con el almacén, la DB y el push de mentira. Devuelve `(resultado, efectos)`."""
    import db
    import hydration_reminders as hr
    import utils_push
    from routers import plans as rp
    efectos = {"estado": dict(estado), "push": [], "apagada": False}
    monkeypatch.setattr(hr, "_leer_estado", lambda _u: dict(efectos["estado"]))
    monkeypatch.setattr(hr, "_guardar_estado", lambda _u, e: efectos.__setitem__("estado", dict(e)))
    monkeypatch.setattr(hr, "_hubo_agua_desde", lambda _u, _d: agua_desde)
    monkeypatch.setattr(db, "user_tz_offset_min", lambda _u: 240)
    monkeypatch.setattr(db, "get_water_intake_glasses_today", lambda _u, _f: vasos)
    monkeypatch.setattr(db, "update_water_tracker_enabled", lambda _u, v: efectos.__setitem__("apagada", v is False) or True)
    monkeypatch.setattr(rp, "_compute_water_goal", lambda _u: {"goal": 9})
    monkeypatch.setattr(utils_push, "send_push_notification",
                        lambda uid, t, b, url="/dashboard", tag=None: efectos["push"].append((b, url, tag)) or True)
    return hr.revisar_usuario("u", "es-DO", con_push, ahora=ahora), efectos


def test_un_tick_avisa_una_vez_por_punto_de_control_y_empieza_la_cuenta(monkeypatch):
    a_las_11_30 = datetime(2026, 9, 20, 15, 30, tzinfo=timezone.utc)   # 11:30 en RD
    r, ef = _tick(monkeypatch, {}, 0, a_las_11_30)
    assert r["accion"] == "aviso" and ef["push"] == [("Hoy no has anotado agua. ¿Empezamos con un vaso?", "/dashboard", "agua")]
    assert ef["estado"]["nudges"] == 1 and ef["estado"]["ignored_since"] == a_las_11_30.isoformat()
    assert ef["estado"]["horas"] == [11]
    r2, ef2 = _tick(monkeypatch, ef["estado"], 0, a_las_11_30 + timedelta(hours=1))
    assert r2["accion"] == "nada" and ef2["push"] == [], "el margen de 2 h no repite el aviso del mismo punto"


def test_quien_va_al_dia_no_recibe_nada_y_no_acumula_ignorados(monkeypatch):
    r, ef = _tick(monkeypatch, {}, 2, datetime(2026, 9, 20, 15, 30, tzinfo=timezone.utc))
    assert r["accion"] == "al_dia" and ef["push"] == [] and "ignored_since" not in ef["estado"]


def test_un_vaso_anotado_pone_la_cuenta_a_cero(monkeypatch):
    estado = {"ignored_since": (T0 - timedelta(hours=60)).isoformat(), "nudges": 5, "fecha": "2026-09-19", "horas": [19]}
    r, ef = _tick(monkeypatch, estado, 0, datetime(2026, 9, 20, 13, 30, tzinfo=timezone.utc), agua_desde=True)
    assert r["accion"] == "nada" and ef["apagada"] is False
    assert "ignored_since" not in ef["estado"] and "nudges" not in ef["estado"]


def test_a_las_48_horas_sin_un_vaso_se_apaga_sola_y_lo_dice(monkeypatch):
    estado = {"ignored_since": (T0 - timedelta(hours=49)).isoformat(), "nudges": 4}
    r, ef = _tick(monkeypatch, estado, 0, T0)
    assert r["accion"] == "apagado" and ef["apagada"] is True
    assert ef["estado"]["auto_off_at"] == T0.isoformat() and "ignored_since" not in ef["estado"]
    assert len(ef["push"]) == 1 and ef["push"][0][1] == "/dashboard/settings" and "Configuración" in ef["push"][0][0]


def test_solo_cuenta_a_quien_los_avisos_le_llegan_y_no_toca_el_tope_de_comidas():
    src = _src("hydration_reminders.py")
    i = src.index("def usuarios_alcanzables_con_agua")
    sql = src[i:i + 1400]
    assert "COALESCE(p.water_tracker_enabled, TRUE) = TRUE" in sql
    assert "FROM push_subscriptions s" in sql and "PREFIJO_CANAL_LOCAL" in sql and "make_interval(hours => %s)" in sql
    # `get_daily_nudge_count` cuenta TODAS las filas de nudge_outcomes contra el tope de 4 avisos de comida al día
    codigo = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert "INSERT INTO nudge_outcomes" not in codigo and "log_nudge_outcome(" not in codigo


def test_el_cron_esta_registrado_y_tiene_interruptor(monkeypatch):
    import hydration_reminders as hr
    app = _src("app.py")
    assert re.search(r'_add_job_jittered\(scheduler, run_hydration_checks, "cron", minute=32,\s*id="hydration_reminders", replace_existing=True\)', app)
    monkeypatch.setenv("MEALFIT_HYDRATION_REMINDERS", "false")
    monkeypatch.setattr(hr, "usuarios_alcanzables_con_agua", lambda: (_ for _ in ()).throw(AssertionError("apagado: ni lee")))
    assert hr.run_hydration_checks()["usuarios"] == 0


def test_el_telefono_recibe_el_agua_junto_a_las_comidas_y_queda_marcado_como_alcanzable(monkeypatch):
    import db
    import hydration_reminders as hr
    import meal_reminders as mr
    from routers import notifications as rn
    from routers import plans as rp
    marcas = []
    monkeypatch.setattr(db, "get_user_profile", lambda _u: {"health_profile": {"scheduleType": "standard"}, "locale": "es-DO"})
    monkeypatch.setattr(db, "user_tz_offset_min", lambda _u: 240)
    monkeypatch.setattr(db, "get_consumed_meals_today", lambda _u, date_str=None, tz_offset_mins=None: [])
    monkeypatch.setattr(db, "get_water_tracker_enabled", lambda _u: True)
    monkeypatch.setattr(db, "get_water_intake_glasses_today", lambda _u, _f: 1)
    monkeypatch.setattr(rp, "_compute_water_goal", lambda _u: {"goal": 9})
    monkeypatch.setattr(mr, "horario_de_avisos", lambda _u, locale=None, consumed_today=None: [])
    monkeypatch.setattr(hr, "marcar_canal_local", lambda u: marcas.append(u))
    r = rn._meal_reminders_sync("u", "local")
    assert marcas == ["u"]
    assert r["water"]["enabled"] is True and r["water"]["days"] == 3 and len(r["water"]["reminders"]) == 3
    assert r["water"]["reminders"][0]["body"] == "Llevas 1 de 9 vasos hoy. ¿Un vaso de agua ahora?"
    assert rn._meal_reminders_sync("u")["water"]["enabled"] is True and marcas == ["u"], "sin `canal=local` no se marca"
    monkeypatch.setattr(db, "get_water_tracker_enabled", lambda _u: False)
    assert rn._meal_reminders_sync("u")["water"] == {"enabled": False, "reminders": []}


def test_encenderla_de_nuevo_pone_la_cuenta_a_cero_y_el_dashboard_sabe_que_se_apago_sola():
    pref = _src("routers/preferences.py")
    i = pref.index('@router.patch("/water-tracker")')
    assert "await asyncio.to_thread(hydration_reminders.al_encender, verified_user_id)" in pref[i:i + 1600]
    planes = _src("routers/plans.py")
    assert '"auto_off_at": _auto_off_at,' in planes
    assert "if not _agua_encendida:" in planes, "solo se lee con el interruptor apagado: el GET normal no paga otra lectura"


def test_el_estado_por_usuario_caduca_y_se_borra_con_la_cuenta():
    import cron_tasks
    import db_profiles
    prefijos = {s["prefix"]: s for s in cron_tasks._KV_SWEEP_PREFIXES}
    for p in ("hydration_state:", "avisos_locales:", "plan_invite:"):
        assert p in prefijos and p in db_profiles._USER_SCOPED_KV_PREFIXES
    assert prefijos["plan_invite:"]["clamp"][0] >= 192, "nunca por debajo de 8 días: barrerla antes re-muestra la tarjeta"
    assert prefijos["avisos_locales:"]["clamp"][0] >= 72, "el alcance del canal local son 72 h"
    assert 'DELETE FROM app_kv_store WHERE key = ANY(%s) RETURNING key' in _src("db_profiles.py")


# ── 4 · «ni se inmuta»: el interruptor de alertas en el binario que YA trae el plugin ───────────────────────────────────
def test_el_plugin_de_capacitor_viaja_en_una_caja_nunca_suelto():
    """Una función async que DEVUELVE el plugin de Capacitor (un Proxy que responde a `.then` con una llamada nativa que
    jamás resuelve) cuelga su promesa para siempre: `estadoDeAvisos()` no volvía, el canal quedaba en `null` y el
    interruptor nacía deshabilitado. Solo fallaba en el binario CON el plugin. Contrato vivo: `lote135.proxy.test.js`."""
    import pytest
    f = _BACKEND.parent / "frontend" / "src" / "utils" / "avisosDeComida.js"
    if not f.exists():
        pytest.skip("sin el repo del frontend al lado")
    av = f.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert "return mod.LocalNotifications ? { LN: mod.LocalNotifications } : null;" in av
    assert "return mod.LocalNotifications || null;" not in av
    assert av.count("const LN = (await _pluginLocal())?.LN;") == 5
    assert "const LN = await _pluginLocal();" not in av
    assert (f.parent.parent / "__tests__" / "lote135.proxy.test.js").exists()


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', _src("app.py"))
    assert m and int(m.group(1)) >= 135
