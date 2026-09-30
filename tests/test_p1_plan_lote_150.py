# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-150 · 2026-09-21] El recordatorio llega a TU hora, y tú decides si lo quieres.

Los casos funcionales viven en los tests de los lotes 72, 73 y 133, que se actualizaron en vez de borrarse: cada uno
seguía protegiendo algo real y lo que cambió fue la hora esperada. Aquí, las anclas del cambio."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8").replace("\r\n", "\n")


def _front(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip("sin el repo del frontend al lado")
    return p.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_el_aviso_se_ADELANTA_y_la_tasa_de_respuesta_ya_no_mueve_la_hora():
    import proactive_agent as pa
    assert pa._antelacion_del_aviso_h() == 0.25
    src = _src("proactive_agent.py")
    assert "delay_hours = -_antelacion_del_aviso_h()" in src
    assert "delay_hours = 2.5" not in src, "retrasar el aviso de quien lo ignora lo hacía aún más inútil"


def test_el_minuto_sale_de_la_hora_en_LAS_DOS_vias():
    """El coach truncaba («6:49») y el teléfono redondeaba («6:50»): el mismo aviso con dos horas distintas.

    [P1-PLAN-LOTE-223] Ya no son dos expresiones iguales sino UNA función (`minuto_del_dia`) que llaman las dos vías."""
    import proactive_agent as pa
    assert pa.minuto_del_dia(6.8249) == 6 * 60 + 49 and pa.minuto_del_dia(6.8334) == 6 * 60 + 50, "redondea, no trunca"
    assert pa.minuto_del_dia(23.9999) == 0, "un 59,99 no acaba en «:60»: vuelve al reloj"
    assert "_aviso_min = minuto_del_dia(_hora_aviso)" in _src("proactive_agent.py")
    assert "hours, mins = divmod(_aviso_min, 60)" in _src("proactive_agent.py")
    assert "return divmod(pa.minuto_del_dia(hora_aviso), 60)" in _src("meal_reminders.py")


def test_el_texto_ya_no_da_por_hecho_que_comiste():
    import meal_reminders as mr
    for comida in mr.COMIDAS:
        cuerpo = mr.texto_del_aviso(comida, "es-DO")[1]
        assert cuerpo.startswith("Es tu hora de "), comida
        assert "Ya " not in cuerpo, f"{comida}: pregunta en vez de animar"


def test_los_dos_interruptores_los_leen_las_DOS_vias():
    import proactive_agent as pa
    assert pa.avisos_de_comida_activos({}) is True, "ausente ⇒ encendido: nadie los pierde por un despliegue"
    assert pa.avisos_de_comida_activos({"avisos_comida": False}) is False
    assert pa.avisos_de_agua_activos({"avisos_agua": False}) is False
    # el cron de comidas, el endpoint que programa el teléfono, y el cron del agua (en su propia consulta)
    assert "if not avisos_de_comida_activos(health):" in _src("proactive_agent.py")
    assert "pa.avisos_de_comida_activos(health)" in _src("routers/notifications.py")
    assert "COALESCE((p.health_profile->>'avisos_agua')::boolean, TRUE) = TRUE" in _src("hydration_reminders.py")


# [P1-PLAN-LOTE-831 · 2026-09-29] La intención de la guarda: los avisos NO son una COLUMNA (viven en `health_profile`).
# Antes prohibía el literal `avisos_comida` en cualquier migración, y la 837 (el trigger del historial de ajustes) lo
# nombra legítimamente como CLAVE jsonb en su `WHEN` (`health_profile -> 'avisos_comida'`): nombrar la clave no la
# convierte en columna. Lo que sigue prohibido es AÑADIR una columna `avisos_*`.
_ANADE_COLUMNA_AVISOS = re.compile(r"ADD\s+COLUMN\s+(?:IF\s+NOT\s+EXISTS\s+)?avisos_", re.IGNORECASE)


def _sin_comentarios_sql(texto: str) -> str:
    return re.sub(r"--[^\n]*", "", texto)


def test_sin_columna_nueva_ni_migracion():
    """Viven en `health_profile`, que ya es jsonb libre y ya tiene su endpoint de merge: ninguna migración AÑADE una
    columna `avisos_*` (nombrar la clave jsonb, como el trigger de la 837, no cuenta)."""
    culpables = [p.name for p in sorted((_BACKEND / "migrations").glob("*.sql"))
                 if _ANADE_COLUMNA_AVISOS.search(_sin_comentarios_sql(p.read_text(encoding="utf-8", errors="ignore")))]
    assert not culpables, f"migraciones que añaden una columna avisos_*: {culpables}"


def test_la_guarda_de_columnas_avisos_distingue_columna_de_clave():
    """La guarda no puede pasar por vacía: caza las formas de añadir la columna y deja pasar la clave jsonb."""
    for anade in ("ALTER TABLE public.user_profiles ADD COLUMN avisos_comida boolean;",
                  "alter table t add column if not exists avisos_agua boolean default true;",
                  "ALTER TABLE t\n    ADD COLUMN   IF  NOT  EXISTS\n    avisos_por_comida jsonb;",
                  "ALTER TABLE t ADD COLUMN AVISOS_X int;"):
        assert _ANADE_COLUMNA_AVISOS.search(anade), anade
    for clave in ("OR (OLD.health_profile -> 'avisos_comida') IS DISTINCT FROM (NEW.health_profile -> 'avisos_comida')",
                  "FOREACH v_clave IN ARRAY ARRAY['avisos_comida', 'avisos_agua']::text[]",
                  "ALTER TABLE t ADD COLUMN otra_cosa boolean;"):
        assert not _ANADE_COLUMNA_AVISOS.search(clave), clave
    # un comentario que hable de la columna prohibida no es una migración que la añada
    assert not _ANADE_COLUMNA_AVISOS.search(_sin_comentarios_sql("-- nunca ADD COLUMN avisos_comida\nSELECT 1;"))
    # y la 837 SÍ nombra la clave (es la que motivó afinar la guarda): sigue pasando porque no añade ninguna columna
    mig = (_BACKEND / "migrations" / "p1_plan_lote_837_ajustes_cambios_2026_09_29.sql").read_text(encoding="utf-8")
    assert "avisos_comida" in mig and not _ANADE_COLUMNA_AVISOS.search(_sin_comentarios_sql(mig))


def test_el_frontend_los_guarda_en_el_perfil_y_reprograma():
    st = _front("src/pages/Settings.jsx")
    # [P1-PLAN-LOTE-162] En el perfil, sí — pero SOLO su clave: `safeUpdateHealthProfile` mandaba el formulario entero
    # y la copia local congelada de la otra preferencia la devolvía a su valor viejo.
    assert "body: JSON.stringify({ health_profile: { [clave]: valor } })," in st
    assert "await sincronizarAvisosLocales();" in st


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', _src("app.py"))
    assert m and int(m.group(1)) >= 150
