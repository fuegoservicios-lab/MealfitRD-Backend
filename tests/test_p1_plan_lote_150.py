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
    """El coach truncaba («6:49») y el teléfono redondeaba («6:50»): el mismo aviso con dos horas distintas."""
    assert "hours, mins = divmod(int(round(nudge_hour * 60)) % (24 * 60), 60)" in _src("proactive_agent.py")
    assert "total = int(round(float(nudge_hour) * 60)) % (24 * 60)" in _src("meal_reminders.py")


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


def test_sin_columna_nueva_ni_migracion():
    """Viven en `health_profile`, que ya es jsonb libre y ya tiene su endpoint de merge."""
    assert "avisos_comida" not in "\n".join(
        p.read_text(encoding="utf-8", errors="ignore") for p in (_BACKEND / "migrations").glob("*.sql"))


def test_el_frontend_los_guarda_en_el_perfil_y_reprograma():
    st = _front("src/pages/Settings.jsx")
    assert "safeUpdateHealthProfile({ [clave]: valor })" in st
    assert "await sincronizarAvisosLocales();" in st


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', _src("app.py"))
    assert m and int(m.group(1)) >= 150
