# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-102 · 2026-09-18] Un icono de cámara, «Progreso» y «Tus macros y micros de hoy», y peso/altura en tiempo real.

La sección del contador se llama «Tus macros y micros de hoy» (antes «Progreso en Tiempo Real») y la pestaña «Progreso»
(antes «Hoy»). El coach remite al usuario a esa sección POR SU NOMBRE (prompts/chat_agent.py): si el prompt
conservara el viejo, mandaría al usuario a buscar algo que ya no existe con ese nombre — la misma trampa que cerró
`todayRemaining.p1_i18n_eaten_claim`. En modo contador, Configuración guarda peso/altura/edad/sexo al momento
(sin regenerar) y las metas (`/api/nutrition/targets`, que lee health_profile) cambian en tiempo real."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def _front(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip("sin el repo del frontend al lado")
    return p.read_text(encoding="utf-8")


def _sin_comentarios_py(src: str) -> str:
    return "\n".join(l for l in src.split("\n") if not l.lstrip().startswith("#"))


def test_el_coach_nombra_la_seccion_como_la_ve_el_usuario():
    prompt = _sin_comentarios_py((_BACKEND / "prompts" / "chat_agent.py").read_text(encoding="utf-8"))
    assert "Tus macros y micros de hoy" in prompt
    assert "Progreso en Tiempo Real" not in prompt, "el coach mandaría al usuario a una sección que ya no se llama así"
    tp = _front("src/components/dashboard/TrackingProgress.jsx")
    assert "t('Tus macros y micros de hoy')" in tp and "t('Progreso en Tiempo Real')" not in tp


def test_pestana_progreso_y_un_solo_icono_de_camara():
    nav = _front("src/config/dashboardNav.js")
    assert "trackingMode ? t('Progreso') : t('Plan|nav')" in nav
    scan = _front("src/components/dashboard/ScanMealModal.jsx")
    i = scan.index('id="scan-meal-title"')
    assert "titleIco" not in scan[i:i + 200] and "<Camera" not in scan[i:i + 200]


def test_modo_contador_guarda_al_momento_y_las_metas_se_repiden():
    s = _front("src/pages/Settings.jsx")
    assert "const enModoContador = isTrackingMode(userProfile);" in s
    h = s[s.index("const handleSaveTracking = async () => {"):s.index("const handleUpdatePlanWithMetrics = async () => {")]
    assert "regeneratePlan(" not in h and "getFreshPlanCount(" not in h
    assert "window.dispatchEvent(new Event('mealfit:targets-changed'));" in h
    d = _front("src/components/dashboard/DashboardTracking.jsx")
    assert "window.addEventListener('mealfit:targets-changed', cargar);" in d
    # y las metas salen de health_profile: lo que Configuración escribe es lo que el contador lee
    ud = (_BACKEND / "routers" / "user_data.py").read_text(encoding="utf-8")
    assert 'hp = (profile or {}).get("health_profile") or {}' in ud


def test_marcador_y_documento():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 102
    assert "P1-PLAN-LOTE-102" in (_BACKEND / "docs" / "modo_seguimiento_ui.md").read_text(encoding="utf-8")
