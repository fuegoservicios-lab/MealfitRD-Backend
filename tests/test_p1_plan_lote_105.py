# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-105 · 2026-09-18] Macros y micros en UNA tarjeta, y el diario de días anteriores completo.

El dueño: «¿y qué tal si fusionas lo de micros con lo de macro? [...] en "ver días anteriores" quiero también ver
el historial de micros y solo se ve el de macros [...] revisa si el sistema de "días anteriores" está en su 100%
posible [...] y lo de la fototeca directa en la app nativa Capacitor hazlo también».

Lo que cierra:
 · «Micros de hoy» deja de ser tarjeta con fetch propio: `MicrosList` pinta lo que ya trae el fetch del día
   (`TrackingProgress`) y el cajón de días anteriores pinta la misma lista con `totals.micros` de ESE día;
 · la tarjeta se llama «Tus macros y micros de hoy» (también en el prompt del coach);
 · el cajón mostraba las cuatro franjas + `snack`, y el componedor registra `extra` por defecto: esa comida
   contaba en el total y no salía en ninguna fila → grupo «Extras y snacks» para todo lo que no es franja;
 · borrar desde cualquier día, registrar en el día que miras (`initialDaysAgo`, hasta 7), ver hasta 90 días,
   media de la semana, y refresco al cambiar el diario desde fuera;
 · `/api/nutrition/targets` devuelve las metas de micros aunque falten campos del contador (solo piden sexo y
   edad): en modo plan el perfil de seguimiento puede estar incompleto;
 · en la app nativa «Elegir de galería» abre la fototeca directa con el plugin (la hoja de tres opciones de iOS
   la dispara el `<input type=file>` y no se puede evitar desde la web).
"""
from __future__ import annotations

import json
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


def test_una_sola_tarjeta_macros_y_micros():
    tp = _front("src/components/dashboard/TrackingProgress.jsx")
    assert "t('Tus macros y micros de hoy')" in tp and "t('Tus macros de hoy')" not in tp
    assert "import { resumirMicros, useMicrosSubtitulo } from './microsShared';" in tp
    # los micros van en el MISMO snapshot que las macros (borrado optimista coherente)
    assert "const { micros, coverage } = resumirMicros(meals);" in tp
    assert "microsCoverage: coverage," in tp
    assert "metas={microTargets}" in tp and "targetMicros={microTargets}" in tp
    # la tarjeta de micros aparte ya no existe ni se monta
    assert not (_FRONT / "src/components/dashboard/MicrosTracker.jsx").exists()
    dt = _front("src/components/dashboard/DashboardTracking.jsx")
    assert "MicrosTracker" not in dt
    assert "microTargets={targets?.micros || null}" in dt
    ml = _front("src/components/dashboard/MicrosList.jsx")
    for k in ("fiber_g", "sodium_mg", "potassium_mg", "calcium_mg", "iron_mg", "vit_c_mg", "vit_a_mcg", "vit_d_mcg"):
        assert f"key: '{k}'" in ml
    assert "export const resumirMicros" in _front("src/components/dashboard/microsShared.js")


def test_el_diario_de_dias_anteriores_esta_completo():
    dh = _front("src/components/dashboard/DiaryHistory.jsx")
    # micros del día con la misma lista
    assert "<MicrosList micros={totales.micros || null} coverage={coberturaMicros} metas={targetMicros} compact showNotes={false} />" in dh
    # lo que no es franja (extra, snack, desconocido) se ve
    assert "const k = franjas.has(raw) ? raw : OTROS;" in dh
    assert "t('Extras y snacks')" in dh
    # borrar desde cualquier día, avisando a la tarjeta
    assert "fetchWithAuth(`/api/diary/consumed/${meal.id}`, { method: 'DELETE' })" in dh
    assert "detail: { source: 'diary-history', date: selected }" in dh
    # registrar en el día que miras, hasta el tope del backend
    assert "const DIAS_ATRAS_REGISTRO = 7;" in dh
    assert "<LogMealModal onClose={cerrarComponedor} initialDaysAgo={atras} />" in dh
    # más días (tope del endpoint) y la semana
    assert "const DIAS_MAX = 90;" in dh and "t('Ver 2 semanas más')" in dh
    assert "'Últimos 7 días: media de {kcal} kcal en {n} días con registro'" in dh
    # se refresca cuando el diario cambia desde fuera
    assert "window.addEventListener('mealfit:refresh-inventory', refrescar);" in dh
    assert "window.addEventListener('mealfit:diary-changed', refrescar);" in dh
    # el backend acepta hasta 7 días atrás: el cajón no promete más
    diary = (_BACKEND / "routers" / "diary.py").read_text(encoding="utf-8")
    assert "days_ago: int = Field(default=0, ge=0, le=7)" in diary
    assert re.search(r"min\(int\(days\),\s*90\)", diary)


def test_la_tarjeta_escucha_el_borrado_del_cajon_sin_reaccionar_al_suyo():
    tp = _front("src/components/dashboard/TrackingProgress.jsx")
    assert "if (isMounted && e?.detail?.source !== 'tracking-progress') fetchConsumed();" in tp
    assert "new CustomEvent('mealfit:diary-changed', { detail: { source: 'tracking-progress' } })" in tp


def test_el_componedor_abre_en_el_dia_pedido():
    lm = _front("src/components/dashboard/LogMealModal.jsx")
    assert "initialDaysAgo = 0" in lm
    assert "useState(() => Math.max(0, Math.min(7, Number(initialDaysAgo) || 0)))" in lm
    assert "options={_getDayOptionsCon(t, initialDaysAgo)}" in lm


def test_metas_de_micros_aunque_falten_campos_del_contador():
    ud = (_BACKEND / "routers" / "user_data.py").read_text(encoding="utf-8")
    i = ud.index('out = {"ok": False, "missing_fields": faltan}')
    assert 'out["micros"] = _metas_micros_de(hp)' in ud[i:i + 400]


def test_fototeca_directa_en_la_app_nativa():
    sm = _front("src/components/dashboard/ScanMealModal.jsx")
    assert "import { chooseNativeGalleryImage, isNativePickerCancellation } from '../../utils/nativeChatImagePicker';" in sm
    assert "if (!isNativeApp()) { galleryInputRef.current?.click(); return; }" in sm
    assert "onClick={openGallery}" in sm
    pk = _front("src/utils/nativeChatImagePicker.js")
    assert "export async function chooseNativeGalleryImage()" in pk
    assert "allowMultipleSelection: false," in pk
    # el plugin ya está en el proyecto iOS y el permiso declarado
    assert '"@capacitor/camera"' in _front("package.json")
    assert "NSPhotoLibraryUsageDescription" in _front("ios/App/App/Info.plist")


def test_el_coach_sabe_donde_se_ve_y_se_borra_otro_dia():
    from prompts.chat_agent import build_tools_instructions, build_tools_instructions_stream
    for txt in (build_tools_instructions("u-1"), build_tools_instructions_stream("u-1")):
        assert "Tus macros y micros de hoy" in txt and "Tus macros de hoy'" not in txt
        assert "'Ver dias anteriores'" in txt.split("P1-CHAT-DIARY-WHERE")[1][:700]


def test_i18n_y_marcador():
    for loc in ("en-US", "pt-BR", "fr-FR", "it-IT"):
        d = json.loads(_front(f"src/i18n/locales/{loc}.json"))
        for k in ("Tus macros y micros de hoy", "Micros", "Extras y snacks", "Ver 2 semanas más", "Registrar en este día",
                  "Solo se puede registrar hasta 7 días atrás.", "Ninguna de estas comidas trae micros (foto o macros propias)."):
            assert d.get(k), f"{loc}: {k}"
        assert d.get("Últimos 7 días: media de {kcal} kcal en {n} días con registro", {}).get("other"), loc
        for viejo in ("Micros de hoy", "Tus macros de hoy", "Snacks"):
            assert viejo not in d, f"{loc}: huérfana {viejo}"
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 105
    assert "P1-PLAN-LOTE-105" in (_BACKEND / "docs" / "modo_seguimiento_ui.md").read_text(encoding="utf-8")
