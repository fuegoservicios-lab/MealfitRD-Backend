# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-136 · 2026-09-20] Configuración con el generador de planes APAGADO: sin contradicciones.

El dueño: «revisa en general el apartado de configuración cuando el generador de planes está apagado: ¿funciona todo al
100 %, listo para producción? ¿No hay ninguna contradicción?». Auditoría de la pantalla (Settings.jsx entero) y de cada
endpoint que dispara: ni caídas, ni paywall, ni generación o créditos gastados al guardar — pero sí contradicciones:

  1. «Modo automático» (no pausar el plan si dejas de registrar) visible en modo contador: su único lector es el worker de
     bloques, apagado ahí. Se pinta solo con el generador encendido.
  2. Con un plan EN PAUSA, «Plan & Objetivo» enseñaba las kcal congeladas del plan y vendía «Evaluar de nuevo» (1 crédito);
     el contador usa /api/nutrition/targets. Contador manda también ahí, y el botón REANUDA (gratis).
  3. El interruptor no sincronizaba el perfil en memoria (sin plan no hay recarga): media pantalla seguía en el modo viejo.
  4. El diálogo de pausa prometía «tu plan queda en el Historial» a quien nunca tuvo plan.
  5. Si fallaba la lectura del modo, el interruptor —la única puerta de vuelta— desaparecía sin aviso.
  6. «0 kcal · 0 g» en el móvil sin metas; el panel no seguía a «Guardar».
  7. «Guardar» escribía el formulario ENTERO (campos nunca preguntados, `appMode`) en `health_profile`.
  8. SERVIDOR: reanudar revivía también los bloques de un plan ya SUSTITUIDO (gasto de IA en un plan que nadie ve).

Contrato fino del cliente: `frontend/src/__tests__/lote136.test.jsx`."""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def _front(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip("sin el repo del frontend al lado")
    return p.read_text(encoding="utf-8").replace("\r\n", "\n")


# ── servidor ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_reanudar_solo_revive_los_bloques_del_plan_vigente(monkeypatch):
    import cron_tasks as ct
    import db_core
    import plan_mode as pm

    ejecutadas = []
    cursor = MagicMock()
    cursor.execute.side_effect = lambda sql, params=None: ejecutadas.append((" ".join(str(sql).split()), params))
    cursor.fetchone.side_effect = [{"health_profile": {"age": 30}}, {"d": 3, "d0": "2026-09-20", "days": []}]
    cursor.fetchall.return_value = [{"id": "c1", "meal_plan_id": "plan-b", "days_count": 4}]
    pool = MagicMock()
    pool.connection.return_value.__enter__.return_value.transaction.return_value.__enter__.return_value = MagicMock()
    pool.connection.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = cursor
    monkeypatch.setattr(db_core, "connection_pool", pool)
    monkeypatch.setattr(ct, "_rebase_pending_chunk_offsets_sql", lambda cur, pid, dias: 0)

    assert pm._revive_paused_chunks("u-1") == {"revived": 1, "plans": 1}
    sql, params = next((s, p) for s, p in ejecutadas if "SET status = 'pending'" in s)
    assert "AND meal_plan_id = ( SELECT id FROM meal_plans WHERE user_id = %s ORDER BY created_at DESC LIMIT 1 )" in sql, \
        "la firma de la pausa sobrevive en las filas de un plan ya sustituido: sin este filtro vuelven a `pending`"
    assert params == ("u-1", pm.PAUSE_CANCEL_REASON, "u-1")


# ── pantalla (contratos cruzados; el detalle vive en el test del frontend) ─────────────────────────────────────────────
def test_modo_automatico_solo_con_el_generador_encendido():
    st = _front("src/pages/Settings.jsx")
    i = st.index("{t('Modo automático')}")
    assert "{!enModoContador && (" in st[i - 1400:i]


def test_plan_y_objetivo_contador_manda_tambien_con_plan_en_pausa():
    st = _front("src/pages/Settings.jsx")
    assert "if ((planData && !enModoContador) || isGuest) return undefined;" in st
    assert "const _goalKcal = (planData?.calories && !enModoContador)" in st
    assert st.count("if (enModoContador && planData) { reanudarPlanes(); return; }") == 2, "móvil y escritorio: reanudar es gratis"
    assert st.count("planData && isLimitReached && !enModoContador ? renderPlanLimitBlock()") == 2
    # la lectura del modo vive ANTES del efecto que la usa (en su array de dependencias, después sería un TDZ)
    assert st.index("const enModoContador = isTrackingMode(userProfile);") < st.index("fetchWithAuth('/api/nutrition/targets')")
    assert st.count("const enModoContador = isTrackingMode(userProfile);") == 1


def test_el_interruptor_obedece_al_servidor_y_refresca_el_perfil_sin_plan():
    st = _front("src/pages/Settings.jsx")
    h = st[st.index("const handleTogglePlanMode"):st.index("const handleTogglePlanMode") + 6500]
    assert "const quedo = (data.plan_mode === 'plan' || data.plan_mode === 'tracking') ? data.plan_mode : next;" in h
    assert h.index("if (quedo !== next) {") < h.index("safeLocalStorageSet('mealfit_plan_mode', next);")
    assert re.search(r"\} else \{[\s\S]{0,700}await refreshProfileAndPlan\(\);", h)
    assert re.search(r"description: planData\s*\? t\('La app pasa a modo contador", h), "sin plan no se promete un Historial"
    assert "{!isGuest && planModeState === null && planModeLoadFailed && (" in st


def test_guardar_en_modo_contador_no_escribe_el_formulario_entero():
    st = _front("src/pages/Settings.jsx")
    g = st[st.index("const handleSaveTracking = async () => {"):st.index("const handleUpdatePlanWithMetrics = async () => {")]
    assert "{ ...(userProfile?.health_profile || {}), ...overrides }" in g
    assert "buildHealthProfilePayload(formData, overrides, session)" not in g
    po = _front("src/components/settings/PlanObjetivo.jsx")
    assert "{kcal == null ? '—' : formatNumber(Number(kcal))}" in po, "sin metas va «—», no «0 kcal»"


def test_marcador_y_documento():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 136
    assert "P1-PLAN-LOTE-136" in (_BACKEND / "docs" / "modo_seguimiento_ui.md").read_text(encoding="utf-8")
