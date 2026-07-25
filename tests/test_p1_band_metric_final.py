"""[P1-BAND-METRIC-FINAL · 2026-07-24] La telemetría de banda medía un plan que no se entregó.

Origen: le reporté al owner que un plan había caído a banda **0.75** y construí una hipótesis de
causa raíz encima. El plan entregado estaba en **1.00** (12/12 celdas). Las dos fuentes:

    pipeline_metrics node='clinical_band'      0.75  ← lectura PRE-finalize
    plan_data.clinical_band_score              1.00  ← lo que el usuario recibió

`refresh_clinical_band_score_post_finalize` ya recomputaba el score sobre el estado ENTREGADO y
sobrescribía `plan_data`, pero **no re-emitía la fila de telemetría** — su propio docstring lo
admitía. Consecuencia: toda la telemetría de banda de la flota sub-reporta.

Segundo agujero, peor: la emisión de `clinical_band` vive en el bloque de streaming. Si el SSE
muere a media generación (caso común, ver P1-PLAN-HYDRATE-ON-COMPLETE), el pipeline sigue, el plan
se persiste por el fallback… y la métrica nunca se emite. De 4 planes del 24/07 (todos 1.00
persistido) **solo 1 tenía fila**. Con ese volumen el cron `_clinical_band_drift_alert_job`
probablemente nunca dispara.

El fix cuelga la emisión del camino **pre-INSERT**, que corre para todo plan persistido — con SSE
vivo o muerto — bajo `node='clinical_band_final'`, con `pre_finalize_score` en metadata para poder
medir el sesgo de la serie vieja.

**Regla operacional**: para analizar banda, leer `plan_data.clinical_band_score`. La serie
`clinical_band` (sin sufijo) es pre-finalize y está sesgada a la baja.
"""
from pathlib import Path

import graph_orchestrator as go

SRC = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
DBP = (Path(go.__file__).resolve().parent / "db_plans.py").read_text(encoding="utf-8")


def test_emite_nodo_propio_no_pisa_la_serie_vieja():
    i = SRC.index("def _emit_clinical_band_final_metric")
    body = SRC[i:i + 3000]
    assert '"clinical_band_final"' in body, (
        "nodo propio: reusar 'clinical_band' mezclaría lecturas pre y post finalize en la misma serie"
    )


def test_lleva_el_score_pre_finalize_en_metadata():
    i = SRC.index("def _emit_clinical_band_final_metric")
    body = SRC[i:i + 3000]
    assert '"pre_finalize_score": prev_score' in body, (
        "sin el par (pre, post) no se puede medir cuánto sesgaba la serie vieja"
    )


def test_se_invoca_desde_el_refresh_post_finalize():
    i = SRC.index("def refresh_clinical_band_score_post_finalize")
    body = SRC[i:i + 4000]
    assert "_emit_clinical_band_final_metric(_bs, _prev_val, plan_data, user_id, surface)" in body
    # …y sobre `_bs` (la lectura fresca), nunca sobre `_prev`.
    assert "_emit_clinical_band_final_metric(_prev" not in body


def test_el_call_site_pre_insert_pasa_user_id():
    """Una fila sin user_id no es atribuible: el scoreboard por cohorte se vuelve inútil."""
    i = DBP.index("refresh_clinical_band_score_post_finalize as _rbs")
    assert "user_id=data.get(\"user_id\")" in DBP[i:i + 700]
    assert "surface=surface" in DBP[i:i + 700]


def test_camino_independiente_del_sse():
    """La razón de existir del fix: el pre-INSERT corre aunque el cliente se haya desconectado."""
    i = DBP.index("refresh_clinical_band_score_post_finalize as _rbs")
    ctx = DBP[max(0, i - 900):i + 700]
    assert "P1-BAND-METRIC-FINAL" in ctx
    assert "SSE" in ctx, "documentar POR QUÉ cuelga de aquí, o el próximo refactor lo devuelve al stream"


def test_gate_de_invitados_y_knob():
    i = SRC.index("def _emit_clinical_band_final_metric")
    body = SRC[i:i + 3000]
    # Mismo gate P1-Q10 que `_emit_progress`: sin él, un invitado dispara IntegrityError por plan.
    assert "_is_guest_metrics_enabled()" in body
    assert 'BAND_METRIC_FINAL_EMIT = _env_bool("MEALFIT_BAND_METRIC_FINAL_EMIT", True)' in SRC
    assert "if BAND_METRIC_FINAL_EMIT:" in SRC


def test_fail_safe():
    """Nunca puede bloquear el INSERT del plan: la telemetría es observabilidad, no producto."""
    i = SRC.index("def _emit_clinical_band_final_metric")
    body = SRC[i:i + 3000]
    assert "except Exception" in body and "logger.debug" in body
