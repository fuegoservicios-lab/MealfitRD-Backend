# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-186 · 2026-09-23] La puerta de banda del revisor mide lo que se guarda.

Batería real rd12 (adulto mayor con HTA): la puerta de banda (`P2-BAND-RETRY-GATE`) rechazó el plan con la grasa del día 1
en 0,887 y las kcal en 0,949 → reintento completo. Reproducido sin IA sobre el mismo plan: la cadena de calidad del
guardado (`db.apply_plan_quality_finalize_chain`, la misma que corre en la cola del ensamblado y al guardar) lo deja en
1,0 en las cuatro celdas y es exactamente el plan que se persistió. Entre la cadena del ensamblado y la puerta hay pases
que mueven cantidades (autofixes tardíos, humanizado, validaciones), así que la puerta medía un estado que el guardado
todavía iba a cerrar. Ahora, si ve alguna celda fuera de banda, re-cierra con esa cadena (idempotente) y vuelve a medir.
Sólo cuesta cuando hay algo fuera. Knob `MEALFIT_REVIEW_BAND_RECLOSE` (True)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def test_la_puerta_re_cierra_con_la_cadena_del_guardado_antes_de_decidir():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("async def review_plan_node(state: PlanState) -> dict:")
    cuerpo = src[i:src.index("\nasync def ", i + 10) if "\nasync def " in src[i + 10:] else len(src)]
    medir = cuerpo.index("_bsr = compute_clinical_band_score(plan, {})")
    recierre = cuerpo.index('apply_plan_quality_finalize_chain, plan, surface="review-band-gate", form_data=form_data')
    decidir = cuerpo.index("_bsr_used_mo = bool(BAND_GATE_USE_MACROS_ONLY")
    assert medir < recierre < decidir
    assert 'MEALFIT_REVIEW_BAND_RECLOSE' in cuerpo[medir:decidir] and "P1-PLAN-LOTE-186-BANDA-COMO-SE-GUARDA" in cuerpo


def test_confirmar_la_tolerancia_es_aviso():
    """rd15, alergia a mariscos: «el reporte recomienda confirmar la tolerancia individual antes de consumirlos»."""
    import graph_orchestrator as go
    issue = ("El plan incluye atún, mero y otros pescados. Aunque la alergia declarada es a mariscos y no implica "
             "necesariamente alergia al pescado, el reporte recomienda confirmar la tolerancia individual antes de consumirlos.")
    aprobado, reales, _s, avisos = go._downgrade_reviewer_verification_demands(False, [issue], "critical")
    assert aprobado and reales == [] and avisos == [issue]


def test_knob_documentado():
    doc = (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    assert "| `MEALFIT_REVIEW_BAND_RECLOSE` | `True` |" in doc


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 186 and m.group(2) >= "2026-09-23"
