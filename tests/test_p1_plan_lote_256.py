# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-256 · 2026-09-25] El «high» que viene de rebajar un crítico reintenta; no es «contextual».

Batería rd252, estatina + amlodipino + HTA: el revisor rechazó como crítico «½ cucharadita de sal en el día 3», el 230
lo rebajó a «high» PARA reintentar, y `_classify_high_severity` lo mandó a abortar en el intento 1 porque el texto dice
«hipertensión» (la lista «contextual» es de restricciones que no cambian entre intentos: despensa, alergia…).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402

RAZON = ("Para la hipertensión, el día 3 acumula ½ cucharadita de sal (¼ cucharadita en el desayuno y otra ¼ en la "
         "merienda). Esto puede elevar demasiado el sodio diario; reducir la sal.")


def _estado(**kw):
    st = {"review_passed": False, "_rejection_severity": "high", "rejection_reasons": [RAZON], "attempt": 1,
          "pipeline_start": time.time(), "plan_result": {"days": []}, "form_data": {},
          "_surgical_reject_attempted": True, "_marker_regen_attempted": True}
    st.update(kw)
    return st


def test_la_razon_sigue_siendo_contextual_por_texto():
    assert go._classify_high_severity([RAZON]) == "contextual"


def test_el_critico_rebajado_reintenta(monkeypatch):
    monkeypatch.setattr(go, "_emit_plan_quality_degraded_alert", lambda *a, **k: None)
    assert go.should_retry(_estado(_soft_reject_retry=True)) != "end"


def test_sin_la_marca_sigue_abortando(monkeypatch):
    monkeypatch.setattr(go, "_emit_plan_quality_degraded_alert", lambda *a, **k: None)
    assert go.should_retry(_estado()) == "end"


def test_las_tres_ramas_marcan_y_el_estado_lo_declara():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert src.count('severity = "high"; _soft_retry = True') == 3
    assert '"_soft_reject_retry": _soft_retry,' in src
    assert 'if _retry_class == "contextual" and not state.get("_soft_reject_retry"):' in src
    assert "_soft_reject_retry" in go.PlanState.__annotations__


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 256
