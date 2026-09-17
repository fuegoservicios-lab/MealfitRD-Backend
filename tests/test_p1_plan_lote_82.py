# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-82 · 2026-09-17] Tolerancia del RECHAZO del piso de proteína, no del piso. Bench real del 17-sep
(90 kg, target 198 g, piso 178,2 g): intento 1 a 177 g (1,2 g bajo el piso) → rechazo «high» → 167 → 176 → max_attempts
y entrega DEGRADADA con «regenéralo», mientras el plan entregado medía 184/199/201 g. El déficit lo abren los topes de
porción DESPUÉS del cerrador (el cap tiene la última palabra por diseño): otro intento del LLM cae en el mismo recorte.
Un día entre el 85 y el 90 % se entrega aprobado y queda escrito; por debajo del 85 % se sigue rechazando; bariátrica
(80 %) y renal no cambian; 0 = conducta anterior."""
from __future__ import annotations

import re
from pathlib import Path

import graph_orchestrator as go

_BACKEND = Path(__file__).resolve().parents[1]


def _plan(dias, target="198g", renal=False):
    p = {"macros": {"protein": target, "carbs": "220g", "fats": "60g"},
         "days": [{"day": i, "meals": [{"protein": g}]} for i, g in enumerate(dias, 1)]}
    if renal:
        p["renal_protein_cap"] = {"applied": True}
    return p


def test_el_helper_sin_tolerancia_no_cambia():
    # 177 de 198 = 89,4 % < 90 % → corto (conducta anterior intacta)
    assert go._protein_floor_shortfall(_plan([177, 199]), renal_capped=False) == [(1, 177, 198)]


def test_dentro_de_la_tolerancia_no_se_rechaza_y_por_debajo_si():
    tol = 0.05   # umbral de rechazo 85 % → 168,3 g
    assert go._protein_floor_shortfall(_plan([177, 199]), renal_capped=False, tolerance_pct=tol) == []
    assert go._protein_floor_shortfall(_plan([176, 167]), renal_capped=False, tolerance_pct=tol) == [(2, 167, 198)]
    assert go._protein_floor_shortfall(_plan([152]), renal_capped=False, tolerance_pct=tol) == [(1, 152, 198)]
    assert go._protein_floor_shortfall(_plan([177]), renal_capped=False, tolerance_pct=0.0) == [(1, 177, 198)]


def test_bariatrica_y_renal_no_cambian():
    bari = {"medicalConditions": ["Cirugía bariátrica"]}
    plan = _plan([63, 75], target="80")          # 63 = 78,75 % < 80 % → corto aunque haya tolerancia
    assert go._protein_floor_shortfall(plan, renal_capped=False, form_data=bari, tolerance_pct=0.05) == [(1, 63, 80)]
    assert go._protein_floor_shortfall(_plan([100], renal=True), renal_capped=True, tolerance_pct=0.05) == []


def test_el_knob_y_su_validador():
    assert go.PROTEIN_FLOOR_RETRY_TOLERANCE_PCT == 0.05
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert 'PROTEIN_FLOOR_RETRY_TOLERANCE_PCT = _env_float("MEALFIT_PROTEIN_FLOOR_RETRY_TOLERANCE_PCT", 0.05' in src
    assert "validator=lambda v: 0.0 <= v <= 0.10" in src
    # el piso de los cerradores y el del último recorte NO cambian (tolerancia del rechazo, no del piso)
    assert 'PROTEIN_FLOOR_HARD_PCT = _env_float("MEALFIT_PROTEIN_FLOOR_HARD_PCT", 0.90)' in src
    lw = (_BACKEND / "protein_floor_last_word.py").read_text(encoding="utf-8")
    assert "_PISO_POR_DEFECTO = 0.90" in lw and "tolerance" not in lw.lower()


def test_los_tres_sitios_que_deciden_otro_intento_pasan_la_tolerancia():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert src.count("tolerance_pct=PROTEIN_FLOOR_RETRY_TOLERANCE_PCT)   # [P1-PLAN-LOTE-82]") == 3
    i = src.index("_short_days = _protein_floor_shortfall(plan, renal_capped=_renal_capped_plan")
    assert "tolerance_pct=PROTEIN_FLOOR_RETRY_TOLERANCE_PCT" in src[i:i + 300]
    j = src.index("_short = _protein_floor_shortfall(new_plan_result, renal_capped=_renal_capped")
    assert "tolerance_pct=PROTEIN_FLOOR_RETRY_TOLERANCE_PCT" in src[j:j + 300]
    k = src.index("_pf_short = _protein_floor_shortfall(plan_result, renal_capped=_renal_capped_final")
    assert "tolerance_pct=PROTEIN_FLOOR_RETRY_TOLERANCE_PCT" in src[k:k + 400]


def test_lo_tolerado_queda_escrito_en_el_plan():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("[P1-PLAN-LOTE-82] Lo tolerado queda escrito")
    body = src[i:i + 1600]
    assert 'plan["_protein_floor_tolerated"]' in body and "_tolerados" in body
    assert "se entrega sin otro intento" in body
    # va DESPUÉS del bloque del rechazo (la ventana de 2.200 del test del mensaje sigue viendo lo suyo)
    assert i > src.index('severity = _severity_max(severity, "high")\n\n    # [P1-PLAN-LOTE-82]') - 5


def test_marcador_y_documento():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 82
    assert "P1-PLAN-LOTE-82" in (_BACKEND / "docs" / "llm_tier_routing.md").read_text(encoding="utf-8")
    assert "MEALFIT_PROTEIN_FLOOR_RETRY_TOLERANCE_PCT" in (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
