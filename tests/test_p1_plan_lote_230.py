# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-230 · 2026-09-25] Un crítico del revisor sin peligro agudo no tira el plan real (G18 para toda condición).

Batería real del 25-sep: colesterol alto («6½ yemas»), gastritis («ajo, cebolla, vinagre»), hipotiroidismo («yuca y casabe
repetidos») → dos críticos seguidos → «Pollo y Arroz / Pescado y Batata», 3 comidas, sin básicos ni suplementos.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as g  # noqa: E402

# Razones REALES de la batería del 25-sep (texto del revisor, recortado).
NO_AGUDOS = [
    "El día 2 incluye aproximadamente 6½ yemas de huevo, superando la recomendación indicada en el reporte de limitar "
    "las yemas a 3–4 por semana para la hipercolesterolemia.",
    "Día 2, almuerzo: incluye vinagre blanco, señalado como ingrediente a excluir por la gastritis.",
    "Día 1, almuerzo: incluye ajo, señalado como posible agravante de la gastritis.",
    "El menú incluye yuca y casabe repetidamente durante los tres días, pese a la recomendación clínica de evitar su "
    "consumo diario y moderar la carga por el hipotiroidismo.",
]
AGUDOS = [
    "Para la enfermedad renal, el plan acumula fuentes de potasio potencialmente elevadas: 375 g de yautía.",
    "El informe clínico recomienda evitar soya texturizada y tofu hasta una evaluación alergológica.",
    "Arroz blanco con ceviche fresco: pescado crudo en embarazo.",
    "Queso parmesano curado con fenelzina (IMAO): la tiramina puede causar una crisis hipertensiva.",
    "Cena sin carbohidratos con insulina nocturna: riesgo de hipoglucemia.",
]


def test_lo_cronico_no_es_agudo():
    for t in NO_AGUDOS:
        assert g._critical_is_non_acute([t]), t
    assert g._critical_is_non_acute(NO_AGUDOS)


def test_lo_agudo_se_queda_critico():
    for t in AGUDOS:
        assert not g._critical_is_non_acute([t]), t
    # una sola razón aguda entre varias crónicas basta para no degradar
    assert not g._critical_is_non_acute(NO_AGUDOS + [AGUDOS[0]])
    assert not g._critical_is_non_acute([])


def _state(sev, issues, attempt=1):
    return {
        "review_passed": False, "_rejection_severity": sev, "rejection_reasons": issues, "attempt": attempt,
        "form_data": {"medicalConditions": ["Colesterol Alto"]}, "pipeline_start": time.time(),
        "plan_result": {"days": [{"meals": []}]}, "_marker_regen_attempted": True,
    }


def test_el_high_reintenta_y_al_final_entrega(monkeypatch):
    monkeypatch.setattr(g, "MAX_ATTEMPTS", 3)
    assert g.should_retry(_state("high", NO_AGUDOS[:1], attempt=2)) == "retry"
    assert g.should_retry(_state("high", NO_AGUDOS[:1], attempt=3)) == "end"


def test_cableado_y_knob():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("P1-BARIATRIC-CRITICAL-RETRY] Bariátrico: crítico de elección de comida degradado")
    j = src.index("and _critical_is_non_acute(issues)):", i)
    k = src.index("        result = {\n            \"review_passed\": False,", i)
    assert i < j < k, "el gate va encadenado tras el bariátrico y antes de armar el resultado"
    bloque = src[j - 400:j]
    for flag in ("not _had_allergen_critical", "not _had_renal_critical", "not _had_diet_critical",
                 "not plan.get(\"_schema_invalid\")"):
        assert flag in bloque, flag
    assert g.NON_ACUTE_CRITICAL_SOFT_REJECT is True
    assert "tooltip-anchor: P1-PLAN-LOTE-230-CRITICO-NO-AGUDO" in src


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 230
