"""[P2-BAND-RETRY-GATE · 2026-06-21] La banda de macros como gate de RETRY (Fase 6 del build
"todo terreno").

El owner pidió "duro en las 4 macros". El band-score gate existente (BAND_SCORE_GATE) marcaba
`_quality_degraded` POST-grafo (banner) sin forzar retry. Este gate fuerza un RETRY cuando el
`clinical_band_score` cae bajo BAND_RETRY_THRESHOLD, dándole al motor determinista + al LLM otra
pasada para acercar las 4 macros a la banda [90,112]%.

HONESTIDAD FÍSICA anclada en el diseño: la banda PERFECTA (todas las celdas día×macro) NO es
alcanzable por la granularidad de porción cocinable (techo ~66.7% all-4-en-banda MEDIDO). Por eso
el umbral default (0.5) ataca solo la cola genuinamente pobre — no mass-retry. Reusa el MISMO score
que el banner gate (cero drift).
"""
import graph_orchestrator as go


def _plan(factor, p=150, c=200, f=60, cals=2000):
    """1 comida/día = el total diario. factor=1.0 → en banda; factor=0.5 → fuera."""
    return {
        "macros": {"protein": f"{p}g", "carbs": f"{c}g", "fats": f"{f}g"},
        "calories": cals,
        "days": [{"meals": [{
            "protein": round(p * factor), "carbs": round(c * factor),
            "fats": round(f * factor), "cals": round(cals * factor),
        }]}],
    }


# ---------------------------------------------------------------------------
# El score (reusado por el gate) — fuente de verdad compartida con el banner
# ---------------------------------------------------------------------------
def test_plan_en_banda_score_alto():
    sc = go.compute_clinical_band_score(_plan(1.0), {})
    assert sc["score"] == 1.0, "Macros = target → todas las celdas en banda."


def test_plan_fuera_de_banda_score_cero():
    sc = go.compute_clinical_band_score(_plan(0.5), {})
    assert sc["score"] == 0.0, "Macros al 50% del target → todas fuera de banda."
    assert sc["score"] < go.BAND_RETRY_THRESHOLD, "Un score 0 cae bajo el umbral → dispararía retry."


def test_usa_plan_macros_sin_nutrition():
    # compute_clinical_band_score cae a plan['macros'] como target cuando nutrition={} → el gate
    # puede correr en review_plan_node sin depender de `nutrition` en scope.
    sc = go.compute_clinical_band_score(_plan(1.0), {})
    assert sc.get("per_macro", {}).get("protein") == 1.0


# ---------------------------------------------------------------------------
# Knobs + gate anclados en el source
# ---------------------------------------------------------------------------
def test_knobs_existen():
    assert hasattr(go, "BAND_RETRY_GATE_ENABLED")
    assert hasattr(go, "BAND_RETRY_THRESHOLD")
    assert 0.0 <= go.BAND_RETRY_THRESHOLD <= 1.0


def test_gate_ancla_en_review():
    src = open(go.__file__, encoding="utf-8").read()
    assert "P2-BAND-RETRY-GATE" in src
    # El gate debe: computar el score, comparar contra el umbral, y forzar retry (severity high).
    idx = src.find("if BAND_RETRY_GATE_ENABLED:")
    assert idx > -1, "El gate de retry de banda debe existir en review_plan_node."
    # [P1-BAND-GATE-ALL4 · 2026-07-01] ventana 1400→3000: los comments del umbral macros-only re-tuneado
    # desplazaron el cierre del gate (severity a offset ~2643); el contrato anclado no cambió.
    region = src[idx: idx + 3000]
    assert "compute_clinical_band_score(plan" in region
    assert "BAND_RETRY_THRESHOLD" in region
    assert "_severity_max(severity, \"high\")" in region


def test_honestidad_fisica_documentada():
    # El diseño DEBE documentar por qué el umbral no es 1.0 (porción cocinable) — protege contra un
    # refactor futuro que suba el umbral a 1.0 y cause mass-retry/degradación.
    src = open(go.__file__, encoding="utf-8").read()
    idx = src.find("BAND_RETRY_GATE_ENABLED = _env_bool")
    region = src[max(0, idx - 1400): idx]
    assert "cocinable" in region and "66.7%" in region
