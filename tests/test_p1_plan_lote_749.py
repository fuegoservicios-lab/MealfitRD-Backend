"""[P1-PLAN-LOTE-749 · 2026-09-28] Arnés del benchmark del landing: lo que la auditoría encontró roto.

QUÉ ESTABA MAL (medido en el código, no supuesto):

  1. **La matriz se cortaba en 20.** `live`/`remote` cortaban con `profiles[:n]` y los dos workflows
     mandaban `n=20` por defecto: los perfiles 21-25 (renal, anemia, gota, hígado graso e IMAO —
     justo los que P1-MEDICAL-SCOPE-GATE añadió porque el formulario no podía expresarlos) NUNCA
     entraban en una corrida por defecto. El protocolo del landing exige 25 perfiles por corrida.
  2. **No había sección de nutrición.** El contrato del landing (B-01) pide MAPE por macro, MAPE del
     peor macro y «días con los 4 macros en banda» sobre el plan ENTREGADO; el arnés no calculaba
     ninguno (solo el eje `banda` del gym, con una banda que no es la del motor: sin el techo kcal
     de ganancia muscular y dividiendo por días vacíos).
  3. **La telemetría contaba lo que no se entregó.** `banda_entregada` promediaba las filas
     `assemble-tail` (lectura INTERMEDIA, pre-review: 269 de 404 filas en 30 días) y
     `fallback_rate`/`generacion_latencia` salían de `clinical_band`, que se emite por CORRIDA del
     pipeline — incluidas las que acaban en un fallback que el router descarta (422/503).
  4. **Textos de routing muertos** («gpt-5.6», «cuota GLM») en el runner, la doc y los workflows.
  5. **El workflow remote no cabía**: `--conc 1` y 170 min para 25 generaciones de ~6-10 min.
  6. **Sin trazabilidad**: el reporte no decía de qué commit salió ni bajo qué protocolo; el
     importador del landing (schema v2) lo rechaza sin eso.

Los tests son funcionales sobre funciones puras (sin LLM, sin DB) salvo dos anclas de paridad con el
motor (`compute_clinical_band_score`), que es exactamente lo que esas funciones prometen imitar.
tooltip-anchor: P1-PLAN-LOTE-749
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

from landing_benchmarks import (
    LANDING_BENCHMARK_PROTOCOL_VERSION,
    LANDING_BENCHMARK_SCHEMA_VERSION,
    LANDING_REPORT_SECTIONS,
    aggregate_nutrition,
    aggregate_reliability,
    build_landing_profiles,
    build_report,
    build_run_meta,
    classify_remote_error,
    derive_expectations,
    engine_band_definition,
    plan_delivery_state,
    score_plan_nutrition,
    telemetry_queries,
)

_BACKEND = Path(__file__).resolve().parent.parent
_RUNNER_PATH = _BACKEND / "scripts" / "landing_benchmark.py"
_RUNNER_SRC = _RUNNER_PATH.read_text(encoding="utf-8")
_DOC = (_BACKEND / "docs" / "landing_benchmarks.md").read_text(encoding="utf-8")
_WF_REMOTE = (_BACKEND / ".github" / "workflows" / "landing-benchmark-remote-guest.yml").read_text(encoding="utf-8")
_WF_OPENAI = (_BACKEND / ".github" / "workflows" / "landing-benchmark-openai.yml").read_text(encoding="utf-8")

# La banda del motor, escrita a mano SOLO para los tests puros (la paridad con el motor se
# comprueba abajo contra `compute_clinical_band_score`, no contra esta copia).
_BAND = {"macro": [0.90, 1.12], "kcal": [0.95, 1.05], "kcal_upper_gain_muscle": 1.10}

# Campos que el importador del landing (bioboros-cinematic/benchmark_import.py, schema v2) exige en
# `run`. Es la lista del CONTRATO, no una copia del importador.
_IMPORTER_RUN_FIELDS = ("id", "architecture", "protocol_version", "started_at", "finished_at",
                        "source_commit", "country_scope", "full_profile_count", "cohort_status",
                        "publication_status", "parameters", "profile_count", "source_dirty")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]+$")


def _load_runner():
    spec = importlib.util.spec_from_file_location("landing_benchmark_cli_749", _RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _meal(k, p, c, f, name="Plato"):
    return {"name": name, "meal": "Almuerzo", "cals": k, "protein": f"{p}g", "carbs": f"{c}g",
            "fats": f"{f}g", "ingredients": ["100g de arroz"], "recipe": ["Cocinar."]}


def _plan(days, *, kcal=2000, p=150, c=200, f=60, goal="Mantenimiento"):
    return {"calories": kcal, "macros": {"protein": f"{p}g", "carbs": f"{c}g", "fats": f"{f}g"},
            "main_goal": goal,
            "days": [{"day": i + 1, "meals": meals} for i, meals in enumerate(days)]}


# ---------------------------------------------------------------------------
# 1. Nutrición: MAPE por macro, peor macro y 4-en-banda sobre el plan ENTREGADO
# ---------------------------------------------------------------------------
def test_nutricion_mape_y_cuatro_en_banda():
    plan = _plan([
        [_meal(1000, 75, 100, 30), _meal(1000, 75, 100, 30)],   # día exacto al objetivo
        [_meal(1000, 60, 100, 30), _meal(1000, 60, 100, 30)],   # proteína −20 %: fuera de banda
    ])
    r = score_plan_nutrition(plan, band=_BAND)
    assert r["scored"] is True and r["days_evaluated"] == 2
    assert r["per_macro_mape_pct"] == {"kcal": 0.0, "protein": 10.0, "carbs": 0.0, "fats": 0.0}
    assert r["four_macros_in_band_days"] == 1 and r["four_macros_in_band_pct"] == 50.0
    assert r["worst_macro"] == "protein" and r["worst_macro_mape_pct"] == 10.0

    agg = aggregate_nutrition([r], n_attempted=3, band=_BAND)
    assert agg["n_scored"] == 1 and agg["n_attempted"] == 3
    assert agg["per_macro_mape_pct"]["protein"] == 10.0
    assert agg["macro_mape_pct"] == 2.5, "media de los 4 MAPE por macro"
    assert agg["worst_macro_mape_pct"] == 10.0 and agg["worst_macro"] == "protein"
    assert agg["four_macros_in_band_pct"] == 50.0 and agg["days_evaluated"] == 2


def test_nutricion_agrega_dias_de_todos_los_planes_no_promedio_de_planes():
    """Se agrega por DÍA evaluado (como el nightly y como define B-01), no por plan: un plan de 3
    días y otro de 1 día pesan 3:1."""
    bueno = _plan([[_meal(2000, 150, 200, 60)]] * 3)
    malo = _plan([[_meal(2000, 120, 200, 60)]])   # proteína −20 %
    agg = aggregate_nutrition([score_plan_nutrition(bueno, band=_BAND),
                               score_plan_nutrition(malo, band=_BAND)], band=_BAND)
    assert agg["days_evaluated"] == 4
    assert agg["per_macro_mape_pct"]["protein"] == 5.0          # 20 % / 4 días
    assert agg["four_macros_in_band_pct"] == 75.0


def test_nutricion_techo_kcal_de_ganancia_muscular():
    dia = [[_meal(2160, 150, 200, 60)]]                          # kcal 1.08× objetivo
    estricto = score_plan_nutrition(_plan(dia, goal="Mantenimiento"), band=_BAND)
    bulk = score_plan_nutrition(_plan(dia, goal="Ganancia Muscular (Superávit 8% · ritmo gradual)"),
                                band=_BAND)
    assert estricto["four_macros_in_band_days"] == 0
    assert bulk["gain_muscle"] is True and bulk["four_macros_in_band_days"] == 1


def test_nutricion_sin_objetivos_no_puntua_pero_cuenta_en_el_denominador():
    sin = {"days": [{"meals": [_meal(2000, 150, 200, 60)]}]}
    r = score_plan_nutrition(sin, band=_BAND)
    assert r["scored"] is False and r.get("reason")
    agg = aggregate_nutrition([r], n_attempted=1, band=_BAND)
    assert agg["n_scored"] == 0 and agg["n_attempted"] == 1
    assert "macro_mape_pct" not in agg, "sin planes puntuados no se inventa un MAPE"


def test_nutricion_paridad_con_la_banda_del_motor():
    """La promesa es «la definición de banda del motor»: los días 4-en-banda y la fracción por macro
    tienen que coincidir con `compute_clinical_band_score` sobre el MISMO plan."""
    from graph_orchestrator import compute_clinical_band_score
    band = engine_band_definition()
    fixtures = [
        _plan([[_meal(1000, 75, 100, 30)] * 2, [_meal(1000, 60, 100, 30)] * 2, []]),
        _plan([[_meal(2160, 150, 200, 60)]], goal="Ganancia Muscular (Superávit 15% · ritmo decidido)"),
        _plan([[_meal(2160, 150, 200, 60)]], goal="Pérdida de Grasa (Déficit 12% · ritmo gradual)"),
        _plan([[_meal(1900, 140, 230, 66)], [_meal(2100, 170, 180, 55)]]),
    ]
    for plan in fixtures:
        eng = compute_clinical_band_score(plan, {})
        mine = score_plan_nutrition(plan, band=band)
        assert mine["four_macros_in_band_days"] == eng["all4_days"], (plan, eng)
        assert mine["days_evaluated"] == eng["days_total"]
        for mac, frac in (eng["per_macro"] or {}).items():
            if frac is None:
                continue
            mine_frac = mine["per_macro"][mac]["in_band"] / mine["per_macro"][mac]["n"]
            assert round(mine_frac, 3) == frac, (mac, plan)


def test_la_banda_se_lee_del_motor():
    import graph_orchestrator as go
    b = engine_band_definition()
    assert b["macro"] == [go.BAND_SCORE_LOWER, go.BAND_SCORE_UPPER]
    assert b["kcal_upper_gain_muscle"] == go.GAINMUSCLE_KCAL_BAND_UPPER
    assert b["source"] == "graph_orchestrator"


# ---------------------------------------------------------------------------
# 2. Lo entregado y lo que no
# ---------------------------------------------------------------------------
def test_estado_de_entrega_del_plan():
    assert plan_delivery_state({"days": []}) == "delivered"
    assert plan_delivery_state({"_is_fallback": True}) == "discarded_fallback", (
        "un fallback sin `_partial_repair` lo DESCARTA el router (422/503): no llega al usuario")
    assert plan_delivery_state({"_is_fallback": True, "_partial_repair": True}) == "delivered_fallback"
    assert plan_delivery_state(None) == "error"


def test_error_remoto_de_fallback_descartado_se_distingue_de_un_error_de_red():
    assert classify_remote_error("SSE error code=critical_restriction: x") == "discarded_fallback"
    assert classify_remote_error("SSE error code=llm_unavailable: x") == "discarded_fallback"
    assert classify_remote_error("HTTP 503 en /api/plans/analyze: IA saturada") == "discarded_fallback"
    assert classify_remote_error("RuntimeError: stream excedió el presupuesto total de 1200s") == "error"


def test_confiabilidad_cuenta_fallos_en_el_denominador_y_en_la_latencia():
    rows = [
        {"delivery": "delivered", "duration_s": 300.0},
        {"delivery": "delivered_fallback", "duration_s": 500.0},
        {"delivery": "discarded_fallback", "duration_s": 700.0},
        {"delivery": "error", "duration_s": 1200.0},
    ]
    r = aggregate_reliability(rows)
    assert r["n_attempted"] == 4 and r["n_delivered"] == 2
    assert r["delivery_rate_pct"] == 50.0
    assert r["fallback_rate_pct"] == 50.0, "fallbacks entregados Y descartados (contrato B-04)"
    assert r["latency_all_s"]["n"] == 4 and r["latency_all_s"]["p50"] in (500.0, 700.0)
    assert r["latency_delivered_s"]["n"] == 2


# ---------------------------------------------------------------------------
# 3. Expectativas clínicas derivadas del formulario (para re-puntuar corpus reales)
# ---------------------------------------------------------------------------
def test_expectativas_derivadas_igualan_a_la_matriz_escrita_a_mano():
    for p in build_landing_profiles():
        assert derive_expectations(p) == p["_expect"], (p["_id"], p["_label"])


# ---------------------------------------------------------------------------
# 4. Reporte schema v2 compatible con el importador del landing
# ---------------------------------------------------------------------------
def test_reporte_schema_v2_con_run_nutrition_y_reliability():
    assert LANDING_BENCHMARK_SCHEMA_VERSION == 2
    for sec in ("run", "nutrition", "reliability"):
        assert sec in LANDING_REPORT_SECTIONS
    r = build_report("remote", run={"id": "x"}, nutrition={"aggregate": {}})
    assert r["schema_version"] == 2 and r["run"] == {"id": "x"}


def test_run_meta_trae_todo_lo_que_el_importador_exige():
    full = [p["_id"] for p in build_landing_profiles()]
    run = build_run_meta(
        mode="remote", started_at="2026-09-28T10:00:00Z", finished_at="2026-09-28T12:00:00Z",
        source_commit="2490e35abc", source_dirty=False, architecture="v2.2",
        protocol_version=LANDING_BENCHMARK_PROTOCOL_VERSION, country_scope=["DO"],
        profile_ids=full, full_profile_ids=full, parameters={"conc": 2})
    for k in _IMPORTER_RUN_FIELDS:
        assert k in run, k
    assert _ID_RE.fullmatch(run["id"]), run["id"]
    assert run["publication_status"] == "candidate"
    assert run["cohort_status"] == "complete" and run["profile_count"] == 25
    assert run["full_profile_count"] == 25

    parcial = build_run_meta(
        mode="remote", started_at="2026-09-28T10:00:00Z", finished_at="2026-09-28T12:00:00Z",
        source_commit=None, source_dirty=None, architecture="unspecified",
        protocol_version=LANDING_BENCHMARK_PROTOCOL_VERSION, country_scope=["DO"],
        profile_ids=full[:20], full_profile_ids=full, parameters={})
    assert parcial["cohort_status"] == "partial" and parcial["profile_count"] == 20
    corpus = build_run_meta(
        mode="score", started_at="2026-09-28T10:00:00Z", finished_at="2026-09-28T10:05:00Z",
        source_commit="2490e35", source_dirty=True, architecture="unspecified",
        protocol_version=LANDING_BENCHMARK_PROTOCOL_VERSION, country_scope=["DO", "MX"],
        profile_ids=["rd10__embarazo"], full_profile_ids=full, parameters={}, cohort="corpus")
    assert corpus["cohort_status"] == "not_applicable"


# ---------------------------------------------------------------------------
# 5. La matriz entera por defecto + workflows que caben
# ---------------------------------------------------------------------------
def test_por_defecto_corre_la_matriz_entera():
    mod = _load_runner()
    assert len(mod._select_profiles()) == 25
    assert len(mod._select_profiles(0)) == 25
    assert [p["_id"] for p in mod._select_profiles(ids={21, 25})] == [21, 25]


def test_los_workflows_no_cortan_la_matriz_en_20():
    for wf in (_WF_REMOTE, _WF_OPENAI):
        m = re.search(r"\n\s+n:\n\s+description:[^\n]*\n\s+default:\s*\"(\d+)\"", wf)
        assert m, "no encuentro el input `n` del workflow"
        assert m.group(1) == "0", "el default debe ser 0 = la matriz entera (25)"
        assert "los 20 de la matriz" not in wf


def test_dispatch_sin_ids_no_hereda_los_ids_del_smoke():
    """`ids || PUSH_RUN_IDS` corría también en un dispatch con `ids` vacío: «vacío = usar n»
    acababa corriendo los 5 ids del smoke. El dispatch lee SOLO sus inputs."""
    m = re.search(r'if \[ "\$\{\{ github\.event_name \}\}" = "workflow_dispatch" \]; then(.*?)\n\s+else',
                  _WF_REMOTE, re.DOTALL)
    assert m, "el paso del benchmark debe separar dispatch y push"
    assert "PUSH_RUN" not in m.group(1)
    assert "|| ''" not in _WF_REMOTE.split("Benchmark remote (guest) contra el deploy")[1].split(
        "Resumen al step summary")[0], "sin fallbacks `||` que mezclen dispatch con push"


def test_workflow_remote_con_conc_2_y_techo_que_cabe():
    assert re.search(r"--conc\s+2\b", _WF_REMOTE), "remote debe correr con --conc 2"
    techo = int(re.search(r"timeout-minutes:\s*(\d+)", _WF_REMOTE).group(1))
    assert 170 < techo <= 360, techo
    assert 'PUSH_RUN_N: "2"' in _WF_REMOTE   # el smoke barato del push sigue igual


def test_sin_nombres_de_modelo_muertos():
    for nombre, src in (("runner", _RUNNER_SRC), ("doc", _DOC), ("remote.yml", _WF_REMOTE),
                        ("openai.yml", _WF_OPENAI)):
        assert "gpt-5.6" not in src, f"{nombre} aún dice gpt-5.6"
        assert "cuota GLM" not in src, f"{nombre} aún habla de la cuota GLM"


# ---------------------------------------------------------------------------
# 6. Telemetría: solo lo entregado
# ---------------------------------------------------------------------------
def test_telemetria_solo_cuenta_lo_entregado():
    q = telemetry_queries(30)
    banda_sql = q["banda_entregada"][0]
    assert "clinical_band_final" in banda_sql and "pre-INSERT" in banda_sql and "chunk-T1" in banda_sql
    assert "assemble-tail" not in banda_sql
    lat_sql = q["generacion_latencia"][0]
    assert "delivered_was_fallback" in lat_sql and "<> 'true'" in lat_sql
    fb_sql = q["fallback_rate"][0]
    assert "meal_plans" in fb_sql and "_is_fallback" in fb_sql, (
        "el fallback ENTREGADO sale de los planes persistidos, no de las corridas del pipeline")
    assert "no_entregados" in q, "lo descartado se reporta aparte, no mezclado"
    for name, (sql, params) in q.items():
        assert params == (30,), name


# ---------------------------------------------------------------------------
# 7. Modo score: re-puntuar planes guardados (matriz y corpus) sin gastar IA
# ---------------------------------------------------------------------------
def test_score_de_un_corpus_con_formularios(tmp_path):
    mod = _load_runner()
    corpus = tmp_path / "cola"
    corpus.mkdir()
    (tmp_path / "rd9").mkdir()
    plan = _plan([[_meal(2000, 150, 200, 60, name="Arroz con pollo")]] * 3)
    (corpus / "rd9__hta.json").write_text(json.dumps({"final_plan": plan}), encoding="utf-8")
    form = {"country": "MX", "allergies": ["Ninguna"], "dietType": "balanced", "mainGoal": "lose_fat",
            "medicalConditions": ["Hipertensión"], "medications": ["Losartán"]}
    (tmp_path / "rd9" / "hta.json").write_text(json.dumps({"form": form, "final_plan": {}}),
                                               encoding="utf-8")
    sections, ctx = mod._score_sections(plans_glob=str(corpus / "*.json"), forms_root=str(tmp_path))
    assert ctx["cohort"] == "corpus" and ctx["profile_ids"] == ["rd9__hta"]
    assert ctx["country_scope"] == ["MX"]
    assert sections["safety"]["aggregate"]["n"] == 1
    assert sections["nutrition"]["aggregate"]["n_scored"] == 1
    assert sections["nutrition"]["aggregate"]["four_macros_in_band_pct"] == 100.0
    assert sections["gym"]["aggregate"]["n"] == 1
    assert sections["safety"]["per_profile"][0].get("professional_review_expected") is True


def test_score_de_planes_de_la_matriz_respeta_el_denominador(tmp_path):
    mod = _load_runner()
    plan = _plan([[_meal(2000, 150, 200, 60)]] * 2)
    f = tmp_path / "landing_plans.json"
    f.write_text(json.dumps({"attempted_ids": [1, 2, 3],
                             "plans": [{"id": 1, "label": "baseline_m", "plan": plan,
                                        "delivery": "delivered"},
                                       {"id": 2, "label": "baseline_f", "plan": {"_is_fallback": True},
                                        "delivery": "discarded_fallback"}]}), encoding="utf-8")
    sections, ctx = mod._score_sections(plans_path=str(f))
    assert ctx["cohort"] == "matrix" and ctx["profile_ids"] == [1, 2, 3]
    assert sections["nutrition"]["aggregate"]["n_attempted"] == 3
    assert sections["safety"]["aggregate"]["n"] == 1, "el fallback descartado no cuenta como entregado"


def test_marcadores():
    lb = (_BACKEND / "landing_benchmarks.py").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-749" in lb and "P1-PLAN-LOTE-749" in _RUNNER_SRC and "P1-PLAN-LOTE-749" in _DOC
