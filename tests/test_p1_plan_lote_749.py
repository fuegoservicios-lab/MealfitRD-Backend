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
     `assemble-tail` (lectura INTERMEDIA, pre-review: 269 de 408 filas en 30 días) y
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
    # [ronda 1] El cuerpo REAL del 503 del FALLBACK-GUARD (JSON de FastAPI), no un texto inventado:
    # la clasificación ahora mira el `detail` (ver test_clasificacion_de_errores_remotos).
    assert classify_remote_error('HTTP 503 en /api/plans/analyze: {"detail":"La IA está '
                                 'temporalmente saturada y no pudimos generar tu plan."}'
                                 ) == "discarded_fallback"
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
    # [ronda 1] Antes solo se pedía que la latencia filtrara `delivered_was_fallback`: eso fijaba el
    # defecto (entraban bloques en segundo plano y reintentos no entregados). Ahora la latencia sale
    # de las corridas EMPAREJADAS con una entrega (test_latencia_solo_de_corridas_con_entrega_y_por_tipo).
    lat_sql = q["generacion_latencia"][0]
    assert "FROM pares p" in lat_sql and "ON pe.corrida_id = c.id" in lat_sql
    fb_sql = q["fallback_rate"][0]
    assert "meal_plans" in fb_sql and "_is_fallback" in fb_sql, (
        "el fallback ENTREGADO sale de los planes persistidos, no de las corridas del pipeline")
    assert "corridas_por_entrega" in q and "banda_excluida_sin_corrida" in q, (
        "lo no entregado se reporta aparte, no mezclado")
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


# ===========================================================================
# RONDA 1 DE LA REVISIÓN ADVERSARIA (2026-09-28). Cada test reproduce un defecto que la revisión
# encontró en la primera versión de esta rama — y fallaba contra ella.
# tooltip-anchor: P1-PLAN-LOTE-749-R1
# ===========================================================================
import shutil
import subprocess


def _run_step(wf: str, name: str) -> str:
    """Cuerpo (hasta el siguiente `- name:`/`- uses:`) del paso `name` de un workflow."""
    tail = wf.split(f"- name: {name}", 1)[1]
    return re.split(r"\n\s+- (?:name|uses):", tail, maxsplit=1)[0]


# --- Defecto 1 (CRÍTICO): la corrida pagada salía «sucia» y el importador la rechazaba ------------
def test_workflows_no_escriben_en_el_arbol_antes_de_medir_git_status():
    """`| tee run_stdout.txt` creaba un fichero SIN trackear en la raíz del checkout mientras el
    runner arrancaba y medía `git status --porcelain`: `source_dirty=True` en 3 de 3 corridas, y el
    importador del landing rechaza «la corrida proviene de un worktree sucio». El tee va fuera del
    repo ($RUNNER_TEMP) y el resumen lee de ahí."""
    for nombre, wf in (("remote.yml", _WF_REMOTE), ("openai.yml", _WF_OPENAI)):
        targets = re.findall(r"\|\s*tee\s+(\S+)", wf)
        assert targets, f"{nombre}: no encuentro el tee del stdout"
        for t in targets:
            assert t.startswith('"$RUNNER_TEMP/'), f"{nombre}: tee a {t} ensucia el checkout"
        assert re.search(r'tail -\d+ "\$RUNNER_TEMP/run_stdout\.txt"', wf), (
            f"{nombre}: el resumen debe leer el stdout de $RUNNER_TEMP")


def test_las_salidas_del_benchmark_no_ensucian_git_status(tmp_path):
    """Funcional: un repo con el `.gitignore` del backend + las salidas que escribe el benchmark
    (`--out`, `--save-plans`, el stdout) sigue LIMPIO para `_git_source_info`; un fichero ajeno no."""
    if not shutil.which("git"):
        pytest.skip("git no disponible")
    mod = _load_runner()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text((_BACKEND / ".gitignore").read_text(encoding="utf-8"),
                                     encoding="utf-8")
    (repo / "x.py").write_text("x = 1\n", encoding="utf-8")
    g = ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t",
         "-c", "commit.gpgsign=false"]
    subprocess.run(g + ["init", "-q"], check=True)
    subprocess.run(g + ["add", "-A"], check=True)
    subprocess.run(g + ["commit", "-q", "-m", "base"], check=True)
    for f in ("run_stdout.txt", "landing_benchmark_remote.json", "landing_benchmark_openai.json",
              "landing_plans_4242.json"):
        (repo / f).write_text("{}", encoding="utf-8")
    commit, dirty = mod._git_source_info(str(repo))
    assert re.fullmatch(r"[0-9a-f]{40}", commit or "")
    assert dirty is False, "las salidas propias del benchmark no pueden marcar la corrida como sucia"
    (repo / "otro.py").write_text("y = 2\n", encoding="utf-8")
    assert mod._git_source_info(str(repo))[1] is True, "un cambio ajeno SÍ ensucia"


def test_workflow_openai_declara_la_arquitectura():
    """Sin input `architecture`, toda corrida OpenAI salía `unspecified` — que el importador rechaza."""
    assert re.search(r"\n\s+architecture:\n\s+description:", _WF_OPENAI)
    paso = _run_step(_WF_OPENAI, "Landing benchmark — live --provider openai (perfiles guest)")
    assert "--architecture" in paso


def test_los_inputs_no_se_interpolan_en_el_script_del_shell():
    """El input `architecture` iba sin comillas dentro del `run:` (`--architecture $ARCH` tras
    `${{ github.event.inputs.architecture }}`): texto libre del dispatch ejecutado como shell. Los
    inputs viajan por `env:` y se citan."""
    for nombre, wf, paso in (
            ("remote.yml", _WF_REMOTE, "Benchmark remote (guest) contra el deploy"),
            ("openai.yml", _WF_OPENAI, "Landing benchmark — live --provider openai (perfiles guest)")):
        cuerpo = _run_step(wf, paso)
        script = cuerpo.split("run: |", 1)[1]
        assert "github.event.inputs" not in script, f"{nombre}: input interpolado en el script"
        assert re.search(r'--architecture "\$', script), f"{nombre}: --architecture sin comillas"
        assert "set -o pipefail" in script, f"{nombre}: sin pipefail, `| tee` esconde un fallo del runner"


def test_arquitectura_se_valida():
    mod = _load_runner()
    base = ["remote", "--api-base", "https://x"]
    assert mod._parse_args(base + ["--architecture", "v2.2"]).architecture == "v2.2"
    assert mod._parse_args(base + ["--architecture", "v1"]).architecture == "v1"
    assert mod._parse_args(base).architecture == "unspecified"
    with pytest.raises(SystemExit):
        mod._parse_args(base + ["--architecture", "v3; rm -rf /"])


def test_run_parameters_solo_los_del_modo():
    """`run.parameters` arrastraba `transport`, `timeout`, `days` y `provider` en corridas
    `score`/`telemetry`: parámetros que esa corrida no usó."""
    mod = _load_runner()
    p = mod._run_parameters(mod._parse_args(["score", "--plans-glob", "/tmp/c/*.json",
                                             "--forms-root", "/tmp"]))
    assert p == {"plans_glob": "/tmp/c/*.json", "forms_root": "/tmp"}
    t = mod._run_parameters(mod._parse_args(["telemetry", "--days", "7"]))
    assert t == {"days": 7}
    r = mod._run_parameters(mod._parse_args(["remote", "--api-base", "https://x", "--changes"]))
    assert r["conc"] == 1 and r["n"] == 0 and r["changes"] is True and r["transport"] == "sse"
    assert "days" not in r and "provider" not in r and "plans_glob" not in r
    lv = mod._run_parameters(mod._parse_args(["live", "--provider", "openai"]))
    assert lv["conc"] == 2 and lv["provider"] == "openai" and "api_base" not in lv


# --- Defectos 2 y 3 (IMPORTANTES): la telemetría contaba filas que no son entregas ----------------
def _placeholders(sql: str) -> int:
    return len(re.findall(r"(?<!%)%s", sql))


def test_banda_entregada_exige_una_corrida_real_detras():
    """Medido en prod (solo lectura, 28-sep): de 81 filas `pre-INSERT` en 30 días, 67 (todas del
    2-7 sep) no tenían corrida `clinical_band` ni `meal_plans` detrás (no eran entregas: todas con
    `user_id` NULL y sesión `post-finalize`, que es lo que deja cualquier llamada a
    `_finalize_plan_data_for_insert` fuera de una generación). La media 0,954 salía casi toda de
    ellas (emparejadas y tipadas por superficie —ronda 2—: plan inicial n=14, media 0,905). Cada
    fila de banda se empareja ahora con la corrida del pipeline que la produjo (mismo usuario —
    para `pre-INSERT`, vía el `meal_plans` de ese usuario—, ≤5 min antes); las que no tienen
    corrida se cuentan APARTE."""
    q = telemetry_queries(30)
    sql = q["banda_entregada"][0]
    assert "JOIN LATERAL" in sql and "node = 'clinical_band'" in sql and "meal_plans" in sql
    assert "FROM pares" in sql
    assert "banda_excluida_sin_corrida" in q
    assert "NOT EXISTS" in q["banda_excluida_sin_corrida"][0]


def test_latencia_solo_de_corridas_con_entrega_y_por_tipo():
    """49 corridas contra 14 planes iniciales + 12 merges T1: entraban los bloques del chunk worker
    y reintentos que nunca se entregaron. La latencia cuenta solo corridas con una entrega
    emparejada y separa plan inicial de bloque posterior — por la SUPERFICIE de la entrega (ronda 2:
    la sesión `unknown` solo apartaba los `rolling_refill`; ver test_tipo_de_entrega_por_superficie)."""
    q = telemetry_queries(30)
    sql = q["generacion_latencia"][0]
    assert "FROM pares p" in sql and "bloque_posterior" in sql and "plan_inicial" in sql
    from landing_benchmarks import _delivery_pairing_ctes
    assert "session_id" not in sql.replace(_delivery_pairing_ctes(), ""), (
        "el tipo no sale de la sesión de la corrida")
    assert "no_entregados" not in q, "el nombre mentía: contaba corridas CON fallback, no no-entregas"
    cpe = q["corridas_por_entrega"][0]
    for col in ("entregadas_plan_inicial", "entregadas_bloque_posterior", "sin_entrega",
                "con_fallback", "sin_usuario"):
        assert col in cpe, col


def test_cada_consulta_de_telemetria_tiene_un_solo_parametro():
    for name, (sql, params) in telemetry_queries(30).items():
        assert params == (30,), name
        assert _placeholders(sql) == 1, (name, _placeholders(sql))


# --- Defecto 4 (IMPORTANTE latente): el modo corpus no veía las alergias escritas a mano ---------
def test_corpus_une_el_texto_libre_como_produccion(tmp_path):
    """Producción une `otherAllergies`/`otherConditions` a sus listas (`_merge_other_text_fields`);
    el modo corpus armaba el perfil con el formulario crudo y `score_plan_safety` solo lee
    `allergies`: camarones a quien escribió «frutos del mar» salía «seguro»."""
    mod = _load_runner()
    corpus = tmp_path / "cola"
    corpus.mkdir()
    (tmp_path / "rdX").mkdir()
    camarones = {"name": "Camarones al ajillo", "meal": "Almuerzo", "cals": 2000,
                 "protein": "150g", "carbs": "200g", "fats": "60g",
                 "ingredients": ["250 g de camarones cocidos"], "recipe": ["Saltear."]}
    plan = _plan([[camarones]])
    for nombre, alergias in (("texto", []), ("centinela", ["Ninguna"])):
        (corpus / f"rdX__{nombre}.json").write_text(json.dumps({"final_plan": plan}), encoding="utf-8")
        (tmp_path / "rdX" / f"{nombre}.json").write_text(json.dumps({"form": {
            "allergies": alergias, "otherAllergies": "frutos del mar", "dietType": "balanced",
            "mainGoal": "maintenance", "medicalConditions": [], "medications": []}}), encoding="utf-8")
    sections, ctx = mod._score_sections(plans_glob=str(corpus / "*.json"), forms_root=str(tmp_path))
    per = {p["profile_id"]: p for p in sections["safety"]["per_profile"]}
    assert per["rdX__texto"]["safe"] is False, "«frutos del mar» escrito a mano debe cazar camarones"
    # Con el centinela «Ninguna» producción DESCARTA el texto (P0-FORM-1): el scorer sigue esa regla,
    # pero la corrida lo deja a la vista en vez de callarlo.
    assert per["rdX__centinela"]["safe"] is True
    assert ctx["texto_libre_descartado"] == ["rdX__centinela"]


def test_expectativas_del_corpus_con_texto_libre():
    from landing_benchmarks import corpus_profile
    prof = corpus_profile({"allergies": [], "otherAllergies": "Maní"}, "p1")
    assert prof["_expect"]["allergens"] == ["Maní"] and prof["_id"] == "p1"
    assert prof["_free_text_discarded"] == []
    prof2 = corpus_profile({"allergies": ["Ninguna"], "otherAllergies": "Maní"}, "p2")
    assert "allergens" not in prof2["_expect"] and prof2["_free_text_discarded"] == ["otherAllergies"]


# --- Defecto 5 (IMPORTANTE): en remote, `source_commit` no es el motor medido ----------------------
def test_version_del_servidor_conserva_git_sha(monkeypatch):
    import httpx
    mod = _load_runner()

    class _R:
        status_code = 200

        def json(self):
            return {"git_sha": "2490e35abcdef0123456789", "git_short_sha": "2490e35",
                    "last_known_pfix": "P1-PLAN-LOTE-765 · 2026-09-28", "drift": False,
                    "deploy_timestamp": "2026-09-28T10:00:00Z", "knobs_sample": {"x": 1}}

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _R())
    v = mod._server_version("https://x")
    assert v["git_sha"] == "2490e35abcdef0123456789" and v["deploy_timestamp"]
    assert "knobs_sample" not in v


def test_remote_captura_el_servidor_al_inicio_y_al_final(monkeypatch):
    mod = _load_runner()
    llamadas = []
    versiones = iter([{"git_sha": "aaaaaaa1", "last_known_pfix": "A"},
                      {"git_sha": "aaaaaaa1", "last_known_pfix": "A"}])

    def _sv(api_base):
        llamadas.append("version")
        return next(versiones)

    def _one(api_base, p, *a, **k):
        llamadas.append("gen")
        return {"id": p["_id"], "label": p["_label"], "delivery": "error", "duration_s": 1.0,
                "error": "x"}

    monkeypatch.setattr(mod, "_server_version", _sv)
    monkeypatch.setattr(mod, "_run_one_remote", _one)
    sections, ctx = mod._remote_sections("https://x", 2, 1, False, None, 10)
    assert llamadas[0] == "version" and llamadas[-1] == "version", llamadas
    sv = sections["meta"]["server_version"]
    assert sv["inicio"]["git_sha"] == "aaaaaaa1" and sv["fin"]["git_sha"] == "aaaaaaa1"
    assert ctx["engine"]["engine_commit"] == "aaaaaaa1"
    assert ctx["source_commit_role"] == "scorers"


def test_identidad_del_motor():
    from landing_benchmarks import engine_identity
    ok = engine_identity({"git_sha": "abc1234"}, {"git_sha": "abc1234"})
    assert ok == {"engine_commit": "abc1234", "engine_commit_status": "verified"}
    unk = engine_identity({"git_sha": "unknown"}, {"git_sha": "unknown"})
    assert unk["engine_commit"] is None and unk["engine_commit_status"] == "not_exposed"
    cambio = engine_identity({"git_sha": "abc1234"}, {"git_sha": "def5678"})
    assert cambio["engine_commit"] is None and cambio["engine_commit_status"] == "changed_during_run"
    assert engine_identity(None, None)["engine_commit_status"] == "unreachable"


def test_run_meta_dice_de_quien_es_el_commit():
    full = [p["_id"] for p in build_landing_profiles()]
    run = build_run_meta(
        mode="remote", started_at="2026-09-28T10:00:00Z", finished_at="2026-09-28T12:00:00Z",
        source_commit="2490e35abc", source_dirty=False, architecture="v2.2",
        protocol_version=LANDING_BENCHMARK_PROTOCOL_VERSION, country_scope=["DO"],
        profile_ids=full, full_profile_ids=full, parameters={},
        source_commit_role="scorers",
        engine={"engine_commit": None, "engine_commit_status": "not_exposed"})
    assert run["source_commit_role"] == "scorers"
    assert run["engine_commit"] is None and run["engine_commit_status"] == "not_exposed"
    live = build_run_meta(
        mode="live", started_at="2026-09-28T10:00:00Z", finished_at="2026-09-28T12:00:00Z",
        source_commit="2490e35abc", source_dirty=False, architecture="v2.2",
        protocol_version=LANDING_BENCHMARK_PROTOCOL_VERSION, country_scope=["DO"],
        profile_ids=full, full_profile_ids=full, parameters={})
    assert live["source_commit_role"] == "engine_and_scorers"
    assert live["engine_commit"] == "2490e35abc" and live["engine_commit_status"] == "in_process"


# --- Defecto 6 (MENOR): 422/503 del síncrono no son todos fallbacks -------------------------------
@pytest.mark.parametrize("msg, esperado", [
    ('RuntimeError: HTTP 422 en /api/plans/analyze: {"detail":"No pudimos generar un plan que '
     'respete tus restricciones declaradas"} | diag: {"fallback_reason": "x"}', "discarded_fallback"),
    ('RuntimeError: HTTP 503 en /api/plans/analyze: {"detail":"La IA está temporalmente saturada y '
     'no pudimos generar tu plan."}', "discarded_fallback"),
    ('RuntimeError: HTTP 503 en /api/plans/analyze: {"detail":"El servicio de IA no está disponible '
     'en este momento."}', "discarded_fallback"),
    ('RuntimeError: HTTP 422 en /api/plans/analyze: {"detail":{"code":"missing_required_fields",'
     '"missing_fields":["age"]}}', "rejected_request"),
    ('RuntimeError: HTTP 422 en /api/plans/analyze: {"detail":{"code":"invalid_biometric_range"}}',
     "rejected_request"),
    ('RuntimeError: HTTP 422 en /api/plans/analyze: {"detail":{"code":"budget_insufficient"}}',
     "rejected_request"),
    ('RuntimeError: HTTP 422 en /api/plans/analyze: {"detail":{"code":"too_many_medical_conditions",'
     '"max":3}}', "rejected_request"),
    ('RuntimeError: HTTP 422 en /api/plans/analyze: {"detail":{"code":"clinical_scope_exceeded"}}',
     "rejected_request"),
    ('RuntimeError: HTTP 503 en /api/plans/analyze: {"detail":"Generamos tu plan pero no pudimos '
     'guardarlo por un problema temporal."}', "error"),
    ('RuntimeError: HTTP 503 en /api/plans/analyze: {"detail":{"code":"server_busy_generating"}}',
     "error"),
    ("RuntimeError: HTTP 503 en /api/plans/analyze: <html><body><h1>503 Service Temporarily "
     "Unavailable</h1></body></html>", "error"),
    ("RuntimeError: SSE error code=plan_persist_failed: Generamos tu plan pero...", "error"),
    ("RuntimeError: SSE error code=llm_unavailable_fallback: x", "error"),
    ("RuntimeError: SSE error code=critical_restriction: x", "discarded_fallback"),
])
def test_clasificacion_de_errores_remotos(msg, esperado):
    assert classify_remote_error(msg) == esperado


def test_rechazo_de_la_peticion_cuenta_aparte():
    r = aggregate_reliability([{"delivery": "delivered", "duration_s": 1.0},
                               {"delivery": "rejected_request", "duration_s": 0.2}])
    assert r["n_rejected_request"] == 1 and r["n_delivered"] == 1
    assert r["fallback_rate_pct"] == 0.0 and r["delivery_rate_pct"] == 50.0


# --- Defecto 7 (MENOR): copias del motor sin test de igualdad --------------------------------------
def test_copias_del_motor_iguales_al_motor():
    import graph_orchestrator as go
    import landing_benchmarks as lb
    assert lb._GAINMUSCLE_GOAL_TOKENS == go._GAINMUSCLE_GOAL_TOKENS
    for x in ("154g", "464 kcal", None, "", "abc", 12, 12.5, "nan", "inf", " 30 G ", True):
        assert lb._macro_num(x) == go._meal_macro_num(x), x
    for goal in ("Ganancia Muscular (Superávit 8%)", "Pérdida de Grasa", "gain_muscle", "BULK",
                 "Mantenimiento", None, ""):
        plan = {"main_goal": goal}
        assert lb._goal_is_gain_muscle(plan) == go._plan_goal_is_gainmuscle(plan), goal


def test_kcal_solo_de_cals_como_el_motor():
    """El motor suma `cals`; la copia sumaba `calories` cuando faltaba `cals` — otro número."""
    from graph_orchestrator import compute_clinical_band_score
    meal = {"name": "x", "calories": 2000, "protein": "150g", "carbs": "200g", "fats": "60g"}
    plan = _plan([[meal]])
    eng = compute_clinical_band_score(plan, {})
    mine = score_plan_nutrition(plan, band=engine_band_definition())
    assert mine["four_macros_in_band_days"] == eng["all4_days"] == 0
    assert mine["per_macro"]["kcal"]["in_band"] == 0


# --- Defecto 8 (MENOR): la doc reproduce el replay tal como se hizo --------------------------------
def test_doc_del_replay_reproducible():
    assert "cola744*/" not in _DOC, "ese glob mete también `_rec` (67 duplicados: 493 ficheros)"
    assert "481/481" not in _DOC, "481 contaba dos veces los 67 del corpus reciente; únicos = 414"
    assert "cero GLM" not in _DOC and "cero GLM" not in _WF_OPENAI, (
        "no se afirma «cero GLM» mientras reviewer/day-gen/swap dependen de los knobs del entorno")



# ===========================================================================
# RONDA 2 DE LA REVISIÓN (2026-09-28). Cada test reproduce lo que la re-verificación encontró en la
# ronda 1 — y fallaba contra ella (`b981255c`).
# tooltip-anchor: P1-PLAN-LOTE-749-R2
# ===========================================================================
import sqlite3


def _sqlite_case(case_sql: str, surface: str):
    """Evalúa la expresión CASE que va al SQL de producción con un `p.surface` dado. Usa solo
    SQL estándar (`IN`, `LIKE`), así que sqlite la evalúa igual que Postgres (el `%%` de psycopg es
    en sqlite dos comodines seguidos: el mismo patrón)."""
    con = sqlite3.connect(":memory:")
    try:
        return con.execute(f"SELECT {case_sql} FROM (SELECT ? AS surface) p", (surface,)).fetchone()[0]
    finally:
        con.close()


# --- R2-1 (IMPORTANTE): el tipo de entrega sale de la SUPERFICIE, no de la sesión ------------------
@pytest.mark.parametrize("surface, tipo", [
    ("pre-INSERT", "plan_inicial"),
    ("chunk-T1 semana 1", "plan_inicial"),
    ("chunk-T1 semana 2", "bloque_posterior"),
    ("chunk-T1 semana 3", "bloque_posterior"),
    ("chunk-T1 semana 10", "bloque_posterior"),
    ("chunk-T1 semana 12", "bloque_posterior"),
])
def test_tipo_de_entrega_por_superficie(surface, tipo):
    """`session_id='unknown'` solo aparta los `rolling_refill` (su form no lleva sesión). Medido en
    prod por la re-verificación: los 5 «planes iniciales entregados por chunk-T1» eran bloques de
    semana 2-3 con `chunk_kind='initial_plan'` (los días 8-30 del horizonte, CON sesión), generados
    3-8 días después de crear el plan. El bloque 1 por la cola (`chunk_kind='initial'`) se rellena
    por `fill_placeholder_meal_plan_atomic` → superficie `pre-INSERT`."""
    from landing_benchmarks import delivery_kind, delivery_kind_sql
    assert delivery_kind(surface) == tipo
    assert _sqlite_case(delivery_kind_sql("p.surface"), surface) == tipo


def test_telemetria_no_clasifica_por_sesion():
    from landing_benchmarks import delivery_kind_sql
    q = telemetry_queries(30)
    for name in ("banda_entregada", "generacion_latencia", "corridas_por_entrega"):
        assert "session_id = 'unknown' THEN 'bloque_posterior'" not in q[name][0], name
    assert delivery_kind_sql("p.surface") in q["banda_entregada"][0]
    assert delivery_kind_sql("pe.surface") in q["generacion_latencia"][0], (
        "la latencia se tipa por la superficie de la ENTREGA emparejada con la corrida")
    assert "DISTINCT ON (p.corrida_id)" in q["generacion_latencia"][0], (
        "una corrida cuenta una vez en la latencia aunque casara con dos filas")


def test_corridas_por_entrega_nombra_lo_que_mide():
    """R2-1 + R2-4: sin entrega no hay superficie, así que una corrida sin entrega no tiene tipo que
    se pueda probar. Las filas se agrupan por lo que SÍ se mide (`origen`: sin usuario / sesión
    `unknown` / con sesión) y el tipo va en columnas, solo para las que tienen entrega. La
    categoría `invitado` solo significaba `user_id IS NULL`: sus 7 filas caían en la misma ventana
    del 2-7 sep que las de scripts."""
    cpe = telemetry_queries(30)["corridas_por_entrega"][0]
    assert "'invitado'" not in cpe
    assert "'sin_usuario'" in cpe and "AS origen" in cpe
    for col in ("entregadas_plan_inicial", "entregadas_bloque_posterior", "sin_entrega",
                "con_fallback"):
        assert col in cpe, col


def test_doc_y_comentarios_sin_la_clasificacion_por_sesion():
    lb = (_BACKEND / "landing_benchmarks.py").read_text(encoding="utf-8")
    for nombre, src in (("doc", _DOC), ("landing_benchmarks.py", lb)):
        assert "un plan inicial también se entrega por" not in src, nombre
    assert "n=19, media 0,904" not in _DOC and "0,977" not in _DOC


# --- R2-2 (MENOR): `cambio_durante_la_corrida` con `git_sha:"unknown"` -----------------------------
_V0 = {"git_sha": "unknown", "git_short_sha": "unknown", "deploy_timestamp": "unknown",
       "process_started_at": "2026-09-28T10:00:00+00:00",
       "last_known_pfix": "P1-PLAN-LOTE-765 · 2026-09-28"}


def test_cambio_de_servidor_sin_git_sha():
    from landing_benchmarks import server_change_during_run
    r = server_change_during_run(_V0, dict(_V0, process_started_at="2026-09-28T11:30:00+00:00"))
    assert r["cambio"] is True and r["claves"] == ["process_started_at"]
    r = server_change_during_run(_V0, dict(_V0, last_known_pfix="P1-PLAN-LOTE-766 · 2026-09-28"))
    assert r["cambio"] is True and r["claves"] == ["last_known_pfix"]
    assert server_change_during_run(_V0, dict(_V0)) == {"cambio": False, "claves": []}
    # Nada comparable (todo «unknown») o un extremo ilegible: NO se puede afirmar que no cambió.
    assert server_change_during_run({"git_sha": "unknown"}, {"git_sha": "unknown"})["cambio"] is None
    assert server_change_during_run({"error": "HTTP 502"}, _V0)["cambio"] is None
    assert server_change_during_run(None, _V0)["cambio"] is None


def test_remote_marca_un_redeploy_a_mitad_aunque_no_haya_git_sha(monkeypatch):
    mod = _load_runner()
    versiones = iter([_V0, dict(_V0, process_started_at="2026-09-28T11:30:00+00:00",
                               last_known_pfix="P1-PLAN-LOTE-766 · 2026-09-28")])
    monkeypatch.setattr(mod, "_server_version", lambda api_base: next(versiones))
    monkeypatch.setattr(mod, "_run_one_remote", lambda api_base, p, *a, **k: {
        "id": p["_id"], "label": p["_label"], "delivery": "error", "duration_s": 1.0, "error": "x"})
    sections, ctx = mod._remote_sections("https://x", 1, 1, False, None, 10)
    sv = sections["meta"]["server_version"]
    assert sv["cambio_durante_la_corrida"] is True
    assert sv["claves_que_cambiaron"] == ["last_known_pfix", "process_started_at"]
    assert ctx["engine"]["engine_commit_status"] == "not_exposed"


# --- R2-3 (MENOR): ningún input del dispatch dentro de un script del shell --------------------------
def test_ningun_input_del_dispatch_se_interpola_en_un_script():
    """`BASE="${{ github.event.inputs.api_base }}"` seguía en el paso «Descubrir API base»; el test
    de la ronda 1 solo miraba el paso del benchmark. Todo input viaja por `env:` (una línea
    `NOMBRE: ${{ ... }}`), en TODO el workflow."""
    env_line = re.compile(r"\s+[A-Z][A-Z0-9_]*:\s*\$\{\{\s*(?:github\.event\.inputs|inputs)\.\w+\s*\}\}\s*")
    for nombre, wf in (("remote.yml", _WF_REMOTE), ("openai.yml", _WF_OPENAI)):
        for ln in wf.splitlines():
            if re.search(r"\$\{\{\s*(?:github\.event\.inputs|inputs)\.", ln):
                assert env_line.fullmatch(ln), f"{nombre}: input interpolado fuera de env: {ln.strip()}"
    paso = _run_step(_WF_REMOTE, "Descubrir API base del deploy")
    assert 'BASE="$IN_API_BASE"' in paso


# --- R2-5 (MENOR): con SSE, un rechazo real del stream no se reenvía al síncrono --------------------
class _StreamResp:
    def __init__(self, status, body, ctype="application/json"):
        self.status_code = status
        self.headers = {"content-type": ctype}
        self.text = body

    def read(self):
        return self.text.encode("utf-8")

    def iter_lines(self):
        return iter(())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _run_sse(monkeypatch, status, body, ctype="application/json"):
    import httpx
    mod = _load_runner()
    sync = []

    def _post(api_base, path, payload, timeout_s, *a, **k):
        sync.append(path)
        raise RuntimeError("HTTP 500 en /api/plans/analyze: sync-llamado")

    monkeypatch.setattr(httpx, "stream", lambda *a, **k: _StreamResp(status, body, ctype))
    monkeypatch.setattr(mod, "_remote_post", _post)
    prof = build_landing_profiles()[0]
    row = mod._run_one_remote("https://x", prof, False, 10, transport="sse", band=None)
    return row, sync


@pytest.mark.parametrize("status, body, esperado", [
    (503, '{"detail":{"code":"server_busy_generating","message":"Estamos generando muchos planes"}}',
     "error"),
    (422, '{"detail":{"code":"missing_required_fields","missing_fields":["age"]}}', "rejected_request"),
    (422, '{"detail":{"code":"budget_insufficient"}}', "rejected_request"),
    (502, "<html><body>502 Bad Gateway</body></html>", "error"),
])
def test_sse_no_reenvia_un_rechazo_del_stream_al_sincrono(monkeypatch, status, body, esperado):
    """El síncrono no tiene el tope de concurrencia (`server_busy_generating` solo existe en el
    stream, routers/plans.py): reenviar ahí se salta el tope y cuenta una SEGUNDA generación. Solo
    se cae al síncrono si el deploy no sirve el stream (404/405/501 o un 200 que no es SSE)."""
    row, sync = _run_sse(monkeypatch, status, body)
    assert sync == [], f"reenviado al síncrono: {sync}"
    assert row["delivery"] == esperado, row


@pytest.mark.parametrize("status, ctype", [(404, "application/json"), (405, "application/json"),
                                           (200, "text/html")])
def test_sse_cae_al_sincrono_solo_si_no_hay_stream(monkeypatch, status, ctype):
    row, sync = _run_sse(monkeypatch, status, '{"detail":"Not Found"}', ctype)
    assert sync == ["/api/plans/analyze"]


@pytest.mark.parametrize("msg, esperado", [
    ('RuntimeError: HTTP 503 en /api/plans/analyze/stream: {"detail":{"code":"server_busy_generating"}}',
     "error"),
    ('RuntimeError: HTTP 422 en /api/plans/analyze/stream: {"detail":{"code":"budget_insufficient"}}',
     "rejected_request"),
])
def test_clasificacion_de_errores_del_stream(msg, esperado):
    assert classify_remote_error(msg) == esperado
