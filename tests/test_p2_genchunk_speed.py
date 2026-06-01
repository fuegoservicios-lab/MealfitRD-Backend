"""[P2-GENCHUNK-SPEED · 2026-06-01] Anclas del audit de velocidad (sin perder
calidad) de los 3 subsistemas pedidos: generación de planes, chunks regenerativos
con aprendizaje continuo, y agente IA.

Workflow find→verify (34 slices grep-first, 56 agentes, verificación adversaria):
22 hallazgos verificados → 12 implement, 7 defer, 3 reject. La lente fue
latencia/tokens de LLM (el cuello real a escala diminuta — meal_plans≈4 filas hace
los N+1 de DB teóricos) bajo la restricción dura "sin perder calidad".

Implementados (9 — A=P2, resto P3; D diferido por riesgo en sección locked-merge):
  A  prune del `plan_data` crudo (4 shopping arrays + 2 coherence keys + _archived_days
     + calc_household_multiplier) antes de serializarlo al system prompt del chat
     EN CADA turno (agent.py ambos paths). Dead-context puro.
  B  `_safe_ainvoke`: el emit de usage-event (INSERT psycopg bloqueante) ya NO corre
     síncrono en el event loop — fire-and-forget vía `_submit_best_effort_metric`.
  C  título creativo del plan (LLM Flash-Lite, hasta 30s tail) diferido fuera del
     critical path de time-to-plan-visible en el chunked save (placeholder + UPDATE bg).
  E  `calculate_plan_quality_score` hoisteado FUERA del lock FOR UPDATE en
     `_persist_nightly_learning_signals` + dedup del doble-fetch likes/rejections en retro.
  F  cold-start: `get_latest_meal_plan()` gateado en `_quality_data_sufficient`
     (ambos consumidores ya AND-gatean ese flag).
  G  `modify_single_meal`: pasar el `plan_data` ya mergeado por `update_plan_data_atomic`
     en el tool result → `execute_tools` no re-SELECTea el plan recién escrito.
  H  preámbulo del chat: sentiment+rag_router (fase 1) y text+multimodal embeddings
     (fase 2) corren concurrentes (eran seriales) en ambos paths del chat.
  I  prompt del reviewer: eliminado el dump plano de ingredientes duplicado
     (`--- TODOS LOS INGREDIENTES DEL PLAN ---`) — `all_meals_summary` ya los lista.
  J  catálogo de precios gateado en señal de presupuesto en el prompt del planner.

Diferido (D): pre-LLM `_merged_chunk_ids` guard en el chunk worker. La verificación
manual (lección del repo "el veredicto del workflow sobreestima") halló que el
fast-path propuesto tropieza con el guard GAP E de consistencia de conteo (L~28618)
ANTES de llegar al merged-check (L~28634), y un fix correcto exige mover la detección
de merge ADENTRO de la sección locked del merge — refactor race-critical sobre el
worker (incidente I8: status=complete+days=0 vivió ~14h en prod). Riesgo > recompensa
para un path de excepción raro (P3). Diferido con razón documentada.

Cada test parsea el SOURCE de prod (sin import/DB — el entorno es DB-less) para que
un refactor que revierta un fix falle ANTES de degradar prod.

Detalle: ~/.claude/projects/.../memory/project_p2_genchunk_speed_2026_06_01.md
Tooltip-anchor: P2-GENCHUNK-SPEED.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_AGENT_PY = _BACKEND_ROOT / "agent.py"
_TOOLS_PY = _BACKEND_ROOT / "tools.py"
_GRAPH_PY = _BACKEND_ROOT / "graph_orchestrator.py"
_CRON_PY = _BACKEND_ROOT / "cron_tasks.py"
_SERVICES_PY = _BACKEND_ROOT / "services.py"
_APP_PY = _BACKEND_ROOT / "app.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _slice_fn(src: str, header: str) -> str:
    """Cuerpo de la función `header` hasta el siguiente def top-level."""
    start = src.find(header)
    assert start >= 0, f"no se encontró {header!r}"
    after = src[start + len(header):]
    nxt = re.search(r"\n(?:def |async def )\w", after)
    return after[: nxt.start()] if nxt else after


def _code_only(src: str) -> str:
    """Quita comentarios `#` para que las aserciones NEGATIVAS no matcheen el
    propio comentario explicativo del fix (lección repetida del repo)."""
    out = []
    for line in src.splitlines():
        h = line.find("#")
        out.append(line if h < 0 else line[:h])
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Marker: el slug del marker debe resolver a este archivo + no retroceder.
# ---------------------------------------------------------------------------
def test_marker_floor_2026_06_01():
    app = _read(_APP_PY)
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"[^"]*(\d{4})-(\d{2})-(\d{2})"', app)
    assert m, "No se encontró _LAST_KNOWN_PFIX con fecha parseable."
    assert (int(m.group(1)), int(m.group(2)), int(m.group(3))) >= (2026, 6, 1), (
        "El marker _LAST_KNOWN_PFIX retrocedió antes de 2026-06-01 (P2-GENCHUNK-SPEED)."
    )


# ---------------------------------------------------------------------------
# A (P2): prune del plan_data crudo en el system prompt del chat (ambos paths).
# ---------------------------------------------------------------------------
def test_a_chat_plan_prune_keys_and_helper():
    src = _read(_AGENT_PY)
    # El set de claves a podar contiene las 8 derivadas/pesadas.
    expected = [
        "aggregated_shopping_list",
        "aggregated_shopping_list_weekly",
        "aggregated_shopping_list_biweekly",
        "aggregated_shopping_list_monthly",
        "_shopping_coherence_block",
        "_shopping_coherence_block_history",
        "_archived_days",
        "calc_household_multiplier",
    ]
    block = src[src.find("_CHAT_PLAN_PRUNE_KEYS"): src.find("_CHAT_PLAN_PRUNE_KEYS") + 600]
    for k in expected:
        assert f'"{k}"' in block, f"A: clave {k!r} ausente de _CHAT_PLAN_PRUNE_KEYS."
    assert "def _prune_plan_for_chat(" in src, "A: falta el helper _prune_plan_for_chat."


def test_a_both_chat_paths_use_prune():
    src = _read(_AGENT_PY)
    # Ambos callsites deben serializar el plan PODADO, no el crudo.
    assert src.count("json.dumps(_prune_plan_for_chat(current_plan))") >= 2, (
        "A: ambos paths del chat deben usar json.dumps(_prune_plan_for_chat(current_plan))."
    )
    # Y NO debe quedar un json.dumps(current_plan) crudo.
    assert "json.dumps(current_plan)}" not in src, (
        "A: quedó un json.dumps(current_plan) crudo sin podar en el system prompt del chat."
    )


# ---------------------------------------------------------------------------
# B (P3): usage-event emit fire-and-forget (no bloquea el event loop).
# ---------------------------------------------------------------------------
def test_b_submit_best_effort_metric_defined():
    src = _read(_GRAPH_PY)
    assert "def _submit_best_effort_metric(" in src, "B: falta el helper _submit_best_effort_metric."
    helper = _slice_fn(src, "def _submit_best_effort_metric(")
    assert "_METRICS_EXECUTOR.submit" in helper, "B: el helper debe despachar al _METRICS_EXECUTOR."


def test_b_safe_ainvoke_emit_offloaded():
    src = _read(_GRAPH_PY)
    body = _slice_fn(src, "async def _safe_ainvoke(")
    # El success path despacha el emit vía el helper off-loop.
    assert "_submit_best_effort_metric(" in body, (
        "B: _safe_ainvoke ya no usa _submit_best_effort_metric — el emit volvió a bloquear el loop."
    )
    # Y NO debe quedar la llamada síncrona directa al emit de usage-event.
    code = _code_only(body)
    assert "_emit_llm_usage_event_best_effort(\n            llm=llm" not in code, (
        "B: quedó la llamada SÍNCRONA _emit_llm_usage_event_best_effort en el success path."
    )


# ---------------------------------------------------------------------------
# C (P3): título creativo diferido en el chunked save path.
# ---------------------------------------------------------------------------
def test_c_defer_title_helpers_and_knob():
    src = _read(_SERVICES_PY)
    assert "def _deterministic_plan_title_placeholder(" in src, "C: falta el placeholder determinista."
    assert "def _defer_creative_plan_title(" in src, "C: falta el helper de título diferido."
    assert "MEALFIT_DEFER_PLAN_TITLE" in src, "C: falta el knob MEALFIT_DEFER_PLAN_TITLE."


def test_c_deferred_update_is_guarded():
    src = _read(_SERVICES_PY)
    body = _slice_fn(src, "def _defer_creative_plan_title(")
    # UPDATE escalar de la columna name, guardado contra rename del usuario + user_id.
    assert "UPDATE meal_plans SET name = %s" in body, "C: el título diferido debe UPDATE-ar la columna name."
    assert "AND user_id = %s" in body, "C: el UPDATE diferido debe filtrar AND user_id (I2)."
    assert "AND name = %s" in body, (
        "C: el UPDATE diferido debe guardar AND name = <placeholder> para no pisar un rename del usuario."
    )


def test_c_save_partial_uses_placeholder_under_knob():
    body = _slice_fn(_read(_SERVICES_PY), "def save_partial_plan_get_id(")
    assert "_deterministic_plan_title_placeholder(plan_data)" in body, (
        "C: save_partial_plan_get_id debe usar el placeholder determinista bajo el knob."
    )
    assert "_defer_creative_plan_title(" in body, "C: debe disparar el título creativo en background."


# ---------------------------------------------------------------------------
# E (P3): quality score fuera del lock + dedup likes/rejections.
# ---------------------------------------------------------------------------
def test_e_quality_score_signature_accepts_prefetched():
    src = _read(_CRON_PY)
    sig = src[src.find("def calculate_plan_quality_score("): src.find("def calculate_plan_quality_score(") + 260]
    assert "recent_likes" in sig and "recent_rejections" in sig, (
        "E: calculate_plan_quality_score debe aceptar recent_likes/recent_rejections para evitar el doble-fetch."
    )
    # El fetch interno debe ser condicional al param.
    fn = _slice_fn(src, "def calculate_plan_quality_score(")
    assert "recent_likes if recent_likes is not None" in fn, (
        "E: el fetch de likes debe reusar recent_likes cuando se pasa."
    )


def test_e_quality_score_computed_outside_lock():
    body = _slice_fn(_read(_CRON_PY), "def _persist_nightly_learning_signals(")
    call_idx = body.find("calculate_plan_quality_score(")
    lock_idx = body.find("with connection_pool.connection()")
    assert call_idx >= 0, "E: no se encontró la llamada a calculate_plan_quality_score."
    assert lock_idx >= 0, "E: no se encontró el bloque transaccional FOR UPDATE."
    assert call_idx < lock_idx, (
        "E: calculate_plan_quality_score debe invocarse FUERA del lock FOR UPDATE "
        "(antes de `with connection_pool.connection()`), no dentro — alargaba el hold-time."
    )
    # Y adentro del lock NO debe re-invocarse (solo persistir el valor ya computado).
    in_lock = body[lock_idx:]
    assert "calculate_plan_quality_score(" not in in_lock, (
        "E: calculate_plan_quality_score NO debe re-invocarse dentro del lock."
    )


# ---------------------------------------------------------------------------
# F (P3): cold-start fetch gateado en _quality_data_sufficient.
# ---------------------------------------------------------------------------
def test_f_cold_start_fetch_gated():
    body = _slice_fn(_read(_CRON_PY), "def _inject_advanced_learning_signals(")
    assert "get_latest_meal_plan(user_id) if _quality_data_sufficient else None" in body, (
        "F: el fetch get_latest_meal_plan del attribution tracker debe gatearse en _quality_data_sufficient "
        "(en cold-start ambos consumidores lo descartan)."
    )


# ---------------------------------------------------------------------------
# G (P3): modify_single_meal reusa el plan_data ya mergeado (no re-SELECT).
# ---------------------------------------------------------------------------
def test_g_tool_returns_merged_plan_data():
    body = _slice_fn(_read(_TOOLS_PY), "def execute_modify_single_meal(")
    assert '_resp["plan_data"] = merged_plan_data' in body, (
        "G: execute_modify_single_meal debe incluir el plan_data ya mergeado en su return."
    )


def test_g_agent_uses_inband_plan_before_refetch():
    src = _read(_AGENT_PY)
    # El branch modify usa el plan_data in-band antes de caer al re-fetch.
    assert 'parsed_mod.get("plan_data")' in src, (
        "G: execute_tools debe usar parsed_mod.get('plan_data') antes de re-SELECTear."
    )
    # El re-fetch sigue presente como fallback (back-compat) — no se elimina del todo.
    assert "get_latest_meal_plan_with_id(" in src, "G: el fallback get_latest_meal_plan_with_id debe conservarse."


# ---------------------------------------------------------------------------
# H (P3): preámbulo del chat paralelizado (sentiment+router, embeddings).
# ---------------------------------------------------------------------------
def test_h_stream_preamble_parallel():
    body = _slice_fn(_read(_AGENT_PY), "def chat_with_agent_stream(")
    # Fase 1: sentiment + rag_router concurrentes.
    assert "ThreadPoolExecutor(max_workers=2)" in body, "H: falta el executor de paralelización en el stream."
    assert "submit(classify_sentiment, prompt)" in body, "H: classify_sentiment debe submitirse al executor (fase 1)."
    assert "submit(rag_query_router, prompt)" in body, "H: rag_query_router debe submitirse al executor (fase 1)."
    # Fase 2: embeddings text + multimodal concurrentes.
    assert "_rag_text_unit" in body and "_rag_visual_unit" in body, (
        "H: los embeddings text/multimodal deben correr como unidades concurrentes en el stream."
    )


def test_h_nonstream_embeddings_parallel():
    body = _slice_fn(_read(_AGENT_PY), "def chat_with_agent(")
    assert "_rag_text_unit" in body and "_rag_visual_unit" in body, (
        "H: el path non-stream debe paralelizar los embeddings text/multimodal."
    )
    assert "ThreadPoolExecutor(max_workers=2)" in body, "H: falta el executor de embeddings en el non-stream."


# ---------------------------------------------------------------------------
# I (P3): reviewer prompt sin el dump plano duplicado de ingredientes.
# ---------------------------------------------------------------------------
def test_i_reviewer_prompt_no_duplicate_ingredient_dump():
    src = _read(_GRAPH_PY)
    # `_code_only` strip de comentarios: el comentario del fix cita la cadena
    # eliminada a propósito — la aserción negativa debe mirar solo el CÓDIGO.
    code = _code_only(src)
    assert "--- TODOS LOS INGREDIENTES DEL PLAN ---" not in code, (
        "I: el bloque duplicado '--- TODOS LOS INGREDIENTES DEL PLAN ---' debe estar eliminado del review_prompt."
    )
    assert "P3-GENCHUNK-SPEED-REVIEWER-DEDUP" in src, "I: falta el tooltip-anchor del dedup del reviewer."
    # all_meals_summary (la vista agrupada por comida) sigue presente.
    assert "all_meals_summary" in src, "I: la vista agrupada all_meals_summary debe conservarse."


# ---------------------------------------------------------------------------
# J (P3): catálogo de precios gateado en señal de presupuesto.
# ---------------------------------------------------------------------------
def test_j_prices_context_gated_on_budget():
    src = _read(_GRAPH_PY)
    assert 'build_prices_context() if (str(form_data.get("budget") or "").strip())' in src, (
        "J: build_prices_context debe gatearse en la señal de presupuesto (mismo predicado que build_budget_context)."
    )
    assert "P3-GENCHUNK-SPEED-PRICES-GATE" in src, "J: falta el tooltip-anchor del gate de precios."
