# Surfaces que escriben `aggregated_shopping_list*` y status del guard

> Tabla canónica movida de CLAUDE.md (P3-NEXT-5 · 2026-05-11) por presión de tamaño. CLAUDE.md mantiene el header + marker + menciones de cada `action_taken` canónico para que el test parser-based `test_p3_next_5_coherence_surfaces_table.py` siga verde. El detalle completo (modo guard por surface, condiciones de retry, notas operacionales) vive aquí.

[P3-NEXT-5 · 2026-05-11] Tabla canónica de **dónde se ejecuta el coherence guard** (cuándo se construye o se modifica `aggregated_shopping_list*` y qué garantías ofrece cada surface). Esta es la tabla "negativa" que faltaba: enumera explícitamente los surfaces que NO bloquean (solo telemetría warn) vs los que SÍ bloquean (retry forzado), para que un futuro refactor no asuma que el guard es universal.

| # | Surface | Archivo | Modo guard | Bloquea retry? | `action_taken` | Notas |
|---|---|---|---|---|---|---|
| 1 | `assemble_plan_node` (LangGraph full pipeline, planes ≤7d) | [`graph_orchestrator.py:6185+`](../backend/graph_orchestrator.py#L6185) | `block` (default P1-NEW-1) | **Sí** — `_shopping_coherence_block` consumido por `review_plan_node:7704` | `not_applicable` (warn) / `degrade` / `reject_minor` / `reject_high` (block) | Única surface en el pipeline LangGraph que puede forzar retry vía `should_retry`. |
| 2 | `_recompute_aggregates_after_swap` (post-swap revalidation) | [`graph_orchestrator.py:7860+`](../backend/graph_orchestrator.py#L7860) | `warn` | **No** (post-review) | `post_swap_revalidation` + opcional `post_swap_critical_alerted` si escala a `system_alerts` | Telemetría P2-B + alert per-user con cooldown 6h (P2-2). |
| 3 | `_chunk_worker` T2 (multi-week chunked plans) | [`cron_tasks.py:22678+`](../backend/cron_tasks.py#L22678) | `warn` con `block_severe_only=True` | **Sí (selectivo)** — escala warn→block si hay divergencias severas (cap_swallowed_modifier o magnitudes >50%); reusa el retry loop existente `_SHOP_MAX_RETRIES` con backoff. NO escala otras divergencias (unknown extras, magnitudes <50%) | `warn_only_chunk_t2` (history persiste igual) | P1-NEXT-2 + **P2-COHERENCE-1** (2026-05-11). Knob `MEALFIT_COHERENCE_T2_BLOCK_SEVERE_ONLY` (default True) kill switch. |
| 4 | `/recalculate-shopping-list` (Pantry add/delete + Dashboard) | [`routers/plans.py:4006+`](../backend/routers/plans.py#L4006) | `warn` | **No** — caller síncrono, 400 rompería UX | `warn_only_recalc` | P1-NEXT-2. Telemetría para post-mortem si recurrente; cliente sigue viendo lista usable. |
| 5 | `tools.modify_single_meal` (agent tool swap) | [`tools.py:514+`](../backend/tools.py#L514) | `warn` | **No** — agente ya entregó respuesta; bloquear caro en tokens | `warn_only_agent_tool` | P1-NEXT-2. |
| 6 | Cron diario `_shopping_coherence_alert_job` (04:00 UTC) | [`cron_tasks.py:685+`](../backend/cron_tasks.py#L685) | `warn` (forzado vía `mode_override='warn'`) | **No** — NO setea `_shopping_coherence_block` retroactivo sobre planes ya entregados | `warn_only_cron_daily` | P2-NEXT-2. Persiste history para planes legacy pre-P1-NEXT-2. Knob `MEALFIT_COHERENCE_CRON_PERSIST_HISTORY` kill switch. |

**Conclusiones operacionales:**
- Solo el flujo `assemble_plan_node → review_plan_node` (surface #1) puede generar retries del LLM por divergencias críticas.
- Las 5 surfaces auxiliares (#2-6) emiten **solo telemetría**. Si una divergencia escapó al assemble (por ejemplo: cache hit reusó plan con shape vieja), las surfaces auxiliares la capturan post-hoc en `_shopping_coherence_block_history`, NO la corrigen.
- Si futuras divergencias en multi-week chunks justifican retry (no es el caso hoy: T2 ya retry con backoff), revisar este contrato antes de cambiar mode a `block` en surface #3 — el cambio implica decidir qué hacer cuando T2 falla N veces (re-encolar? abandonar? alertar?). Está OUT-OF-SCOPE del audit 2026-05-11.

**Tests anchor:**
- [`test_p1_next_2_guard_at_persist_sites.py`](../backend/tests/test_p1_next_2_guard_at_persist_sites.py) — surfaces #3, #4, #5 (helper invocation).
- [`test_p2_next_2_cron_persists_coherence_history.py`](../backend/tests/test_p2_next_2_cron_persists_coherence_history.py) — surface #6 (helper + persist + knob).
- [`test_p3_next_4_coherence_metrics_surface_breakdown.py`](../backend/tests/test_p3_next_4_coherence_metrics_surface_breakdown.py) — buckets agregados por surface en P3-B.
- [`test_p3_next_5_coherence_surfaces_table.py`](../backend/tests/test_p3_next_5_coherence_surfaces_table.py) — drift detection bidireccional (CLAUDE.md menciona los 6 action_taken canónicos + cada uno emite en código).
