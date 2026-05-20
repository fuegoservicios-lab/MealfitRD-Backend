# Agent tools cubiertas por el override `user_id` (P0-AGENT-1)

> Tabla canónica movida de CLAUDE.md (P2-CHAT-CLEANUP · 2026-05-20) por presión de tamaño + drift histórico (la tabla decía "9 tools" cuando `tools.py::agent_tools` ya exportaba 11 — gap #6 del audit prod-readiness 2026-05-20). CLAUDE.md mantiene header + 1-line + link. El test parser-based [`test_p2_chat_cleanup.py`](../tests/test_p2_chat_cleanup.py) enforza paridad bidireccional: si añades una tool a `agent_tools` SIN entry acá (o viceversa), el test falla.

[P0-AGENT-1 · 2026-05-11 · paridad enforzada P2-CHAT-CLEANUP · 2026-05-20]

El nodo LangGraph `execute_tools` ([`backend/agent.py`](../agent.py)) force-overridea `tool_args["user_id"] = _trusted_uid` para CADA tool_call ANTES de cualquier branch específico del if/elif/else. Cubre TODAS las tools de `agent_tools` ([`backend/tools.py::agent_tools`](../tools.py)) porque el override es genérico al tope del loop — no depende de que cada branch nuevo se "acuerde" de hacerlo.

**Razón**: la LLM recibe el `user_id` autenticado en plano dentro del system prompt vía `build_tools_instructions(user_id)` ([`prompts/chat_agent.py:128, 148`](../prompts/chat_agent.py#L128)). Eso es **prompt-trustable, NO enforced**. Una entrada adversaria del usuario (mensaje hostil, contenido importado vía `vision_agent`, recetas externas) puede inducir a la LLM a emitir `tool_call` con `user_id` ajeno, abriendo IDOR cross-user sobre `user_inventory`, `consumed_meals`, `user_facts`, `health_profile`, `meal_plans`, `hydration_log`.

Es la simétrica de las invariantes I2/I6 (filtros server-side `AND user_id = %s` en SQL + endpoints backend que no aceptan user_id arbitrario del cliente) aplicada al chat-agent layer.

## Las 11 tools cubiertas

| # | Tool | Mutación cross-user que el override impide |
|---|---|---|
| 1 | `update_form_field` | `update_user_health_profile_atomic` + `delete_user_facts_by_metadata` |
| 2 | `generate_new_plan_from_chat` | pipeline completo + `save_new_meal_plan_robust` |
| 3 | `log_consumed_meal` | `db_log_consumed_meal` + `deduct_consumed_meal_from_inventory` |
| 4 | `modify_single_meal` | `update_meal_plan_data` (full plan_data overwrite) |
| 5 | `search_deep_memory` | leak de summaries cross-user |
| 6 | `check_shopping_list` | leak de pantry/plan cross-user |
| 7 | `check_current_pantry` | leak de pantry cross-user |
| 8 | `modify_pantry_inventory` | `add_or_update_inventory_item` + `deduct_consumed_meal_from_inventory` |
| 9 | `mark_shopping_list_purchased` | `restock_inventory` |
| 10 | `check_hydration_today` | leak de `hydration_log` cross-user (read-only) |
| 11 | `log_water_glass` | INSERT/UPDATE en `hydration_log` cross-user |

## Cómo verificar

```bash
# Override + funcional (P0-AGENT-1, pre-existente):
pytest backend/tests/test_p0_agent_1_user_id_override.py -v

# Paridad bidireccional tabla ↔ tools.py (P2-CHAT-CLEANUP, nuevo):
pytest backend/tests/test_p2_chat_cleanup.py -v
```

## Telemetría + si añades tool nueva

El override emite `WARN [P0-AGENT-1]` con `tool=/llm_user_id=/trusted=` para identificar prompt-injection attempts en logs. **Si añades una tool nueva a `agent_tools`**, el override ya la cubre automáticamente (no requiere cambio en `execute_tools`); SÍ requiere añadir entry en la tabla arriba — el test `test_p2_chat_cleanup.py` falla si la paridad se rompe. Si la tool nueva acepta otra identidad sensitiva (e.g. `session_id`), añadir override análogo + branch correspondiente en `test_p0_agent_1_user_id_override.py`. Narrativa de cierre del audit: [`runbook_security_antipatterns.md`](~/.claude/projects/.../memory/runbook_security_antipatterns.md).
