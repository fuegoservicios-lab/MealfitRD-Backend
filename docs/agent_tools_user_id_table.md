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
| 2 | `log_consumed_meal` | `db_log_consumed_meal` + `deduct_consumed_meal_from_inventory` |
| 3 | `search_deep_memory` | leak de summaries cross-user |
| 4 | `check_shopping_list` | leak de pantry/plan cross-user |
| 5 | `check_current_pantry` | leak de pantry cross-user |
| 6 | `modify_pantry_inventory` | `add_or_update_inventory_item` + `deduct_consumed_meal_from_inventory` |
| 7 | `mark_shopping_list_purchased` | `restock_inventory` |
| 8 | `check_hydration_today` | leak de `hydration_log` cross-user (read-only) |
| 9 | `log_water_glass` | INSERT/UPDATE en `hydration_log` cross-user |
| 10 | `suggest_foods_for_nutrient` | leak del `health_profile` cross-user (lee alergias/dislikes/dieta del `user_id` para filtrar la sugerencia del catálogo) |
| 11 | `check_clinical_profile` | leak del `health_profile.clinical_profile` cross-user (laboratorios, historial de peso — el dato más sensible del sistema) (P1-CHAT-CLINICAL-TOOL · 2026-07-12) |

### Retiradas temporalmente del set activo (P1-CHAT-PLAN-TOOLS-OFF · 2026-07-12)

Decisión del owner: el chat NO muta el plan por ahora. Las tools siguen definidas en `tools.py`
(y cubiertas por el override P0-AGENT-1 si vuelven) pero solo se anexan a `agent_tools` con
`MEALFIT_CHAT_PLAN_TOOLS_ENABLED=true` (+ restart). Sin formato de fila numerada a propósito —
el parser de paridad de `test_p2_chat_cleanup.py` solo cuenta filas `| n | tool |`.

- **generate_new_plan_from_chat** — pipeline completo + `save_new_meal_plan_robust`
- **modify_single_meal** — `update_meal_plan_data` (full plan_data overwrite)
- **regenerate_full_day** — invoca `api_regenerate_day` con `verified_user_id=user_id` — sin el override, regeneraría (y COBRARÍA 1 crédito) el día del plan de una víctima (P1-CHAT-DAY-REGEN-TOOL · 2026-07-12)

## Cómo verificar

```bash
# Override + funcional (P0-AGENT-1, pre-existente):
pytest backend/tests/test_p0_agent_1_user_id_override.py -v

# Paridad bidireccional tabla ↔ tools.py (P2-CHAT-CLEANUP, nuevo):
pytest backend/tests/test_p2_chat_cleanup.py -v
```

## Telemetría + si añades tool nueva

El override emite `WARN [P0-AGENT-1]` con `tool=/llm_user_id=/trusted=` para identificar prompt-injection attempts en logs. **Si añades una tool nueva a `agent_tools`**, el override ya la cubre automáticamente (no requiere cambio en `execute_tools`); SÍ requiere añadir entry en la tabla arriba — el test `test_p2_chat_cleanup.py` falla si la paridad se rompe. Si la tool nueva acepta otra identidad sensitiva (e.g. `session_id`), añadir override análogo + branch correspondiente en `test_p0_agent_1_user_id_override.py`. Narrativa de cierre del audit: [`runbook_security_antipatterns.md`](~/.claude/projects/.../memory/runbook_security_antipatterns.md).
