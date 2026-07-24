# Cómo funciona MealfitRD — diagramas del sistema

[SYSTEM-OVERVIEW · 2026-07-24] Vista de conjunto derivada del **código en HEAD**, no de
memoria. Cada diagrama cita los archivos que lo respaldan para poder re-verificarlo.

Docs relacionadas: [`llm_tier_routing.md`](llm_tier_routing.md) (routing/pricing),
[`coherence_surfaces_table.md`](coherence_surfaces_table.md) (guard de coherencia),
[`agent_tools_user_id_table.md`](agent_tools_user_id_table.md) (tools del chat),
[`dreaming_consolidation.md`](dreaming_consolidation.md) (memoria offline).

---

## 1. Vista general

```mermaid
flowchart TB
    U["Usuario · navegador / PWA iOS"]

    subgraph VPS["VPS Oracle ARM · Ubuntu 24.04 · nginx + TLS"]
        FE["Frontend React 19 + Vite<br/>SPA · service worker · Sentry"]
        BE["Backend FastAPI + uvicorn<br/>systemd: mealfit-backend"]
        CR["APScheduler<br/>~40 crons in-process"]
    end

    subgraph EXT["Servicios externos"]
        NEON[("Neon Postgres<br/>+ pgvector")]
        NAUTH["Neon Auth · Better Auth<br/>JWT EdDSA / JWKS"]
        DS["DeepSeek V4<br/>API OpenAI-compatible"]
        CO["Cohere embed-v4.0<br/>1536 dims"]
        PP["PayPal<br/>suscripciones + webhook"]
    end

    U -->|"HTTPS"| FE
    FE -->|"fetch + SSE · cookie __Host-mf_session"| BE
    FE -.->|"login / registro"| NAUTH
    BE -->|"verify_neon_jwt · JWKS"| NAUTH
    BE -->|"execute_sql_* · psycopg"| NEON
    BE -->|"generación / chat / revisor"| DS
    BE -->|"embeddings de user_facts"| CO
    BE <-->|"verify + webhook firmado"| PP
    CR --> NEON
    CR --> DS

    classDef ext fill:#1f2937,stroke:#4b5563,color:#e5e7eb
    class NEON,NAUTH,DS,CO,PP ext
```

**Reglas duras del borde** — el backend es la única frontera de confianza:
`auth.py::get_verified_user_id` valida firma server-side (P0-AUDIT-1), toda mutación
filtra `AND user_id = %s` (invariante I2), y el frontend **no escribe** a `meal_plans`
(invariante I6: solo endpoints con `jsonb_set` quirúrgico).

---

## 2. Generación de un plan — de la solicitud a la entrega

```mermaid
flowchart TB
    A["Usuario completa el formulario<br/>7 · 15 · 30 días"] --> B["POST /api/plans/analyze/stream<br/>SSE · fallback /analyze"]
    B --> C["arun_plan_pipeline<br/>fija user_id_var · resuelve tier"]
    C --> D["Grafo LangGraph<br/>PLAN_CHUNK_SIZE = 3 días por chunk"]

    subgraph G["Grafo · build_plan_graph"]
        direction TB
        N1["preflight_optimization"] --> N2["reflection"]
        N2 --> N3["context_compression"]
        N3 --> N4{"semantic_cache_check"}
        N4 -->|"hit"| N9["assemble_plan"]
        N4 -->|"miss"| N5["plan_skeleton"]
        N5 --> N6["generate_days_parallel<br/>map · días en paralelo"]
        N6 --> N7["adversarial_judge"]
        N7 --> N8["self_critique"]
        N8 --> N9
        N9 --> N10{"review_plan<br/>fact-checker + revisor médico"}
        N10 -->|"retry"| N11["retry_reflection"] --> N5
        N10 -->|"marker_regen"| N12["surgical_marker_regen"] --> N9
        N10 -->|"end"| FIN(["Plan aprobado o<br/>entregado degradado"])
    end

    D --> G
    FIN --> P["_save_plan_and_track_background<br/>INSERT meal_plans · el plan_id nace aquí"]
    P --> Q["SSE: plan al frontend"]
    Q --> R["Chunks 2..N encolados<br/>plan_chunk_queue"]
```

- `assemble_plan` construye `aggregated_shopping_list` y corre el **guard de coherencia**
  recetas ↔ lista; `review_plan` es quien lo consume y puede forzar retry.
- Si el revisor rechaza y se agota el presupuesto, el plan **se entrega igual** marcado
  `review_passed=False` → alert `plan_quality_degraded:<user>:<plan>` (invariante I5).
- Fuentes: [`graph_orchestrator.py::build_plan_graph`](../graph_orchestrator.py),
  [`services.py`](../services.py), [`constants.py`](../constants.py).

---

## 3. Planes largos: chunking en segundo plano

```mermaid
flowchart LR
    A["Plan de 15 o 30 días"] --> B["Chunk 1 · 3 días<br/>SÍNCRONO por SSE"]
    B --> C["Usuario ya ve su plan"]
    A --> D[("plan_chunk_queue<br/>chunks 2..N · status=pending")]
    D --> E["Cron process_plan_chunk_queue<br/>max_instances=1 · coalesce"]
    E --> F["_chunk_worker"]

    subgraph W["Worker · dos transacciones"]
        F --> T1["T1 · FOR UPDATE<br/>merge days + shopping list<br/>+ learning_metrics"]
        T1 --> T2["T2 · guard de coherencia<br/>+ status=completed"]
    end

    T2 --> G{"¿faltan chunks?"}
    G -->|"sí"| E
    G -->|"no"| H["generation_status = complete"]

    I["Crons centinela<br/>stuck · dead-letter · lag · zombie"] -.->|"vigilan"| D
```

Estados canónicos de `plan_chunk_queue.status` y sus transiciones viven en
[`cron_tasks.py`](../cron_tasks.py) (state-machine documentada en
`process_plan_chunk_queue`). El CHECK de DB `meal_plans_complete_requires_days`
(invariante I8) impide marcar `complete` con `days = 0`.

---

## 4. Qué modelo se usa y cuándo

```mermaid
flowchart TB
    A["Llamada LLM"] --> B{"¿knob per-feature<br/>MEALFIT_&lt;FEATURE&gt;_MODEL?"}
    B -->|"sí · siempre gana"| Z["Modelo del knob<br/>rollback / A-B sin redeploy"]
    B -->|"no"| C{"¿superficie clínica?"}
    C -->|"revisor médico risk-tier"| PRO1["deepseek-v4-pro FIJO<br/>thinking ON · effort medium"]
    C -->|"fact-checker clínico"| PRO2["thinking ON · effort high<br/>bind_tools nativo"]
    C -->|"aux barato<br/>títulos · RAG · nudges · resúmenes"| FL1["deepseek-v4-flash FIJO"]
    C -->|"pipeline de plan / chat"| D{"plan_tier del usuario"}
    D -->|"gratis · guest · NULL · fallo de lookup"| FL2["deepseek-v4-flash<br/>$0.14 in · $0.28 out /1M"]
    D -->|"basic · plus · ultra"| PRO3["deepseek-v4-pro<br/>$0.435 in · $0.87 out /1M"]

    classDef pro fill:#1e3a5f,stroke:#3b82f6,color:#dbeafe
    classDef fl fill:#14532d,stroke:#22c55e,color:#dcfce7
    class PRO1,PRO2,PRO3 pro
    class FL1,FL2 fl
```

Dos invariantes que este diagrama codifica:

- **fail-cheap**: cualquier duda (guest, blip de DB, tier corrupto) resuelve a FLASH —
  un fallo de lookup jamás encarece la llamada.
- **la seguridad clínica no se degrada por plan de pago**: el revisor médico de riesgo
  va a PRO para *todos* los tiers, incluido el gratis.

El *thinking mode* nativo de V4 está **OFF global** y se re-activa solo en superficies de
juicio con output chico; en generación grande revienta el timeout. Detalle y mediciones
A/B: [`llm_tier_routing.md`](llm_tier_routing.md).

---

## 5. Vida del plan después de entregado

```mermaid
flowchart TB
    P["Plan en meal_plans<br/>plan_data JSONB"]

    P --> SL["Lista de compras agregada<br/>con precios y costo del ciclo"]
    SL --> NE["Nevera Inteligente<br/>user_inventory"]
    NE -->|"POST /restock · ya compré"| NE
    NE -->|"POST /inventory/consume"| NE
    NE -.->|"snapshot de despensa"| REN["Renovación pantry-aware"]

    P --> SW["POST /swap-meal → /swap-meal/persist<br/>cambiar una comida"]
    P --> RD["POST /{id}/regenerate-day"]
    P --> RE["POST /recipe/expand<br/>receta completa"]
    P --> SH["POST /shift-plan<br/>mueve la ventana rodante"]
    P --> PDF["PDF del plan + lista"]
    P --> HI["Historial · /history-list"]

    SW --> RC["Recalculo de agregados<br/>+ guard de coherencia post-swap"]
    RD --> RC
    RE --> RC
    RC --> P

    classDef ep fill:#1f2937,stroke:#6b7280,color:#e5e7eb
    class SW,RD,RE,SH,PDF,HI ep
```

Todas esas flechas son **endpoints backend**: el cliente nunca escribe `plan_data`
(invariante I6). Las escrituras full-overwrite van bajo advisory lock o
`update_plan_data_atomic` (`SELECT … FOR UPDATE` + callback) para cerrar la ventana
lost-update (invariante I7).

Los endpoints de mantenimiento (`/restock`, `/inventory/consume`, `/shift-plan`,
`/recalculate-shopping-list`, historial) están **exentos del paywall** a propósito: cero
costo LLM, y cobrarlos congelaba funciones de un plan ya pagado. Se protegen con
`RateLimiter` por bucket, no con el cap mensual.

---

## 6. Chat del agente

```mermaid
flowchart TB
    M["Mensaje del usuario<br/>texto o foto de comida"] --> A["Grafo del chat<br/>agent.py"]

    subgraph G["chat_builder"]
        CM["call_model<br/>modelo por tier"] --> RT{"route_tools"}
        RT -->|"tool_calls"| ET["execute_tools"]
        ET --> CM
        RT -->|"sin tools"| END(["Respuesta"])
    end

    A --> G
    ET -.->|"P0-AGENT-1<br/>user_id FORZADO del JWT"| T["11 tools<br/>nevera · consumos · perfil · micros"]
    CM -.->|"RAG"| F[("user_facts + pgvector<br/>Cohere embed-v4")]
    CM -.->|"perfil consolidado"| DR["user_memory_profile<br/>dreaming nocturno"]
    M -.->|"foto"| VI["Visión local gemma<br/>vía túnel Ollama · costo 0"]
```

El override de `user_id` en `execute_tools` es la defensa central: el modelo **recibe**
el `user_id` en el prompt, pero eso es *prompt-trustable, no enforced* — una inyección
podría emitir un `tool_call` con identidad ajena, así que el nodo lo sobrescribe con el
valor autenticado antes de invocar cualquier tool.

---

## 7. Trabajo de fondo (crons)

```mermaid
flowchart LR
    S["APScheduler<br/>register_plan_chunk_scheduler"] --> A["Chunks<br/>procesar · rescatar · GC"]
    S --> B["Coherencia<br/>métricas horarias + alerta diaria"]
    S --> C["Memoria<br/>dreaming · drenar facts · resúmenes"]
    S --> D["Salud del sistema<br/>deploy lag · circuit breaker · pool"]
    S --> E["Producto<br/>nudges · KPIs · benchmark nocturno"]

    A --> AL[("system_alerts")]
    B --> AL
    C --> AL
    D --> AL
    E --> AL
    AL --> OP["Operador: /health/version<br/>y endpoints admin"]
```

Una alert "vive" mientras `resolved_at IS NULL`; el catálogo completo de ~32 `alert_key`
con su productor y su modelo de resolución está en
[`system_alerts_resolution_table.md`](system_alerts_resolution_table.md).

---

## Cómo re-verificar este documento

| Diagrama | Fuente de verdad |
|---|---|
| 2 · grafo | `grep -n "add_node\|add_edge\|add_conditional_edges" backend/graph_orchestrator.py` |
| 3 · chunking | `PLAN_CHUNK_SIZE` en [`constants.py`](../constants.py) + `_chunk_worker` en [`cron_tasks.py`](../cron_tasks.py) |
| 4 · modelos | [`llm_provider.py`](../llm_provider.py) (`DEEPSEEK_FLASH` / `DEEPSEEK_PRO`) + [`llm_tier_routing.md`](llm_tier_routing.md) |
| 5 · endpoints | `grep -n "@router.post\|@router.get" backend/routers/plans.py` |
| 6 · chat | `chat_builder` al final de [`agent.py`](../agent.py) |
| 7 · crons | `grep -on "id=[\"'][a-z0-9_]*[\"']" backend/cron_tasks.py` |
