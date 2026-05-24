# Runbooks operacionales

> Anchor: P3-RUNBOOK-CONSOLIDATION · 2026-05-12.

Procedimientos paso-a-paso para SRE / operador on-call. Estos archivos
viven en el repo (no solo en la memoria local de Claude) para que el
operador pueda consultarlos directamente desde GitHub sin depender de
la IA.

## Cuándo consultar cada runbook

| Síntoma | Runbook |
|---|---|
| Alert `scheduler_cascade_missed` no se resuelve / scheduler saturado / hay que decidir qué cron sacrificar | [cron_priority_and_synthetic_cleanup.md](cron_priority_and_synthetic_cleanup.md) |
| Plan Sintético% acumulados en prod (cron `_sweep_synthetic_test_plans` no llegó) | [cron_priority_and_synthetic_cleanup.md § P3-LIVE-12](cron_priority_and_synthetic_cleanup.md#p3-live-12--sop-plan-sintético-cleanup-manual) |
| Alert `plan_data_corrupted:<plan_id>:<field>` (manual) | [system_alerts_sops.md](system_alerts_sops.md) |
| Alert `deploy_lag_drift_vs_expected` (auto pero requiere acción humana) | [system_alerts_sops.md § deploy_lag_drift](system_alerts_sops.md#sop-resolver-deploy_lag_drift_vs_expected) |
| Limpieza one-shot de alerts huérfanas pre-deploy | [system_alerts_sops.md § Limpieza](system_alerts_sops.md#limpieza-one-shot-de-alerts-huérfanas) |
| Circuit breaker LLM stale (fila `is_open=true` por >24h, modelo OFF) | [llm_circuit_breaker_kv_lifecycle.md](llm_circuit_breaker_kv_lifecycle.md) |
| Reset manual de una fila `llm_circuit_breaker:<model>` | [llm_circuit_breaker_kv_lifecycle.md § SOP-2](llm_circuit_breaker_kv_lifecycle.md#sop-2-forzar-reset-manual-de-una-fila-cb) |
| Migración Supabase falló durante apply / regresión post-apply / data corruption | [migration_rollback.md](migration_rollback.md) |
| Validar pool DB bajo carga antes de lanzamiento / diagnosticar saturación | [db_pool_load_test.md](db_pool_load_test.md) |
| Reproducir/migrar imagen productiva (EasyPanel → Fly.io / k8s / local) | [dockerfile_deployment.md](dockerfile_deployment.md) |
| Auditar cobertura de auth por endpoint / añadir nuevo endpoint público | [endpoint_auth_coverage.md](endpoint_auth_coverage.md) |
| Diagnosticar "cache devuelve stale" / SOP de invalidación manual Redis | [cache_invalidation_policy.md](cache_invalidation_policy.md) |

## Convenciones

1. **Rutas de archivos**: las referencias `backend/file.py:N` son
   relativas a la raíz del repo. Para clickearlas en GitHub, navegá
   desde la root del repo. En render local (VSCode preview) los links
   pueden no resolver — eso es cosmético, no funcional.

2. **Sincronización con CLAUDE.md**: estos runbooks son el detalle largo
   de secciones referenciadas en `../CLAUDE.md`. Si un cambio operacional
   afecta el SOP (e.g., nuevo knob, nuevo alert_key), actualizar AMBOS:
   el runbook + la tabla resumen en CLAUDE.md.

3. **Nada en estos runbooks reemplaza la lectura del código fuente**.
   Si el SOP dice `_atomic_reset_db` y el código fue refactorizado, el
   código manda — actualizar el runbook como parte del refactor.

## Cómo añadir un runbook nuevo

Cuando una sección de CLAUDE.md pase de >50 líneas de SOPs detallados,
considerar moverla acá:

1. Crear `<topic>.md` con el detalle completo.
2. Reducir la sección en CLAUDE.md a 1 párrafo + tabla de knobs/keys +
   link `[ver runbook](backend/docs/runbooks/<topic>.md)`.
3. Si hay tests parser-based que escanean CLAUDE.md por anchors
   específicos, verificar que los anchors mínimos sigan en CLAUDE.md
   tras el trim (el test atrapa drift).
4. Añadir entry a la tabla "Cuándo consultar" de este README.
