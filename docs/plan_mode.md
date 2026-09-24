# Modo seguimiento (`plan_mode`) — la app sin generar planes

[P1-PLAN-MODE · 2026-08-11] Este párrafo vivía completo en `CLAUDE.md` y se movió aquí en una pasada doc-first
(política `P3-CLAUDEMD-MARGIN-RESTORE`: CLAUDE.md se auto-carga en cada turn, así que conserva la regla condensada
y el detalle íntegro se archiva en `docs/`). Texto verbatim, nada perdido:

> El usuario puede usar la app SOLO como contador de macros/diario (estilo MyFitnessPal): paso 0 del wizard (¿plan
> o contador?, rama corta de 10 pasos cuyos 12 campos saltados quedan AUSENTES — no se inventan) e interruptor en
> Configuración → Capacidades con el plan ya creado. La pausa es DOS capas: gate SQL en el pickup del chunk worker
> (`plan_mode='tracking'` en `user_profiles` — LA que detiene el gasto, porque el pickup no lee flags del jsonb) +
> cancelación de la cola (los 5 estados resucitables, INCLUIDO `pending_user_action`: el recovery cron los revive a
> las 12h). Orden flag-first en ambas direcciones. El plan pausado conserva `plan_data` con snapshot
> `_paused_prev_generation_status` (guard I8: jamás restaurar `complete` con days=[]). Motor SSOT
> [`backend/plan_mode.py`](../plan_mode.py); knob `MEALFIT_PLAN_MODE_SWITCH`; ventana de reanudación
> `MEALFIT_PLAN_PAUSE_MAX_RESUME_DAYS`. Nav del dashboard por modo: SSOT `frontend/src/config/dashboardNav.js`.
> Tests [`test_p1_plan_mode.py`](../tests/test_p1_plan_mode.py) (backend, 22) +
> `frontend/src/__tests__/PlanMode.contract.test.jsx` (15).

La Nevera en modo contador (encenderla/apagarla, apagado automático a las 48 h vacía) es un feature aparte de
este modo, no parte de él: ver [`nevera_opcional.md`](nevera_opcional.md).
