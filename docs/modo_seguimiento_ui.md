# Modo seguimiento: qué se muestra y qué no (`P1-PLAN-LOTE-86` · 2026-09-17)

**Sin medidor de créditos en el contador.** Pregunta del dueño con captura («Créditos 10/10» con el generador apagado):
¿en qué se usan? En nada. El crédito mensual (`api_usage`; gratis 10, basic 50, plus 200, ultra 500) solo lo consumen
las superficies del generador, las que llevan `Depends(verify_api_quota)`: generar plan (`POST /api/plans`), `/analyze`
(+stream), `/swap-meal`, `/recipe/expand`, `/{plan_id}/regenerate…`, `/fix-sodium`, `/retry-chunk`, `/chunks/…` y
`/regen-degraded`. El coach tiene su cuota mensual aparte (`P1-COACH-QUOTA-METER`, 60 en gratis) y escanear, estimar o
anotar comidas va a `llm_usage_events`, nunca a `api_usage` (tabla de exenciones en CLAUDE.md).

Con `plan_mode = 'tracking'` ninguna de esas superficies es alcanzable (el plan está en pausa y la navegación por modo no
las ofrece), así que `DashboardTracking` deja de montar `CreditsMeter`. El dashboard de plan (`Dashboard.jsx`) lo
conserva: al reanudar el plan, el medidor vuelve con la cifra real.

Tests: `frontend/src/__tests__/DashboardTracking.no_credits_meter.test.jsx` y `backend/tests/test_p1_plan_lote_86.py`
(que además ancla que el diario y el chat NO cuelgan de `verify_api_quota`: si alguien lo hace, la premisa cae).
