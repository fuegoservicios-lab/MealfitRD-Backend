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

## El contador en el teléfono (`P1-PLAN-LOTE-87` · 2026-09-17)

Tres capturas del dueño a 392px: «en móviles todavía le falta; la señal de que quieres que la IA te arme tu plan debería
estar arriba de primero; se siente muy encogido lo del contador de macros».

- **La invitación a encender el plan va primera en el teléfono.** Era la última tarjeta de la columna lateral (debajo de
  la hidratación), que en móvil se apila bajo el contador. Ahora es hija DIRECTA de la rejilla (`.turnOnSlot`) y el orden
  lo dan las áreas: escritorio `"main side" / "main plan"` (bajo la hidratación, como antes); móvil `"plan" / "main" /
  "side"`. Ni segundo render ni `order` dentro de la columna.
- **La tarjeta de macros recupera aire en ≤480px.** `P1-MACRO-CARD-DENSITY` (2026-08-10) la apretó para que la lista de
  comidas cupiera sin scroll; el dueño la siente encogida. Se afloja donde no cuesta legibilidad: relleno 1 → 1,15 rem,
  hueco entre bloques 1 → 1,3 rem, entre macros 0,75 → 1 rem, cabecera 1/0,75 → 1,25/0,9 rem (y en ≤768px 1,15 → 1,4 y
  0,9 → 1,15). Letras y altura de barra intactas: sus dos correcciones de agosto («no las vuelvas más pequeñas») siguen.

Tests: `frontend/src/__tests__/DashboardTracking.mobile_order.test.jsx` y `backend/tests/test_p1_plan_lote_87.py`.
