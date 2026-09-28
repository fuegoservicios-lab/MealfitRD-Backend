# Regalos de la cuenta — créditos extra y planes de cortesía

[P1-PLAN-LOTE-771..777 · 2026-09-28] Spec: `docs/superpowers/specs/2026-09-28-admin-cuentas-regalos-design.md` (raíz).
Plan: `docs/superpowers/plans/2026-09-28-admin-cuentas-regalos.md`.

## Qué es

Desde `/admin` → «Cuentas» el dueño busca una cuenta por su correo EXACTO (no hay lista) y le regala créditos de
planes, mensajes del coach o un plan de cortesía (Básico / Plus / Max, con fecha de fin opcional).

## La regla que no se rompe

Lo pagado y lo regalado NUNCA se mezclan. PayPal escribe `user_profiles.plan_tier`; los regalos viven en
`public.account_grants` y `regalos_cuenta.superponer` los superpone AL LEER dentro de `get_user_profile` (y
`get_user_plan_tier` para el enrutado de modelos). El perfil lleva `plan_tier` (efectivo: el mayor de los dos),
`plan_tier_pagado` (el de PayPal: las pantallas de cobro deciden con él), `cortesia` y `creditos_extra`. Nada escribe
el efectivo: solo `routers/billing.py` y la degradación de `get_user_profile` escriben `plan_tier` (guard en
`test_p1_plan_lote_772.py`). Un regalo de créditos no borra uso: sube el tope, y el panel de costes sigue siendo
verdad.

## Tabla

Migración `p1_plan_lote_771_account_grants_2026_09_28.sql`. Un regalo se revoca, no se borra; se va con la cuenta
(`ON DELETE CASCADE`); una sola cortesía viva por persona (índice único parcial). CHECK: `admin` no es regalable, los
créditos van de 1 a 1000 y siempre caducan, motivo de 3 a 300 caracteres.

## Reglas

- Créditos: válidos hasta fin de este mes o del siguiente (día 1, 00:00 UTC, la MISMA ventana que el contador
  `get_monthly_api_usage` vía `regalos_cuenta.inicio_de_mes`). Suben el tope de cada mes en que están vigentes; no
  son un saldo que se acumula.
- «Recargar al completo» = lo gastado del cupo del plan (usados − regalo vigente): deja disponible exactamente el
  cupo del plan.
- Cortesía: plan mejor que el pagado; «hasta el X» incluye todo el día X en la hora de RD. Caduca sola (sin cron).
- Una cuenta `admin` no recibe regalos.

## Rutas (`routers/admin.py`, `require_admin` en el router)

`POST /api/admin/cuentas/buscar`, `GET /api/admin/cuentas/{user_id}`, `POST /api/admin/cuentas/{user_id}/creditos`,
`POST /api/admin/cuentas/{user_id}/cortesia`, `POST /api/admin/regalos/{grant_id}/revocar`. Todo POST exige la
cabecera `X-Admin-Accion: 1`. Cada búsqueda, vista y cambio se anota en `admin_access_log` ANTES de responder o
escribir; si no se puede anotar ⇒ 503 y no hay cambio.

## Lo que ve la persona

`GET /api/user/credits/{uid}` devuelve `limit`, `bonus`, `bonus_hasta` y `regalos_recientes` (14 días). El medidor usa
ese tope; `AvisoRegalos` anuncia cada regalo una vez por dispositivo (toast + centro de notificaciones) y, al regalar,
sale una push ya traducida (`avisos_regalo.py`). El último día que vale un regalo se muestra fijo en la hora de Santo
Domingo (`ultimoDiaDeRegalo`, `frontend/src/utils/regalosCuenta.js`) — el MISMO día que dice la push, sea cual sea el
huso del dispositivo. En Configuración, la escalera de «Otros planes» y las acciones de cobro (cancelar, «Tu plan
actual») deciden por el plan PAGADO (`planDeCobro`); el nombre del plan, la pastilla de estado y las funciones
premium siguen al plan efectivo.

## Privacidad

[P1-PLAN-LOTE-777 · 2026-09-28] Los regalos aparecen en la exportación de datos del usuario
(`GET /api/account/export`, tabla `account_grants`) sin los ids del personal que los otorgó o
revocó (`granted_by`/`revoked_by` van en `_ACCOUNT_EXPORT_STRIPPED_KEYS`; el motivo, `reason`,
SÍ se exporta). La Política de Privacidad §5 declara el acceso del equipo de soporte a la cuenta.

## Knob

`MEALFIT_ACCOUNT_GRANTS` (default `True`). En `False`: la superposición y los topes extra se ignoran (cada cuenta
queda con lo que paga) y regalar responde 503; revertir sigue funcionando.

## Tests

`test_p1_plan_lote_771.py` (tabla y reglas), `_772` (superposición y cuotas), `_773` (aviso), `_774` (panel),
`_777_doc`, `_777_export` (exportación e higiene de ids del personal); frontend `lote775.test.jsx` (panel) y
`lote776.test.jsx` (usuario).
