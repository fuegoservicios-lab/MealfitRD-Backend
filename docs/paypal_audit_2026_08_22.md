# Auditoría de PayPal — 2026-08-22

Auditoría de extremo a extremo del cobro: código, configuración del VPS, la API LIVE de
PayPal, la base de producción y los logs. Doc canónica de lo verificado, lo que se
arregló y lo que queda abierto.

---

## El hecho que enmarca todo lo demás

**Nunca se ha completado un solo pago en producción.** PayPal registra 13 suscripciones
creadas entre el 3 y el 17 de agosto de 2026, **todas** en `APPROVAL_PENDING`, y hoy las
13 devuelven `404 RESOURCE_NOT_FOUND` (PayPal purga las que nadie aprueba). Cero
`ACTIVATED`, cero `PAYMENT.SALE.COMPLETED`, cero `"Subscripcion Verificada B2B"` en el
journal, y `user_profiles.paypal_subscription_id` es NULL en los 14 perfiles.

El tramo *approve → onApprove → /verify → tier* no tiene **ni una** ejecución real. Todo
lo que sigue está verificado por inspección y por sondas en vivo; nada por un cobro
verdadero. Cuando llegue el primero, verificar a mano que el tier se aplica.

---

## Lo verificado en vivo (todo correcto)

| Comprobación | Resultado |
|---|---|
| OAuth contra `api-m.paypal.com` con las credenciales del VPS | OK |
| Los 5 planes que se venden | `ACTIVE`; 9.99/19.99/49.99 ×`MONTH`, 89.99/179.99 ×`YEAR` |
| Precios de PayPal ↔ `PRICING` en `frontend/src/config/plans.js` | Idénticos |
| Plan IDs: PayPal ↔ `.env` del VPS ↔ `frontend/.env.production` ↔ bundle desplegado | Los 5 coinciden, sin drift |
| Max Anual (el de `P0-ANNUAL-PLANS-MISCONFIGURED`, 449.99 **cada mes**) | `INACTIVE` en PayPal + comentado en ambos lados + `ANNUAL_DISABLED_TIERS` lo incluye |
| Webhook | `PAYPAL_WEBHOOK_ID` coincide; URL `https://bioboros.com/api/webhooks/paypal`; suscrito a `*` |
| Verificación de firma en producción | **Activa** — un evento forjado se rechaza (probado); 3 rechazos reales más en el log |
| `/verify`, `/cancel`, `/discount/validate` sin token | 401 los tres |
| CSP + `Permissions-Policy` en `app.bioboros.com` | Permiten PayPal (`payment=(self "https://www.paypal.com")`, `frame-src`/`connect-src` con paypal) |
| SDK con el client-id LIVE, `intent=subscription&vault=true` | HTTP 200 |
| Dedup de webhooks (`P1-WEBHOOK-DEDUP-ATOMIC`) | 4 marcadores, todos `done` |

### Dos falsos positivos descartados, por si vuelven a sonar

- **`PayPalButtons` sin `forceReRender`.** Parece el clásico closure obsoleto (aplicar un
  cupón después de que el botón renderiza cobraría el precio viejo). **No lo es**:
  `@paypal/react-paypal-js` 8.9.2 pasa las props-función por un `Proxy` que se reasigna en
  cada render, así que `createSubscription`/`onApprove` son siempre los del render actual.
  `forceReRender` solo hace falta para props que NO son funciones.
- **La CSP estricta de `bioboros.com`.** El apex sirve `default-src 'self'` sin ninguna
  excepción para PayPal, lo que romperia el checkout… si el checkout viviera ahí. No vive:
  el apex es marketing estático (sin precios, sin SDK) y enlaza a `app.bioboros.com`, que
  tiene su propia CSP correcta. Son dos server blocks distintos en el mismo nginx.

---

## Lo que se arregló

### `P1-BILLING-ORPHAN-RECOVERY` — un cobro cuyo `/verify` no llegó ya no se pierde

`POST /api/subscription/verify` era el **único** camino por el que una suscripción de
PayPal llegaba a `user_profiles`, y lo dispara el **navegador** desde `onApprove`. Si
entre la aprobación y esa llamada se caía la red, el usuario cerraba la pestaña, o
`/verify` devolvía 500/409:

- `paypal_subscription_id` quedaba NULL para ese usuario, y
- los webhooks `ACTIVATED` / `PAYMENT.SALE.COMPLETED` filtran por
  `WHERE paypal_subscription_id = %s` → 0 filas → **no-op silencioso**.

Cobro recurrente sin acceso, sin alerta y sin cron de reconciliación: la pérdida era
permanente. Y sin manera automática de atribuir la suscripción a nadie, porque el `create`
del frontend solo mandaba `plan_id` — la única pista era el email del comprador, a mano.

**El fix son dos mitades y las dos hacen falta:**

1. `PaymentModal` estampa `custom_id: <user_id>` al crear la suscripción. PayPal nos lo
   devuelve **firmado** dentro del webhook.
2. El webhook `BILLING.SUBSCRIPTION.ACTIVATED` que no matchea ninguna fila adopta al
   huérfano usando ese `custom_id` (`_recover_orphan_subscription`).

**Lo que NO concede, y por qué:**

- El **tier** sigue derivándose server-side del `plan_id` vía `_build_paypal_plan_tier_map`
  (I-Billing-1). El `custom_id` dice **a quién**, jamás **qué**. Un `plan_id` que no mapea
  no otorga nada.
- Si la sub **ya está** en algún perfil no es huérfana: el UPDATE no matchó por el guard de
  `P1-BILLING-REACTIVATE-NOT-CANCELLED`, y resucitar una `CANCELLED` por esta vía sería la
  puerta trasera que ese fix cerró.
- Si el perfil destino ya tiene **otra** suscripción viva, no la pisa: cambiar el handle sin
  cancelar la vieja en PayPal es el doble cobro de `P1-BILLING-UPGRADE-FAIL-LOUD`.
- **Solo en `ACTIVATED`.** `PAYMENT.SALE.COMPLETED` no trae `plan_id` (no hay tier que
  derivar) y su guard es `PAYMENT_RETRYING`-only, así que 0 filas es su caso **normal**:
  llamar ahí convertiría cada pago de renovación en una alerta falsa.
- `custom_id` se valida como UUID antes de tocar `WHERE id = %s`. `user_profiles.id` es
  `uuid`: un valor basura propagaría `invalid input syntax for type uuid` → 503 → PayPal
  reintentando 25 veces contra un error determinista.

Knob `MEALFIT_BILLING_ORPHAN_RECOVERY` (default `True`) = kill-switch.
Alertas: `billing_orphan_subscription_recovered:<user>:<sub>` (warning) y
`billing_orphan_subscription_unrecoverable:<sub>` (critical, con `metadata.motivo`).
Test funcional: [`test_p1_billing_orphan_recovery.py`](../tests/test_p1_billing_orphan_recovery.py).

> **La primera versión de esos tests era una coartada: las tres mutaciones sobrevivieron.**
> Faltaba que el doble de BD supiera que `id` es un `uuid` (sin eso, el test del `custom_id`
> basura no distinguía un backend que valida de uno que no), y que el caso `CANCELLED`
> asegurara también el **silencio** — porque el guard de «¿la tiene ya alguien?» gobierna
> además la alerta, y sin él cada cancelación normal emitía un crítico falso.

### `P1-CHECKOUT-CREDITS-TRUTH` — la pantalla de pago decía cifras que no existen

`getPlanFeatures` en `PaymentModal.jsx` tenía los números escritos a mano, del ladder
**viejo** (cuando Gratis eran 15 créditos):

| Tier | Decía | Es |
|---|---|---|
| Básico | «3× más que Gratis» | 5× (50/10) |
| Plus | «13× más que Gratis» | 20× (200/10) |
| Max | «Créditos Ilimitados» + «Generación Ilimitada de Planes» | 500/mes (`auth._TIER_LIMITS`) |

`P1-CREDITS-LADDER` (31-jul) cambió el ladder y actualizó la landing y `/upgrade`, que
**derivan** de `TIER_CREDITS` vía `creditsVsPredecessor`. Esta pantalla no. El resultado
medible: el usuario leía «500 Créditos al mes» en la tarjeta de Max, hacía clic, y la
pantalla donde pone la tarjeta le prometía «ilimitado» — una contradicción dentro del mismo
embudo, justo en el paso del dinero.

El comentario `P2-PAYMENT-FEATURES-ALIGN` decía que esta pantalla se alineó con
`Pricing.jsx`, y era cierto: se alineó en **mayo**. Por eso ahora no se copia el resultado
sino la **fuente**. Test: `frontend/src/__tests__/PaymentModal.credits_truth.test.jsx`.

---

## Lo que queda abierto (decisión del dueño, no deuda oculta)

### 1. La manipulación de monto se alerta pero, tal como está escrito, no puede bloquearse

`MEALFIT_BILLING_VERIFY_AMOUNT` no está seteado en el VPS → modo `warn`. Pero incluso en
`block`, `_verify_subscription_amount` exige `discount_pct is not None` para marcar
`proven_underpaid`: **un override sin cupón nunca se bloquea**, por diseño *fail-cheap*
(`P1-BILLING-AMOUNT-FP-FIX`, para no rechazar un pago legítimo cuyo `coupon_code` no llegó).

El supuesto de ese diseño era que un override sin cupón es **ambiguo**. Hoy no lo es:
`discount_codes` está **vacía**, así que ningún cupón puede validar y por tanto **cualquier**
override es ilegítimo por definición. Quien llame a `actions.subscription.create` desde la
consola con su propio `billing_cycles` paga $0.01 y obtiene Max; queda un crítico en
`system_alerts`, pero el acceso se concede.

Cierre disponible cuando se decida: bloquear el override cuando no existe **ningún** cupón
activo aplicable al tier. Deja intacto el caso que el fail-cheap protege.

### 2. `plan_tier` nace `'free'` en la DB y el código escribe `'gratis'`

El default del esquema es `'free'::text`; 13 de 14 perfiles lo tienen. El degradador
(`db_profiles.get_user_profile`) escribe `'gratis'`. Hoy no rompe nada porque cada
consumidor tiene un default seguro — `_TIER_LIMITS.get(t, _TIER_LIMITS["gratis"])` da 10,
`Upgrade.jsx` normaliza a `'gratis'` lo que no esté en su lista, `isPremium` excluye ambos —
pero son dos nombres para el mismo escalón conviviendo en la misma columna.

### 3. Higiene

- 6 usuarios `e2e-test-*` vivos en la base de **producción** (lo avisa el detector de
  residuo al correr la suite).
- Un perfil `plus` sin `paypal_subscription_id` (`00d90458`, 31-jul): tier concedido a mano.
  Correcto que exista, pero PayPal no lo gobierna y ningún webhook lo degradará nunca.
- `discount_codes` vacía: el campo de cupón del checkout siempre responde «Código no
  encontrado o inactivo».
