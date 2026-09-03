# Lista de comprobación: la primera compra real (PayPal)

[P3-PAYPAL-FIRST-PURCHASE-CHECKLIST · 2026-09-03] Nunca se ha completado un pago en producción
(ver [`paypal_audit_2026_08_22.md`](paypal_audit_2026_08_22.md)): el tramo *approve → /verify →
tier* y los webhooks de renovación/cancelación están verificados por inspección y sondas, no por un
cobro. Esta lista es para ejecutar UNA compra real con la cuenta del dueño, mirar cada eslabón y
cancelar. Coste: 9,99 USD (Basic mensual), reembolsable desde PayPal.

## Antes

- [ ] En el VPS, `/opt/mealfit/backend/.env` tiene `PAYPAL_CLIENT_ID`, `PAYPAL_SECRET`,
      `PAYPAL_WEBHOOK_ID`, `PAYPAL_PLAN_BASIC_ID` y `ENVIRONMENT=production` (verificado 2026-09-03).
- [ ] La cuenta de prueba es la del dueño en tier `free`, con `paypal_subscription_id` NULL:
      `SELECT plan_tier, subscription_status, paypal_subscription_id FROM user_profiles WHERE email = …`.
- [ ] Journal abierto en una terminal: `sudo journalctl -u mealfit-backend -f | grep -iE 'paypal|billing|webhook|verificad'`.
- [ ] Anotar la hora UTC de inicio (para acotar el log de nginx después).

## La compra (web, no la app nativa: en nativo no existe comercio)

1. Configuración → Suscripción → **Basic mensual** → botón de PayPal → aprobar con la cuenta
   personal de PayPal (no la de negocio que cobra).
2. Al volver a la app:
   - [ ] La pantalla dice suscripción activada (sin quedarse en «Verificando…»).
   - [ ] Journal: `POST /api/subscription/verify … 200` y la línea `Subscripcion Verificada`.
   - [ ] DB: `plan_tier = 'basic'`, `subscription_status = 'ACTIVE'`, `paypal_subscription_id = I-…`.
   - [ ] Créditos: Configuración muestra 50 disponibles y el chip del menú de motivos dice `x/50`.
3. En los siguientes minutos (PayPal reintenta hasta 3 días si el servidor no responde):
   - [ ] Journal: `BILLING.SUBSCRIPTION.ACTIVATED` procesado (dedup `done`) y, cuando cobre,
         `PAYMENT.SALE.COMPLETED`. Ambos devuelven 200 en nginx: `sudo grep 'webhooks/paypal' /var/log/nginx/access.log | tail`.
   - [ ] `system_alerts`: ninguna alerta nueva de billing (`SELECT alert_key, created_at FROM system_alerts WHERE created_at > <inicio> ORDER BY 2 DESC`).

## Si /verify no llegó (pestaña cerrada, red)

- [ ] El webhook `ACTIVATED` adopta al huérfano por `custom_id` (P1-BILLING-ORPHAN-RECOVERY): el
      tier debe aparecer igual en la DB sin que el navegador haya vuelto. Si no aparece a los 5 min,
      buscar en el journal `orphan` / `custom_id`.

## La cancelación

1. Configuración → Suscripción → **Cancelar**.
   - [ ] Journal: `POST /api/subscription/cancel … 200` (o 204/404 idempotente).
   - [ ] DB: `subscription_status = 'CANCELLED'`, `subscription_end_date` = la fecha del próximo cobro
         (`next_billing_time`); `plan_tier` sigue `basic` (gracia hasta el fin del ciclo).
   - [ ] La app sigue mostrando Basic y sus 50 créditos hasta esa fecha.
2. Para no esperar un mes: simular el vencimiento
   `UPDATE user_profiles SET subscription_end_date = now() - interval '1 minute' WHERE email = …;`
   y recargar el perfil:
   - [ ] DB: `plan_tier = 'gratis'` (el degradador perezoso de `get_user_profile`).
   - [ ] La app vuelve a mostrar el tier gratis y 10 créditos.

## Después

- [ ] Reembolsar los 9,99 desde PayPal si se desea (no afecta a la app: la cancelación ya se procesó;
      si PayPal envía `PAYMENT.SALE.REFUNDED`, se registra sin degradar dos veces).
- [ ] Dejar el perfil en `free` con `paypal_subscription_id` NULL o con el `I-…` cancelado, a elección.
- [ ] Anotar en `paypal_audit_2026_08_22.md` la fecha y el resultado: a partir de ahí el ciclo tiene
      una ejecución real.

## Lo que esta compra NO prueba

- La **renovación mensual** (`PAYMENT.SALE.COMPLETED` del segundo cobro): solo ocurre al mes. Si se
  quiere probarla sin esperar, dejar la suscripción viva un ciclo y repetir la sección «webhooks».
- El **pago fallido** (`PAYMENT.FAILED` → `PAYMENT_RETRYING` sin degradar): requiere una tarjeta que
  falle; queda cubierto por `test_p2_billing_payment_failed_grace.py`.
