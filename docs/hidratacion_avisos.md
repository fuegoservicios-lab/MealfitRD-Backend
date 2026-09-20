# Avisos de hidratación, apagado automático e invitación semanal al plan (lote 135)

[P1-PLAN-LOTE-135 · 2026-09-20] Tres encargos del dueño con la captura del contador en su iPhone.

## 1 · Avisos de hidratación

Motor: [`hydration_reminders.py`](../hydration_reminders.py). Cero LLM, cero mensajes en el chat.

| Punto de control (hora LOCAL) | Fracción de la meta esperada | Con meta 9 |
|---|---|---|
| 11 h | 25 % | 2 vasos |
| 15 h | 55 % | 4 vasos |
| 19 h | 80 % | 7 vasos |

Si el usuario va por debajo sale el aviso («Llevas 2 de 9 vasos hoy. ¿Un vaso de agua ahora?»); si va al día, silencio.
Cada punto sigue tocando 2 h (un tick perdido no pierde el aviso) y sale una sola vez por día.

Dos canales, una cuenta (igual que los recordatorios de comida, [`recordatorios_de_comida.md`](recordatorios_de_comida.md)):

- **Navegador / PWA**: Web Push desde el cron `run_hydration_checks` (APScheduler, a y 32, `id="hydration_reminders"`), `tag="agua"`.
- **App nativa**: notificaciones LOCALES. `GET /api/notifications/meal-reminders?canal=local` trae `water: {enabled, days, goal,
  glasses, reminders[]}`; el teléfono programa HOY (solo lo que no va al día, con la cuenta real) y los 2 días siguientes
  (texto genérico), y re-sincroniza al anotar agua. Solo 3 días: si el usuario no vuelve, a las 48 h la hidratación se
  apaga y el teléfono ya no tiene nada más programado.

Con `scheduleType` `night_shift`/`variable` no hay avisos (la misma puerta que las comidas).

## 2 · Apagado automático a las 48 h

`evaluar_apagado` exige las tres cosas:

1. **El usuario es alcanzable**: suscripción push o `avisos_locales:<user_id>` tocado en las últimas 72 h (lo marca el
   teléfono al pedir su horario). A quien no le llegó nada no se le acusa de ignorarlo.
2. **≥ 48 h desde el primer aviso sin respuesta y ≥ 3 avisos** (`ignored_since`, `nudges`).
3. **Ni un vaso anotado desde entonces** — se mira `water_intake_log` (`glasses > 0 AND updated_at >= ignored_since`), así
   que vale igual el botón del contador que el coach. Un vaso pone la cuenta a cero.

Al apagarse: `user_profiles.water_tracker_enabled = false`, push «Pausamos la hidratación…» (si hay suscripción) y
`auto_off_at` en el estado; `GET /api/plans/water-intake` lo devuelve (solo con el interruptor apagado) y el dashboard lo
dice UNA vez. Volver a encenderla en Configuración borra el estado (`al_encender`): sin eso se apagaría de nuevo en el
siguiente tick.

**Por qué NO `nudge_outcomes`**: `get_daily_nudge_count` cuenta todas sus filas contra el tope de 4 avisos de comida al
día; tres avisos de agua habrían dejado al usuario sin el recordatorio de la cena.

## 3 · Invitación «¿Quieres que la IA te arme el plan?»: una vez por semana y por usuario

Motor: [`plan_invite.py`](../plan_invite.py); `GET/PATCH /api/user/preferences/plan-invite` (`action`: `seen` | `dismiss`).
El descarte vivía en el `localStorage` de cada dispositivo: cada binario de TestFlight, cada navegador y la PWA traen un
almacén distinto, así que «Ahora no» valía para ese almacén y la tarjeta volvía.

- nunca vista → se muestra; esa vista abre la semana;
- vista → 24 h a la vista, luego se esconde sola hasta cumplir 7 días desde esa vista;
- «Ahora no» → escondida en el acto, hasta 7 días después.

La tarjeta de «Tu plan está en pausa» (reanudar es gratis) sigue la misma regla.

## Estado y knobs

Estado por usuario en `app_kv_store` (sin DDL): `hydration_state:`, `avisos_locales:`, `plan_invite:` + `<user_id>`.
Caduca por TTL (`cron_tasks._KV_SWEEP_PREFIXES`) y se borra con la cuenta (`db_profiles._USER_SCOPED_KV_PREFIXES`).

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_HYDRATION_REMINDERS` | `True` | kill switch de los avisos de agua Y del apagado automático |
| `MEALFIT_HYDRATION_AUTO_OFF_HOURS` | `48` | horas sin un vaso desde el primer aviso ignorado, clamp [24, 336] |
| `MEALFIT_HYDRATION_AUTO_OFF_MIN_NUDGES` | `3` | avisos mínimos antes de apagar, clamp [1, 20] |
| `MEALFIT_PLAN_INVITE_EVERY_DAYS` | `7` | días entre invitaciones, clamp [1, 90] |
| `MEALFIT_KV_TTL_HYDRATION_STATE_HOURS` / `…_AVISOS_LOCALES_HOURS` / `…_PLAN_INVITE_HOURS` | 720 / 168 / 720 | TTL del barrido |

Tests: [`test_p1_plan_lote_135.py`](../tests/test_p1_plan_lote_135.py), `frontend/src/__tests__/lote135.test.js`.
