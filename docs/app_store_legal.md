# Preparación legal y App Store (lotes P1-PLAN-LOTE-840 … 849, 29-sep-2026)

Objetivo del dueño: «lo legal al 100 % para producción y la App Store». Especificación con evidencia archivo:línea:
`docs/superpowers/specs/2026-09-29-legal-appstore-auditoria.md` (raíz). Plan: `docs/superpowers/plans/2026-09-29-legal-appstore.md`.

## Qué cierra el código

| Lote | Qué | Doc / test ancla |
|---|---|---|
| 840/842 | PostHog sin autocapture en la app; lista CERRADA de eventos propios (sin datos de salud; fuera `meal_name`); `ph-no-capture` como segunda capa | frontend `lote840.test.js`, `lote842*.test.*` |
| 841 | Rastro del equipo (`admin_access_log`) seudonimizado al cerrar la cuenta y purgado a los `MEALFIT_ADMIN_LOG_RETENTION_DAYS` (730) | `tests/test_p1_plan_lote_841.py` |
| 843 | Permiso EXPLÍCITO antes de mandar datos a la IA de terceros (Apple 5.1.2(i), RGPD 9.2.a/49.1.a): 428 en cada endpoint de IA, filtro en la recogida de bloques y en segundo plano, retirada con pausa, invitados por cabecera | `docs/consentimiento_ia.md`, `tests/test_p1_plan_lote_843*.py` |
| 844 | La hoja «Tus datos y la IA», puertas en cada punto de entrada, el 428 en el cliente, «IA de terceros» en Configuración, textos versionados | `docs/consentimientos/ia/ia-2026-10/*.md`, frontend `lote844*` |
| 845 | Acceso del revisor de Apple (código fijo SOLO para un correo, nunca admin, ráfagas alertadas y bloqueadas); la app nativa sin salidas al comercio (legales en `/app/*`, bot de ayuda sin precios) | `review_login.py`, `tests/test_p1_plan_lote_845*.py`, frontend `lote845.test.jsx` |
| 846 | Recordatorio médico visible (formulario, coach, plan y contador; banner profesional que no desaparece) y edad mínima 18 en formulario, backend y coach | `edad_minima.py`, `tests/test_p1_plan_lote_846_edad.py` |
| 847 | Analítica y Replay de Sentry solo con permiso (opt-in); la persona de PostHog se borra con la cuenta | `tests/test_p1_plan_lote_847*.py`, frontend `lote847*` |
| 848 | iOS: cadenas de permiso en 5 idiomas (+ `es.lproj`), `PrivacyInfo.xcprivacy` completo, solo iPhone, PHPicker nativo (`MfFotos`) sin permiso de fototeca, revocación de Sign in with Apple al borrar la cuenta | `apple_tokens.py`, `tests/test_p1_plan_lote_848*.py`, frontend `lote848*` |
| 849 | Términos dentro de la app sin «dónde se compra»; la casilla de analítica respeta lo elegido | este doc |

## Despliegue (en este orden)

1. Migraciones `p1_plan_lote_843_user_consents` y `p1_plan_lote_848_apple_tokens` aplicadas ANTES del backend
   (`scripts/apply_migration.py --status` sale 0). Sin la 843 la recogida de bloques falla para todos.
2. `MEALFIT_AI_CONSENT_GATE=log` en el `.env` del VPS; `deploy-mealfit.ps1 all`.
3. Verificar: `GET /api/consents` responde 401 (no 404), los bloques se recogen, aceptar y retirar funciona en web y app.
4. `MEALFIT_AI_CONSENT_GATE=block` y reiniciar; `POST /api/help/chat` sin sesión ni cabecera → 428.
5. Enviar a revisión SOLO con `block` activo.

## Secretos del VPS (los pone el dueño)

| Variables | Para qué | Sin ellas |
|---|---|---|
| `APPLE_SIWA_KEY_ID`, `APPLE_SIWA_KEY_FILE` o `APPLE_SIWA_PRIVATE_KEY` (.p8 con Sign in with Apple), `MEALFIT_TOKEN_ENC_KEY` (Fernet); el Team ID cae a `APNS_TEAM_ID` | canjear y revocar los tokens de Apple | no se revoca al borrar la cuenta (5.1.1(v)) |
| `POSTHOG_PERSONAL_API_KEY`, `POSTHOG_PROJECT_ID` (`POSTHOG_HOST` opcional) | borrar la persona de PostHog con la cuenta | quedan sus eventos |
| `MEALFIT_REVIEW_LOGIN_EMAIL`, `MEALFIT_REVIEW_LOGIN_CODE_SHA256` (solo durante la revisión) | que el revisor entre con un código fijo de 6 cifras | el revisor no puede entrar |

## Abierto (decisión del dueño, no de código)

- Transferencia a DeepSeek (China) de datos de usuarios de la UE: hoy se ampara en el consentimiento explícito
  (art. 49.1.a), que el Comité Europeo interpreta de forma restrictiva. Lo robusto: cláusulas tipo con DeepSeek o
  enrutar la UE a otro proveedor. Con un abogado.
- Facturación de la API de Gemini (la gratuita usa los datos para mejorar productos de Google).
- Cuenta de Apple Developer Individual para una app con datos de salud (5.1.1(ix)): riesgo de rechazo.
- Dispositivo médico regulado (campo obligatorio de App Store Connect) y comerciante en la UE (DSA).
- Compras dentro de la app (3.1.3): la app refleja el plan de la web sin venderlo; respuesta preparada en la auditoría §3.5.
- Plazo del rastro del equipo (24 meses por defecto) y conservación de los permisos de invitados.
