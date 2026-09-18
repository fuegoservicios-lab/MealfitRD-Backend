# La sesión first-party y su marcador (`P1-PLAN-LOTE-90` · 2026-09-17)

## El incidente

Reporte del dueño (PWA de iOS instalada en la pantalla de inicio): «agrego el correo, verifico el código en el login y no me
inicia sesión». Reconstruido con el journal del backend y el access log de nginx (20:33-20:37 UTC, todo de solo lectura):

| Hora | Qué pidió el teléfono | Resultado |
|---|---|---|
| 20:33:25 | `GET /api/auth/me` (cookie + token de la cuenta purgada el 16-sep, `sub=f47126cb…`) | 401 `P1-AUTH-CUENTA-BORRADA`: correcto |
| 20:34:10-22 | Google OAuth | entró con la OTRA cuenta de Google del teléfono → identidad nueva `0f3ca99f…`; cerró sesión |
| 20:35:15 y 20:35:51 | `POST /api/auth/email-otp/verify` | **200, sesión emitida para `61a13831…`** |
| 20:35:16 y 20:35:52 | recarga a `/` | vuelve a `/login` en <1 s, **sin ninguna llamada a `/api/auth/me`** |

El último OTP que funcionó (11-sep, otro iPhone) sí llamó a `/me` justo después de verificar. La diferencia está en el
cliente: `checkFirstPartySession` solo preguntaba si encontraba el token en `localStorage` (`NO-401-NOISE`, 2026-06-23), y en
ese arranque no estaba. Comprobado contra producción que con un token presente el arranque SÍ pregunta. **Por qué el token no
estaba no se puede ver desde el servidor** (almacenamiento del PWA); lo que sí se ve es el diseño: el token de `localStorage`
nació como RESPALDO de la cookie (el PWA de iOS la pierde entre lanzamientos) y había acabado siendo la ÚNICA puerta. El
comentario del atajo lo decía: «caso raro cookie-sin-localStorage: el usuario simplemente vuelve a iniciar sesión» — y volver a
iniciar sesión repetía exactamente lo mismo.

## Qué cambió

1. **Marcador `__Host-mf_has_session=1`.** `set_session_cookie` lo emite junto a `__Host-mf_session` (misma vida, `Secure`,
   `Path=/`, `SameSite=Strict`) y `clear_session_cookie` lo borra con ella. No lleva secreto y NO es HttpOnly: existe para que
   el JS lo lea. El cliente pregunta a `/me` si hay token **o** marcador; sin ninguno de los dos sigue sin llamar (el
   visitante anónimo no genera un 401). Como viaja con la cookie, no depende de que `localStorage` haya guardado nada.
2. **El marcador también cae desde el cliente** (`clearSessionMarker`) al cerrar sesión y ante un 401 de `/me`: un logout sin
   red no llega al servidor, y con el marcador puesto el siguiente arranque habría vuelto a entrar solo.
3. **Una cookie de identidad borrada ya no decide sola.** En `get_verified_user_id` la rama de la cookie devolvía `None` en
   cuanto el guard `P1-AUTH-CUENTA-BORRADA` la rechazaba, sin mirar `X-MF-Session`: una cookie vieja tapaba un token nuevo y
   válido de otra cuenta en el mismo navegador. Ahora sigue al header, que se verifica por su cuenta (firma + guard): solo
   AÑADE una segunda credencial comprobada. El orden Bearer → cookie → header no cambia y una cookie viva sigue mandando.
4. **`/me` toma el `iat` de la credencial que ES del uid** (antes `mf_session or x_mf_session`: con una cookie ajena heredaba
   su antigüedad). Si el uid llegó por Bearer y la credencial first-party es de otra cuenta, re-emite fresca.
5. **Diagnóstico**: `/me` registra `[P1-PLAN-LOTE-90] /me resuelto solo por cookie` cuando llega cookie sin header — la huella
   de un cliente sin token en `localStorage`. Si el problema del PWA se repite, el journal ya lo dice.

## Lo que NO cambia

`P0-AUDIT-1` (jamás un `sub` sin firma verificada), el guard de cuenta borrada en las cuatro puertas, `HttpOnly` +
`SameSite=Strict` en la cookie de sesión, y el orden de resolución. El marcador no autentica a nadie: con el marcador y sin
cookie válida, `/me` responde 401 y lo borra.

Tests: `backend/tests/test_p1_plan_lote_90.py` (comportamiento del resolvedor, de `/me` y de las cookies + ancla del cliente) y
`frontend/src/__tests__/firstPartySession.marker.test.js` (anónimo no pregunta · sin token pero con marcador entra · 401 limpia
los dos · logout sin red también cierra).

## Que Google pregunte siempre qué cuenta usar: no se puede desde este lado (`P1-PLAN-LOTE-95` → revertido en `P1-PLAN-LOTE-96`)

Decidido por el dueño: «Continuar con Google» entraba con la sesión de Google activa en el dispositivo (en su iPhone creó
la identidad vacía `0f3ca99f…` con su otro correo). El lote 95 intentó forzar el selector con `prompt=select_account`: pedía
la URL a Better Auth con `disableRedirect: true` y se la añadía si el host era `accounts.google.com`.

**No hacía nada en producción**, y se vio al comprobarlo con la Navigation API tras desplegar:

- La URL que devuelve Neon no es la de Google sino un salto en SU dominio,
  `…/neondb/auth/sign-in/social/init?token=<uuid>`, que redirige a Google desde el servidor.
- Añadirle `&prompt=select_account` a ese salto tampoco llega: en las URLs finales de Google
  (`/v3/signin/accountchooser?…`) están client_id, PKCE, scope y state, y ningún `prompt`.
- El proveedor de Google de Neon solo acepta client ID y secret (`neonctl neon-auth oauth-provider add|update`:
  `--oauth-client-id`, `--oauth-client-secret`); el endpoint `/sign-in/social` solo acepta `loginHint` y
  `additionalData`, y el adaptador Supabase no reenvía `queryParams`.

El envoltorio se revirtió (código que no podía actuar, con un test que afirmaba lo contrario) y la explicación quedó junto a
`signInWithOAuth` en `authClient.js`. Queda pedírselo a Neon (que su proveedor acepte `prompt`).

## El aviso «¿Es la cuenta que querías?» (`P1-PLAN-LOTE-97` · 2026-09-18)

Lo que sí se puede desde este lado, elegido por el dueño. El dispositivo recuerda con qué cuentas se entra
(`utils/cuentasDelDispositivo.js`, clave `mf_cuentas_dispositivo`: id, correo ENMASCARADO «an***@gmail.com» y cuándo se
vio; máximo 5, nunca el correo completo porque la lista sobrevive al cierre de sesión a propósito). El login marca el
inicio de un acceso con Google (`mf_google_inicio`, en localStorage porque en el PWA de iOS la vuelta del OAuth no
siempre conserva la pestaña). Al cargar el perfil, `AvisoCuentaGoogle` (montado en `App`, perezoso y fuera del apex)
pregunta **solo** si: el acceso vino de Google hace menos de 15 min, el dispositivo ya conocía OTRA cuenta, y esta no la
había visto nunca. Dice si la cuenta es nueva (creada hace menos de 15 min) o solo nueva en el teléfono.

- «Sí, seguir con esta» → la cuenta pasa a conocida; no se vuelve a preguntar por ella en ese dispositivo.
- «No, salir» → cierre de sesión SIN recordarla, de vuelta al login. La nota explica que para entrar con la de antes
  hay que usar «Continuar con correo» (Google volvería a elegir la cuenta abierta del teléfono).
- Cerrar con Escape o tocando fuera no decide: el próximo acceso con Google a esa cuenta vuelve a preguntar.
- El marcador de Google se consume siempre: un aviso por acceso, no uno por recarga. Un login con correo no pregunta
  nunca (es deliberado) y enseña al dispositivo la cuenta habitual.

Tests: `frontend/src/__tests__/cuentasDelDispositivo.test.js`, `frontend/src/__tests__/AvisoCuentaGoogle.test.jsx` y
`backend/tests/test_p1_plan_lote_97.py`.

Tests: `backend/tests/test_p1_plan_lote_96.py` (impide que el envoltorio vuelva sin una vía que funcione).

## Abierto (decisión del dueño)

- Esa identidad vacía sigue existiendo; borrarla es una escritura en producción (o «Eliminar cuenta» desde la propia app).
