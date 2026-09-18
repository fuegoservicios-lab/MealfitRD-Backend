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

## Google pregunta siempre qué cuenta usar (`P1-PLAN-LOTE-95` · 2026-09-17)

Decidido por el dueño. «Continuar con Google» entraba con la sesión de Google activa en el dispositivo: en su iPhone creó
una identidad nueva (`0f3ca99f…`, otro correo) sin preguntar. Ahora la URL de autorización lleva `prompt=select_account`.

No se puede pedir por petición: el endpoint `/sign-in/social` de Better Auth 1.4.18 solo acepta `loginHint` y
`additionalData` (el `prompt` es configuración del proveedor en el servidor de Neon) y el adaptador Supabase de
`@neondatabase/auth` no reenvía `queryParams`. El cliente (`authClient.js`) pide la URL con `disableRedirect: true` —el
servidor responde `redirect: false` y el `redirectPlugin` del cliente no navega—, le añade el parámetro con
`conSelectorDeCuenta` (solo en `accounts.google.com`, sin duplicar, respetando `none`) y navega. Misma petición, mismas
cookies, mismo cliente. Si el cliente Better Auth no está expuesto, el acceso sigue por el camino de siempre.

Tests: `frontend/src/__tests__/authClient.google_select_account.test.js` y `backend/tests/test_p1_plan_lote_95.py`.

## Abierto (decisión del dueño)

- Esa identidad vacía sigue existiendo; borrarla es una escritura en producción (o «Eliminar cuenta» desde la propia app).
