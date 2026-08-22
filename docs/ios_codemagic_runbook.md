# iOS → TestFlight vía Codemagic — runbook del dueño

[P1-IOS-CODEMAGIC · 2026-08-22] Fase 2 de `P1-IOS-NATIVE-SHELL`. El pipeline vive en
`frontend/codemagic.yaml` (test ancla `test_p1_ios_codemagic.py`). Lo de abajo es lo que
**sólo puede hacer el dueño** con su cuenta de Apple, en orden. Versión interactiva con
casillas: artefacto «Bioboros a TestFlight» (22-ago-2026).

## Valores que el YAML espera (copiar exactos)

| Dónde | Valor |
|---|---|
| Bundle ID (App ID + App Store Connect) | `com.bioboros.app` — **inmutable tras la primera subida** |
| Nombre de la integración App Store Connect en Codemagic | `bioboros_asc` |
| Env var en Codemagic | `APP_STORE_APP_ID` = Apple ID numérico de la app (General → App Information) |
| Grupo de TestFlight (Internal Testing) | `Equipo Bioboros` |
| Capabilities del App ID | Push Notifications + Sign in with Apple (marcarlas YA: añadirlas después invalida el perfil) |
| API key | Access **App Manager** (Developer no basta para subir builds). El `.p8` sólo se descarga una vez |

## Los 6 pasos

1. **App ID** — developer.apple.com → Identifiers → `+` → App → Explicit `com.bioboros.app` → capabilities Push + SIWA.
2. **App en App Store Connect** — Apps → `+` → iOS, «Bioboros», idioma Spanish (Mexico), SKU `bioboros-ios-2026`, Full Access. Apuntar el **Apple ID** numérico.
3. **API key** — Users and Access → Integrations → App Store Connect API → `+`, nombre «Codemagic», App Manager. Guardar Issuer ID, Key ID y el `.p8`.
4. **Codemagic** — registro con GitHub, **cuenta Individual** (Team es pago por uso y pide tarjeta; los 500 min/mes gratis van con Individual). Instalar la app de GitHub en la org `fuegoservicios-lab` con acceso sólo a `MealfitRD` (repo del frontend, donde vive el YAML).
5. **Subir la key** — avatar → Settings → Integrations → App Store Connect → Add key → nombre `bioboros_asc`.
6. **Variable + grupo + build** — env var `APP_STORE_APP_ID`; grupo TestFlight `Equipo Bioboros` con el dueño dentro; instalar TestFlight en el iPhone; Start new build → `iOS → TestFlight` → `main`.

## Estado: build #6 VERDE (2026-08-22 04:47) — `App.ipa` 3,95 MB en App Store Connect

Los 6 pasos del dueño se ejecutaron en vivo esa noche. Hicieron falta **seis builds**; los
cinco fallos eran todos del pipeline o del proyecto, ninguno del dueño. Cronología y lección de
cada uno en la memoria `project_p1_ios_codemagic_2026_08_22.md`. Dos variables de entorno en
Codemagic (grupo `default`, ambas secretas): `APP_STORE_APP_ID` y `CERTIFICATE_PRIVATE_KEY_B64`
(clave RSA en base64 — la semilla con la que Codemagic crea los certificados; la API key de
Apple autoriza, no firma. Entregar por FICHERO, nunca pegada desde un chat: la primera llegó
corrupta). Tiempo por build: ~3 min.

## Lo que esperar del primer build (histórico — ya ocurrió)

**Probablemente falle.** No hay Mac en el equipo: el proyecto Xcode nunca ha compilado en
macOS y lo típico es provisioning o `cap sync`. Se corrige leyendo el log de Codemagic;
uno o dos intentos. Cada intento consume ~10-15 min de los 500/mes gratuitos.

## Lo que se verifica con la app en el iPhone (antes de la ficha)

- Login funciona: la sesión viaja por `X-MF-Session` (localStorage), no por la cookie
  `__Host-` con `SameSite=Strict`, que no sale de `capacitor://`. Previsto, no probado.
- El escáner abre la cámara nativa (`NSCameraUsageDescription` en español).
- En Ajustes no aparece precio, «Mejorar plan» ni PayPal (gate `nativeHidesCommerce`).
- Las políticas abren en `bioboros.com` (P1-LEGAL-LINKS-APEX), no en la copia interna.

## Después, en orden

Sign in with Apple (4.8, obligatorio por tener Google) → ficha + capturas + cuestionario
de privacidad (derivado de `paypal_audit` y `politicas_verdad`) → cuenta de demo para
el revisor → envío. APNs y deep links no bloquean el envío.
