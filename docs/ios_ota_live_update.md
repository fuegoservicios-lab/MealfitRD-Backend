# Actualizaciones en vivo (OTA) de la app nativa

[P1-PLAN-LOTE-108 · 2026-09-19]

## Qué resuelve

El binario de iOS lleva la web **empaquetada** (`webDir: 'dist'`, sin `server`). Hasta este lote, un
arreglo desplegado en la web no llegaba al iPhone sin un build manual de Codemagic + el procesado de
Apple — y, ya en la App Store, su revisión de 1-2 días. Con OTA, **cada `deploy-mealfit.ps1 frontend`
llega también a la app nativa**, sin build ni revisión. Apple lo permite para HTML/CSS/JS mientras no
cambie el propósito de la app.

El build de Codemagic sigue haciendo falta SOLO cuando cambia algo nativo: un plugin, un permiso del
`Info.plist`, el icono, la versión de Capacitor.

## Cómo funciona

| Pieza | Dónde | Qué hace |
|---|---|---|
| Productor | `frontend/scripts/build-ota-bundle.mjs` (`npm run build:ota`) | Construye el bundle nativo en `dist-native` con su id horneado, repite la sanidad de Codemagic (sin IDs de la pasarela, API absoluta, id propio presente), lo comprime a `dist/ota/<id>.zip` y escribe `dist/ota/latest.json` con sha256 |
| Despliegue | `deploy-mealfit.ps1` | Lo corre en el VPS tras el build web y ANTES de publicar: zip y manifiesto viajan en la misma release atómica. Un fallo NO aborta la web, pero se grita en rojo |
| Consumidor | `frontend/src/native/liveUpdate.js` | Confirma el paquete que corre (`ready()`), lee el manifiesto por HTTP nativo, decide (`decidirOta`), descarga con checksum y lo deja para el siguiente arranque en frío |
| Plugin | `@capawesome/capacitor-live-update` (MIT, autoalojado, sin telemetría) | Descomprime, cambia el `serverBasePath` y vuelve atrás si el paquete no confirma |
| SSOT | `frontend/ota.config.json` | `enabled`, `reset`, `minNativeBuild`, foto de las deps nativas |

El **id** del paquete es un sello UTC ordenable (`20260919-045520`). La app solo acepta un paquete
**posterior** al que corre; el binario de Codemagic se sella con la hora de su build.

## Las redes de seguridad

1. **Vuelta atrás automática.** `readyTimeout: 10000` en `capacitor.config.ts`: un paquete que no
   llama a `ready()` en 10 s se descarta y la app vuelve sola al del binario. `ready()` se llama
   cuando la app PINTÓ (`mealfit:app-ready`), no al cargar el módulo. El paquete que provocó la
   vuelta atrás se apunta en `mf_ota_bloqueados` y no se reintenta; el suceso va a Sentry
   (`component: liveUpdate`).
2. **Compatibilidad con el binario.** `minNativeBuild` contra el build number real de la app
   (`CFBundleVersion`). Un paquete que usa un plugin nativo nuevo NO llega a binarios que no lo traen.
3. **Origen fijo.** La URL del zip tiene que ser exactamente `https://app.bioboros.com/ota/<id>.zip`.
4. **Sin recarga en caliente.** `reload()` puede dejar la pantalla en blanco si un recurso web tiene
   candado (lo documenta el plugin) y no se puede medir sin un iPhone: se aplica al siguiente
   arranque en frío. Al actualizar el binario, Capacitor descarta solo el paquete OTA anterior.

## Operación

- **Apagar el OTA** (dejar de ofrecer paquetes): `"enabled": false` en `ota.config.json` + deploy
  del frontend. Quien ya tenga un paquete se queda con él.
- **Botón del pánico** (devolver a TODOS al paquete del binario): `"reset": true` + deploy. Mientras
  esté en `true` nadie recibe paquetes: devuélvelo a `false` cuando el problema esté arreglado.
- **Al añadir o actualizar un plugin nativo**: `lote108.test.js` falla hasta que actualices
  `nativeDeps` — y ese es el momento de SUBIR `minNativeBuild` al número del próximo build de
  Codemagic y lanzar ese build. Orden: build de Codemagic primero, deploy del frontend después.
- **Comprobar qué se publicó**: `curl -s https://app.bioboros.com/ota/latest.json`.
- **El deploy dice «OTA FAIL»**: la web se publicó; la app nativa no recibe ese despliegue. Log en
  `/tmp/npm-ota.log` del VPS.

## Lo que NO está medido

Nada de esto se ha visto correr en un iPhone: el primer binario con el plugin es el build de
Codemagic posterior a este lote (≥ 13). Prueba de humo: instalar ese build, desplegar el frontend
con un cambio visible, abrir la app (descarga), cerrarla del todo y volver a abrirla (aplica).
La firma criptográfica del paquete (`publicKey` del plugin) queda fuera: exige custodiar una clave
privada, y HTTPS + HSTS sobre nuestro propio dominio es el mismo nivel de confianza que la PWA.
