# `infra/` — la capa de entrega, versionada

[P2-INFRA-EDGE-SSOT · 2026-08-14]

Hasta hoy, **toda** la capa de entrega de producción vivía únicamente en el disco
de un VPS Oracle Always Free: los server blocks de los dos dominios, TLS, los 6
security headers replicados en cada `location`, el fallback del SPA, los timeouts
de 3600 s que el SSE de generación necesita, la unidad de systemd y el
`publish-marker.sh` que el deploy invoca en cada despliegue. `find . -name "*.conf"`
en el repo devolvía **cero**.

No había drift —`curl -sI https://bioboros.com/` devolvía los 6 headers vivos y
correctos, HSTS con `preload` incluido— así que esto nunca fue un incidente. Era
riesgo de reconstrucción: si ese disco desaparece, el estado de producción no se
puede reproducir, y ningún cambio en él pasa por revisión.

## Qué hay aquí

| Fichero | Destino en el VPS |
|---|---|
| `nginx/mealfit.conf` | `/etc/nginx/sites-enabled/mealfit` |
| `nginx/snippets/mealfit-security.conf` | `/etc/nginx/snippets/mealfit-security.conf` |
| `systemd/mealfit-backend.service` | `/etc/systemd/system/mealfit-backend.service` |
| `scripts/publish-marker.sh` | `/opt/mealfit/publish-marker.sh` |

## Qué NO hay aquí, y no por olvido

Certificados, claves privadas y ficheros `.env`. Los gestiona certbot y el propio
VPS; versionarlos sería exactamente el fallo que `P2-DEPLOY-ENV-GUARD` cierra en
el otro extremo.

## ⚠️ Esto no es un espejo automático

Estos ficheros son la **fuente**, pero nada los sincroniza solo. Un cambio hecho
a mano por SSH deja el repo mintiendo, y un fichero versionado que nadie aplica
es igual de inútil.

Y ojo con la clase de test que NO sirve aquí: un guard parser-based leería este
directorio, no lo que nginx está sirviendo — daría verde con producción
divergida. El único detector honesto es un `curl` contra el sitio real, que es lo
que hace `verificar-edge.sh`.

## Aplicar un cambio

```bash
# 1. Editar aquí, no en el VPS.
# 2. Subir y VALIDAR antes de recargar (nginx -t es obligatorio, no opcional:
#    una config rota que se recarga tumba los dos dominios).
scp -i ~/.ssh/mealfit-vps.key infra/nginx/mealfit.conf ubuntu@132.145.160.173:/tmp/
ssh -i ~/.ssh/mealfit-vps.key ubuntu@132.145.160.173 '
  sudo cp /etc/nginx/sites-enabled/mealfit /etc/nginx/sites-enabled/mealfit.bak.$(date +%s) &&
  sudo cp /tmp/mealfit.conf /etc/nginx/sites-enabled/mealfit &&
  sudo nginx -t && sudo systemctl reload nginx'

# 3. Verificar contra producción (no contra el fichero).
bash infra/verificar-edge.sh
```

Si `nginx -t` falla, la recarga no ocurre y el `.bak` de la línea anterior es el
rollback: `sudo cp <el .bak más reciente> /etc/nginx/sites-enabled/mealfit`.
