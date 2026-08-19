#!/usr/bin/env bash
# [P2-INFRA-EDGE-SSOT · 2026-08-14] Verifica la capa de entrega contra PRODUCCIÓN.
#
# Por qué un script de `curl` y no un test parser-based: un test que leyera
# `infra/nginx/mealfit.conf` comprobaría lo que el repo DICE, no lo que nginx
# está sirviendo. Daría verde con producción divergida, que es precisamente el
# fallo que este directorio existe para hacer visible.
#
# Uso:  bash infra/verificar-edge.sh [origen]
set -u
ORIGEN="${1:-https://bioboros.com}"
fallos=0

ok()   { printf '  \033[32mOK\033[0m    %s\n' "$1"; }
mal()  { printf '  \033[31mFALLA\033[0m %s\n' "$1"; fallos=$((fallos+1)); }

echo "== Verificando $ORIGEN =="

cab="$(curl -sS -I --max-time 20 "$ORIGEN/" || true)"
[ -z "$cab" ] && { echo "  no hubo respuesta"; exit 1; }

echo "-- Headers de seguridad (los 6) --"
for h in "strict-transport-security" "x-content-type-options" "x-frame-options" \
         "referrer-policy" "permissions-policy"; do
  grep -qi "^$h:" <<<"$cab" && ok "$h" || mal "$h ausente"
done

# [P3-CSP-ENSAYO . 2026-08-19] La Report-Only es un ENSAYO, y sobra cuando ya se
# estrena. Este bucle la exigia siempre, y por eso daba rojo contra el apex:
# desde P1-01 ese host sirve IMPUESTA la politica estricta completa —medido:
# `default-src self`, `script-src self` sin `unsafe-inline`, `object-src none`,
# `frame-ancestors none`—. Pedirle ademas un ensayo de lo que ya hace es pedir
# una cabecera sin funcion, y un rojo permanente por algo que no hay que
# arreglar entrena a ignorar todo lo demas.
#
# Asi que la regla es condicional: si la impuesta AUN no es estricta, la
# Report-Only es obligatoria (es el unico modo de endurecer sin romper). Si ya
# lo es, es opcional y se dice.
csp_enf="$(grep -i "^content-security-policy:" <<<"$cab" || true)"
if grep -qi "default-src" <<<"$csp_enf" && ! grep -qi "script-src[^;]*unsafe-inline" <<<"$csp_enf"; then
  ok "CSP impuesta ya estricta (la Report-Only es opcional aqui)"
else
  grep -qi "^content-security-policy-report-only:" <<<"$cab" \
    && ok "content-security-policy-report-only (ensayo de la politica estricta)" \
    || mal "la CSP impuesta no es estricta y NO hay Report-Only con la que endurecerla"
fi

echo "-- CSP enforced (P2-CSP-ENFORCE movimiento 1) --"
# `-x` para no confundirla con la Report-Only, que también empieza por ese nombre.
csp="$(grep -i "^content-security-policy:" <<<"$cab" || true)"
if [ -n "$csp" ]; then
  ok "cabecera presente"
  for d in "base-uri" "form-action" "object-src" "frame-ancestors"; do
    grep -qi "$d" <<<"$csp" && ok "  $d" || mal "  $d ausente de la CSP enforced"
  done
  # [P3-CSP-HOST . 2026-08-19] PayPal se exige donde VIVE el checkout, no aqui.
  #
  # Esta comprobacion daba rojo contra el apex, y el rojo era falso: desde que la
  # portada se mudo al sitio estatico, en bioboros.com no hay ningun formulario
  # —medido: cero `<form action=>` en /precios—. El pago vive en
  # app.bioboros.com/dashboard/upgrade, cuya CSP si lo permite.
  #
  # Exigir PayPal en la CSP del apex es pedir un permiso que ese host no necesita,
  # y ampliar una politica sin motivo la debilita. Un verificador que falla siempre
  # por algo que no hay que arreglar entrena a ignorar su salida entera.
  case "$ORIGEN" in
    *app.*)
      grep -qi "paypal" <<<"$csp" && ok "  form-action deja pasar PayPal" \
        || mal "  form-action SIN PayPal: rompe el checkout" ;;
    *)
      grep -qi "paypal" <<<"$csp" \
        && mal "  el apex permite PayPal en form-action y aqui no hay checkout" \
        || ok "  form-action sin PayPal (correcto: el checkout no vive en este host)" ;;
  esac
else
  mal "no hay CSP enforced (sólo Report-Only)"
fi

echo "-- Compresión --"
# Se lee de la CABECERA y no de `-w %{content_encoding}`: esa variable de curl no
# existe en todas las versiones (la del entorno de desarrollo no la tiene) y
# devolvía vacío, o sea un FALLA contra un servidor que comprimía bien.
codificacion() { curl -sS -I -H 'Accept-Encoding: gzip' --max-time 20 "$1" \
                 | grep -i '^content-encoding:' | tr -d '\r' | awk '{print $2}'; }

enc="$(codificacion "$ORIGEN/" || true)"
[ "$enc" = "gzip" ] && ok "index.html gzip" || mal "index.html sin gzip (fue: '$enc')"

# El primer .js que referencie el HTML: es lo que de verdad pesa.
js="$(curl -sS --max-time 20 "$ORIGEN/" | grep -oE '/assets/[A-Za-z0-9._-]+\.js' | head -1 || true)"
if [ -n "$js" ]; then
  enc="$(codificacion "$ORIGEN$js" || true)"
  [ "$enc" = "gzip" ] && ok "$js gzip" || mal "$js SIN gzip"
fi

echo "-- Cache-Control (el que rompe despliegues si se equivoca) --"
cc="$(curl -sS -I --max-time 20 "$ORIGEN/" | grep -i '^cache-control:' || true)"
grep -qi 'no-cache' <<<"$cc" && ok "index.html no-cache" \
  || mal "index.html CACHEABLE ('$cc') -- servirá bundles viejos tras cada deploy"

if [ -n "${js:-}" ]; then
  cc="$(curl -sS -I --max-time 20 "$ORIGEN$js" | grep -i '^cache-control:' || true)"
  grep -qi 'immutable' <<<"$cc" && ok "assets immutable" || mal "assets sin immutable ('$cc')"
fi

echo "-- Meta por ruta (P2-LANDING-PRERENDER-META) --"
# Se comprueba SIN seguir redirects a propósito: el primer intento sirvió el
# contenido correcto detrás de un 301 a la ruta con barra final, y un `curl -L`
# lo habría dado por bueno.
for ruta in precios motor supermercado; do
  cod="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 20 "$ORIGEN/$ruta" || true)"
  if [ "$cod" != "200" ]; then
    mal "/$ruta responde $cod (¿301 de barra final? mira try_files)"
    continue
  fi
  cuerpo="$(curl -sS --max-time 20 "$ORIGEN/$ruta" || true)"
  url="$(printf '%s' "$cuerpo" | tr '>' '\n' | sed -n 's/.*og:url" content="\([^"]*\)".*/\1/p' | head -1)"
  case "$url" in
    */"$ruta") ok "/$ruta se declara canónica de sí misma" ;;
    *) mal "/$ruta declara og:url='$url' (debería terminar en /$ruta)" ;;
  esac
done

echo "-- Sourcemaps NO servibles (segunda barrera) --"
if [ -n "${js:-}" ]; then
  cod="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 20 "$ORIGEN$js.map" || true)"
  [ "$cod" = "404" ] && ok ".map devuelve 404" || mal ".map devuelve $cod -- ¡fuente público!"
fi

# [P3-BROTLI . 2026-08-19] La compresion, contra produccion y no contra el .conf.
#
# Se comprueban las DOS ramas a proposito. Un cliente que anuncia `br` debe
# recibir brotli; uno que solo anuncia gzip debe seguir recibiendo gzip. La forma
# facil de romper esto es dejar solo brotli: los clientes viejos se quedarian sin
# comprimir y nadie lo notaria, porque el navegador del que mira SI habla brotli.
#
# Y `Vary: Accept-Encoding` es lo que impide que una cache intermedia le sirva
# brotli a quien no lo entiende. Sin esa cabecera, esto no es una optimizacion:
# es una pagina rota para una parte del publico, de forma intermitente.
if [ -n "${js:-}" ]; then
  enc_br="$(curl -sS -o /dev/null -H 'Accept-Encoding: br' -w '%{content_type}' --max-time 20 "$ORIGEN$js" >/dev/null 2>&1; curl -sSI -H 'Accept-Encoding: br' --max-time 20 "$ORIGEN$js" | grep -i "^content-encoding:" | tr -d "" | awk '{print $2}')"
  case "$enc_br" in
    br) ok "brotli servido a quien lo anuncia" ;;
    *)  mal "con Accept-Encoding: br llego content-encoding='${enc_br:-ninguno}'" ;;
  esac

  enc_gz="$(curl -sSI -H 'Accept-Encoding: gzip' --max-time 20 "$ORIGEN$js" | grep -i "^content-encoding:" | tr -d "" | awk '{print $2}')"
  case "$enc_gz" in
    gzip) ok "gzip intacto para quien no habla brotli" ;;
    *)    mal "con Accept-Encoding: gzip llego content-encoding='${enc_gz:-ninguno}'" ;;
  esac

  vary="$(curl -sSI -H 'Accept-Encoding: br' --max-time 20 "$ORIGEN$js" | grep -i "^vary:" | tr -d "")"
  case "$vary" in
    *Accept-Encoding*|*accept-encoding*) ok "Vary: Accept-Encoding presente" ;;
    *) mal "falta Vary: Accept-Encoding -- una cache puede servir brotli a quien no lo entiende" ;;
  esac
fi
echo
if [ "$fallos" -eq 0 ]; then
  echo "TODO OK contra $ORIGEN"
else
  echo "$fallos comprobacion(es) FALLARON contra $ORIGEN"
fi
exit "$fallos"
