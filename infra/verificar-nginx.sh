#!/usr/bin/env bash
# [P1-NGINX-RECONSTRUIBLE · 2026-08-19] La configuración viva se puede reconstruir
# desde el repositorio, y esto lo comprueba.
#
# POR QUÉ. `infra/nginx/` era una copia de cortesía: nadie garantizaba que
# estuviera completa ni al día. Medido el 2026-08-19, faltaba
# `snippets/bioboros-v2-mirror.conf` —un fichero que nginx CARGA— porque su única
# copia vivía en el repo del landing con otro nombre. Reconstruir desde este
# directorio habría dado un servidor sin las cabeceras del espejo, y nadie lo
# habría notado hasta que un buscador indexara v2 como original.
#
# LA AUTORIDAD ES `nginx -T`, NO UN LISTADO DE DIRECTORIO. Esa diferencia es todo
# el diseño. `/etc/nginx` está lleno de señuelos: `sites-available/mealfit` es la
# configuración de la era mealfitrd.com que ya nadie sirve —195 líneas contra las
# 694 vivas—, hay cinco `mealfit.bak.*` y un `.mealfit.antes-de-la-mudanza` dentro
# de `sites-enabled` que nginx ignora porque su glob no casa los ocultos.
# Comparar contra lo que hay en el disco produce alarmas falsas: a mí me convenció
# durante un rato de que la configuración principal había divergido en 560 líneas
# cuando el fichero vivo era byte-idéntico al repo. `nginx -T` imprime EXACTAMENTE
# los ficheros que la configuración efectiva incluye.
#
# Uso:  bash infra/verificar-nginx.sh
set -uo pipefail

VPS="${MEALFIT_VPS:-ubuntu@132.145.160.173}"
KEY="${MEALFIT_VPS_KEY:-$HOME/.ssh/mealfit-vps.key}"
AQUI="$(cd "$(dirname "$0")" && pwd)"
REPO="$AQUI/nginx"

fallos=0
ok()  { printf '  \033[32mOK\033[0m    %s\n' "$1"; }
mal() { printf '  \033[31mFALLA\033[0m %s\n' "$1"; fallos=$((fallos+1)); }

# Ficheros que la configuración carga pero NO son nuestros. Cada exclusión lleva
# su razón: una lista de exclusiones sin motivo es donde se esconde el fichero que
# sí importa.
ajeno() {
  case "$1" in
    */mime.types)                       return 0 ;;  # del paquete de Ubuntu
    /etc/nginx/modules-enabled/*)       return 0 ;;  # apt, suben con nginx
    /etc/letsencrypt/*)                 return 0 ;;  # certbot los reescribe al renovar
    /etc/nginx/nginx.conf)              return 0 ;;  # del sistema; NUESTRA parte
                                                     # (los `map`) se versiona
                                                     # aparte en nginx-http-maps.conf
    *) return 1 ;;
  esac
}

# Dónde vive en el repo cada fichero vivo.
espejo() {
  case "$1" in
    /etc/nginx/sites-enabled/mealfit)     echo "$REPO/mealfit.conf" ;;
    /etc/nginx/sites-enabled/bioboros-v2) echo "$REPO/bioboros-v2.conf" ;;
    /etc/nginx/snippets/*)                echo "$REPO/snippets/$(basename "$1")" ;;
    *) echo "" ;;
  esac
}

echo "== Reconstruibilidad de nginx desde el repositorio =="

cargados="$(ssh -i "$KEY" -o ConnectTimeout=20 "$VPS" \
  "sudo nginx -T 2>/dev/null | grep -oE '^# configuration file [^:]+' | sed 's/# configuration file //' | sort -u")"

if [ -z "$cargados" ]; then
  echo "  no pude leer la configuración efectiva del servidor"
  exit 2
fi

# Cuantos DEBERIAN revisarse, contado antes del bucle. Sin esta cifra, un bucle
# que termina antes de tiempo es indistinguible de uno que no tenia mas trabajo.
esperados=0
while read -r f; do
  [ -z "$f" ] && continue
  ajeno "$f" || esperados=$((esperados + 1))
done <<< "$cargados"

nuestros=0
while read -r f; do
  [ -z "$f" ] && continue
  if ajeno "$f"; then continue; fi
  nuestros=$((nuestros + 1))
  local_f="$(espejo "$f")"
  if [ -z "$local_f" ]; then
    mal "$f: nginx lo carga y no sé dónde debería vivir en el repo"
    continue
  fi
  if [ ! -f "$local_f" ]; then
    mal "$f: nginx lo carga y NO existe en el repo ($local_f)"
    continue
  fi
  # `-n` OBLIGATORIO: sin el, ssh hereda stdin y se traga el resto del bucle.
  # La primera version comprobo UN fichero de seis y dijo «TODO OK» — el mismo
  # modo de fallo que este directorio existe para impedir, cometido aqui dentro.
  ssh -n -i "$KEY" -o ConnectTimeout=20 "$VPS" "sudo cat '$f'" > /tmp/nginx-vivo.$$ 2>/dev/null
  if diff -q <(tr -d '\r' < /tmp/nginx-vivo.$$) <(tr -d '\r' < "$local_f") >/dev/null 2>&1; then
    ok "$(basename "$f")"
  else
    n="$(diff <(tr -d '\r' < /tmp/nginx-vivo.$$) <(tr -d '\r' < "$local_f") | grep -c '^[<>]')"
    mal "$(basename "$f"): DIFIERE del repo en $n línea(s) — $local_f"
  fi
  rm -f /tmp/nginx-vivo.$$
done <<< "$cargados"

# Y al revés: un fichero en el repo que nginx no carga es, o bien un olvido de
# despliegue, o bien basura que alguien copiará creyendo que está viva.
echo "-- Ficheros del repo que la configuración NO carga --"
for l in "$REPO"/*.conf "$REPO"/snippets/*.conf; do
  [ -f "$l" ] || continue
  b="$(basename "$l")"
  case "$b" in nginx-http-maps.conf) continue ;; esac  # fragmento de nginx.conf, a propósito
  if grep -qF "/$b" <<< "$cargados" || grep -qE "/(sites-enabled)/${b%.conf}$" <<< "$cargados"; then
    :
  else
    mal "$b está en el repo y nginx NO lo carga (¿sin desplegar, o ya muerto?)"
  fi
done

if [ "$nuestros" -ne "$esperados" ]; then
  mal "revise $nuestros fichero(s) de $esperados: el bucle termino antes de tiempo"
fi

echo
if [ "$fallos" -eq 0 ]; then
  echo "TODO OK — $nuestros fichero(s) nuestros, todos reconstruibles desde el repo"
else
  echo "$fallos problema(s) de reconstruibilidad"
fi
exit "$fallos"
