#!/usr/bin/env bash
# [P1-ARQ25-F1-CLOSE · 2026-09-02] Drain cooperativo ANTES de reiniciar el backend (§5.5).
#
# El deploy de las 12:27 UTC reinició el servicio con un chunk inicial a mitad del ensamblado:
# systemd remata a los 10 s y un pipeline dura minutos, así que el drain del shutdown no llega.
# Este script corre en el VPS, tras extraer el tarball y antes de `systemctl restart`: pide al
# worker que deje de reclamar (`POST /api/system/admin/worker-drain`) y espera, en tramos de
# 20 s y hasta 12 min, a que no quede ningún tick en vuelo.
#
# Vive como fichero (no inline en deploy-mealfit.ps1) porque una cadena entre comillas simples
# de PowerShell con comillas anidadas llegó rota al bash remoto («syntax error near `('»).
#
# Salidas: SIEMPRE 0. 404 (binario viejo sin endpoint) o servicio caído ⇒ seguir; timeout ⇒
# avisar y seguir (el zombie rescue recupera lo que quede, como hasta hoy).
set -u
ENV_FILE="${ENV_FILE:-/opt/mealfit/backend/.env}"
URL="${DRAIN_URL:-http://127.0.0.1:3001/api/system/admin/worker-drain}"
MAX_ROUNDS="${DRAIN_MAX_ROUNDS:-36}"   # 36 × 20 s = 12 min
WAIT_S="${DRAIN_WAIT_S:-20}"

TOK=$(grep -E '^CRON_SECRET=' "$ENV_FILE" 2>/dev/null | cut -d= -f2- | tr -d '"'"'"' \r')
if [ -z "$TOK" ]; then
  echo "drain: sin CRON_SECRET en $ENV_FILE -> sigo sin drenar"
  exit 0
fi

for i in $(seq 1 "$MAX_ROUNDS"); do
  RESP=$(curl -s -m 30 -w '\n%{http_code}' -X POST "$URL" \
    -H "Authorization: Bearer $TOK" -H 'Content-Type: application/json' \
    -d "{\"wait_s\":$WAIT_S}" 2>/dev/null) || RESP=$'\n000'
  CODE=$(printf '%s\n' "$RESP" | tail -n 1)
  BODY=$(printf '%s\n' "$RESP" | head -n 1)
  if [ "$CODE" != "200" ]; then
    echo "drain: HTTP $CODE (sin endpoint o servicio caido) -> sigo"
    exit 0
  fi
  N=$(printf '%s' "$BODY" | sed -E 's/.*"ticks_in_flight": *([0-9]+).*/\1/')
  echo "drain: ticks en vuelo=$N (intento $i/$MAX_ROUNDS)"
  if [ "$N" = "0" ]; then
    exit 0
  fi
done
echo "drain: TIMEOUT con ticks en vuelo -> reinicio igual (zombie rescue)"
exit 0
