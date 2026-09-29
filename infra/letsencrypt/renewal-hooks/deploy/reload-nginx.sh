#!/bin/sh
# [P1-PLAN-LOTE-797 · 2026-09-28] recarga nginx tras renovar un certificado: sin esto seguia sirviendo el viejo
systemctl reload nginx
