# backend/admin_acceso.py
"""[P1-PLAN-LOTE-574 · 2026-09-27] Quién entra al panel de administración y el rastro de lo que ve.

Spec docs/superpowers/specs/2026-09-27-panel-admin-design.md §1. Entra SOLO quien esté en MEALFIT_ADMIN_USER_IDS (lista
en el .env del VPS) con MEALFIT_ADMIN_PANEL encendido; a cualquier otro el panel le responde 404: no se anuncia. Toda
vista de contenido de un usuario (capas 2-3) se anota ANTES de responder y, si no se puede anotar, no se responde.
"""
from __future__ import annotations

import json
from typing import Optional

from fastapi import Depends, HTTPException

from auth import get_verified_user_id
from db import execute_sql_write
from knobs import _env_bool, _env_str


def panel_encendido() -> bool:
    return _env_bool("MEALFIT_ADMIN_PANEL", False)


def admin_ids() -> frozenset:
    crudo = _env_str("MEALFIT_ADMIN_USER_IDS", "")
    return frozenset(x.strip().lower() for x in crudo.split(",") if x.strip())


def es_admin(user_id: Optional[str]) -> bool:
    return bool(user_id) and panel_encendido() and str(user_id).strip().lower() in admin_ids()


async def require_admin(verified_user_id: Optional[str] = Depends(get_verified_user_id)) -> str:
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="Autenticación requerida.")
    if not es_admin(verified_user_id):
        raise HTTPException(status_code=404, detail="Not Found")
    return verified_user_id


def registrar_acceso(admin_user_id: str, accion: str, objetivo: Optional[str] = None,
                     detalle: Optional[dict] = None) -> None:
    """Una fila en admin_access_log. Si no puede escribirla, LANZA: quien llama no responde sin rastro."""
    execute_sql_write(
        "INSERT INTO public.admin_access_log (admin_user_id, action, target, detail) VALUES (%s, %s, %s, %s::jsonb)",
        (admin_user_id, str(accion)[:64], objetivo, json.dumps(detalle or {}, ensure_ascii=False)),
    )
