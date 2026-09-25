# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-290 · 2026-09-25] SSOT de los suplementos en la Alacena (`user_inventory.kind = 'supplement'`).

La etiqueta por porción la decide el ENVASE (foto o marca); sin ella, un estimado genérico marcado como tal. Las
macros de una toma las calcula este módulo (etiqueta × porciones), nunca el modelo. Todo lector que trata las filas de
la Nevera como INGREDIENTES filtra `kind = 'food'` (guard por AST en tests/test_p1_plan_lote_290.py).
Spec: docs/superpowers/specs/2026-09-25-suplementos-alacena-design.md. tooltip-anchor: P1-PLAN-LOTE-290
"""
from __future__ import annotations

UNIDADES = ("scoop", "capsula", "porcion", "g")
FUENTES = ("foto", "marca", "estimado")
_CAMPOS = ("kcal", "protein_g", "carbs_g", "fats_g")

# Estimados genéricos POR PORCIÓN (unidad típica del producto). Claves = constants.SUPPLEMENT_NAMES.
ESTIMADOS = {
    "whey_protein":  {"unidad": "scoop",   "serving_g": 30, "kcal": 120, "protein_g": 24, "carbs_g": 3, "fats_g": 1.5},
    "vegan_protein": {"unidad": "scoop",   "serving_g": 33, "kcal": 130, "protein_g": 21, "carbs_g": 6, "fats_g": 2.5},
    "creatine":      {"unidad": "g",       "serving_g": 5,  "kcal": 0,   "protein_g": 0,  "carbs_g": 0, "fats_g": 0},
    "bcaa":          {"unidad": "scoop",   "serving_g": 10, "kcal": 10,  "protein_g": 0,  "carbs_g": 2, "fats_g": 0},
    "pre_workout":   {"unidad": "scoop",   "serving_g": 10, "kcal": 5,   "protein_g": 0,  "carbs_g": 1, "fats_g": 0},
    "fat_burner":    {"unidad": "capsula", "serving_g": 1,  "kcal": 0,   "protein_g": 0,  "carbs_g": 0, "fats_g": 0},
    "collagen":      {"unidad": "scoop",   "serving_g": 10, "kcal": 36,  "protein_g": 9,  "carbs_g": 0, "fats_g": 0},
    "multivitamin":  {"unidad": "capsula", "serving_g": 1,  "kcal": 0,   "protein_g": 0,  "carbs_g": 0, "fats_g": 0},
    "omega3":        {"unidad": "capsula", "serving_g": 1,  "kcal": 10,  "protein_g": 0,  "carbs_g": 0, "fats_g": 1},
    "magnesium":     {"unidad": "capsula", "serving_g": 1,  "kcal": 0,   "protein_g": 0,  "carbs_g": 0, "fats_g": 0},
    "probiotics":    {"unidad": "capsula", "serving_g": 1,  "kcal": 0,   "protein_g": 0,  "carbs_g": 0, "fats_g": 0},
    "electrolytes":  {"unidad": "porcion", "serving_g": 7,  "kcal": 10,  "protein_g": 0,  "carbs_g": 2, "fats_g": 0},
}


def _num(v):
    try:
        f = float(v)
        return f if f == f and f >= 0 else None
    except (TypeError, ValueError):
        return None


def etiqueta_valida(etiqueta) -> dict | None:
    """La etiqueta normalizada, o None si falta o es inverosímil (kcal > 9·g + 20, o macros > gramos + 1)."""
    if not isinstance(etiqueta, dict):
        return None
    g = _num(etiqueta.get("serving_g"))
    out = {"serving_g": g if g is not None else 0}
    for k in _CAMPOS:
        v = _num(etiqueta.get(k, 0))
        if v is None:
            return None
        out[k] = v
    if g:
        if out["kcal"] > 9 * g + 20:
            return None
        if out["protein_g"] + out["carbs_g"] + out["fats_g"] > g + 1:
            return None
    return out


def macros_de_porciones(etiqueta, porciones) -> dict | None:
    """Etiqueta × porciones, redondeado a 1 decimal. None sin etiqueta válida o sin porciones."""
    e = etiqueta_valida(etiqueta)
    n = _num(porciones)
    if not e or not n:
        return None
    return {k: round(e[k] * n, 1) for k in _CAMPOS}


# ── [P1-PLAN-LOTE-291] Escritura y lectura de los potes ──────────────────────────────────────────────────────────────

def buscar(user_id: str, nombre: str) -> dict | None:
    """El pote de `nombre` en la Alacena del usuario (mismo criterio de nombre que la Nevera), o None."""
    from db_core import execute_sql_query
    from constants import pantry_names_match
    filas = execute_sql_query(
        # [SUPLEMENTOS-OK: lee justamente los suplementos]
        "SELECT id, ingredient_name, brand, quantity::float8 AS quantity, unit, serving_unit, serving_label, label_source "
        "FROM user_inventory WHERE user_id = %s AND kind = 'supplement'",
        (user_id,), fetch_all=True,
    ) or []
    return next((f for f in filas if pantry_names_match(f.get("ingredient_name") or "", nombre or "")), None)


def _upsert(user_id, nombre, marca, porciones, unidad, etiqueta, fuente):
    import json
    from db_core import execute_sql_write
    execute_sql_write(
        """
        INSERT INTO user_inventory (user_id, ingredient_name, quantity, unit, kind, serving_label, serving_unit,
                                    label_source, brand, source, last_mutation_type)
        VALUES (%s, %s, %s, %s, 'supplement', %s::jsonb, %s, %s, %s, 'chat', 'manual')
        ON CONFLICT (user_id, ingredient_name, unit) DO UPDATE
           SET quantity = CASE WHEN EXCLUDED.quantity > 0 THEN EXCLUDED.quantity ELSE user_inventory.quantity END,
               kind = 'supplement',
               serving_label = COALESCE(EXCLUDED.serving_label, user_inventory.serving_label),
               serving_unit = EXCLUDED.serving_unit,
               label_source = COALESCE(EXCLUDED.label_source, user_inventory.label_source),
               brand = COALESCE(EXCLUDED.brand, user_inventory.brand),
               updated_at = now()
        """,
        (user_id, nombre, float(porciones or 0), unidad, json.dumps(etiqueta) if etiqueta else None, unidad,
         fuente, marca),
    )


def guardar(user_id, nombre, marca=None, porciones=None, unidad="scoop", etiqueta=None, fuente="estimado",
            clave=None, forzar_nevera=False, usar_estimado=True) -> dict:
    """Guarda (o actualiza) el pote. Pasa antes por la regla de la Nevera: si la apagó el usuario, no escribe y
    devuelve ok=False con estado 'preguntar'. Etiqueta inverosímil o ausente ⇒ el estimado de `clave` (si
    `usar_estimado`) marcado como tal, o sin etiqueta."""
    from nevera_opcional import encender_por_uso
    estado = encender_por_uso(user_id, forzar=bool(forzar_nevera))
    if estado == "preguntar":
        return {"ok": False, "estado_nevera": estado, "etiqueta": None, "fuente": fuente}
    unidad = unidad if unidad in UNIDADES else "porcion"
    e = etiqueta_valida(etiqueta)
    if e is None or fuente not in FUENTES:
        fuente = "estimado"
        base = ESTIMADOS.get(clave or "") if usar_estimado else None
        e = etiqueta_valida(base) if base else None
    _upsert(user_id, str(nombre).strip(), marca, porciones, unidad, e, fuente if e else None)
    return {"ok": True, "estado_nevera": estado, "etiqueta": e, "fuente": fuente}


def descontar(user_id: str, fila_id, porciones) -> float:
    """Resta `porciones` del pote (nunca por debajo de 0) y devuelve las que quedan. Lo tomado manda: si tomó más de lo
    que el pote decía, se registra igual y el pote queda en 0."""
    from db_core import execute_sql_query
    fila = execute_sql_query(
        # [SUPLEMENTOS-OK: descuenta del pote de un suplemento]
        "UPDATE user_inventory SET quantity = GREATEST(0, quantity - %s), updated_at = now() "
        "WHERE id = %s AND user_id = %s AND kind = 'supplement' RETURNING quantity::float8 AS quantity",
        (float(porciones or 0), fila_id, user_id), fetch_one=True,
    )
    return float((fila or {}).get("quantity") or 0)
