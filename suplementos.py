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
    "whey_protein":  {"unidad": "scoop",   "gramos_porcion": 30, "kcal": 120, "protein_g": 24, "carbs_g": 3, "fats_g": 1.5},
    "vegan_protein": {"unidad": "scoop",   "gramos_porcion": 33, "kcal": 130, "protein_g": 21, "carbs_g": 6, "fats_g": 2.5},
    "creatine":      {"unidad": "g",       "gramos_porcion": 5,  "kcal": 0,   "protein_g": 0,  "carbs_g": 0, "fats_g": 0},
    "bcaa":          {"unidad": "scoop",   "gramos_porcion": 10, "kcal": 10,  "protein_g": 0,  "carbs_g": 2, "fats_g": 0},
    "pre_workout":   {"unidad": "scoop",   "gramos_porcion": 10, "kcal": 5,   "protein_g": 0,  "carbs_g": 1, "fats_g": 0},
    "fat_burner":    {"unidad": "capsula", "gramos_porcion": 1,  "kcal": 0,   "protein_g": 0,  "carbs_g": 0, "fats_g": 0},
    "collagen":      {"unidad": "scoop",   "gramos_porcion": 10, "kcal": 36,  "protein_g": 9,  "carbs_g": 0, "fats_g": 0},
    "multivitamin":  {"unidad": "capsula", "gramos_porcion": 1,  "kcal": 0,   "protein_g": 0,  "carbs_g": 0, "fats_g": 0},
    "omega3":        {"unidad": "capsula", "gramos_porcion": 1,  "kcal": 10,  "protein_g": 0,  "carbs_g": 0, "fats_g": 1},
    "magnesium":     {"unidad": "capsula", "gramos_porcion": 1,  "kcal": 0,   "protein_g": 0,  "carbs_g": 0, "fats_g": 0},
    "probiotics":    {"unidad": "capsula", "gramos_porcion": 1,  "kcal": 0,   "protein_g": 0,  "carbs_g": 0, "fats_g": 0},
    "electrolytes":  {"unidad": "porcion", "gramos_porcion": 7,  "kcal": 10,  "protein_g": 0,  "carbs_g": 2, "fats_g": 0},
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
    g = _num(etiqueta.get("gramos_porcion"))
    out = {"gramos_porcion": g if g is not None else 0}
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
    elif out["kcal"] > 600:
        return None          # sin gramos no hay techo físico: una porción de más de 600 kcal es una lectura rota
    # [P1-PLAN-LOTE-292 · revisión I3] Atwater: las kcal no pueden pasar de lo que dan sus macros (+ margen de redondeo).
    if out["kcal"] > 4 * out["protein_g"] + 4 * out["carbs_g"] + 9 * out["fats_g"] + 40:
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
    # [P1-PLAN-LOTE-300 · detalle M5] Primero el nombre EXACTO (sin mayúsculas ni tildes); si no, un parecido solo si es
    # el ÚNICO: «proteína» con whey y vegana en la Alacena es ambiguo y no se adivina.
    from constants import strip_accents
    _n = lambda s: " ".join(strip_accents(str(s or "")).lower().split())
    exacto = [f for f in filas if _n(f.get("ingredient_name")) == _n(nombre)]
    if exacto:
        return exacto[0]
    _pal = lambda s: set(_n(s).replace("(", " ").replace(")", " ").replace("/", " ").split())
    _q = _pal(nombre)
    parecidos = [f for f in filas
                 if (_q and _q <= _pal(f.get("ingredient_name")))       # «whey» ⊂ «Proteína Whey»
                 or pantry_names_match(f.get("ingredient_name") or "", nombre or "")]
    return parecidos[0] if len(parecidos) == 1 else None


def unidad_de_fila(unidad: str) -> str:
    """[P1-PLAN-LOTE-292 · revisión C2/C3] La `unit` con la que el pote vive en `user_inventory`: con prefijo `sup_`,
    así un pote y un alimento del mismo nombre y unidad («Creatina» en g) nunca chocan en el ON CONFLICT
    (user_id, ingredient_name, unit) — ni el alimento suma a las porciones del pote, ni el pote convierte al alimento.
    La unidad real vive en `serving_unit`."""
    return f"sup_{unidad}"


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
               serving_label = COALESCE(EXCLUDED.serving_label, user_inventory.serving_label),
               serving_unit = EXCLUDED.serving_unit,
               label_source = COALESCE(EXCLUDED.label_source, user_inventory.label_source),
               brand = COALESCE(EXCLUDED.brand, user_inventory.brand),
               updated_at = now()
        """,
        (user_id, nombre, float(porciones or 0), unidad_de_fila(unidad), json.dumps(etiqueta) if etiqueta else None,
         unidad, fuente, marca),
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
    # [P1-PLAN-LOTE-300 · detalle M5] Si ya hay un pote con ese nombre (o uno parecido e inequívoco), se actualiza ESE:
    # «proteína whey» no crea un segundo pote al lado de «Proteína Whey».
    try:
        _existente = buscar(user_id, nombre)
    except Exception:
        _existente = None
    if _existente:
        nombre = _existente.get("ingredient_name") or nombre
        unidad = _existente.get("serving_unit") or unidad
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
        (max(0.0, float(porciones or 0)), fila_id, user_id), fetch_one=True,   # [revisión I5] nunca rellena
    )
    return float((fila or {}).get("quantity") or 0)


# ── [P1-PLAN-LOTE-291] Lo que el coach sabe de suplementos y lo que hay en la Alacena del usuario ───────────────────

BLOQUE_CONOCIMIENTO = (
    "\n\n💊 SUPLEMENTOS (sabes esto; cita la porción de la ETIQUETA y no recetes dosis): creatina 3-5 g al día, 0 kcal, "
    "la hora da igual; whey ~24 g de proteína y ~120 kcal por scoop de 30 g; proteína vegetal parecida con algo más de "
    "carbohidrato; un ganador de peso trae 250-600 kcal por porción; un pre-entreno es cafeína (de noche aplica la "
    "regla T); el colágeno NO cuenta como proteína completa para su meta. Para registrar una toma de algo de su "
    "Alacena usa log_consumed_meal con suplemento=<nombre> y porciones=<n>: las macros salen de la etiqueta del pote. "
    "Para guardar un pote nuevo (lo pide, o manda la foto de la etiqueta y dice que es suyo), guardar_suplemento.")


def _potes(user_id):
    from db_core import execute_sql_query
    return execute_sql_query(
        # [SUPLEMENTOS-OK: lee justamente los suplementos]
        "SELECT ingredient_name, brand, quantity::float8 AS quantity, serving_unit, serving_label, label_source "
        "FROM user_inventory WHERE user_id = %s AND kind = 'supplement' ORDER BY ingredient_name LIMIT 20",
        (user_id,), fetch_all=True,
    ) or []


def bloque_para_chat(user_id: str) -> str:
    """El bloque 💊 del system prompt: el conocimiento fijo + SU ALACENA (cada pote con porciones y etiqueta)."""
    try:
        potes = _potes(user_id)
    except Exception:
        potes = []
    out = BLOQUE_CONOCIMIENTO
    if potes:
        lineas = []
        for p in potes:
            e = etiqueta_valida(p.get("serving_label"))
            u = p.get("serving_unit") or "porción"
            et = (f"1 {u}: {int(round(e['kcal']))} kcal, {e['protein_g']:g} g proteína "
                  f"({p.get('label_source') or 'estimado'})") if e else "sin etiqueta"
            marca = f" ({p['brand']})" if p.get("brand") else ""
            lineas.append(f"{p.get('ingredient_name')}{marca} — ~{int(float(p.get('quantity') or 0))} {u} — {et}")
        out += " SU ALACENA: " + "; ".join(lineas) + "."
    return out


# ── [P1-PLAN-LOTE-292] El formulario: «¿qué tomas?» y «¿te recomendamos?» ──────────────────────────────────────────

# La IA nunca los RECOMIENDA (poca evidencia o riesgo); si el usuario ya los toma, se respetan como suyos.
NO_RECOMENDAR = frozenset({"fat_burner", "pre_workout", "bcaa"})
RECOMENDABLES = frozenset(ESTIMADOS) - NO_RECOMENDAR


def normalizar_suplementos(fd) -> dict:
    """{'toma': [claves], 'recomendar': bool} desde el formulario nuevo (`currentSupplements` +
    `recommendSupplements`) o el viejo (`includeSupplements` + `selectedSupplements`: lo elegido era «lo quiero»,
    y el interruptor sin elección, «recomiéndame»). Espejo: frontend/src/utils/normalizarSuplementos.js."""
    fd = fd if isinstance(fd, dict) else {}
    if isinstance(fd.get("currentSupplements"), list):
        return {"toma": [s for s in fd["currentSupplements"] if isinstance(s, str)],
                "recomendar": bool(fd.get("recommendSupplements"))}
    if not fd.get("includeSupplements"):
        return {"toma": [], "recomendar": False}
    sel = [s for s in (fd.get("selectedSupplements") or []) if isinstance(s, str)]
    if sel:
        return {"toma": sel, "recomendar": False}
    return {"toma": [], "recomendar": True}


def suplementos_activos(fd) -> bool:
    """¿El plan lleva suplementos (los que toma o los que se le recomiendan)?"""
    n = normalizar_suplementos(fd)
    return bool(n["toma"] or n["recomendar"])
