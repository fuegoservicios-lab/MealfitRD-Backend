"""[P1-PLAN-LOTE-1160 · 2026-11-21] Papelera de la Nevera: lo que el usuario borra se recupera durante 30 días.

Caso vivo del dueño (21-nov): «Borrar todos» vació la Nevera y los tres potes de su Despensa con su etiqueta (DELETE
duro). Le pidió al coach recuperarlos; el coach no tenía de dónde, dijo «Recuperé…» sin haber recuperado nada y acabó
anotando los suplementos como TOMADOS. Se rescataron desde una instantánea de Neon.

Aquí vive todo borrado que pide el usuario (uno, «Borrar todos», el coach): el DELETE y el archivo de la fila ENTERA
(`to_jsonb`: etiqueta, marca, tipo, unidad) van en la MISMA sentencia, así que no hay borrado sin copia. Restaurar
reinserta la fila tal cual; si entretanto volvió a añadir el mismo alimento, se suman las cantidades y la etiqueta
que faltara se rellena. Tabla `user_inventory_papelera` (migración p1_plan_lote_1160_papelera_nevera.sql).

Fail-open: si la tabla aún no existe (deploy antes que la migración), el borrado se hace igual, sin copia, y se avisa
en el log — no recuperar algo es malo, pero no poder borrar es peor. tooltip-anchor: P1-PLAN-LOTE-1160-PAPELERA
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

DIAS_PAPELERA = 30
VIAS = ("uno", "todos", "coach")


def _write(sql: str, params: tuple) -> list:
    from db import execute_sql_write
    return execute_sql_write(sql, params, returning=True) or []


def _query(sql: str, params: tuple) -> list:
    from db import execute_sql_query
    return execute_sql_query(sql, params, fetch_all=True) or []


def _falta_la_tabla(exc: Exception) -> bool:
    texto = f"{type(exc).__name__} {exc}".lower()
    return "user_inventory_papelera" in texto and ("does not exist" in texto or "undefinedtable" in texto)


def borrar_y_archivar(user_id: str, *, ids: Optional[Iterable[int]] = None, solo_alimentos: bool = False,
                      via: str = "uno") -> list:
    """Borra filas de la Nevera del usuario y las archiva. Devuelve una por fila borrada:
    `{original_id, papelera_id, ingredient_name, kind}` (`papelera_id` None si no se pudo archivar).

    - `ids`: solo esas filas (None = todas las del usuario).
    - `solo_alimentos`: deja los suplementos (lo usa «Borrar todos»: vacía la Nevera, no la Despensa de potes).
    """
    if via not in VIAS:
        raise ValueError(f"via desconocida: {via!r}")
    lista_ids = None if ids is None else [int(i) for i in ids]
    if lista_ids is not None and not lista_ids:
        return []
    filtro, params = "user_id = %s", [user_id]
    if lista_ids is not None:
        filtro += " AND id = ANY(%s)"
        params.append(lista_ids)
    if solo_alimentos:
        filtro += " AND kind = 'food'"
    sql = (
        # [SUPLEMENTOS-OK: borra lo que eligió el usuario; `solo_alimentos` decide si toca los potes]
        f"""
        WITH borradas AS (
            DELETE FROM public.user_inventory WHERE {filtro} RETURNING *
        ), archivadas AS (
            INSERT INTO public.user_inventory_papelera (user_id, original_id, fila, via)
            SELECT b.user_id, b.id, to_jsonb(b), %s FROM borradas b
            RETURNING id AS papelera_id, original_id, fila->>'ingredient_name' AS ingredient_name, fila->>'kind' AS kind
        )
        SELECT * FROM archivadas
        """
    )
    try:
        filas = _write(sql, (*params, via))
    except Exception as exc:
        if not _falta_la_tabla(exc):
            raise
        logger.warning("🗑️ [P1-PLAN-LOTE-1160] sin papelera (¿migración pendiente?): borrado SIN copia")
        filas = _write(
            # [SUPLEMENTOS-OK: mismo borrado, sin papelera]
            f"DELETE FROM public.user_inventory WHERE {filtro} "
            "RETURNING NULL::bigint AS papelera_id, id AS original_id, ingredient_name, kind",
            tuple(params),
        )
        return [dict(f) for f in filas]
    _purgar(user_id)
    return [dict(f) for f in filas]


def _purgar(user_id: str) -> None:
    try:
        from db import execute_sql_write
        execute_sql_write(
            "DELETE FROM public.user_inventory_papelera WHERE user_id = %s "
            f"AND borrado_at < now() - interval '{int(DIAS_PAPELERA)} days'",
            (user_id,),
        )
    except Exception as exc:   # la purga es higiene: nunca tumba el borrado
        logger.debug(f"[P1-PLAN-LOTE-1160] purga de la papelera falló: {exc!r}")


def listar(user_id: str, limite: int = 60) -> list:
    """Lo borrado en los últimos 30 días, lo más reciente primero (un nombre+unidad aparece UNA vez: la última)."""
    try:
        filas = _query(
            f"""
            SELECT DISTINCT ON (fila->>'ingredient_name', fila->>'unit')
                id AS papelera_id, original_id, via,
                to_jsonb(borrado_at)#>>'{{}}' AS borrado_at,
                fila->>'ingredient_name' AS ingredient_name,
                (fila->>'quantity')::float8 AS quantity,
                fila->>'unit' AS unit,
                COALESCE(fila->>'kind', 'food') AS kind,
                fila->>'brand' AS brand,
                (fila->'serving_label') IS NOT NULL AND jsonb_typeof(fila->'serving_label') = 'object' AS con_etiqueta,
                borrado_at AS _orden
            FROM public.user_inventory_papelera
            WHERE user_id = %s AND borrado_at >= now() - interval '{int(DIAS_PAPELERA)} days'
            ORDER BY fila->>'ingredient_name', fila->>'unit', borrado_at DESC
            """,
            (user_id,),
        )
    except Exception as exc:
        if _falta_la_tabla(exc):
            return []
        raise
    filas = sorted((dict(f) for f in filas), key=lambda f: f.get("_orden"), reverse=True)
    for f in filas:
        f.pop("_orden", None)
    return filas[: max(1, int(limite))]


def buscar_por_nombres(user_id: str, nombres: Iterable[str]) -> tuple[list, list]:
    """(encontradas, no_encontradas): cada nombre contra lo borrado, por el SSOT de identidad de la Nevera
    (`constants.pantry_names_match`) y, si no casa, por inclusión de palabras («multi» → «DESIGNS FOR SPORT MULTI»)."""
    from constants import pantry_names_match, strip_accents
    disponibles = listar(user_id)
    encontradas, faltan, vistos = [], [], set()

    def _palabras(t: str) -> set:
        return {w for w in strip_accents(str(t or "")).lower().replace("+", " ").split() if len(w) > 2}

    for nombre in nombres:
        nombre = str(nombre or "").strip()
        if not nombre:
            continue
        casan = [f for f in disponibles if pantry_names_match(nombre, f["ingredient_name"])]
        if not casan:
            buscadas = _palabras(nombre)
            casan = [f for f in disponibles if buscadas and buscadas <= _palabras(f["ingredient_name"])]
        if not casan:
            faltan.append(nombre)
            continue
        for f in casan:
            if f["papelera_id"] not in vistos:
                vistos.add(f["papelera_id"])
                encontradas.append(f)
    return encontradas, faltan


def restaurar(user_id: str, *, papelera_ids: Optional[Iterable[int]] = None,
              original_ids: Optional[Iterable[int]] = None) -> list:
    """Devuelve a la Nevera lo archivado. Devuelve las filas restauradas `{id, ingredient_name, kind, original_id}`.
    Si ya existe una fila con el mismo nombre y unidad (lo volvió a añadir), suma la cantidad y rellena la etiqueta."""
    p_ids = [int(i) for i in (papelera_ids or [])]
    o_ids = [int(i) for i in (original_ids or [])]
    if not p_ids and not o_ids:
        return []
    sql = (
        # [SUPLEMENTOS-OK: restaura las filas que eligió el usuario, alimentos y potes]
        """
        WITH cand AS (
            SELECT DISTINCT ON (fila->>'ingredient_name', fila->>'unit') id
            FROM public.user_inventory_papelera
            WHERE user_id = %s AND (id = ANY(%s) OR original_id = ANY(%s))
            ORDER BY fila->>'ingredient_name', fila->>'unit', borrado_at DESC
        ), sacadas AS (
            DELETE FROM public.user_inventory_papelera p USING cand
            WHERE p.id = cand.id AND p.user_id = %s
            RETURNING p.original_id, p.fila
        ), filas AS (
            SELECT s.original_id, (jsonb_populate_record(NULL::public.user_inventory, s.fila)).*
            FROM sacadas s
        ), puestas AS (
            INSERT INTO public.user_inventory
                (user_id, ingredient_name, quantity, unit, master_ingredient_id, source, category, brand, kind,
                 serving_label, serving_unit, label_source, reserved_quantity, reservation_details,
                 last_mutation_type, created_at, updated_at)
            SELECT %s, ingredient_name, COALESCE(quantity, 0), unit, master_ingredient_id,
                   COALESCE(source, 'manual'), category, brand, COALESCE(kind, 'food'),
                   serving_label, serving_unit, label_source, 0, '{}'::jsonb,
                   'restored', COALESCE(created_at, now()), now()
            FROM filas
            ON CONFLICT (user_id, ingredient_name, unit) DO UPDATE SET
                quantity = public.user_inventory.quantity + EXCLUDED.quantity,
                serving_label = COALESCE(public.user_inventory.serving_label, EXCLUDED.serving_label),
                serving_unit = COALESCE(public.user_inventory.serving_unit, EXCLUDED.serving_unit),
                label_source = COALESCE(public.user_inventory.label_source, EXCLUDED.label_source),
                brand = COALESCE(public.user_inventory.brand, EXCLUDED.brand),
                last_mutation_type = 'restored',
                updated_at = now()
            RETURNING id, ingredient_name, unit, kind
        )
        SELECT p.id, p.ingredient_name, p.kind, f.original_id
        FROM puestas p LEFT JOIN filas f ON f.ingredient_name = p.ingredient_name AND f.unit = p.unit
        """
    )
    try:
        return [dict(f) for f in _write(sql, (user_id, p_ids, o_ids, user_id, user_id))]
    except Exception as exc:
        if _falta_la_tabla(exc):
            return []
        raise
