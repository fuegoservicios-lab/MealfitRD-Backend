# food_search.py
"""[P1-MANUAL-FOOD-LOG · 2026-08-11] El resolutor del registro manual de comida.

EL CONTRATO, y es la doctrina de `consumed-from-plan`: el cliente manda REFERENCIAS,
nunca macros. `ref` puede ser:

    "dish:<slug>"   un plato criollo curado de `data/dominican_dishes.json`
                    (60 platos cocidos, `per_100g` + `finished_g` = 1 ración +
                    `constituents` en gramos CRUDOS con nombres exactos del catálogo)
    "food:<id>"     una fila de `master_ingredients` (por id; nombre como respaldo)
    "custom"        un alimento que el catálogo no conoce — el usuario declara nombre
                    y macros, la misma autoridad que ya tiene el escáner de fotos.

La ARITMÉTICA vive aquí y corre UNA vez al enviar, no por tecla: unidad→gramos con las
densidades del catálogo, macros desde `per_100g`, y expansión de `constituents` para la
Nevera. El cliente solo multiplica `qty × grams_per_qty` de las porciones precomputadas
que `/api/catalog` le sirve — nunca lleva su copia del motor.

LA REGLA DURA DE SERIALIZACIÓN (Nevera): cada línea sale como
`"NNN g de <Nombre exacto del catálogo>"` — gramos, y el nombre TAL CUAL, sin adjetivos
de cocción. `deduct_consumed_meal_from_inventory` parsea con `_parse_quantity`, que
aplica `_calculate_yield_multiplier`: «180 g de habichuelas cocidas» se convertiría en
63 g y la Nevera bajaría un tercio de lo debido. Los `constituents` de los platos ya
vienen CRUDOS, así que la cadena correcta preserva yield 1.0. El test ancla esta regla
con los nombres reales.

Si una `ref` no resuelve, `resolve_line` lanza `LineaIrresoluble` y el endpoint devuelve
422 SIN registrar a medias: el usuario vería 949 kcal en pantalla y 589 en el diario.
"""
from __future__ import annotations

import json
import logging
import threading
import unicodedata
from pathlib import Path

logger = logging.getLogger("mealfit")

_MAX_QTY = 50.0          # 50 raciones/tazas/unidades por línea ya es un banquete
_MAX_GRAMS = 5000.0      # y 5 kg por línea, un error de dedo

# Espejo de los clamps de `ConsumedMealRequest` (routers/diary.py): la vía manual no
# puede aceptar por `custom` lo que la vía de fotos rechaza en la frontera HTTP.
_MACRO_CAPS = {"kcal": 10000.0, "protein": 1000.0, "carbs": 2000.0, "fats": 1000.0}


class LineaIrresoluble(ValueError):
    """Una `ref` que no existe o una unidad sin densidad para ese alimento."""


def _sin_acentos(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn")


def _norm(s: str) -> str:
    return _sin_acentos(str(s or "").strip().lower())


# ── Los 60 platos curados ────────────────────────────────────────────────────
_dishes_cache: dict | None = None
_dishes_lock = threading.Lock()


def load_dishes() -> dict:
    """`{slug: plato}` desde `data/dominican_dishes.json`. Cache de proceso: el archivo
    es parte del deploy, no cambia en runtime."""
    global _dishes_cache
    if _dishes_cache is not None:
        return _dishes_cache
    with _dishes_lock:
        if _dishes_cache is not None:
            return _dishes_cache
        ruta = Path(__file__).resolve().parent / "data" / "dominican_dishes.json"
        try:
            data = json.loads(ruta.read_text(encoding="utf-8"))
            dishes = data.get("dishes") if isinstance(data, dict) else None
            _dishes_cache = dishes if isinstance(dishes, dict) else {}
        except Exception as e:  # noqa: BLE001 — sin platos el registro por alimento sigue vivo
            logger.error(f"❌ [P1-MANUAL-FOOD-LOG] No se pudo cargar dominican_dishes.json: {e}")
            _dishes_cache = {}
        return _dishes_cache


def dishes_for_client() -> list[dict]:
    """La vista pública de los platos: SIN `constituents`. La deducción de Nevera es
    server-side y el cliente no necesita saber de qué está hecho el moro."""
    out = []
    for slug, d in load_dishes().items():
        per = d.get("per_100g") or {}
        out.append({
            "slug": slug,
            "label": d.get("label") or slug,
            "finished_g": d.get("finished_g"),
            "per_100g": {
                "kcal": per.get("kcal"), "protein": per.get("protein"),
                "carbs": per.get("carbs"), "fats": per.get("fats"),
            },
        })
    out.sort(key=lambda x: _norm(x["label"]))
    return out


# ── Índices del catálogo ─────────────────────────────────────────────────────

def _index_catalog(rows: list[dict]) -> tuple[dict, dict]:
    por_id, por_nombre = {}, {}
    for r in rows or []:
        rid = str(r.get("id") or "").strip()
        if rid:
            por_id[rid] = r
        por_nombre[_norm(r.get("name"))] = r
    return por_id, por_nombre


def _grams_for(row: dict, qty: float, unit: str) -> float:
    """unidad→gramos con las densidades del CATÁLOGO. Sin densidad para esa unidad no
    se adivina: se lanza. Adivinar 240 g/taza para «arroz» cuando la tabla dice 185 es
    exactamente la clase de drift que el catálogo existe para impedir."""
    u = _norm(unit)
    if u in ("g", "gramos", "gramo", "gr"):
        return float(qty)
    if u in ("taza", "tazas", "cup"):
        dens = row.get("density_g_per_cup")
        if not dens:
            raise LineaIrresoluble(f"«{row.get('name')}» no tiene densidad por taza en el catálogo")
        return float(qty) * float(dens)
    if u in ("unidad", "unidades", "ud"):
        dens = row.get("density_g_per_unit")
        if not dens:
            raise LineaIrresoluble(f"«{row.get('name')}» no tiene peso por unidad en el catálogo")
        return float(qty) * float(dens)
    raise LineaIrresoluble(f"unidad desconocida: {unit!r}")


def _clamp_macros(m: dict) -> dict:
    out = {}
    for k, cap in _MACRO_CAPS.items():
        try:
            v = float(m.get(k) or 0.0)
        except (TypeError, ValueError):
            v = 0.0
        if v != v or v in (float("inf"), float("-inf")):  # NaN/Inf
            v = 0.0
        out[k] = max(0.0, min(cap, v))
    return out


def resolve_line(line: dict, catalog_rows: list[dict]) -> dict:
    """Una línea del componedor → `{name, grams, macros, pantry_lines}`.

    `pantry_lines` es lo que viaja a `consumed_meals.ingredients` y de ahí a la
    deducción: `list[str]` en el formato que la Nevera parsea (ver cabecera). Para un
    plato son sus `constituents` CRUDOS escalados por la porción; para un alimento, él
    mismo; para un `custom`, NADA — no se puede descontar lo que el catálogo no conoce.
    """
    ref = str(line.get("ref") or "").strip()
    try:
        qty = float(line.get("qty") or 0)
    except (TypeError, ValueError):
        qty = 0.0
    unit = str(line.get("unit") or "g")
    if qty <= 0:
        raise LineaIrresoluble("cantidad vacía o negativa")
    if qty > (_MAX_GRAMS if _norm(unit) in ("g", "gramos", "gramo", "gr") else _MAX_QTY):
        raise LineaIrresoluble(f"cantidad fuera de rango: {qty:g} {unit}")

    # ── plato curado ──
    if ref.startswith("dish:"):
        slug = ref[5:]
        d = load_dishes().get(slug)
        if not d:
            raise LineaIrresoluble(f"plato desconocido: {slug!r}")
        finished = float(d.get("finished_g") or 0) or 100.0
        u = _norm(unit)
        if u in ("racion", "raciones", "porcion", "porciones", "plato", "platos"):
            grams = qty * finished
        elif u in ("g", "gramos", "gramo", "gr"):
            grams = qty
        else:
            raise LineaIrresoluble(f"un plato se mide en raciones o gramos, no en {unit!r}")
        per = d.get("per_100g") or {}
        factor = grams / 100.0
        escala = grams / finished
        pantry = []
        for c in d.get("constituents") or []:
            g = round(float(c.get("g") or 0) * escala)
            if g >= 1 and c.get("name"):
                # El nombre del constituyente YA es el del catálogo, crudo y sin
                # adjetivos — la regla dura de la cabecera se cumple por construcción.
                pantry.append(f"{g} g de {c['name']}")
        return {
            "name": d.get("label") or slug,
            "grams": round(grams, 1),
            "macros": _clamp_macros({
                "kcal": float(per.get("kcal") or 0) * factor,
                "protein": float(per.get("protein") or 0) * factor,
                "carbs": float(per.get("carbs") or 0) * factor,
                "fats": float(per.get("fats") or 0) * factor,
            }),
            "pantry_lines": pantry,
        }

    # ── alimento del catálogo ──
    if ref.startswith("food:"):
        clave = ref[5:]
        por_id, por_nombre = _index_catalog(catalog_rows)
        row = por_id.get(clave) or por_nombre.get(_norm(clave))
        if not row:
            raise LineaIrresoluble(f"alimento desconocido: {clave!r}")
        grams = _grams_for(row, qty, unit)
        if grams > _MAX_GRAMS:
            raise LineaIrresoluble(f"cantidad fuera de rango: {grams:g} g")
        factor = grams / 100.0
        g_ent = round(grams)
        return {
            "name": row.get("name"),
            "grams": round(grams, 1),
            "macros": _clamp_macros({
                "kcal": float(row.get("kcal_per_100g") or 0) * factor,
                "protein": float(row.get("protein_g_per_100g") or 0) * factor,
                "carbs": float(row.get("carbs_g_per_100g") or 0) * factor,
                "fats": float(row.get("fats_g_per_100g") or 0) * factor,
            }),
            "pantry_lines": ([f"{g_ent} g de {row.get('name')}"] if g_ent >= 1 else []),
        }

    # ── alimento declarado por el usuario ──
    if ref == "custom":
        nombre = str(line.get("name") or "").strip()[:120]
        if not nombre:
            raise LineaIrresoluble("un alimento personalizado necesita nombre")
        return {
            "name": nombre,
            "grams": None,
            "macros": _clamp_macros(line.get("macros") or {}),
            "pantry_lines": [],
        }

    raise LineaIrresoluble(f"ref desconocida: {ref!r}")


def merge_pantry_lines(lineas: list[str]) -> list[str]:
    """Funde duplicados sumando gramos: dos líneas de «Arroz blanco» (el moro y el
    locrio comparten arroz) deben llegar a la Nevera como UNA resta, no como dos filas
    que el parser trata por separado."""
    import re
    acumulado: dict[str, tuple[str, float]] = {}
    orden: list[str] = []
    for l in lineas or []:
        m = re.match(r"^(\d+(?:\.\d+)?)\s*g\s+de\s+(.+)$", str(l).strip())
        if not m:
            clave = _norm(l)
            if clave not in acumulado:
                acumulado[clave] = (str(l), 0.0)
                orden.append(clave)
            continue
        g, nombre = float(m.group(1)), m.group(2).strip()
        clave = _norm(nombre)
        if clave in acumulado:
            prev_nombre, prev_g = acumulado[clave]
            acumulado[clave] = (prev_nombre, prev_g + g)
        else:
            acumulado[clave] = (nombre, g)
            orden.append(clave)
    out = []
    for clave in orden:
        nombre, g = acumulado[clave]
        out.append(f"{round(g)} g de {nombre}" if g > 0 else nombre)
    return out


def derive_meal_name(nombres: list[str], tope: int = 200) -> str:
    """«A», «A y B», «A, B y C» — con el tope de `meal_name` (200) respetado a nivel de
    ELEMENTO: cortar la cadena final por la mitad de una palabra se lee como un fallo."""
    limpios = [str(n).strip() for n in nombres or [] if str(n).strip()]
    if not limpios:
        return "Comida registrada"
    if len(limpios) == 1:
        return limpios[0][:tope]
    nombre = ", ".join(limpios[:-1]) + " y " + limpios[-1]
    if len(nombre) <= tope:
        return nombre
    while len(limpios) > 1 and len(nombre) > tope - 2:
        limpios = limpios[:-1]
        nombre = (", ".join(limpios[:-1]) + " y " + limpios[-1]) if len(limpios) > 1 else limpios[0]
    return (nombre[: tope - 1] + "…") if len(nombre) > tope else nombre
