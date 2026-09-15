"""Batería determinista de los escáneres (sin LLM, sin escrituras) — encargo del 15-sep.

Sin la clave de visión en local, lo que se puede medir es la capa que va DESPUÉS del modelo:
  · escáner de comida (`/api/diary/upload` y fotos del chat): `vision_agent._coerce_meal_scan`
    con salidas del modelo simuladas (tres de ellas copiadas de los logs de producción);
  · escáner de Nevera / formulario (`/api/inventory/photo-scan`): `_sane_scan_qty` y
    `_match_catalog` contra el catálogo REAL (`master_ingredients`, lectura).

Uso: python scripts/coach_battery/scanner_battery.py --label antes
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.environ.get("MEALFIT_BATTERY_ENV", str(BACKEND / ".env")))

import psycopg  # noqa: E402

import vision_agent as va  # noqa: E402
from constants import meal_name_backed_by_description  # noqa: E402
from routers.user_data import _match_catalog, _sane_scan_qty  # noqa: E402

MEALS = [
    {"id": "M1", "nota": "PROD 06-sep 16:48 (foto en el chat): el nombre mete habichuelas que no están",
     "raw": {"is_food": True, "photo_kind": "plato", "meal_name": "Arroz con espaguetis, carne guisada y habichuelas",
             "description": "Plato servido con arroz blanco, espaguetis guisados con salsa de tomate y aceitunas, carne de res guisada y plátano maduro frito.",
             "calories": 1065, "protein": 51, "carbs": 130, "healthy_fats": 35}},
    {"id": "M2", "nota": "PROD 05-sep 22:39 (descripción cortada a 120 en el log)",
     "raw": {"is_food": True, "photo_kind": "plato", "meal_name": "Papas con carne mechada, huevo y habichuelas",
             "description": "Plato servido compuesto por papas asadas en gajos, porción de carne mechada coronada con huevo escalfado y salsa blanca.",
             "calories": 720, "protein": 42, "carbs": 55, "healthy_fats": 34}},
    {"id": "M3", "nota": "PROD 04-sep 19:57: «vegetal» no aparece literal en el inventario",
     "raw": {"is_food": True, "photo_kind": "plato", "meal_name": "Sándwich vegetal en pan integral",
             "description": "Sándwich preparado en pan de molde integral relleno de lechuga fresca, rodajas de tomate, rodajas de remolacha cocida, pepino y queso.",
             "calories": 380, "protein": 16, "carbs": 48, "healthy_fats": 12}},
    {"id": "M4", "nota": "nombre propio criollo",
     "raw": {"is_food": True, "photo_kind": "plato", "meal_name": "Los tres golpes",
             "description": "Mangú de plátano verde con cebolla, salami frito, queso frito y huevo frito.",
             "calories": 850, "protein": 34, "carbs": 70, "healthy_fats": 48}},
    {"id": "M5", "nota": "víveres criollos",
     "raw": {"is_food": True, "photo_kind": "plato", "meal_name": "Sancocho con casabe",
             "description": "Sancocho de pollo con yautía, auyama y plátano, acompañado de casabe.",
             "calories": 560, "protein": 32, "carbs": 70, "healthy_fats": 14}},
    {"id": "M6", "nota": "guandules",
     "raw": {"is_food": True, "photo_kind": "plato", "meal_name": "Moro de guandules con pollo",
             "description": "Moro de guandules con coco y pollo guisado.",
             "calories": 640, "protein": 36, "carbs": 80, "healthy_fats": 16}},
    {"id": "M7", "nota": "no es comida",
     "raw": {"is_food": False, "photo_kind": "otro", "meal_name": "", "description": "Un gato durmiendo sobre un sofá gris.",
             "calories": 300, "protein": 5}},
    {"id": "M8", "nota": "etiqueta nutricional (no es un plato)",
     "raw": {"is_food": False, "photo_kind": "otro", "meal_name": "",
             "description": "Etiqueta de información nutricional de un yogurt: 150 kcal y 8 g de proteína por porción.",
             "calories": 150, "protein": 8}},
    {"id": "M9", "nota": "macros absurdas / negativas",
     "raw": {"is_food": True, "photo_kind": "plato", "meal_name": "Pollo guisado", "description": "Pollo guisado con arroz blanco.",
             "calories": 99999, "protein": -5, "carbs": "mucho", "healthy_fats": None}},
    {"id": "M10", "nota": "compra con cantidades absurdas (peso impreso leído como piezas)",
     "raw": {"is_food": True, "photo_kind": "items", "description": "Compra del súper.",
             "items": [{"name": "arroz", "quantity": 500, "unit": "paquete"},
                       {"name": "huevos", "quantity": 30, "unit": "unidad"},
                       {"name": "", "quantity": 2, "unit": "lb"},
                       {"name": "salami", "quantity": 1, "unit": "unidad"}]}},
    {"id": "M11", "nota": "compra sin nada identificable",
     "raw": {"is_food": True, "photo_kind": "items", "description": "Fundas cerradas.", "items": []}},
    {"id": "M12", "nota": "salida vieja sin photo_kind",
     "raw": {"is_food": True, "meal_name": "Habichuelas con dulce", "description": "Un vaso de habichuelas con dulce con galletitas.",
             "calories": 420, "protein": 9, "carbs": 70, "healthy_fats": 12}},
]

PANTRY_NAMES = [
    "guandules", "auyama", "yautía", "casabe", "salami", "pechuga de pollo", "plátano verde", "guineo",
    "pan", "pan de agua", "leche", "arroz blanco", "habichuelas rojas", "queso de freír", "huevos",
    "aceite", "sazón completo", "longaniza", "batata", "ñame", "cilantro", "ají", "orégano", "chinola",
    "lechosa", "mantequilla", "avena", "sardinas", "atún en lata", "refresco", "cerveza", "Coca-Cola",
    "detergente", "cloro", "papel de baño",
]
QTY = [(500, "paquete"), (2, "paquete"), (30, "unidad"), (0, "unidad"), (2.5, "lb"), (40, "lb"),
       (0, "g"), (900, "g"), (None, "botella"), ("tres", "lata")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    args = ap.parse_args()
    out_dir = HERE / "out" / f"scanner_{args.label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    meals = []
    for c in MEALS:
        res = va._coerce_meal_scan(dict(c["raw"]))
        meals.append({"id": c["id"], "nota": c["nota"], "nombre_modelo": c["raw"].get("meal_name"),
                      "respaldado": meal_name_backed_by_description(c["raw"].get("meal_name") or "", c["raw"].get("description") or ""),
                      "salida": {k: res.get(k) for k in ("photo_kind", "is_food", "meal_name", "calories", "protein", "carbs", "healthy_fats", "items")}})

    with psycopg.connect(os.environ["NEON_DATABASE_URL"], autocommit=True) as conn:
        conn.execute("SET default_transaction_read_only = on")  # conexión directa propia, solo esta sesión
        cur = conn.execute("SELECT id::text AS id, name, aliases, market_container, default_unit FROM master_ingredients")
        cols = [d.name for d in cur.description]
        catalog = [dict(zip(cols, r)) for r in cur.fetchall()]
    pantry = []
    for n in PANTRY_NAMES:
        m = _match_catalog(n, catalog)
        pantry.append({"detectado": n, "catalogo": m["name"] if m else None})
    qty = [{"quantity": q, "unit": u, "saneado": _sane_scan_qty(q, u)} for q, u in QTY]

    data = {"label": args.label, "catalogo_filas": len(catalog), "comidas": meals, "nevera_match": pantry, "cantidades": qty}
    (out_dir / "results.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    sys.stdout.write(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
