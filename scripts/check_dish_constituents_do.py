# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-13 · 2026-09-12] Validador de la tabla curada `data/dish_constituents_do.json` (el SSOT).

Historia corta. La tabla nació en F6 (2026-09-05) GENERADA por `scripts/build_dish_constituents_do.py` a partir de
las 60 recetas con gramos del diario y de ítems con nombre exacto del catálogo. Desde las curaciones del 09-09/09-10
(vinagre del mangú, queso de papa → gouda, despensa de su mercado, tope de sal, C8 del dueño) se edita A MANO, y al
aplicar C8 se midió que el generador ya no la reproducía: 75 entradas divergían y regenerarla dejaba dos plantillas
sin constituyentes. Un generador que no reproduce su salida es una trampa con la etiqueta «fuente»: se retiró en
este lote (su última versión queda en el historial: `git show e4528a22:scripts/build_dish_constituents_do.py`).

Lo que SÍ hace falta y este guion cubre — sin catálogo, en milisegundos, y anclado por test — es que la tabla siga
siendo COHERENTE con las plantillas:

  · toda plantilla DO sin `constituents` inline tiene su entrada, y no hay entradas huérfanas (plantilla borrada o
    renombrada sin mover la clave — C8 renombró tres y eso es exactamente lo que se rompe en silencio);
  · cada constituyente trae `name` (texto no vacío) y `grams` > 0;
  · `declared_unresolved` está vacío (las cuatro promesas históricas las resolvió el dueño en C8);
  · `origin` está presente (procedencia: `curated` o `rules`), y la nota dice que el JSON es el SSOT.

La RESOLUCIÓN contra el catálogo vivo (¿existe «Jugo de limón»?) NO va aquí: la hace `compile_dish_registry.py
--check`, que abre el pool. Dos preguntas, dos herramientas.

    python scripts/check_dish_constituents_do.py          # exit 0 = coherente · exit 1 = fallos listados
    python scripts/check_dish_constituents_do.py --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
DATA = os.path.join(BACKEND, "data")
TEMPLATES = os.path.join(DATA, "dish_templates.json")
TABLE = os.path.join(DATA, "dish_constituents_do.json")


def check(templates_path: str = TEMPLATES, table_path: str = TABLE) -> dict:
    """Devuelve `{"ok": bool, "fallos": [str], "plantillas": n, "entradas": n}`. Puro: sólo lee los dos JSON."""
    with open(templates_path, encoding="utf-8") as f:
        templates = json.load(f)["templates"]
    with open(table_path, encoding="utf-8") as f:
        doc = json.load(f)
    tabla = doc.get("templates") or {}
    fallos: list[str] = []

    if "ES EL SSOT" not in str(doc.get("_note", "")):
        fallos.append("_note: debe declarar que ESTE JSON ES EL SSOT (el generador se retiró en P1-PLAN-LOTE-13)")

    sin_inline = [t["name"] for t in templates if not t.get("constituents")]
    nombres = {t["name"] for t in templates}
    for name in sin_inline:
        if name not in tabla:
            fallos.append(f"sin entrada en la tabla: {name!r}")
    for name in tabla:
        if name not in nombres:
            fallos.append(f"entrada huérfana (no hay plantilla con ese nombre exacto): {name!r}")

    for name, entry in tabla.items():
        if not isinstance(entry, dict):
            fallos.append(f"{name!r}: la entrada no es un objeto")
            continue
        if entry.get("declared_unresolved"):
            fallos.append(f"{name!r}: declared_unresolved no vacío {entry['declared_unresolved']!r}")
        if entry.get("origin") not in ("curated", "rules"):
            fallos.append(f"{name!r}: origin desconocido {entry.get('origin')!r}")
        cons = entry.get("constituents")
        if not isinstance(cons, list) or not cons:
            fallos.append(f"{name!r}: sin constituyentes")
            continue
        vistos: set[str] = set()
        for c in cons:
            n = c.get("name") if isinstance(c, dict) else None
            g = c.get("grams") if isinstance(c, dict) else None
            if not isinstance(n, str) or not n.strip():
                fallos.append(f"{name!r}: constituyente sin nombre {c!r}")
                continue
            if n in vistos:
                fallos.append(f"{name!r}: constituyente repetido {n!r}")
            vistos.add(n)
            try:
                if not (float(g) > 0):
                    fallos.append(f"{name!r}: {n!r} con gramos no positivos ({g!r})")
            except (TypeError, ValueError):
                fallos.append(f"{name!r}: {n!r} con gramos no numéricos ({g!r})")

    return {"ok": not fallos, "fallos": fallos, "plantillas": len(templates), "sin_inline": len(sin_inline),
            "entradas": len(tabla)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Coherencia de data/dish_constituents_do.json con data/dish_templates.json")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    res = check()
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
    else:
        print(f"plantillas DO: {res['plantillas']} · sin constituyentes inline: {res['sin_inline']} · entradas en la tabla: {res['entradas']}")
        for f in res["fallos"]:
            print("  ✗", f)
        print("OK" if res["ok"] else f"{len(res['fallos'])} fallo(s)")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
