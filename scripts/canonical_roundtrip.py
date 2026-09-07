# -*- coding: utf-8 -*-
"""[P1-ARQ30-F4-CANONICAL · 2026-09-06] ¿Aguanta `IngredientLine` las líneas reales? (ARQ30-P1-01).

`ARQ30-P1-01` propone que texto y compras se DERIVEN de una representación canónica. Antes de mover
esa autoridad hay que saber si la representación soporta lo que el motor escribe hoy — el primer
criterio de cierre del gap lo dice: «roundtrip de legacy con fixtures reales conserva
nutrientes/demanda».

Este guion lo mide sobre los planes vivos: cada línea de ingrediente se convierte a `IngredientLine`
y se vuelve a renderizar. **No cambia nada**; solo cuenta.

    python scripts/canonical_roundtrip.py
    python scripts/canonical_roundtrip.py --json
    python scripts/canonical_roundtrip.py --planes 200 --fallos 30

## Cómo se compara, y por qué

- **exacta** — el render reproduce el texto carácter a carácter.
- **equivalente** — coincide tras normalizar mayúsculas, acentos y espacios. El parser capitaliza el
  nombre y unifica «cocida/cocido», así que exigir identidad byte a byte mediría el formato, no la
  información. **Es la cifra que importa para decidir la migración.**
- **cantidad conservada** — el número y la unidad del original sobreviven. Es lo único que NO puede
  perderse: un plan es una promesa de cantidades.
- **sin cantidad** — líneas como «Sal al gusto». No son fallos; son líneas sin número, y contarlas
  como pérdida inflaría el defecto.
- **perdida** — el parser devolvió `None` sobre una línea que sí traía cantidad. Esas son las que hay
  que leer una a una.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_CANT = re.compile(r"(\d+(?:[.,]\d+)?)\s*(g|gr|ml|kg|l|taza|tazas|cda|cdta|unidad|rebanada)", re.I)


def _norm(s: str) -> str:
    from constants import strip_accents
    s = strip_accents(str(s or "").lower())
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _cantidades(s: str) -> set:
    return {(m.group(1).replace(",", "."), m.group(2).lower()) for m in _CANT.finditer(str(s or ""))}


def medir(planes: int = 120, max_fallos: int = 20) -> dict:
    from dotenv import load_dotenv
    import psycopg
    from psycopg.rows import dict_row

    from canonical_recipe import parse_line, render_line
    from nutrition_db import IngredientNutritionDB
    ndb = IngredientNutritionDB()   # UNA instancia para todo el barrido

    load_dotenv(_BACKEND / ".env")
    with psycopg.connect(os.environ["NEON_DATABASE_URL"], row_factory=dict_row) as conn:
        filas = conn.execute(
            "SELECT id, plan_data FROM meal_plans WHERE plan_data->'days' IS NOT NULL "
            "ORDER BY created_at DESC LIMIT %s", (planes,)).fetchall()

    # [alcance 2026-09-06] Lo que decide si la representacion sirve para COMPRAS y MACROS: que la
    # linea resuelva a un alimento del catalogo y que se puedan derivar gramos. La fidelidad del
    # TEXTO se sigue midiendo, pero como el motivo del alcance, no como criterio.
    with psycopg.connect(os.environ["NEON_DATABASE_URL"], row_factory=dict_row) as conn:
        cat = conn.execute("SELECT name, aliases FROM master_ingredients").fetchall()
    from plan_policy import ingredient_id_for
    ids_cat = {ingredient_id_for(r["name"]) for r in cat}
    for r in cat:
        for a in (r["aliases"] or []):
            ids_cat.add(ingredient_id_for(a))

    tot = exacta = equiv = qty_ok = sin_qty = perdida = 0
    resuelve = gramos_ok = 0
    sin_resolver = collections.Counter()
    fallos, por_estado = [], collections.Counter()
    for fila in filas:
        for dia in ((fila["plan_data"] or {}).get("days") or []):
            for meal in (dia.get("meals") or []):
                for raw in (meal.get("ingredients") or []):
                    raw = str(raw)
                    tot += 1
                    linea = parse_line(raw, ndb)
                    if linea is None:
                        if _cantidades(raw):
                            perdida += 1
                            if len(fallos) < max_fallos:
                                fallos.append({"plan": str(fila["id"])[:8], "raw": raw[:70],
                                               "motivo": "el parser no devolvió línea"})
                        else:
                            sin_qty += 1
                        continue
                    if linea.qty is None:
                        sin_qty += 1
                        continue
                    por_estado[linea.state] += 1
                    if linea.ingredient_id in ids_cat:
                        resuelve += 1
                    else:
                        sin_resolver[linea.ingredient_id] += 1
                    if linea.grams is not None:
                        gramos_ok += 1
                    out = render_line(linea)
                    if out == raw:
                        exacta += 1
                        equiv += 1
                    elif _norm(out) == _norm(raw):
                        equiv += 1
                    elif len(fallos) < max_fallos:
                        fallos.append({"plan": str(fila["id"])[:8], "raw": raw[:70],
                                       "render": out[:70], "motivo": "render distinto"})
                    if _cantidades(raw) <= _cantidades(out) or not _cantidades(raw):
                        qty_ok += 1

    con_qty = tot - sin_qty - perdida
    def pct(n, d):
        return round(100.0 * n / d, 1) if d else None
    return {
        "planes": len(filas), "lineas": tot,
        "sin_cantidad": sin_qty, "perdidas": perdida, "con_cantidad": con_qty,
        "exacta": {"n": exacta, "pct": pct(exacta, con_qty)},
        "equivalente": {"n": equiv, "pct": pct(equiv, con_qty)},
        "cantidad_conservada": {"n": qty_ok, "pct": pct(qty_ok, con_qty)},
        "resuelve_catalogo": {"n": resuelve, "pct": pct(resuelve, con_qty)},
        "gramos_derivables": {"n": gramos_ok, "pct": pct(gramos_ok, con_qty)},
        "sin_resolver_top": dict(sin_resolver.most_common(12)),
        "por_estado": dict(por_estado.most_common()),
        "fallos": fallos,
    }


def render(r: dict) -> str:
    o = [f"planes {r['planes']} · líneas de ingrediente {r['lineas']}",
         f"  sin cantidad («Sal al gusto»): {r['sin_cantidad']}   perdidas: {r['perdidas']}",
         f"  con cantidad (denominador)   : {r['con_cantidad']}", ""]
    o.append("  --- ALCANCE ACORDADO: compras y macros ---")
    for k in ("resuelve_catalogo", "gramos_derivables", "cantidad_conservada"):
        v = r[k]
        tasa = "—" if v["pct"] is None else "{:.1f} %".format(v["pct"])
        o.append("  {:22s} {:>8s}   {} de {}".format(k, tasa, v["n"], r["con_cantidad"]))
    if r.get("sin_resolver_top"):
        o.append("  ids que NO estan en el catalogo (top): " +
                 ", ".join(f"{k}×{n}" for k, n in list(r["sin_resolver_top"].items())[:8]))
    o.append("")
    o.append("  --- fidelidad del TEXTO (el motivo del alcance, no un criterio) ---")
    for k in ("exacta", "equivalente"):
        v = r[k]
        tasa = "—" if v["pct"] is None else "{:.1f} %".format(v["pct"])
        o.append("  {:22s} {:>8s}   {} de {}".format(k, tasa, v["n"], r["con_cantidad"]))
    o += ["", "  estado detectado: " + ", ".join(f"{k}={n}" for k, n in r["por_estado"].items())]
    if r["fallos"]:
        o += ["", f"  primeros {len(r['fallos'])} desajustes (léelos: son el coste de la migración):"]
        for f in r["fallos"]:
            o.append(f"     [{f['plan']}] {f['motivo']}")
            o.append(f"         raw   : {f['raw']}")
            if "render" in f:
                o.append(f"         render: {f['render']}")
    return "\n".join(o)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--planes", type=int, default=120)
    ap.add_argument("--fallos", type=int, default=20)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    from db_core import connection_pool
    if connection_pool is not None:
        try:
            connection_pool.open()   # fuera de FastAPI el pool nace cerrado
        except Exception:
            pass

    r = medir(a.planes, a.fallos)
    print(json.dumps(r, ensure_ascii=False, indent=2) if a.json else render(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
