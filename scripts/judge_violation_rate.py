# -*- coding: utf-8 -*-
"""[P2-JUDGE-RATE-PROBE · 2026-09-06] La tasa de violaciones del juez, por FECHA de creación.

Nace de un error mío: leí un histórico de 7 días como si fuera el estado de hoy y recomendé
atacar algo que ya estaba cerrado. Una tasa agregada mezcla planes anteriores y posteriores a
cada arreglo, así que **la fecha de creación del plan es la unidad**, no la ventana.

READ-ONLY y sin coste de LLM: lee `plan_data->'_culinary_judge_history'`, que el juez ya dejó
escrito en cada plan. Correrlo cuesta segundos y convierte una expectativa en un dato.

    python scripts/judge_violation_rate.py            # 8 días
    python scripts/judge_violation_rate.py --dias 3

Mide tres cosas a la vez porque las tres se decidieron el mismo día:

  1. violaciones por plan y por tipo, por fecha  → ¿bajan tras los arreglos del 06-sep?
  2. líneas de hierba por encima de 150 g        → la predicción de `P1-UNKNOWN-UNIT-NOT-WHOLE`
  3. hints de báscula que contradicen su línea   → la de `P1-STEP-GRAM-HINT-STALE`

La línea base del 06-sep (antes de desplegar) está en
`backend/docs/judge_violation_baseline.md`. Comparar contra ella es el punto.
"""
from __future__ import annotations

import argparse
import collections
import os
import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_HIERBAS = ("cilantro", "cebollin", "perejil", "oregano", "tomillo", "laurel", "albahaca", "romero")
_HINT_RE = re.compile(r"\(\s*[≈~]?\s*(\d+(?:[.,]\d+)?)\s*(?:g|ml)\s*\)", re.IGNORECASE)
_LEAD_G_RE = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*(?:g|gr|gramos?|ml)\b", re.IGNORECASE)


def _conectar():
    from dotenv import load_dotenv
    load_dotenv()
    import psycopg
    url = os.environ.get("NEON_DATABASE_URL")
    if not url:
        sys.exit("NEON_DATABASE_URL no está en el entorno (ver runbook_sql_forensic_sop).")
    # ⚠️ Fuera de FastAPI el pool no está abierto y `master_ingredients` sale vacío.
    try:
        import db_core
        db_core.connection_pool.open()
    except Exception:
        pass
    return psycopg.connect(url)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dias", type=int, default=8, help="ventana en días (default 8)")
    args = ap.parse_args()

    with _conectar() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT created_at, plan_data->'_culinary_judge_history', plan_data->'days'
                 FROM meal_plans
                WHERE jsonb_array_length(coalesce(plan_data->'days','[]'::jsonb)) > 0
                  AND created_at > now() - make_interval(days => %s)
                ORDER BY created_at""",
            (int(args.dias),))
        filas = cur.fetchall()

    try:
        from nutrition_db import IngredientNutritionDB
        db = IngredientNutritionDB()
    except Exception:
        db = None

    from constants import strip_accents as sa

    planes = collections.Counter()
    por_tipo: dict = collections.defaultdict(collections.Counter)
    hierbas_altas = collections.Counter()
    hints_malos = collections.Counter()
    lineas = collections.Counter()

    for creado, hist, dias in filas:
        f = str(creado)[:10]
        planes[f] += 1
        for e in (hist or []):
            if not isinstance(e, dict):
                continue
            for v in (e.get("violations") or []):
                por_tipo[f][str(v.get("tipo"))] += 1
        for d in (dias or []):
            for m in (d.get("meals") or []):
                gmap = {}
                for ln in (m.get("ingredients") or []):
                    s = str(ln)
                    lineas[f] += 1
                    mh = _HINT_RE.search(s) or _LEAD_G_RE.match(s.strip())
                    if mh:
                        cuerpo = re.sub(r"^\s*[\d.,½¼¾⅓⅔]+\s*\w*\.?\s*(?:de\s+|del\s+)?", "", s.strip())
                        tks = [t for t in re.split(r"[^\wáéíóúñü]+", sa(cuerpo.lower())) if len(t) >= 4]
                        if tks:
                            gmap.setdefault(tks[0], float(mh.group(1).replace(",", ".")))
                    if db is not None and any(h in sa(s.lower()) for h in _HIERBAS):
                        try:
                            g = db.grams_from_ingredient_string(s)
                        except Exception:
                            g = None
                        if g and g > 150:
                            hierbas_altas[f] += 1
                for paso in (m.get("recipe") or []):
                    if not isinstance(paso, str):
                        continue
                    for mm in re.finditer(
                            r"(\d+(?:[.,]\d+)?)\s*(?:g|gr)\s+de\s+([A-Za-zÁÉÍÓÚÑáéíóúñ][\wáéíóúñ]*)"
                            r"\s*\(\s*[≈~]?\s*(\d+(?:[.,]\d+)?)\s*g\s*\)", paso):
                        lider, hint = float(mm.group(1)), float(mm.group(3))
                        if abs(lider - hint) > max(1.0, 0.15 * lider):
                            hints_malos[f] += 1

    tipos = sorted({t for f in por_tipo for t in por_tipo[f]})
    print(f"ventana: {args.dias} días · planes: {sum(planes.values())}\n")
    cab = f"{'fecha':12s}{'planes':>7s}" + "".join(f"{t[:13]:>14s}" for t in tipos) + f"{'total':>7s}{'/plan':>7s}"
    print(cab)
    print("-" * len(cab))
    for f in sorted(planes):
        tot = sum(por_tipo[f].values())
        fila = f"{f:12s}{planes[f]:7d}" + "".join(f"{por_tipo[f].get(t, 0):14d}" for t in tipos)
        print(fila + f"{tot:7d}{tot / max(1, planes[f]):7.2f}")

    print("\npredicciones de los arreglos del 06-sep (deben tender a cero):")
    print(f"  {'fecha':12s}{'líneas':>9s}{'hierbas >150 g':>16s}{'hints en contra':>17s}")
    for f in sorted(planes):
        print(f"  {f:12s}{lineas[f]:9d}{hierbas_altas.get(f, 0):16d}{hints_malos.get(f, 0):17d}")
    if db is None:
        print("\n  ⚠ sin catálogo (pool cerrado): la columna de hierbas no es fiable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
