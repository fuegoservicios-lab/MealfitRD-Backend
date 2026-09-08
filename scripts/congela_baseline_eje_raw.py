# -*- coding: utf-8 -*-
"""Congela la fotografía del eje `ingredients` ↔ `ingredients_raw` ANTES del 10-sep.

`baseline_flota_2026_09_07.json` mide el escáner culinario. Los siete arreglos `raw[idx]` del 07-08
de septiembre no tocan ese escáner: tocan **si lo que la receta dice coincide con lo que la lista
compra**. Comparar con la fotografía equivocada el día que llegue contenido nuevo daría un «no
cambió nada» que sólo hablaría de la métrica.

El primer contenido generado con los arreglos vivos llega el **2026-09-10 00:30 local** (el bloque
`off=3` del plan `b9e9671a`, único con cola). A partir de ahí, `created_at` por plan permite partir
la flota en antes/después sin volver a medir a mano.

## Qué se mide, y por qué SÓLO esto

Los defectos se midieron RE-EJECUTANDO las funciones sobre datos vivos (171→2, 38→0, …), y eso no es
una propiedad de lo almacenado: no se puede volver a leer mañana. Lo que sí es observable en el dato
y estable en el tiempo:

  - marcas del tracer (`_misalign_trace`) y huella viva por comida;
  - cantidades imposibles en unidades de cocina dentro de `ingredients_raw`;
  - caso 3 del eje: la receta nombra un alimento que la compra no trae por ningún lado;
  - el espejo: la compra trae un alimento que ninguna comida usa.

Los dos últimos con el instrumento de `test_p2_coherence_eje_ciego` — el que tiene 7 casos de
discriminación verificados, no una heurística nueva.

## Lo que NO se mide, dicho aquí para que nadie lo deduzca del silencio

El cuarto eje —lista ALMACENADA contra comidas— no se incluye: `_resolve_line_food_grams` trunca
nombres compuestos («carne de cangrejo» → `carne`) y produce ~49 % de falsos positivos. Es un
resolvedor de GRAMOS, no de IDENTIDAD.
"""
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_B))
sys.path.insert(0, str(_B / "tests"))
os.chdir(_B)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_B / ".env")
import db_core  # noqa: E402

db_core.connection_pool.open()
import psycopg  # noqa: E402
from psycopg.rows import dict_row  # noqa: E402

import graph_orchestrator as go  # noqa: E402
from test_p2_coherence_eje_ciego import _EXENTO, _foods_de, _mismo_alimento  # noqa: E402

_ABSURDAS = [
    (re.compile(r"^\s*([\d.,]+)\s*cdtas?\b", re.I), 6),
    (re.compile(r"^\s*([\d.,]+)\s*cdas?\b", re.I), 8),
    (re.compile(r"^\s*([\d.,]+)\s*tazas?\b", re.I), 6),
]


def _absurdas(lineas):
    fuera = []
    for s in lineas or []:
        for rx, tope in _ABSURDAS:
            m = rx.match(str(s))
            if m:
                try:
                    if float(m.group(1).replace(",", ".")) > tope:
                        fuera.append(str(s))
                except Exception:
                    pass
                break
    return fuera


def main() -> None:
    with psycopg.connect(os.environ["NEON_DATABASE_URL"], row_factory=dict_row) as cx:
        filas = cx.execute(
            "SELECT id, created_at, plan_data FROM meal_plans "
            "WHERE plan_data->'days' IS NOT NULL AND jsonb_array_length(plan_data->'days') > 0"
        ).fetchall()

    total = Counter()
    por_plan = []
    for f in filas:
        pd = f["plan_data"] or {}
        p = Counter()
        for d in pd.get("days") or []:
            for m in d.get("meals") or []:
                if not isinstance(m, dict):
                    continue
                p["comidas"] += 1
                disp = [str(x) for x in (m.get("ingredients") or [])]
                raw = [str(x) for x in (m.get("ingredients_raw") or [])]
                if not (disp and raw):
                    continue
                p["comparables"] += 1
                if len(disp) != len(raw):
                    p["largo_distinto"] += 1
                if m.get("_misalign_trace"):
                    p["marca_tracer"] += 1
                try:
                    fp = go._misalign_fingerprint(m)
                except Exception:
                    fp = None
                if fp and any(fp.get(k) for k in ("len", "qty", "missing_in_raw")):
                    p["huella_viva"] += 1
                if _absurdas(raw):
                    p["cantidad_absurda_en_compra"] += 1
                fd, fr = _foods_de(disp), _foods_de(raw)
                if fd and fr:
                    def _huerfanos(a, b):
                        return [s for t, s in a
                                if not (t & _EXENTO) and "al gusto" not in s.lower()
                                and not any(_mismo_alimento(t, tb) for tb, _ in b)]
                    if _huerfanos(fd, fr):
                        p["lee_y_no_compra"] += 1
                    if _huerfanos(fr, fd):
                        p["compra_y_no_lee"] += 1
        total.update(p)
        por_plan.append({
            "plan": str(f["id"])[:8],
            "created_at": f["created_at"].isoformat() if f["created_at"] else None,
            **dict(p),
        })

    salida = {
        "congelado_at": datetime.now(timezone.utc).isoformat(),
        "motivo": ("fotografia del eje display<->raw ANTES del primer contenido generado con los 7 "
                   "arreglos raw[idx] (llega 2026-09-10 00:30 local, bloque off=3 del plan b9e9671a)"),
        "instrumento": "test_p2_coherence_eje_ciego (7 casos de discriminacion verificados)",
        "no_medido": ("cuarto eje (lista almacenada vs comidas): _resolve_line_food_grams trunca "
                      "compuestos y da ~49 % de falsos positivos — resolvedor de GRAMOS, no de IDENTIDAD"),
        "planes": len(filas),
        "totales": dict(total),
        "por_plan": por_plan,
    }
    destino = _B / "scripts" / "data" / "baseline_eje_raw_2026_09_08.json"
    destino.write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"   congelado en {destino.name}")
    for k, v in sorted(total.items(), key=lambda x: -x[1]):
        pct = 100 * v / max(1, total["comidas"])
        print(f"      {k:32s} {v:5d}  ({pct:5.2f} % de las comidas)")


if __name__ == "__main__":
    main()
