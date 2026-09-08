# -*- coding: utf-8 -*-
"""Linea base de la flota ANTES de que entren planes con las §20 y §21.

Las dos secciones del prompt solo afectan a planes NUEVOS. Si no se congela el estado de hoy, en
unos dias no habra con que comparar y la unica respuesta posible sera una impresion.

Se guarda por PLAN y por comida con su fecha de creacion, para que la comparacion pueda partir la
flota en «generado antes» y «generado despues» en vez de mezclarlo todo — un promedio sobre la
union no distingue una mejora de una dilucion.
"""
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_B = Path(__file__).resolve().parents[1]
S = _B / "scripts" / "data"
sys.path.insert(0, str(_B))
os.chdir(_B)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from dotenv import load_dotenv

load_dotenv(_B / ".env")
import db_core

db_core.connection_pool.open()
import re
import psycopg
from psycopg.rows import dict_row
from shopping_calculator import get_master_ingredients
import culinary_coherence as cc

CAPAS = ("_v1_verbo_alimento", "_v2_estado_imposible", "_v3_huerfanos",
         "_v4_cantidad_inconsistente", "_v5_paso_usa_lo_que_no_esta",
         "_v6_paso_pide_mas_que_la_lista", "_v7a_lista_compra_de_mas",
         "_v7b_duplicado_incompatible", "_v7c_seco_sin_coccion",
         "_v7d_masa_sobrante", "_v7e_paso_pide_mas_piezas")

MASA = re.compile(r"(croqueta|tortita|bollito|arepita|panqueque|muffin|empanad|albondiga|"
                  r"albóndiga|yaniqueque|tortilla)", re.I)
CARNE = re.compile(r"(pollo|pechuga|muslo|cerdo|chuleta|pavo)", re.I)
SECO = re.compile(r"(horne|al horno|airfryer|fríe|frie|sarten|sartén|plancha)", re.I)
INTERIOR = re.compile(r"salga limpio|sale limpio|palillo|cuchillo entre|centro cuaj|centro coc|"
                      r"centro firme|cuaje por completo|cuaja el centro|sin partes rosad|"
                      r"blandit|tiern|termometro|termómetro|74 ?°|por dentro", re.I)

idx = cc.build_culinary_index(get_master_ingredients() or [])
with psycopg.connect(os.environ["NEON_DATABASE_URL"], row_factory=dict_row) as c:
    filas = c.execute(
        "SELECT id, created_at, plan_data FROM meal_plans "
        "WHERE plan_data->'days' IS NOT NULL AND jsonb_array_length(plan_data->'days') > 0"
    ).fetchall()

tot, por_plan = Counter(), []
comidas = 0
nec_int = con_int = 0
for f in filas:
    ch_plan, n_plan = Counter(), 0
    for d in (f["plan_data"] or {}).get("days") or []:
        for m in d.get("meals") or []:
            if not isinstance(m, dict) or not m.get("recipe"):
                continue
            comidas += 1
            n_plan += 1
            for capa in CAPAS:
                try:
                    for v in (getattr(cc, capa)({}, m, idx) or []):
                        ch_plan[v["check"]] += 1
                except Exception:
                    pass
            txt = " ".join(str(x) for x in (m.get("recipe") or []))
            nom = str(m.get("name") or "")
            if (MASA.search(txt + nom) or CARNE.search(txt + nom)) and SECO.search(txt):
                nec_int += 1
                if INTERIOR.search(txt):
                    con_int += 1
    tot.update(ch_plan)
    por_plan.append({"plan_id": str(f["id"]), "created_at": str(f["created_at"]),
                     "comidas": n_plan, "checks": dict(ch_plan)})

salida = {
    "congelado_at": datetime.now(timezone.utc).isoformat(),
    "motivo": "linea base ANTES de P1-DAYGEN-USE-ALL-INGREDIENTS (§20) y P1-DAYGEN-DONENESS-INSIDE (§21)",
    "planes": len(filas), "comidas": comidas,
    "disparos_por_capa": dict(tot), "disparos_totales": sum(tot.values()),
    "interior": {"necesitan": nec_int, "tienen": con_int,
                 "pct": round(100 * con_int / nec_int, 1) if nec_int else None},
    "por_plan": por_plan,
}
p = S / "baseline_flota_2026_09_07.json"
p.write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")

print(f"planes: {len(filas)} · comidas con receta: {comidas}")
print(f"disparos del escaner: {sum(tot.values())}  {dict(tot)}")
print(f"senal de interior: {con_int}/{nec_int} "
      f"({100*con_int/nec_int:.1f} %)" if nec_int else "")
print(f"congelado en {p.name}")
