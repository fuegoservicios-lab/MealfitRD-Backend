# -*- coding: utf-8 -*-
"""[P0-CULINARY-BASELINE · 2026-09-06] La línea base de coherencia culinaria, ANTES de mejorarla.

Congelar la foto es el primer paso del benchmark, y no es burocracia: **una medición posterior al
efecto no mide el efecto**. Este repo ya lo pagó una vez —«recontar mañana» no era un entregable— y
por eso la línea base se toma, se escribe con fecha y se commitea antes de tocar nada.

    python scripts/culinary_baseline.py                 # la foto de hoy
    python scripts/culinary_baseline.py --json          # para diffear contra la congelada
    python scripts/culinary_baseline.py --congelar      # reescribe docs/culinary_baseline.json

## Qué mide, y por qué las dos capas por separado

- **Contrato determinista** (`culinary_contract_scan`, V1–V5): reglas que no opinan. V1 técnica
  imposible, V2 estado imposible, V3 ingrediente que ningún paso usa, V4 gramos que no cuadran,
  V5 paso que usa lo que la lista no trae.
- **Juez culinario** (LLM, `_culinary_judge_history`): incoherencia de prosa — pasos que se
  contradicen, nombres que prometen lo que el plato no tiene, combinaciones absurdas.

Se publican **separadas y con su solapamiento**, nunca sumadas en un índice único. La medición del
2026-09-06 explica por qué: de las 314 comidas que marcan entre las dos, **solo 23 coinciden**.
Fundirlas en un número escondería que ninguna sustituye a la otra.

## La foto del 2026-09-06 (96 planes · 1.186 comidas)

    contrato determinista ... 133 comidas   11,2 %   V1=80 V3=55 V4=34 V5=11 V2=4
    juez culinario .......... 227 comidas   19,1 %   paso_incoherente=96
                                                     nombre_no_corresponde=53
                                                     combo_absurdo=50
                                                     slot_inapropiado=28
                                                     tecnica_impropia=23
    coinciden ...............  23 comidas
    solo determinista ....... 110   ·   solo juez ... 204

**Cuidado al leer esto como «calidad».** El juez es un LLM sin verdad de referencia: no se conoce su
precisión ni su recall. Que señale el 19,1 % no significa que el 19,1 % de las comidas esté mal;
significa que él lo cree. Convertir esa tasa en una nota, o calibrar un umbral contra ella, es el
overfitting que este repo ya pagó en agosto. Lo que falta es el golden set humano
(`scripts/culinary_golden_sample.py`).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

CONGELADA = _BACKEND / "docs" / "culinary_baseline.json"


def medir(planes: int = 120) -> dict:
    from dotenv import load_dotenv
    import psycopg
    from psycopg.rows import dict_row

    from culinary_coherence import culinary_contract_scan, judgment_covers_delivered

    load_dotenv(_BACKEND / ".env")
    with psycopg.connect(os.environ["NEON_DATABASE_URL"], row_factory=dict_row) as c:
        cat = [dict(r) for r in c.execute(
            "SELECT name, aliases, category, ready_to_eat, prep_methods "
            "FROM master_ingredients").fetchall()]
        filas = c.execute(
            "SELECT id, plan_data FROM meal_plans WHERE plan_data->'days' IS NOT NULL "
            "ORDER BY created_at DESC LIMIT %s", (planes,)).fetchall()

    comidas = 0
    por_check, por_tipo = collections.Counter(), collections.Counter()
    con_det, con_juez = set(), set()
    cobertura = collections.Counter()   # P1-JUDGE-REVISION-STAMP: si / no / desconocido

    for f in filas:
        pid = str(f["id"])
        pd = f["plan_data"] or {}
        for d in (pd.get("days") or []):
            comidas += len(d.get("meals") or [])
        for v in culinary_contract_scan(pd, cat):
            por_check[str(v.get("check"))] += 1
            con_det.add((pid, v.get("day"), str(v.get("meal"))))
        for h in (pd.get("_culinary_judge_history") or []):
            if not isinstance(h, dict):
                continue
            # [P1-JUDGE-REVISION-STAMP · 2026-09-06] ¿Esta entrada juzgó lo que se ENTREGÓ, o una
            # versión que el pipeline reparó después? Tres estados, y el tercero manda: sin sello
            # (todo lo generado antes del P-fix) la pregunta no es decidible, y colapsarlo hacia
            # cualquier lado fabricaría una cifra. Se cuenta aparte y se publica aparte.
            cubre = judgment_covers_delivered(h, pd)
            cobertura["si" if cubre else ("no" if cubre is False else "desconocido")] += 1
            for v in (h.get("violations") or []):
                if isinstance(v, dict):
                    por_tipo[str(v.get("tipo"))] += 1
                    con_juez.add((pid, v.get("day"), str(v.get("meal"))))

    def pct(n):
        return round(100.0 * n / comidas, 1) if comidas else None

    return {
        "planes": len(filas), "comidas": comidas,
        "determinista": {"comidas": len(con_det), "pct": pct(len(con_det)),
                         "por_check": dict(por_check.most_common())},
        "juez": {"comidas": len(con_juez), "pct": pct(len(con_juez)),
                 "por_tipo": dict(por_tipo.most_common())},
        # [P1-JUDGE-REVISION-STAMP] Cuantas entradas del juez se sabe que juzgaron lo
        # ENTREGADO. Va al lado de la tasa, no dentro: no la corrige, la CALIFICA.
        "juez_sobre_lo_entregado": dict(cobertura),
        "solapamiento": {"ambas": len(con_det & con_juez),
                         "solo_determinista": len(con_det - con_juez),
                         "solo_juez": len(con_juez - con_det)},
        # Se guarda EXPLICITO para que nadie lo derive de las tasas y se engañe.
        # [P1-JUDGE-REVISION-STAMP · 2026-09-06] La segunda frase NO es decorativa y por eso vive
        # AQUÍ y no solo en el JSON congelado: `--congelar` reescribe el fichero entero, así que
        # una advertencia enmendada a mano en el JSON se habría borrado en silencio en el próximo
        # congelado. La advertencia tiene que nacer del mismo sitio que el dato.
        "advertencia": ADVERTENCIA,
    }


#: [P1-JUDGE-REVISION-STAMP · 2026-09-06] Dos razones para no leer la tasa del juez como calidad,
#: no una. La segunda se midió DESPUÉS de congelar la foto: parte de sus quejas describen un
#: estado que el pipeline REPARÓ antes de entregar.
ADVERTENCIA = (
    "el juez es un LLM sin verdad de referencia: su tasa NO es la tasa de defectos reales. "
    "Calibrar contra ella es overfitting. Y hay una segunda razon, medida el 2026-09-06: parte de "
    "sus quejas describen un estado que el pipeline REPARO antes de entregar — de 37 quejas "
    "juzgables del tipo «X no aparece en la lista», 6 nombraban algo que SI esta en el plan "
    "entregado, y el sub-patron mas citado («lonjas/pedazos de queso» sobre queso cottage) "
    "aparece 0 veces de 23 en planes vivos. Las entradas SIN `judged_fingerprint` no permiten "
    "saber cuales juzgaron lo entregado: leelas como «cuantas comidas el juez senalo en algun "
    "momento», nunca como «cuantas se entregaron mal»."
)


def render(r: dict, previa: dict | None = None) -> str:
    o = [f"planes {r['planes']} · comidas {r['comidas']}", ""]
    for capa, etiq in (("determinista", "contrato determinista (V1-V5)"), ("juez", "juez culinario")):
        v = r[capa]
        linea = f"  {etiq:32s} {v['comidas']:5d} comidas   {v['pct']} %"
        if previa and previa.get(capa):
            d = v["comidas"] - previa[capa]["comidas"]
            linea += f"   ({d:+d} vs congelada)"
        o.append(linea)
        detalle = v.get("por_check") or v.get("por_tipo") or {}
        o.append("      " + "  ".join(f"{k}={n}" for k, n in detalle.items()))
    s = r["solapamiento"]
    o += ["", f"  coinciden {s['ambas']}   ·   solo determinista {s['solo_determinista']}   ·   "
              f"solo juez {s['solo_juez']}",
          "", "  Las dos capas apenas se solapan: ninguna sustituye a la otra, y por eso NO se",
          "  suman en un indice unico."]
    cob = r.get("juez_sobre_lo_entregado") or {}
    if cob:
        o += ["", "  entradas del juez que juzgaron LO ENTREGADO: "
                  + "  ".join(f"{k}={n}" for k, n in cob.items()),
              "  («desconocido» = sin `judged_fingerprint`, anterior a P1-JUDGE-REVISION-STAMP; "
              "no cuenta a ningun lado)"]
    o += ["", "  " + r["advertencia"]]
    return "\n".join(o)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--planes", type=int, default=120)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--congelar", action="store_true",
                    help="reescribe docs/culinary_baseline.json con la foto de AHORA")
    a = ap.parse_args()

    r = medir(a.planes)
    previa = None
    if CONGELADA.exists():
        try:
            previa = json.loads(CONGELADA.read_text(encoding="utf-8"))
        except Exception:
            previa = None

    if a.congelar:
        CONGELADA.write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8", newline="\n")
        print(f"linea base congelada en {CONGELADA.relative_to(_BACKEND)}")
        return 0
    print(json.dumps(r, ensure_ascii=False, indent=2) if a.json else render(r, previa))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
