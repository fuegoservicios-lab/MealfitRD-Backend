# -*- coding: utf-8 -*-
"""[P0-CULINARY-GOLDEN · 2026-09-06] Muestra estratificada para el golden set humano.

**El paso que ninguna máquina puede dar por sí sola.** Hoy la coherencia culinaria la juzgan dos
capas —el contrato determinista (V1-V5) y un juez LLM— y de ninguna se conoce su precisión ni su
recall, porque **no hay verdad de referencia**. Sin ella:

- «el juez señala el 19,1 % de las comidas» no dice que el 19,1 % esté mal; dice que él lo cree;
- subir o bajar un umbral hasta que la tasa del juez mejore es **overfitting**, y este repo ya lo
  pagó en agosto (un juez «al 89 %» que solo se estaba dando la razón).

Este guion no etiqueta nada: **prepara** las 80 comidas para que una persona las marque.

    python scripts/culinary_golden_sample.py            # escribe docs/culinary_golden_set.json
    python scripts/culinary_golden_sample.py --md       # además una versión legible para revisar

## Por qué estratificada, y por qué 25 comidas SIN hallazgo

| estrato | n | mide |
|---|---|---|
| solo determinista | 20 | precisión de V1-V5 |
| solo juez | 20 | precisión del juez — sus ~200 hallazgos exclusivos son la incógnita |
| ambas | 15 | los casos que las dos capas ven |
| **sin hallazgo** | **25** | **el RECALL: cuántos defectos reales se les escapan a las dos** |

El último estrato es el que se suele omitir y el que impide engañarse: sin comidas «limpias» solo se
mide precisión, y **un detector que no dispara nunca sale perfecto**.

## La muestra no la elige quien la mide

El orden dentro de cada estrato sale del `sha256` de la clave `(plan, día, franja)`, no del azar ni
de la fecha. Es reproducible, y evita la tentación —consciente o no— de escoger los casos que
confirman lo que uno espera.

## Cómo se etiqueta

Cada caso trae el plato, sus ingredientes, sus pasos y **lo que dijo cada máquina**, con dos campos
vacíos:

- `veredicto_humano`: `ok` | `defecto` | `dudoso`
- `nota_humana`: una frase con QUÉ está mal, si lo está

Marca **primero el veredicto y luego lee lo que dijo la máquina** si te sirve — al revés, su opinión
ancla la tuya y el golden set deja de ser independiente. `dudoso` es una respuesta legítima: forzar
un binario donde no lo hay contamina la medición.

Después, `scripts/culinary_golden_score.py` calcula precisión y recall de cada capa contra esas
etiquetas. **Ese** número sí se puede usar para decidir si V5 escala a `block`.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
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

SALIDA = _BACKEND / "docs" / "culinary_golden_set.json"
SALIDA_MD = _BACKEND / "docs" / "culinary_golden_set.md"

CUOTAS = {"solo_determinista": 20, "solo_juez": 20, "ambas": 15, "sin_hallazgo": 25}


def construir(planes: int = 120) -> dict:
    from dotenv import load_dotenv
    import psycopg
    from psycopg.rows import dict_row

    from culinary_coherence import culinary_contract_scan

    load_dotenv(_BACKEND / ".env")
    with psycopg.connect(os.environ["NEON_DATABASE_URL"], row_factory=dict_row) as c:
        cat = [dict(r) for r in c.execute(
            "SELECT name, aliases, category, ready_to_eat, prep_methods "
            "FROM master_ingredients").fetchall()]
        filas = c.execute(
            "SELECT id, plan_data FROM meal_plans WHERE plan_data->'days' IS NOT NULL "
            "ORDER BY created_at DESC LIMIT %s", (planes,)).fetchall()

    comidas, det, juez = {}, collections.defaultdict(list), collections.defaultdict(list)
    for f in filas:
        pid, pd = str(f["id"]), (f["plan_data"] or {})
        for di, d in enumerate(pd.get("days") or [], 1):
            for m in (d.get("meals") or []):
                comidas[(pid, di, str(m.get("meal") or m.get("name")))] = {
                    "dia": di, "franja": str(m.get("meal") or ""), "nombre": str(m.get("name") or ""),
                    "ingredientes": [str(x) for x in (m.get("ingredients") or [])],
                    "pasos": [str(x) for x in (m.get("recipe") or [])]}
        for v in culinary_contract_scan(pd, cat):
            det[(pid, v.get("day"), str(v.get("meal")))].append(
                f"{v.get('check')}: {str(v.get('detail'))[:130]}")
        for h in (pd.get("_culinary_judge_history") or []):
            if not isinstance(h, dict):
                continue
            for v in (h.get("violations") or []):
                if isinstance(v, dict):
                    juez[(pid, v.get("day"), str(v.get("meal")))].append(
                        f"{v.get('tipo')}: {str(v.get('detalle'))[:130]}")

    def estrato(k):
        d, j = k in det, k in juez
        return ("ambas" if (d and j) else "solo_determinista" if d
                else "solo_juez" if j else "sin_hallazgo")

    por_estrato = collections.defaultdict(list)
    for k in comidas:
        por_estrato[estrato(k)].append(k)
    for e in por_estrato:                     # orden por hash: reproducible y no lo elige el medidor
        por_estrato[e].sort(key=lambda k: hashlib.sha256(repr(k).encode()).hexdigest())

    casos, disponibles = [], {}
    for e, n in CUOTAS.items():
        disp = por_estrato.get(e, [])
        disponibles[e] = len(disp)
        for k in disp[:n]:
            c2 = comidas[k]
            casos.append({
                "id": hashlib.sha256(repr(k).encode()).hexdigest()[:10],
                "estrato": e, "plan": k[0][:8], **c2,
                "maquina_determinista": det.get(k, []),
                "maquina_juez": juez.get(k, []),
                "veredicto_humano": "",       # ok | defecto | dudoso
                "nota_humana": "",
            })
    return {"generado": "2026-09-06", "planes_leidos": len(filas),
            "cuotas": CUOTAS, "disponibles_por_estrato": disponibles,
            "como_etiquetar": ("marca `veredicto_humano` como ok/defecto/dudoso ANTES de leer "
                               "`maquina_*`; su opinion ancla la tuya. `dudoso` es legitimo."),
            "casos": casos}


def a_markdown(d: dict) -> str:
    o = ["# Golden set de coherencia culinaria", "",
         f"Generado {d['generado']} · {len(d['casos'])} comidas de {d['planes_leidos']} planes.", "",
         "Marca `veredicto_humano` (**ok** / **defecto** / **dudoso**) en el `.json` hermano.",
         "Hazlo **antes** de leer lo que dijo la máquina: su opinión ancla la tuya.", ""]
    for c in d["casos"]:
        o += [f"## `{c['id']}` — {c['nombre']}", "",
              f"*{c['estrato']} · plan {c['plan']} · día {c['dia']} · {c['franja']}*", "",
              "**Ingredientes**", ""]
        o += [f"- {i}" for i in c["ingredientes"]]
        o += ["", "**Pasos**", ""]
        o += [f"{n}. {p}" for n, p in enumerate(c["pasos"], 1)]
        if c["maquina_determinista"] or c["maquina_juez"]:
            o += ["", "<details><summary>Lo que dijo la máquina (léelo DESPUÉS)</summary>", ""]
            o += [f"- det · {x}" for x in c["maquina_determinista"]]
            o += [f"- juez · {x}" for x in c["maquina_juez"]]
            o += ["", "</details>"]
        o += ["", "---", ""]
    return "\n".join(o)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--planes", type=int, default=120)
    ap.add_argument("--md", action="store_true", help="además, una versión legible en Markdown")
    a = ap.parse_args()

    if SALIDA.exists():
        try:
            previo = json.loads(SALIDA.read_text(encoding="utf-8"))
            if any(str(c.get("veredicto_humano") or "").strip() for c in previo.get("casos", [])):
                print(f"⛔ {SALIDA.name} ya tiene etiquetas humanas. NO se sobrescribe: "
                      f"regenerarlo tiraría el trabajo de la persona que las puso.")
                return 1
        except Exception:
            pass

    d = construir(a.planes)
    SALIDA.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8", newline="\n")
    print(f"escrito {SALIDA.relative_to(_BACKEND)} — {len(d['casos'])} casos")
    for e, n in d["disponibles_por_estrato"].items():
        print(f"   {e:20s} disponibles {n:5d} · en la muestra {min(CUOTAS[e], n)}")
    if a.md:
        SALIDA_MD.write_text(a_markdown(d), encoding="utf-8", newline="\n")
        print(f"escrito {SALIDA_MD.relative_to(_BACKEND)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
