# -*- coding: utf-8 -*-
"""[P0-CULINARY-GOLDEN · 2026-09-06] Muestra estratificada para el golden set humano.

**El paso que ninguna máquina puede dar por sí sola.** Hoy la coherencia culinaria la juzgan dos
capas —el contrato determinista (V1-V7) y un juez LLM— y de ninguna se conocía su precisión ni su
recall, porque **no había verdad de referencia**. Sin ella:

- «el juez señala el 19,1 % de las comidas» no dice que el 19,1 % esté mal; dice que él lo cree;
- subir o bajar un umbral hasta que la tasa del juez mejore es **overfitting**, y este repo ya lo
  pagó en agosto (un juez «al 89 %» que solo se estaba dando la razón).

Este guion no etiqueta nada: **prepara** las 80 comidas para que una persona las marque.

    python scripts/culinary_golden_sample.py            # escribe docs/culinary_golden_set.json
    python scripts/culinary_golden_sample.py --md       # además una versión legible para revisar
    python scripts/culinary_golden_sample.py --ciego    # [C1] la representación CIEGA para un 2º anotador
    python scripts/culinary_golden_sample.py --plantilla # [C1] plantilla de anotación con rúbrica (por id)

## Por qué estratificada, y por qué 25 comidas SIN hallazgo

| estrato | n | mide |
|---|---|---|
| solo determinista | 20 | precisión de V1-V7 |
| solo juez | 20 | precisión del juez — sus ~200 hallazgos exclusivos son la incógnita |
| ambas | 15 | los casos que las dos capas ven |
| **sin hallazgo** | **25** | **el RECALL: cuántos defectos reales se les escapan a las dos** |

El último estrato es el que se suele omitir y el que impide engañarse: sin comidas «limpias» solo se
mide precisión, y **un detector que no dispara nunca sale perfecto**.

## La muestra no la elige quien la mide

El orden dentro de cada estrato sale del `sha256` de la clave `(plan, día, índice, franja)`, no del azar
ni de la fecha. Es reproducible, y evita la tentación —consciente o no— de escoger los casos que
confirman lo que uno espera.

## [P1-PLAN-LOTE-22 · 2026-09-12] (C1 · CUL-P0-01) La identidad es la OCURRENCIA, en las dos orillas

El corpus ya se indexaba por `(plan, día, índice, franja)` desde el 09-07; los hallazgos, no: la capa 1 y el juez
identificaban la comida por su FRANJA, y dos meriendas el mismo día habrían heredado los mismos hallazgos.
Desde C1 la capa 1 emite `meal_index` y el juez lo devuelve cuando la rúbrica se lo pide; las entradas del juez
anteriores a CUL-P0-01 no llevan indice y se resuelven por franja SOLO cuando es única
(`culinary_coherence.resolve_judge_violations`). Lo que no se puede atar a una comida —franja ambigua, día que
ya no existe— no se reparte a nadie: se informa aparte en `hallazgos_sin_correspondencia`.

## Cómo se etiqueta

Cada caso trae el plato, sus ingredientes, sus pasos y **lo que dijo cada máquina**, con dos campos
vacíos:

- `veredicto_humano`: `ok` | `defecto` | `dudoso`
- `nota_humana`: una frase con QUÉ está mal, si lo está

Marca **primero el veredicto y luego lee lo que dijo la máquina** si te sirve — al revés, su opinión
ancla la tuya y el golden set deja de ser independiente. `dudoso` es una respuesta legítima: forzar
un binario donde no lo hay contamina la medición.

[C1] Para el marcador ESTRICTO hace falta además, por defecto: `clase` (una de `RUBRICA` en
`culinary_golden_score.py`), `severidad` (minor/high), `evidencia` (el paso o la línea, copiado) y, si aplica,
`alimento`. `--plantilla` escribe `docs/culinary_golden_anotaciones_pendientes.json` con un hueco por caso; se
rellena a mano y se pasa a `culinary_golden_score.py --estricto --anotaciones <fichero>`. `--ciego` escribe la
versión sin estrato, sin veredicto previo y sin lo que dijo la máquina, para que un segundo anotador no se ancle.

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
SALIDA_CIEGO = _BACKEND / "docs" / "culinary_golden_set_ciego.md"
SALIDA_PLANTILLA = _BACKEND / "docs" / "culinary_golden_anotaciones_pendientes.json"

CUOTAS = {"solo_determinista": 20, "solo_juez": 20, "ambas": 15, "sin_hallazgo": 25}


def indexar_hallazgos(pid: str, pd: dict, cat: list) -> tuple[dict, dict, dict, list]:
    """[C1] Las comidas de un plan por OCURRENCIA y sus hallazgos, atados a esa misma ocurrencia.

    Devuelve `(comidas, det, juez, sin_correspondencia)`: `comidas[(pid, di, mi, franja)]`, `det[(pid, di, mi)]`,
    `juez[(pid, di, mi)]` y la lista de hallazgos que no pudieron atarse a una comida (ambiguos o sin comida).
    Puro: sin DB, sin env — para poder probarlo con un plan sintético de dos meriendas."""
    from culinary_coherence import culinary_contract_scan, resolve_judge_violations

    comidas, det, juez, sueltos = {}, collections.defaultdict(list), collections.defaultdict(list), []
    for pos, d in enumerate(pd.get("days") or [], 1):
        di = d.get("day") if isinstance(d.get("day"), int) else pos
        for mi, m in enumerate(d.get("meals") or []):
            # [P1-MEAL-IDENTITY-BY-OCCURRENCE - 2026-09-07] La clave lleva el INDICE, no solo
            # la franja: dos meriendas el mismo dia colisionaban y una sobrescribia a la otra.
            comidas[(pid, di, mi, str(m.get("meal") or m.get("name")))] = {
                "dia": di, "indice": mi, "franja": str(m.get("meal") or ""), "nombre": str(m.get("name") or ""),
                "ingredientes": [str(x) for x in (m.get("ingredients") or [])],
                "pasos": [str(x) for x in (m.get("recipe") or [])]}
    for v in culinary_contract_scan(pd, cat):
        mi = v.get("meal_index")
        if isinstance(mi, int):
            det[(pid, v.get("day"), mi)].append(f"{v.get('check')}: {str(v.get('detail'))[:130]}")
        else:
            sueltos.append({"capa": "determinista", "day": v.get("day"), "meal": v.get("meal"), "motivo": "sin_indice"})
    for h in (pd.get("_culinary_judge_history") or []):
        if not isinstance(h, dict):
            continue
        for v in resolve_judge_violations(pd, [x for x in (h.get("violations") or []) if isinstance(x, dict)]):
            mi = v.get("meal_index")
            if isinstance(mi, int):
                _marca = " [dudosa]" if str(v.get("certeza") or "") == "dudosa" else ""   # [P1-PLAN-LOTE-28] (CUL-P1-06)
                juez[(pid, v.get("day"), mi)].append(f"{v.get('tipo')}{_marca}: {str(v.get('detalle'))[:130]}")
            else:
                sueltos.append({"capa": "juez", "day": v.get("day"), "meal": v.get("meal"),
                                "motivo": v.get("resolucion")})
    return comidas, det, juez, sueltos


def construir(planes: int = 120) -> dict:
    from dotenv import load_dotenv
    import psycopg
    from psycopg.rows import dict_row

    load_dotenv(_BACKEND / ".env")
    with psycopg.connect(os.environ["NEON_DATABASE_URL"], row_factory=dict_row) as c:
        c.read_only = True
        cat = [dict(r) for r in c.execute(
            "SELECT name, aliases, category, ready_to_eat, prep_methods "
            "FROM master_ingredients").fetchall()]
        filas = c.execute(
            "SELECT id, plan_data FROM meal_plans WHERE plan_data->'days' IS NOT NULL "
            "ORDER BY created_at DESC LIMIT %s", (planes,)).fetchall()

    comidas, det, juez, sin_corr = {}, {}, {}, []
    for f in filas:
        pid, pd = str(f["id"]), (f["plan_data"] or {})
        c_, d_, j_, s_ = indexar_hallazgos(pid, pd, cat)
        comidas.update(c_)
        det.update(d_)
        juez.update(j_)
        sin_corr.extend(s_)

    def _clave_hallazgo(k):
        """(plan, dia, indice) -- la identidad por OCURRENCIA, en las dos orillas desde C1.

        [P1-MEAL-IDENTITY-BY-OCCURRENCE] el corpus ya se indexaba por ocurrencia; los hallazgos identificaban la
        comida por FRANJA. [P1-PLAN-LOTE-22] la capa 1 emite `meal_index` y el juez lo devuelve; las entradas del
        juez anteriores a CUL-P0-01 no llevan indice y `resolve_judge_violations` las ata por franja SOLO cuando es
        unica — lo ambiguo va a `hallazgos_sin_correspondencia`, no a las dos meriendas.
        """
        pid, di, mi, _franja = k
        return (pid, di, mi)

    def estrato(k):
        kh = _clave_hallazgo(k)
        d, j = kh in det, kh in juez
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
                "maquina_determinista": det.get(_clave_hallazgo(k), []),
                "maquina_juez": juez.get(_clave_hallazgo(k), []),
                "veredicto_humano": "",       # ok | defecto | dudoso
                "nota_humana": "",
            })
    return {"generado": "2026-09-12", "planes_leidos": len(filas),
            "cuotas": CUOTAS, "disponibles_por_estrato": disponibles,
            "hallazgos_sin_correspondencia": sin_corr,
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


def a_markdown_ciego(d: dict) -> str:
    """[C1] La representación CIEGA: sin estrato, sin veredicto previo, sin lo que dijo la máquina, y en un orden
    que no revela el estrato (por hash del id). Para el segundo anotador."""
    casos = sorted(d["casos"], key=lambda c: hashlib.sha256(str(c["id"]).encode()).hexdigest())
    o = ["# Golden set de coherencia culinaria — anotación CIEGA", "",
         f"{len(casos)} comidas. Para cada una: `veredicto` (ok / defecto / dudoso) y, si hay defecto, uno o más",
         "defectos con `clase` (rúbrica de `scripts/culinary_golden_score.py`), `severidad` (minor/high), `evidencia`",
         "(el paso o la línea, copiado) y `alimento` si aplica. Aquí NO aparece ni el estrato ni lo que dijo nadie:",
         "es a propósito, para que la anotación sea independiente.", ""]
    for c in casos:
        o += [f"## `{c['id']}` — {c['nombre']}", "", f"*{c['franja']}*", "", "**Ingredientes**", ""]
        o += [f"- {i}" for i in c["ingredientes"]]
        o += ["", "**Pasos**", ""]
        o += [f"{n}. {p}" for n, p in enumerate(c["pasos"], 1)]
        o += ["", "---", ""]
    return "\n".join(o)


def plantilla_anotacion(d: dict, anotador: str = "") -> dict:
    """[C1] Un hueco por caso para la anotación con rúbrica (lo lee `culinary_golden_score.py --anotaciones`).
    Sin veredicto ni defectos prellenados: rellenarlos aquí con el modelo sustituiría la etiqueta humana."""
    return {"anotador": anotador, "generado": "2026-09-12",
            "instrucciones": ("por caso: veredicto ok|defecto|dudoso; defectos: [{clase, severidad, evidencia, alimento?}] "
                              "con clase de RUBRICA (culinary_golden_score.py). Deja la lista vacia si el veredicto es ok."),
            "casos": {str(c["id"]): {"veredicto": "", "defectos": []}
                      for c in sorted(d["casos"], key=lambda c: hashlib.sha256(str(c["id"]).encode()).hexdigest())}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--planes", type=int, default=120)
    ap.add_argument("--md", action="store_true", help="además, una versión legible en Markdown")
    ap.add_argument("--ciego", action="store_true", help="[C1] SOLO la representación ciega del golden set existente")
    ap.add_argument("--plantilla", action="store_true", help="[C1] SOLO la plantilla de anotación con rúbrica")
    ap.add_argument("--anotador", default="", help="[C1] nombre del anotador para la plantilla")
    a = ap.parse_args()

    if a.ciego or a.plantilla:
        if not SALIDA.exists():
            print(f"no existe {SALIDA.name}: genera primero el golden set")
            return 1
        d = json.loads(SALIDA.read_text(encoding="utf-8"))
        if a.ciego:
            SALIDA_CIEGO.write_text(a_markdown_ciego(d), encoding="utf-8", newline="\n")
            print(f"escrito {SALIDA_CIEGO.relative_to(_BACKEND)} — {len(d['casos'])} casos, sin estrato ni máquina")
        if a.plantilla:
            if SALIDA_PLANTILLA.exists():
                previo = json.loads(SALIDA_PLANTILLA.read_text(encoding="utf-8"))
                if any(str((v or {}).get("veredicto") or "").strip() for v in (previo.get("casos") or {}).values()):
                    print(f"⛔ {SALIDA_PLANTILLA.name} ya tiene anotaciones. NO se sobrescribe.")
                    return 1
            SALIDA_PLANTILLA.write_text(json.dumps(plantilla_anotacion(d, a.anotador), ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8", newline="\n")
            print(f"escrito {SALIDA_PLANTILLA.relative_to(_BACKEND)} — {len(d['casos'])} huecos")
        return 0

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
    if d["hallazgos_sin_correspondencia"]:
        print(f"   hallazgos sin comida a la que atarse (no se repartieron): {len(d['hallazgos_sin_correspondencia'])}")
    if a.md:
        SALIDA_MD.write_text(a_markdown(d), encoding="utf-8", newline="\n")
        print(f"escrito {SALIDA_MD.relative_to(_BACKEND)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
