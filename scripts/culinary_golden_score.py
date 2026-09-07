# -*- coding: utf-8 -*-
"""[P0-CULINARY-GOLDEN · 2026-09-06] Precisión y recall de cada capa contra las etiquetas humanas.

Lee `docs/culinary_golden_set.json` YA ETIQUETADO y calcula, por capa, cuánto acierta y cuánto se le
escapa. **Este es el único número con el que se decide si V5 escala a `block`** — no la tasa del
juez, que es un LLM opinando sobre sí mismo.

    python scripts/culinary_golden_score.py
    python scripts/culinary_golden_score.py --json

## Cómo se corrige el sesgo del muestreo

La muestra es estratificada a propósito: 25 comidas «sin hallazgo» de un universo de 919 y 15 «ambas»
de un universo de 23. Contar los aciertos en bruto daría una precisión inventada, porque los estratos
raros están sobre-representados.

Cada caso pesa `disponibles_en_su_estrato / muestreados_en_su_estrato`, así que las cifras se leen
como si fueran de la población. **Se publican las dos**: la cruda (lo que se contó) y la ponderada
(lo que significa). Un solo número aquí escondería el sesgo en vez de corregirlo.

## Lo que NO calcula

Una «nota de calidad culinaria» de 1 a 10. Precisión y recall son propiedades del DETECTOR; la
calidad del plan es otra cosa y necesitaría un criterio de gravedad que hoy nadie ha definido.
Fabricar una nota a partir de estos números sería darle a una opinión la cara de una medición.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_BACKEND = Path(__file__).resolve().parents[1]
GOLDEN = _BACKEND / "docs" / "culinary_golden_set.json"

MINIMO_ETIQUETAS = 20

_DEFECTO = {"defecto", "malo", "mal", "si", "sí"}
_OK = {"ok", "bien", "correcto", "no"}


def _verdad(caso) -> bool | None:
    v = str(caso.get("veredicto_humano") or "").strip().lower()
    if v in _DEFECTO:
        return True
    if v in _OK:
        return False
    return None                     # vacío o «dudoso»: NO se fuerza a binario


def puntuar(d: dict) -> dict:
    casos = d.get("casos") or []
    disp = d.get("disponibles_por_estrato") or {}
    muestreados = collections.Counter(c.get("estrato") for c in casos)

    etiquetados = [c for c in casos if _verdad(c) is not None]
    dudosos = sum(1 for c in casos
                  if str(c.get("veredicto_humano") or "").strip().lower() == "dudoso")
    sin_etiquetar = len(casos) - len(etiquetados) - dudosos

    def peso(c):
        e = c.get("estrato")
        m = muestreados.get(e) or 1
        return (disp.get(e) or m) / m

    salida = {"casos": len(casos), "etiquetados": len(etiquetados),
              "dudosos": dudosos, "sin_etiquetar": sin_etiquetar, "capas": {}}

    for capa, clave in (("determinista", "maquina_determinista"), ("juez", "maquina_juez")):
        tp = fp = fn = tn = 0.0
        tp_n = fp_n = fn_n = tn_n = 0
        for c in etiquetados:
            marco = bool(c.get(clave))
            real = _verdad(c)
            w = peso(c)
            if marco and real:
                tp += w; tp_n += 1
            elif marco and not real:
                fp += w; fp_n += 1
            elif not marco and real:
                fn += w; fn_n += 1
            else:
                tn += w; tn_n += 1

        def r(a, b):
            return round(100.0 * a / (a + b), 1) if (a + b) else None
        salida["capas"][capa] = {
            "crudo": {"tp": tp_n, "fp": fp_n, "fn": fn_n, "tn": tn_n,
                      "precision": r(tp_n, fp_n), "recall": r(tp_n, fn_n)},
            "ponderado": {"precision": r(tp, fp), "recall": r(tp, fn)},
        }
    return salida


def render(r: dict) -> str:
    o = [f"casos {r['casos']} · etiquetados {r['etiquetados']} · dudosos {r['dudosos']} · "
         f"sin etiquetar {r['sin_etiquetar']}", ""]
    if r["etiquetados"] < MINIMO_ETIQUETAS:
        o += ["  ⛔ Con menos de 20 casos etiquetados estas cifras no significan nada.", ""]
    o.append("  capa            precision      recall        (crudo -> ponderado)")
    for capa, v in r["capas"].items():
        c, p = v["crudo"], v["ponderado"]
        o.append(f"  {capa:14s}  {str(c['precision']):>5s} -> {str(p['precision']):<6s} "
                 f"{str(c['recall']):>5s} -> {str(p['recall']):<6s}"
                 f"  (tp={c['tp']} fp={c['fp']} fn={c['fn']} tn={c['tn']})")
    o += ["", "  El ponderado corrige el sesgo del muestreo estratificado; el crudo dice lo que se",
          "  conto de verdad. Se publican los dos a proposito.",
          "", "  `dudoso` no cuenta en ninguna direccion: forzarlo a binario contaminaria la medida."]
    return "\n".join(o)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if not GOLDEN.exists():
        print(f"no existe {GOLDEN.name}: crealo con scripts/culinary_golden_sample.py")
        return 1
    r = puntuar(json.loads(GOLDEN.read_text(encoding="utf-8")))
    print(json.dumps(r, ensure_ascii=False, indent=2) if a.json else render(r))
    # [P1-SCORE-INCOMPLETE-EXIT · 2026-09-07] Un experimento SIN etiquetas suficientes salia con
    # codigo 0 y metricas `null`: para CI y para cualquier consumidor eso es indistinguible de
    # "medido y correcto". El aviso en pantalla solo lo ve una persona que ademas lo lea.
    # Exit 4 = incompleto (no es un fallo del programa, es la ausencia de la referencia humana).
    if r["etiquetados"] < MINIMO_ETIQUETAS:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
