# -*- coding: utf-8 -*-
"""[P2-MONOTONIA-PROBE · 2026-09-06] ¿Repiten los planes la misma comida, o solo lo pareció una vez?

Origen: leyendo el plan vivo `c5880d96` vi ricotta en las tres cenas de los tres días y lo reporté como
monotonía del motor. Antes de construir un guard de variedad medí la flota, y **el hallazgo no sobrevivió**:

    planes donde UNA proteína cubre >=60 % de las comidas fuertes ...  0 de 84
    días con la misma proteína en almuerzo Y cena ................... 13 de 109  (11,9 %)
    series (plan x franja) con la misma proteína >=3 días seguidos ...  3 de 54   ( 5,6 %)
    series con la misma FORMA de plato >=3 días seguidos .............  7 de 335  ( 2,1 %)

(Medido el 2026-09-06 sobre los 84 planes vivos con >=3 días. Vuelve a correrlo antes de citarlo.)

Y las rachas que quedan casi todas son legítimas: yogurt en la merienda, huevos en el desayuno, «vaso frío
de…» como formato de merienda con contenido distinto cada día. Un guard sobre eso empeoraría el producto.

`c5880d96` era un caso, no un patrón — y esta es la diferencia entre una lección y una anécdota. La sonda se
commitea justamente para que la próxima vez que alguien vea tres platos parecidos no haya que volver a
discutirlo de memoria: **se vuelve a correr**. (La lección de P2-JUDGE-RATE-PROBE: «recontar mañana» no es un
entregable; la sonda commiteada sí.)

    python scripts/monotonia_probe.py                # las cuatro medidas
    python scripts/monotonia_probe.py --json         # para diffear entre versiones
    python scripts/monotonia_probe.py --planes 200   # ventana más ancha

Lo que NO mide: si el usuario percibe repetición. Dos platos con nombres distintos y los mismos cinco
ingredientes se leen como repetidos y aquí cuentan como distintos.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

# La consola de Windows llega en cp1252 y esta tabla lleva acentos: sin esto el guion muere al imprimir,
# no al medir, que es la peor forma de perder una medición.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_FUERTE = {"almuerzo", "cena"}

# El primer alimento con gramos de una línea de ingrediente. Las líneas sin cantidad («Sal al gusto») no
# entran: no son lo que hace que un plato se parezca a otro.
_LINEA = re.compile(r"\s*([\d.,/]+)\s*g\s+de\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ ]+)")

_PROTEICOS = (
    "ricotta", "queso", "yogur", "kefir", "kéfir", "huevo", "clara", "pollo", "res", "cerdo", "pavo",
    "atun", "atún", "salmon", "salmón", "tilapia", "sardina", "camaron", "camarón", "lenteja",
    "habichuela", "frijol", "garbanzo", "soya", "tofu", "edamame", "haba", "guandul", "gandul",
    "maní", "mani", "almendra",
)

# Estados de un mismo alimento: «lentejas secas» y «lentejas cocidas» son lentejas.
_ESTADO = re.compile(
    r"\s+(cocid\w*|crud\w*|seca\w*|entero|entera|griego|firme|rallad\w*|en cubos|picad\w*|molid\w*).*$")


def _proteina_principal(meal: dict) -> str | None:
    """El alimento proteico con MÁS gramos de la comida. Con gramos, no por orden: el primero de la lista
    suele ser el cereal."""
    mejor, tope = None, -1.0
    for ing in (meal.get("ingredients") or []):
        m = _LINEA.match(str(ing))
        if not m:
            continue
        nombre = m.group(2).strip().lower()
        if not any(p in nombre for p in _PROTEICOS):
            continue
        try:
            gramos = float(m.group(1).replace(",", "."))
        except Exception:
            continue
        if gramos > tope:
            tope, mejor = gramos, _ESTADO.sub("", nombre).strip()
    return mejor


def _forma(nombre: str | None) -> str | None:
    """La primera palabra del título, sin diminutivo ni plural: «Arepitas horneadas…» y «Arepita de maíz…»
    son la misma forma. Es un proxy grueso a propósito — un plato se reconoce por cómo se sirve."""
    from constants import strip_accents

    palabras = strip_accents(str(nombre or "").strip().lower()).split()
    if not palabras:
        return None
    raiz = re.sub(r"[^a-z]", "", palabras[0])
    return re.sub(r"(itas|itos|ita|ito|es|s)$", "", raiz) or None


def _racha_maxima(seq: list) -> tuple[int, object]:
    mayor, actual, cual = 1, 1, seq[0]
    for a, b in zip(seq, seq[1:]):
        actual = actual + 1 if a == b else 1
        if actual > mayor:
            mayor, cual = actual, b
    return mayor, cual


def medir(planes: int = 90, racha_min: int = 3) -> dict:
    from dotenv import load_dotenv
    import psycopg
    from psycopg.rows import dict_row

    load_dotenv(_BACKEND / ".env")
    with psycopg.connect(os.environ["NEON_DATABASE_URL"], row_factory=dict_row) as conn:
        filas = conn.execute(
            "SELECT id, plan_data FROM meal_plans "
            "WHERE plan_data->'days' IS NOT NULL AND jsonb_array_length(plan_data->'days') >= 3 "
            "ORDER BY created_at DESC LIMIT %s", (planes,)).fetchall()

    dia_tot = dia_repite = 0
    series_prot = racha_prot = 0
    series_forma = racha_forma = 0
    dominados = []
    detalle = []

    for fila in filas:
        dias = (fila["plan_data"] or {}).get("days") or []
        por_franja_prot = collections.defaultdict(list)
        por_franja_forma = collections.defaultdict(list)
        conteo_fuerte, total_fuerte = collections.Counter(), 0

        for dia in dias:
            comidas = dia.get("meals") or []
            for meal in comidas:
                franja = str(meal.get("meal") or "").strip().lower()
                if not franja:
                    continue
                prot = _proteina_principal(meal)
                if prot:
                    por_franja_prot[franja].append(prot)
                    if franja in _FUERTE:
                        conteo_fuerte[prot] += 1
                forma = _forma(meal.get("name"))
                if forma:
                    por_franja_forma[franja].append(forma)
                if franja in _FUERTE:
                    total_fuerte += 1

            fuertes = [_proteina_principal(m) for m in comidas
                       if str(m.get("meal") or "").strip().lower() in _FUERTE]
            fuertes = [p for p in fuertes if p]
            if len(fuertes) >= 2:
                dia_tot += 1
                if len(set(fuertes)) == 1:
                    dia_repite += 1

        for franja, seq in por_franja_prot.items():
            if len(seq) < racha_min:
                continue
            series_prot += 1
            mayor, cual = _racha_maxima(seq)
            if mayor >= racha_min:
                racha_prot += 1
                detalle.append({"plan": str(fila["id"])[:8], "franja": franja, "eje": "proteína",
                                "valor": cual, "dias": mayor, "de": len(seq)})
        for franja, seq in por_franja_forma.items():
            if len(seq) < racha_min:
                continue
            series_forma += 1
            mayor, cual = _racha_maxima(seq)
            if mayor >= racha_min:
                racha_forma += 1
                detalle.append({"plan": str(fila["id"])[:8], "franja": franja, "eje": "forma",
                                "valor": cual, "dias": mayor, "de": len(seq)})

        if total_fuerte >= 4 and conteo_fuerte:
            alimento, veces = conteo_fuerte.most_common(1)[0]
            if veces / total_fuerte >= 0.6:
                dominados.append({"plan": str(fila["id"])[:8], "alimento": alimento,
                                  "veces": veces, "de": total_fuerte})

    return {
        "planes": len(filas),
        "dominados_60pct": dominados,
        "dia_almuerzo_y_cena": {"total": dia_tot, "repiten": dia_repite},
        "racha_proteina": {"series": series_prot, "con_racha": racha_prot},
        "racha_forma": {"series": series_forma, "con_racha": racha_forma},
        "detalle": sorted(detalle, key=lambda d: -d["dias"]),
    }


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f} %" if d else "—"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--planes", type=int, default=90, help="cuántos planes vivos leer (default 90)")
    ap.add_argument("--racha", type=int, default=3, help="días seguidos que cuentan como racha (default 3)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    r = medir(planes=args.planes, racha_min=args.racha)
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return 0

    print(f"planes leídos: {r['planes']}\n")
    d = r["dia_almuerzo_y_cena"]
    p = r["racha_proteina"]
    f = r["racha_forma"]
    print(f"  una proteína cubre >=60 % de las comidas fuertes  {len(r['dominados_60pct']):5d} de {r['planes']:<5d} "
          f"{_pct(len(r['dominados_60pct']), r['planes'])}")
    print(f"  misma proteína en almuerzo Y cena del mismo día   {d['repiten']:5d} de {d['total']:<5d} "
          f"{_pct(d['repiten'], d['total'])}")
    print(f"  misma proteína >={args.racha} días seguidos por franja     {p['con_racha']:5d} de {p['series']:<5d} "
          f"{_pct(p['con_racha'], p['series'])}")
    print(f"  misma forma de plato >={args.racha} días seguidos          {f['con_racha']:5d} de {f['series']:<5d} "
          f"{_pct(f['con_racha'], f['series'])}")

    if r["dominados_60pct"]:
        print("\n  planes dominados por un alimento:")
        for x in r["dominados_60pct"]:
            print(f"     [{x['plan']}] {x['alimento']:24s} {x['veces']}/{x['de']}")
    if r["detalle"]:
        print("\n  rachas más largas (mira si son legítimas antes de llamarlas defecto —")
        print("  yogurt en merienda y huevos en desayuno lo son):")
        for x in r["detalle"][:15]:
            print(f"     [{x['plan']}] {x['franja']:10s} {x['eje']:9s} «{x['valor']}» "
                  f"{x['dias']} de {x['de']} días")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
