# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-34 · 2026-09-13] F6: barrido descripción-USDA ↔ nombre sobre TODAS las filas del catálogo con `fdc_id`.

Un `fdc_id` es una afirmación (`P1-PROVENANCE-TRUTHFUL`): «esta fila ES ese alimento de USDA». La auditoría del 19-ago
cerró los ids COMPARTIDOS (hoy 0: 288 ids, 288 distintos) pero dejó escrito lo que no veía: **un id único mal apuntado es
invisible a un barrido de duplicados** — `Lomo embuchado` apuntaba a lomo CRUDO (110 kcal frente a 321) y lo destapó BEDCA,
no el barrido. Este script mira las dos señales fila a fila:

  1. identidad: qué fracción de los tokens del nombre del catálogo (`name_en`, el gloss en inglés; y `name`) aparece en la
     descripción USDA del id (plural, acentos y palabras vacías fuera);
  2. valores: el desvío máximo de kcal / proteína / grasa / carbohidrato del catálogo frente a USDA.

Veredicto automático (umbrales declarados en el artefacto): `coincide` (identidad ≥ 0,4 y desvío ≤ 10 %), `mal_apuntado`
(identidad < 0,4 y desvío > 25 %: las dos señales en contra) y `revisar` (una de las dos). Un id que USDA no devuelve se
anota `sin_respuesta`: **un 404 no dice que la fila esté mal**. Los `revisar` y `mal_apuntado` se miran a MANO, uno a uno,
y el veredicto final se escribe en `revision` con su razón — el script propone, no decide.

Solo lectura: la conexión a la base se abre con `read_only`; la API de USDA recibe la clave por cabecera (`X-Api-Key`),
nunca en la URL. Las respuestas se guardan en `--cache`, así que la clasificación se puede re-ejecutar sin red.

    python scripts/catalog_fdc_sweep.py --cache <f.json> --out scripts/data/catalog_fdc_sweep_<fecha>.json
    python scripts/catalog_fdc_sweep.py --cache <f.json> --out <artefacto> --sin-red     # re-clasifica desde el caché

[P2-LOGGER-EXEMPT: script CLI de auditoría, la salida a stdout ES el producto]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from datetime import date
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_BACKEND = Path(__file__).resolve().parents[1]
API = "https://api.nal.usda.gov/fdc/v1"
#: número de nutriente USDA → macro del catálogo. La energía de Foundation viene a veces como Atwater (957/958) o kJ (268).
MACROS = {"208": "kcal", "203": "protein", "204": "fats", "205": "carbs"}
ENERGIA_ALT = ("957", "958")
COLUMNAS = {"kcal": "kcal_per_100g", "protein": "protein_g_per_100g", "fats": "fats_g_per_100g", "carbs": "carbs_g_per_100g"}
#: suelo del denominador: un 0,4 g frente a 0,1 g no es un desvío del 300 %.
SUELO = {"kcal": 10.0, "protein": 1.0, "fats": 1.0, "carbs": 1.0}
#: por debajo de esto una diferencia es convención de cálculo o redondeo, no otro alimento.
TOLERANCIA = {"kcal": 12.0, "protein": 1.5, "fats": 1.5, "carbs": 3.0}
UMBRALES = {"identidad_min": 0.4, "desvio_ok_pct": 10.0, "desvio_mal_pct": 25.0}
VACIAS = {"and", "or", "with", "without", "of", "the", "a", "in", "raw", "fresh", "plain", "whole", "all", "types",
          "includes", "ns", "as", "to", "added", "salt", "for", "de", "del", "la", "el", "los", "las", "con", "sin",
          "y", "en", "al", "commercial", "commercially", "prepared", "ready", "eat", "meat", "only", "cooked",
          "unprepared", "frozen", "dry", "dried", "enriched", "unenriched", "regular", "grade", "separable", "lean", "fat"}


def _singular(t: str) -> str:
    """Inglés: «berries» → «berry», «peaches» / «tomatoes» → «peach» / «tomato», «limes» → «lime» (no «lim»)."""
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if re.search(r"(ch|sh|x|ss|z|o)es$", t) and len(t) > 4:
        return t[:-2]
    if t.endswith("s") and not t.endswith("ss") and len(t) > 3:
        return t[:-1]
    return t


def _tokens(s: str) -> set:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    out = set()
    for t in re.findall(r"[a-z]+", s):
        if t in VACIAS or len(t) < 3:
            continue
        out.add(_singular(t))
    return out


def identidad(nombres: list, descripcion: str) -> float:
    """La mejor fracción de tokens de un nombre del catálogo que aparece en la descripción USDA (0 si no hay tokens)."""
    d = _tokens(descripcion)
    mejor = 0.0
    for n in nombres:
        a = _tokens(n)
        if a:
            mejor = max(mejor, len(a & d) / len(a))
    return round(mejor, 3)


def desvio(cat: dict, usda: dict) -> tuple:
    """Desvío máximo en %, con TOLERANCIA ABSOLUTA: una diferencia menor que `TOLERANCIA[k]` no cuenta (34 frente a 28 kcal
    en una verdura es la convención de energía —Atwater general frente a específico, SR Legacy frente a Foundation—, no
    otro alimento)."""
    peor, clave, por_macro = 0.0, None, {}
    for k in COLUMNAS:
        c, u = cat.get(k), usda.get(k)
        if c is None or u is None:
            continue
        diff = abs(float(c) - float(u))
        pct = 0.0 if diff <= TOLERANCIA[k] else diff / max(abs(float(u)), SUELO[k]) * 100.0
        por_macro[k] = round(pct, 1)
        if pct > peor:
            peor, clave = pct, k
    return round(peor, 1), clave, por_macro


def veredicto(ident: float, dev: float) -> str:
    if ident >= UMBRALES["identidad_min"] and dev <= UMBRALES["desvio_ok_pct"]:
        return "coincide"
    if ident < UMBRALES["identidad_min"] and dev > UMBRALES["desvio_mal_pct"]:
        return "mal_apuntado"
    return "revisar"


def filas_catalogo() -> list:
    from dotenv import dotenv_values
    import psycopg
    from psycopg.rows import dict_row

    url = os.environ.get("NEON_DATABASE_URL") or dotenv_values(_BACKEND / ".env").get("NEON_DATABASE_URL")
    if not url:
        raise SystemExit("falta NEON_DATABASE_URL")
    with psycopg.connect(url, row_factory=dict_row) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute("""SELECT name, name_en, fdc_id, nutrition_source, nutrition_source_ref,
                                  kcal_per_100g, protein_g_per_100g, fats_g_per_100g, carbs_g_per_100g
                           FROM master_ingredients WHERE fdc_id IS NOT NULL ORDER BY name""")
            return [dict(r) for r in cur.fetchall()]


def consultar_usda(ids: list, cache: dict) -> None:
    import requests
    from dotenv import dotenv_values

    key = os.environ.get("USDA_API_KEY") or dotenv_values(_BACKEND / ".env").get("USDA_API_KEY") or "DEMO_KEY"
    faltan = [i for i in ids if str(i) not in cache]
    for j in range(0, len(faltan), 20):
        lote = faltan[j:j + 20]
        r = requests.get(f"{API}/foods", timeout=60, headers={"X-Api-Key": key},
                         params={"fdcIds": ",".join(str(i) for i in lote), "format": "abridged",
                                 "nutrients": ",".join(list(MACROS) + list(ENERGIA_ALT) + ["268"])})
        r.raise_for_status()
        vistos = set()
        for f in r.json() or []:
            fid = str(f.get("fdcId"))
            vistos.add(fid)
            vals = {}
            for n in f.get("foodNutrients") or []:
                num = str(n.get("number") or (n.get("nutrient") or {}).get("number") or "")
                amt = n.get("amount") if n.get("amount") is not None else n.get("value")
                if amt is None:
                    continue
                if num in MACROS:
                    vals[MACROS[num]] = float(amt)
                elif num in ENERGIA_ALT and "kcal" not in vals:
                    vals["kcal_atwater"] = float(amt)
                elif num == "268":
                    vals["kj"] = float(amt)
            if "kcal" not in vals:
                if "kcal_atwater" in vals:
                    vals["kcal"] = vals["kcal_atwater"]
                elif "kj" in vals:
                    vals["kcal"] = round(vals["kj"] / 4.184, 1)
            cache[fid] = {"descripcion": f.get("description"), "tipo": f.get("dataType"), "macros": vals}
        for i in lote:
            if str(i) not in vistos:
                cache[str(i)] = None
        print(f"  USDA {j + len(lote)}/{len(faltan)}")


def clasificar(filas: list, cache: dict) -> dict:
    out, cuenta = [], {"coincide": 0, "revisar": 0, "mal_apuntado": 0, "sin_respuesta": 0}
    for r in filas:
        u = cache.get(str(r["fdc_id"]))
        cat = {k: (float(r[c]) if r[c] is not None else None) for k, c in COLUMNAS.items()}
        fila = {"name": r["name"], "name_en": r["name_en"], "fdc_id": r["fdc_id"], "catalogo": cat,
                "nutrition_source": r["nutrition_source"], "nutrition_source_ref": r["nutrition_source_ref"]}
        if not u:
            fila["veredicto"] = "sin_respuesta"
        else:
            ident = identidad([r["name_en"] or "", r["name"] or ""], u["descripcion"] or "")
            dev, peor, por = desvio(cat, u["macros"])
            fila.update({"descripcion_usda": u["descripcion"], "tipo_usda": u["tipo"], "usda": u["macros"],
                         "identidad": ident, "desvio_max_pct": dev, "macro_peor": peor, "desvio_pct": por,
                         "veredicto": veredicto(ident, dev)})
        cuenta[fila["veredicto"]] += 1
        out.append(fila)
    return {"filas": out, "cuenta": cuenta}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sin-red", action="store_true")
    ap.add_argument("--filas", help="JSON con las filas ya leídas (para re-clasificar sin base)")
    a = ap.parse_args(argv)
    cache_p = Path(a.cache)
    cache = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.exists() else {}
    if a.filas:
        filas = json.loads(Path(a.filas).read_text(encoding="utf-8"))
    else:
        filas = filas_catalogo()
    if not a.sin_red:
        consultar_usda([r["fdc_id"] for r in filas], cache)
        cache_p.write_text(json.dumps(cache, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    res = clasificar(filas, cache)
    previo = {}
    outp = Path(a.out)
    if outp.exists():
        previo = json.loads(outp.read_text(encoding="utf-8")).get("revision") or {}
    art = {
        "schema": "2026-09-13.fdc_sweep", "fecha": date.today().isoformat(),
        "metodo": ("cada fila con fdc_id contra la fila USDA de ese id (API FoodData Central, formato abridged): identidad "
                   "= fracción de tokens de name_en/name presentes en la descripción; desvío = máximo de kcal/proteína/"
                   "grasa/carbohidrato con suelo en el denominador"),
        "umbrales": UMBRALES, "suelo_denominador": SUELO,
        "n_filas": len(filas), "n_consultadas": sum(1 for r in filas if cache.get(str(r["fdc_id"]))),
        "n_sin_respuesta": res["cuenta"]["sin_respuesta"], "cuenta": res["cuenta"],
        "filas": res["filas"], "revision": previo,
    }
    outp.write_text(json.dumps(art, ensure_ascii=False, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"{len(filas)} filas · {res['cuenta']} → {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
