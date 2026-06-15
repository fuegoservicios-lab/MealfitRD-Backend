#!/usr/bin/env python
"""[G16-FNDDS-GROUND-TRUTH · 2026-06-15] Construye una REFERENCIA EXTERNA de macros para platos
dominicanos desde USDA FNDDS (Food and Nutrient Database for Dietary Studies) vía la API de FoodData
Central. FNDDS es dominio público (CC0, uso comercial sin restricción) y contiene PLATOS COMPUESTOS
medidos independientemente — el ground-truth NO-CIRCULAR que faltaba para la validación de precisión
(antes se recomputaba desde el MISMO catálogo que genera las porciones → circular).

Mapea cada plato/componente dominicano a su análogo FNDDS (busca primero el nombre criollo —FNDDS tiene
platos hispanos por las encuestas US— y cae a un análogo genérico). Imprime un JSON
{dish_key: {fdc_id, fnds_description, per_100g:{kcal,protein,carbs,fats}, query_used}} a stdout, que se
commitea como backend/data/fndds_dish_reference.json (valores reales FNDDS, CC0). One-shot, NO runtime.

Uso (en el VPS o local con USDA_API_KEY en .env):
    PYTHONPATH=backend python backend/scripts/build_fndds_reference.py > backend/data/fndds_dish_reference.json

[P2-LOGGER-EXEMPT: script CLI one-shot, salida JSON a stdout intencional]
"""
import json
import os
import sys
import time

import dotenv
import requests

dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
USDA_KEY = os.environ.get("USDA_API_KEY", "DEMO_KEY")
SEARCH = "https://api.nal.usda.gov/fdc/v1/foods/search"

# Plato/componente dominicano → (queries FNDDS en orden, keywords que el match DEBE contener). El guard de
# keyword filtra la basura del search fuzzy de FDC ("beans, red"→"Cabbage, red"; "plantains green fried"→
# "Fried green TOMATOES"): si el food Survey no contiene NINGUNA keyword en su descripción, se descarta y se
# prueba la siguiente query; si ninguna pasa, MISS honesto (el plato no tiene análogo FNDDS limpio → queda en
# curación manual). CURADO a 10 platos con análogo Survey DEFENSIBLE (verificado a mano vs el perfil de macros).
# OMITIDOS (sin análogo Survey limpio — el search fuzzy solo da aproximaciones engañosas → curación manual):
#   • mangu                → solo "meat pie"/"cracklings" (no plátano-verde-hervido-majado simple).
#   • tostones             → "cracklings" (sobre-grasa de chicharrón) en vez de plátano frito simple.
#   • platano frito        → "Plantain chips" (densidad ~×3, perfil C/F distinto al maduro frito).
#   • habichuelas guisadas → "Bean SPROUTS" (perfil ≠ habichuelas; los granos ya son ingrediente del catálogo).
_DR_DISH_TO_FNDDS = {
    "moro":                 (["rice and beans", "rice, white, cooked, with beans"], ["bean"]),
    "arroz con habichuelas":(["rice and beans", "rice, white, with pinto beans"], ["bean"]),
    "locrio":               (["rice with chicken", "rice, white, cooked, with chicken"], ["rice"]),
    "sancocho":             (["stew, chicken", "soup, chicken and vegetable, stew type"], ["stew", "soup"]),
    "mofongo":              (["plantain with cracklings", "plantains, green, fried, mashed"], ["plantain"]),
    "pollo guisado":        (["chicken, stewed", "chicken, cooked, stewed"], ["chicken"]),
    "arroz blanco":         (["rice, white, cooked, regular", "rice, white, cooked"], ["rice"]),
    "yuca hervida":         (["cassava, cooked"], ["cassava"]),
    "batata":               (["sweet potato, baked, no added fat", "sweet potato, boiled"], ["potato"]),
    "avena":                (["oatmeal, cooked"], ["oat"]),
}

_N = {  # nutrientName FNDDS → key
    "Energy": "kcal", "Protein": "protein",
    "Carbohydrate, by difference": "carbs", "Total lipid (fat)": "fats",
}


def _macros_from_food(food):
    out = {"kcal": 0.0, "protein": 0.0, "carbs": 0.0, "fats": 0.0}
    for n in food.get("foodNutrients", []) or []:
        nm = n.get("nutrientName")
        if nm in _N:
            v = n.get("value")
            if v is not None:
                # Energy puede venir en kJ en algunas filas; preferimos la fila en kcal (unitName KCAL).
                if nm == "Energy" and str(n.get("unitName", "")).upper() not in ("KCAL", ""):
                    continue
                out[_N[nm]] = float(v)
    return out


def _search_fndds(query, keywords=None):
    """Busca SIN el param dataType (los paréntesis de 'Survey (FNDDS)' dan 400 en el GET) y filtra
    client-side: SOLO Survey (FNDDS) —el plato compuesto medido— NUNCA Branded/Foundation/SR Legacy (los
    Branded son etiquetas crowdsourced → basura tipo 'moro'→'TRUFFLE TORTE'). Además, si `keywords` se da,
    el match DEBE contener alguna en su descripción → filtra la basura semántica del search fuzzy de FDC
    ('beans, red'→'Cabbage, red'; 'plantains green fried'→'Fried green TOMATOES'). Mejor MISS honesto que un
    análogo irrelevante. Devuelve (food, macros, 'Survey (FNDDS)') | 'RATE_LIMITED' | None."""
    params = {"query": query, "api_key": USDA_KEY, "pageSize": 25}
    for _ in range(3):
        r = requests.get(SEARCH, params=params, timeout=25)
        if r.status_code == 429:
            if USDA_KEY == "DEMO_KEY":
                return "RATE_LIMITED"
            time.sleep(65); continue
        if r.status_code >= 400:
            return None
        foods = r.json().get("foods") or []
        for f in foods:
            if str(f.get("dataType")) != "Survey (FNDDS)":
                continue
            desc = str(f.get("description", "")).lower()
            if keywords and not any(kw in desc for kw in keywords):
                continue   # guard semántico: descarta el match fuzzy irrelevante
            m = _macros_from_food(f)
            if (m["protein"] + m["carbs"] + m["fats"]) > 0:
                return f, m, "Survey (FNDDS)"
        return None
    return None


def main():
    ref = {}
    misses = []
    req_count = 0
    rate_limited = False
    # Cap de requests para no quemar la cuota DEMO_KEY (30/hr, 50/día). Con first-hit-wins la mayoría
    # de platos resuelve en 1 request. Con una key propia (1000/hr) el cap no estorba.
    cap = 28 if USDA_KEY == "DEMO_KEY" else 999
    for key, (queries, keywords) in _DR_DISH_TO_FNDDS.items():
        if rate_limited:
            misses.append(key); continue
        hit = None
        for q in queries:
            if req_count >= cap:
                break
            req_count += 1
            res = _search_fndds(q, keywords)
            if res == "RATE_LIMITED":
                rate_limited = True
                print("  ⏳ DEMO_KEY rate-limited — detengo el fetch (re-correr con USDA_API_KEY propia).",
                      file=sys.stderr)
                break
            if res:
                food, m, dtype = res
                hit = {
                    "fdc_id": food.get("fdcId"),
                    "fnds_description": food.get("description"),
                    "data_type": dtype,                 # "Survey (FNDDS)" ideal; otro USDA = proxy honesto
                    "query_used": q,
                    "per_100g": {k: round(v, 2) for k, v in m.items()},
                }
                break
            time.sleep(0.5 if USDA_KEY == "DEMO_KEY" else 0.3)
        if hit:
            ref[key] = hit
        else:
            misses.append(key)
        print(f"  {'OK ' if hit else 'MISS'} {key:24} "
              f"[{hit['data_type'] if hit else '-'}] {hit['fnds_description'][:46] if hit else ''}",
              file=sys.stderr)
    print(f"\n[G16-FNDDS] {len(ref)} platos mapeados ({req_count} requests), "
          f"{len(misses)} sin match: {misses}", file=sys.stderr)
    out = {
        "_note": ("[G16-FNDDS-GROUND-TRUTH] Referencia externa NO-CIRCULAR de macros (per-100g) para platos "
                  "dominicanos desde USDA FNDDS (dominio público / CC0, uso comercial libre). Generado por "
                  "scripts/build_fndds_reference.py. La usa clinical_validation_export.py para medir el PERFIL "
                  "de macros que la app afirma vs un ground-truth independiente del catálogo propio."),
        "_source": "USDA FoodData Central — Survey (FNDDS), CC0 1.0 (public domain)",
        "dishes": ref,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
