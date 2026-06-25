"""[P1-FOOD-DB-EXTENDED-MICROS · 2026-06-25] Backfill de 8 micros nuevos desde USDA FoodData Central
por `fdc_id` (endpoint detail, extracción por nutrient ID con fallback a nombre). Solo toca filas con
fdc_id. Lo no resuelto queda NULL → el panel lo trata como 'estimado' (honesto).

    USDA_API_KEY + NEON_DATABASE_URL(_POOLED) en .env.
    python scripts/backfill_extended_micros.py [--dry-run]
"""
import os, sys, time, json, urllib.request, urllib.parse
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass
import psycopg

USDA_KEY = os.environ.get("USDA_API_KEY", "DEMO_KEY")
NEON = os.environ.get("NEON_DATABASE_URL_POOLED") or os.environ.get("NEON_DATABASE_URL")
DRY = "--dry-run" in sys.argv

# Por cada columna destino: lista ORDENADA de candidate nutrient IDs (prioridad) + nombres (fallback,
# lowercase). Folato prioriza DFE (1190) sobre total (1177); omega-3 prioriza ALA (1404) sobre 18:3 (1270).
TARGETS = {
    "zinc_mg_per_100g":           ([1095],        ["zinc, zn"]),
    "folate_mcg_dfe_per_100g":    ([1190, 1177],  ["folate, dfe", "folate, total"]),
    "vitamin_a_mcg_rae_per_100g": ([1106],        ["vitamin a, rae"]),
    "vitamin_c_mg_per_100g":      ([1162],        ["vitamin c, total ascorbic acid"]),
    "vitamin_e_mg_per_100g":      ([1109],        ["vitamin e (alpha-tocopherol)"]),
    "vitamin_k_mcg_per_100g":     ([1185],        ["vitamin k (phylloquinone)"]),
    "selenium_mcg_per_100g":      ([1103],        ["selenium, se"]),
    "omega3_ala_g_per_100g":      ([1404, 1270],  ["pufa 18:3 n-3 c,c,c (ala)", "18:3 n-3 c,c,c (ala)",
                                                   "18:3 n-3 c,c,c (a.l.a.)"]),
}


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "mealfit-backfill"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch(fdc_id):
    url = f"https://api.nal.usda.gov/fdc/v1/food/{fdc_id}?api_key={urllib.parse.quote(USDA_KEY)}"
    for attempt in range(4):
        try:
            data = _get(url)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(60); continue
            if e.code == 404:
                return None
            raise
        # indexa por id y por nombre
        by_id, by_name = {}, {}
        for n in data.get("foodNutrients", []) or []:
            nut = n.get("nutrient") or {}
            nid = nut.get("id")
            amt = n.get("amount")
            if amt is None:
                amt = n.get("value")
            if amt is None:
                continue
            if nid is not None:
                by_id[nid] = float(amt)
            nm = str(nut.get("name") or "").strip().lower()
            if nm:
                by_name[nm] = float(amt)
        out = {}
        for col, (ids, names) in TARGETS.items():
            val = None
            for i in ids:
                if i in by_id:
                    val = by_id[i]; break
            if val is None:
                for nm in names:
                    if nm in by_name:
                        val = by_name[nm]; break
            if val is not None:
                out[col] = round(val, 3)
        return out
    return None


def main():
    if not NEON:
        print("FATAL: falta NEON_DATABASE_URL(_POOLED)"); sys.exit(1)
    print(f"USDA key: {'DEMO_KEY (lento)' if USDA_KEY == 'DEMO_KEY' else 'custom'}  dry_run={DRY}")
    with psycopg.connect(NEON) as conn:
        rows = conn.execute("SELECT id, name, fdc_id FROM master_ingredients WHERE fdc_id IS NOT NULL ORDER BY name").fetchall()
        print(f"Filas con fdc_id: {len(rows)}")
        ok = empty = err = 0
        per_nutrient = {c: 0 for c in TARGETS}
        for rid, name, fdc in rows:
            try:
                d = fetch(int(fdc))
                time.sleep(0.3)
            except Exception as e:
                print(f"  ERR {name} (fdc {fdc}): {type(e).__name__}: {str(e)[:70]}"); err += 1; continue
            if not d:
                print(f"  -- {name}: sin micros nuevos"); empty += 1; continue
            for c in d:
                per_nutrient[c] += 1
            if not DRY:
                sets = ", ".join(f"{c} = %s" for c in d)
                conn.execute(f"UPDATE master_ingredients SET {sets} WHERE id = %s", (*d.values(), rid))
            ok += 1
            print(f"  OK {name}: " + ", ".join(f"{k.replace('_per_100g','').replace('_mg','').replace('_mcg','').replace('_dfe','').replace('_rae','').replace('_g','').replace('_ala','')}={v}" for k, v in d.items()))
        if not DRY:
            conn.commit()
        print(f"\nRESUMEN: {ok} con datos, {empty} sin micros nuevos, {err} errores. dry_run={DRY}")
        print("Cobertura por nutriente (de las filas con fdc_id):")
        for c, n in per_nutrient.items():
            print(f"  {c:30} {n}/{len(rows)}")


if __name__ == "__main__":
    main()
