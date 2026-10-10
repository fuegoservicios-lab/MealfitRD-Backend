"""Audited FDC composition import. Dry-run default; --apply writes a backup first.

Only catalog/FDC pairs passing the existing identity and macro audit are used.
No name-only guesses, missing-to-zero conversion, or total-folate-to-DFE fallback.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys
import urllib.request
import urllib.parse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diary_nutrients import CATALOG_VERSION, valid_amount

TARGETS = {
    'magnesium_mg': ('magnesium_mg_per_100g', 1090, 'mg'),
    'zinc_mg': ('zinc_mg_per_100g', 1095, 'mg'),
    'b12_mcg': ('vitamin_b12_mcg_per_100g', 1178, 'ug'),
    'folate_mcg': ('folate_mcg_dfe_per_100g', 1190, 'ug'),
    'vit_e_mg': ('vitamin_e_mg_per_100g', 1109, 'mg'),
    'vit_k_mcg': ('vitamin_k_mcg_per_100g', 1185, 'ug'),
    'selenium_mcg': ('selenium_mcg_per_100g', 1103, 'ug'),
    'vit_b6_mg': ('vitamin_b6_mg_per_100g', 1175, 'mg'),
    'iodine_mcg': ('iodine_mcg_per_100g', 1100, 'ug'),
}

def extract(food):
    nutrients = {n.get('nutrient', {}).get('id'): n for n in food.get('foodNutrients', [])}
    out = {}
    for key, (_, nutrient_id, expected) in TARGETS.items():
        item = nutrients.get(nutrient_id, {})
        unit = str(item.get('nutrient', {}).get('unitName', '')).lower().replace('µ', 'u').replace('μ', 'u')
        value = valid_amount(item.get('amount'))
        out[key] = value if unit == expected else None
    return out

def main():
    from dotenv import load_dotenv
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--food-cache')
    args = parser.parse_args()
    audit = json.loads((Path(__file__).parent / 'data/catalog_fdc_sweep_2026_09_13.json').read_text(encoding='utf8'))
    approved = {(r['name'], r['fdc_id']): r for r in audit['filas'] if r['veredicto'] == 'coincide' and r['identidad'] >= .6}
    dsn = os.environ.get('NEON_DATABASE_URL_POOLED') or os.environ['NEON_DATABASE_URL']
    with psycopg.connect(dsn, row_factory=dict_row, autocommit=True) as conn:
        rows = conn.execute('SELECT * FROM master_ingredients WHERE fdc_id IS NOT NULL ORDER BY name').fetchall()
        selected = [r for r in rows if (r['name'], r['fdc_id']) in approved]
        # A pairing audit cannot certify later macro/identity edits to that same row.
        cols = {'kcal': ('kcal_per_100g', 12), 'protein': ('protein_g_per_100g', 1.5), 'carbs': ('carbs_g_per_100g', 3), 'fats': ('fats_g_per_100g', 3)}
        selected = [r for r in selected if all(valid_amount(r.get(col)) is not None and abs(float(r[col]) - approved[(r['name'], r['fdc_id'])]['catalogo'][key]) <= max(tolerance, approved[(r['name'], r['fdc_id'])]['catalogo'][key] * .1) for key, (col, tolerance) in cols.items())]
        ids = sorted({r['fdc_id'] for r in selected})
        cache = Path(args.food_cache) if args.food_cache else None
        foods = {int(k): v for k, v in json.loads(cache.read_text()).items()} if cache and cache.exists() else {}
        pending = [food_id for food_id in ids if food_id not in foods]
        for i in range(0, len(pending), 20):
            url = 'https://api.nal.usda.gov/fdc/v1/foods?api_key=' + urllib.parse.quote(os.environ.get('USDA_API_KEY', 'DEMO_KEY'))
            req = urllib.request.Request(url, data=json.dumps({'fdcIds': pending[i:i+20], 'format': 'full'}).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=45) as response:
                for f in json.load(response):
                    foods[f['fdcId']] = f
            if cache:
                cache.write_text(json.dumps(foods), encoding='utf8')
        iodine_path = Path(__file__).parent / 'data/iodine_release4_ndb.json'
        iodine = json.loads(iodine_path.read_text(encoding='utf8')) if iodine_path.exists() else {}
        changes = []
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        for row in selected:
            food = foods.get(row['fdc_id'])
            if not food:
                continue
            values = extract(food)
            iodine_entry = iodine.get(str(food.get('ndbNumber') or '').zfill(5))
            if values['iodine_mcg'] is None and iodine_entry:
                values['iodine_mcg'] = valid_amount(iodine_entry['value'])
            provenance = dict(row.get('micronutrient_provenance') or {})
            for key, value in values.items():
                provenance[key] = {'source': 'USDA FoodData Central', 'fdc_id': row['fdc_id'],
                    'description': food.get('description'), 'data_type': food.get('dataType'),
                    'retrieved_at': now, 'version': CATALOG_VERSION, 'reported': value is not None,
                    'url': f"https://fdc.nal.usda.gov/food-details/{row['fdc_id']}/nutrients"}
            if iodine_entry and values['iodine_mcg'] is not None:
                provenance['iodine_mcg'] = dict(iodine_entry, retrieved_at=now, version=CATALOG_VERSION, reported=True)
            changes.append({'name': row['name'], 'fdc_id': row['fdc_id'], 'before': {col: row.get(col) for col, _, _ in TARGETS.values()},
                            'before_provenance': row.get('micronutrient_provenance') or {}, 'values': values, 'provenance': provenance})
        output = Path(args.output)
        if output.exists():
            raise RuntimeError('Output must be new: preserve previous audit/backup')
        output.write_text(json.dumps({'version': CATALOG_VERSION, 'retrieved_at': now, 'changes': changes}, indent=2, ensure_ascii=False, default=float), encoding='utf8')
        if args.apply:
            with conn.transaction():
                for change in changes:
                    columns = [col for col, _, _ in TARGETS.values()]
                    conn.execute('UPDATE master_ingredients SET ' + ', '.join(col + '=%s' for col in columns) + ', micronutrient_provenance=%s WHERE name=%s AND fdc_id=%s',
                                 [change['values'][key] for key in TARGETS] + [Jsonb(change['provenance']), change['name'], change['fdc_id']])
        print(json.dumps({'applied': args.apply, 'rows': len(changes), 'coverage': {key: sum(c['values'][key] is not None for c in changes) for key in TARGETS}, 'backup': str(output)}))

if __name__ == '__main__':
    main()
