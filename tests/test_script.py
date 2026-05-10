import urllib.request
import json

env = {}
with open('.env', 'r') as f:
    for line in f:
        if '=' in line:
            k, v = line.strip().split('=', 1)
            env[k] = v.strip('\"\'')

url_base = env.get('SUPABASE_URL')
key = env.get('SUPABASE_SERVICE_ROLE_KEY')

url = f'{url_base}/rest/v1/custom_shopping_items?user_id=eq.c131dd62-2ca4-4ba2-b349-8b8eb3eb3110&select=id,item_name,category,display_name'
req = urllib.request.Request(url, headers={
    'apikey': key,
    'Authorization': f'Bearer {key}'
})

try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        print(f'Total items: {len(data)}')
        if data:
            print(f"First raw: {data[0]}")
        
        excluded_cats = ["Suplementos", "Limpieza y Hogar", "Higiene Personal", "Otros"]
        ingredient_names = [item.get("display_name") or item.get("name") for item in data if item.get("category") not in excluded_cats and (item.get("display_name") or item.get("name"))]
        
        print("\nWhat agent.py extracts:")
        print(ingredient_names)
        
except Exception as e:
    print('Error:', e)
