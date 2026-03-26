import sys
import os
import dotenv
import json

dotenv.load_dotenv('.env')
sys.path.append(os.path.dirname(__file__))

from db import get_custom_shopping_items, supabase

user_id = 'c131dd62-2ca4-4ba2-b349-8b8eb3eb3110'
print(f"URL: {supabase.supabase_url}")

existing = get_custom_shopping_items(user_id)
existing_items = existing.get('data', []) if isinstance(existing, dict) else existing
excluded_cats = ['Suplementos', 'Limpieza y Hogar', 'Higiene Personal', 'Otros']
ingredient_names = [item.get('display_name') or item.get('name') for item in existing_items if item.get('category') not in excluded_cats and (item.get('display_name') or item.get('name'))]

print('Total items in DB:', len(existing_items))
if existing_items:
    print('First item keys:', existing_items[0].keys())

print('Total extracted ingredients:', len(ingredient_names))
print('Extracted:', json.dumps(ingredient_names, indent=2))
