import os, sys, json
sys.path.insert(0, os.path.abspath('.'))
from db_core import supabase

r = supabase.table('meal_plans').select('*').order('created_at', desc=True).limit(1).execute()
if len(r.data) > 0:
    plan_data = r.data[0].get('plan_data', {})
    weekly = plan_data.get('aggregated_shopping_list_weekly', [])
    for it in weekly:
        if 'Pechuga' in it.get('name', ''):
            print(json.dumps(it, indent=2))
else:
    print("No plans found")
