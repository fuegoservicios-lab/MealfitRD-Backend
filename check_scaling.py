import sys
sys.path.append('.')
from db_core import supabase

response = supabase.table('meal_plans').select('id', 'plan_data').order('created_at', desc=True).limit(1).execute()
plan = response.data[0]
data = plan.get('plan_data', {})
scaled = data.get('aggregated_shopping_list_weekly', [])
print([item.get('display_string', str(item)) if isinstance(item, dict) else str(item) for item in scaled][:10])
