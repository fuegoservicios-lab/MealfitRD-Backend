import logging
import asyncio
from db_core import execute_sql_query

def main():
    res = execute_sql_query("SELECT plan_data FROM meal_plans WHERE session_id = 'f3b6214e-8efe-4e1d-bf31-d3e45d3de745' ORDER BY created_at DESC LIMIT 1")
    if not res:
        print("NO PLAN FOUND")
        return
        
    data = res[0]['plan_data']
    for row, day in enumerate(data.get('days', [])):
        print(f"=== DAY {day.get('day')} ===")
        for meal in day.get('meals', []):
            print(f"  [{meal.get('type')}] {meal.get('meal')}")
            ings = [i.get('name') or i.get('item_name') for i in meal.get('ingredients', []) if isinstance(i, dict)]
            for i in meal.get('ingredients', []):
                if isinstance(i, str): ings.append(i)
            print(f"    Ings: {', '.join([str(x) for x in ings])}")

main()
