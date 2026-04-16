from db_core import execute_sql_query
import json

def main():
    print('Testing DB State...')
    
    plans = execute_sql_query("SELECT id, user_id, plan_data FROM meal_plans ORDER BY created_at DESC LIMIT 1")
    if not plans:
        print('No plans')
        return
        
    p = plans[0]
    user_id = p['user_id']
    plan_id = p['id']
    data = p['plan_data']
    if isinstance(data, str):
        data = json.loads(data)
        
    print(f'Latest Plan ID: {plan_id} for User: {user_id}')
    print(f'is_restocked: {data.get("is_restocked")}')
    
    inv = execute_sql_query(f"SELECT ingredient_name, quantity, unit FROM user_inventory WHERE user_id = '{user_id}' ORDER BY ingredient_name")
    print(f'Inventory Has {len(inv)} items')
    for i in inv[:10]:
        print(i)

    print('Shopping List:')
    ag = data.get('aggregated_shopping_list', [])
    for a in ag[:10]:
        print(a)

if __name__ == '__main__':
    main()
