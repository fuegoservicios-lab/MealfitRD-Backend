import sys
# fake db dependencies to allow import
import db_core
db_core.supabase = None
db_core.connection_pool = None
db_core.execute_sql_query = lambda q, p=None: []

import shopping_calculator

# create mock get_master_ingredients
def mock_get_master_ingredients():
    return [
        {"name": "Papa", "aliases": ["papas", "patata", "patatas"]}
    ]
shopping_calculator.get_master_ingredients = mock_get_master_ingredients

print("Testing 'Papas y':", repr(shopping_calculator.normalize_name("Papas y")))
print("Testing 'Papas peladas y picadas':", repr(shopping_calculator.normalize_name("Papas peladas y picadas")))
