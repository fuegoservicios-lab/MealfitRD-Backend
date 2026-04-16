import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from db_core import execute_sql_query

res = execute_sql_query("SELECT name, aliases FROM master_ingredients WHERE name ILIKE '%huevo%' OR name ILIKE '%clara%'", fetch_all=True)
for r in res:
    print(r)
