import sys
sys.path.append('.')
from tools import execute_modify_single_meal

print("Calling execute_modify_single_meal...")
try:
    execute_modify_single_meal(
        user_id='test_user',
        day_number=1,
        meal_type='Cena',
        changes='sin arroz'
    )
except Exception as e:
    import traceback
    traceback.print_exc()
print("Done.")
