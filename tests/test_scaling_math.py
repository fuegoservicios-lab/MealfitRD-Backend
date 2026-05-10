import sys
sys.path.append('.')
from shopping_calculator import get_shopping_list_delta
import logging

logging.basicConfig(level=logging.INFO)

plan = {
    'days': [
        {
            'day': 1,
            'meals': [
                {'ingredients': ['1 lb Pechuga de pavo', '1 lb Yuca']}
            ]
        }
    ]
}

res4 = get_shopping_list_delta('0000', plan, is_new_plan=True, structured=True, multiplier=4.0)
print('\n\n4 PERSONAS:')
for item in res4:
    print(item.get('display_string', str(item)))

res5 = get_shopping_list_delta('0000', plan, is_new_plan=True, structured=True, multiplier=5.0)
print('\n\n5 PERSONAS:')
for item in res5:
    print(item.get('display_string', str(item)))
