import sys
import os

# Agregamos backend al path para que Python encuentre los módulos
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from shopping_calculator import aggregate_and_deduct_shopping_list

# Test 1: Pan (Rebanadas -> Paquete)
print("TEST 1: Pan integral")
list1 = ["12 rebanadas de Pan integral"]
res1 = aggregate_and_deduct_shopping_list(list1, structured=False)
for r in res1: print("- ", r)

# Test 2: Queso cottage (Pote decimal -> Pote entero)
print("\nTEST 2: Queso cottage")
list2 = ["0.4 pote de Queso cottage"]
res2 = aggregate_and_deduct_shopping_list(list2, structured=False)
for r in res2: print("- ", r)

# Test 3: Vainitas (piso absurdo - densidad muy baja)
print("\nTEST 3: Vainitas (test de absurdo de conteo alto)")
list3 = ["20 Uds. de Vainitas"]
res3 = aggregate_and_deduct_shopping_list(list3, structured=False)
for r in res3: print("- ", r)
