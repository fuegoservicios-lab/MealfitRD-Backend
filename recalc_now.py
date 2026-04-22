"""
Script one-shot: Recalcula la lista de compras del plan activo
usando el código actual (con master_ingredients poblado).
"""
import sys, json, logging
logging.basicConfig(level=logging.INFO)

from db_core import supabase
from shopping_calculator import get_shopping_list_delta, invalidate_master_cache

# Forzar recarga del caché de master_ingredients
invalidate_master_cache()

# Buscar plan activo
res = supabase.table("meal_plans").select("id, plan_data, user_id").order("created_at", desc=True).limit(1).execute()

if not res.data:
    print("NO ENCONTRE PLAN ACTIVO")
    sys.exit(1)

plan_id = res.data[0]["id"]
plan_data = res.data[0]["plan_data"]
user_id = res.data[0]["user_id"]
print(f"Plan ID: {plan_id}")
print(f"User ID: {user_id}")
print(f"Dias en el plan: {len(plan_data.get('days', []))}")

# Recalcular con código actual
scaled_7 = get_shopping_list_delta(user_id, plan_data, is_new_plan=True, structured=True, multiplier=1.0)
scaled_15 = get_shopping_list_delta(user_id, plan_data, is_new_plan=True, structured=True, multiplier=2.0)
scaled_30 = get_shopping_list_delta(user_id, plan_data, is_new_plan=True, structured=True, multiplier=4.0)

print(f"Recalculado: {len(scaled_7)} items (7d), {len(scaled_15)} items (15d), {len(scaled_30)} items (30d)")

# Mostrar sample
for it in scaled_7[:10]:
    # Use encode/decode to safely print even if there are weird chars like accent marks
    safestr = it.get('display_string', '?').encode('ascii', 'ignore').decode('ascii')
    print(f"  -> {safestr}")

# Guardar en DB
plan_data["aggregated_shopping_list"] = scaled_7
plan_data["aggregated_shopping_list_weekly"] = scaled_7
plan_data["aggregated_shopping_list_biweekly"] = scaled_15
plan_data["aggregated_shopping_list_monthly"] = scaled_30

supabase.table("meal_plans").update({"plan_data": plan_data}).eq("id", plan_id).execute()
print("Plan modificado y guardado en la Base de Datos :D")
