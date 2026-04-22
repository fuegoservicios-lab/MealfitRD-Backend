import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph_orchestrator import run_plan_pipeline

def test_caching():
    print("🚀 Iniciando prueba de Semantic Caching...")
    
    # Form data simulado
    form_data = {
        "user_id": "test_semantic_user_123",
        "weight": "80",
        "height": "180",
        "age": "30",
        "gender": "male",
        "dietType": "balanced",
        "mainGoal": "lose_fat",
        "activityLevel": "moderate",
        "allergies": [],
        "medicalConditions": [],
        "dislikes": ["Brócoli"],
        "budget": "medium",
        "cookingTime": "30min",
        "mealsPerDay": 3
    }
    
    # 1. Ejecución 1 (Miss - Debería tomar tiempo)
    print("\n--- EJECUCIÓN 1 (Esperando Cache MISS) ---")
    start1 = time.time()
    plan1 = run_plan_pipeline(form_data, [], "", memory_context="")
    end1 = time.time()
    print(f"⏱️ Tiempo Ejecución 1: {round(end1 - start1, 2)}s")
    
    # Simular que se guarda el plan 1 con su embedding para que pueda ser encontrado
    if plan1 and "_profile_embedding" in plan1:
        print("💾 Simulando guardado en DB (Llamando a save_new_meal_plan_robust)...")
        from db_plans import save_new_meal_plan_robust
        profile_embedding = plan1.pop("_profile_embedding")
        insert_data = {
            "user_id": form_data["user_id"],
            "plan_data": plan1,
            "calories": 2000,
            "name": "Plan de Prueba Semantic",
            "profile_embedding": profile_embedding
        }
        save_new_meal_plan_robust(insert_data)
        print("✅ Plan 1 guardado en DB.")
    else:
        print("⚠️ No se generó _profile_embedding en el plan 1.")
        
    print("\n⏳ Esperando 2 segundos para asegurar propagación en DB...")
    time.sleep(2)
    
    # 2. Ejecución 2 (Hit - Debería ser casi instantáneo)
    print("\n--- EJECUCIÓN 2 (Esperando Cache HIT) ---")
    start2 = time.time()
    plan2 = run_plan_pipeline(form_data, [], "", memory_context="")
    end2 = time.time()
    print(f"⏱️ Tiempo Ejecución 2: {round(end2 - start2, 2)}s")
    
    if plan2.get("_is_cached"):
        print("🎯 ¡EXITO! Se detectó un CACHE HIT.")
    else:
        print("❌ FALLO: No se detectó CACHE HIT.")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    test_caching()
