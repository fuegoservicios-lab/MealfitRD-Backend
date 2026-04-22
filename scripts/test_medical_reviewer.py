import asyncio
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph_orchestrator import review_plan_node

async def test_medical_reviewer():
    print("🚀 Iniciando prueba del Agente Revisor Médico Autónomo...")
    
    # Estado simulado
    state = {
        "form_data": {
            "allergies": ["Látex"], # Alergia engañosa
            "medicalConditions": [],
            "dietType": "balanced",
            "dislikes": []
        },
        "plan_result": {
            "calories": 2000,
            "days": [
                {
                    "day": 1,
                    "meals": [
                        {
                            "meal": "Desayuno",
                            "name": "Tostada de Aguacate",
                            "ingredients": ["Pan", "Aguacate", "Huevo", "Sal"] # El aguacate da reacción cruzada con el látex
                        },
                        {
                            "meal": "Almuerzo",
                            "name": "Pollo con Plátano",
                            "ingredients": ["Pollo", "Plátano", "Arroz"] # El plátano también da reacción cruzada con látex
                        }
                    ]
                }
            ]
        },
        "taste_profile": "",
        "attempt": 1
    }
    
    # 1. Ejecutar nodo
    print("\n--- EJECUCIÓN DEL REVISOR MÉDICO ---")
    result_state = await review_plan_node(state)
    
    # 2. Imprimir veredicto
    print("\n✅ Veredicto Final:")
    print(f"Aprobado: {result_state.get('review_passed')}")
    print(f"Problemas detectados: {result_state.get('review_feedback')}")
    print(f"Severidad: {result_state.get('rejection_reasons', [])}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(test_medical_reviewer())
