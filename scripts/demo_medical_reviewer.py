"""[P1-DEMO-NO-ES-TEST . 2026-08-19] Demostracion manual, NO una prueba.

Se llamaba `test_medical_reviewer.py`, asi que pytest lo recogia como test y el
gate del despliegue moria con «async def functions are not natively supported» —
un mensaje que habla del plugin que falta y no de la causa, que es que esto
nunca fue un test.

Su propia ficha en `scripts/README.md` ya lo decia: «Smoke test manual del review
LLM (NO automated)». El nombre contradecia a la documentacion, y el nombre es lo
que lee pytest.

Y no era solo ruido: esto invoca `review_plan_node`, o sea una llamada REAL al
proveedor. Un fichero que se llama `test_*` y gasta dinero es una factura
esperando a que alguien ejecute la suite entera sin mirar.

Uso:  python scripts/demo_medical_reviewer.py
"""
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
