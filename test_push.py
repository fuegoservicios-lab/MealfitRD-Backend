import os
import sys
from dotenv import load_dotenv

# Asegurar que se carguen las variables de entorno
load_dotenv(".env")
os.environ["LOG_LEVEL"] = "DEBUG"

import logging
logging.basicConfig(level=logging.DEBUG)

# Mockear el datetime para saltarnos las restricciones horarias y de spam
import proactive_agent

# Sobrescribimos temporalmente la función de spam para que devuelva False o quite la hora
def _bypass_checks():
    pass

def trigger_manual_notification(meal_to_check, trigger_time_str):
    # Simular la carga del cron forzando que se verifique TODO
    print(f"Forzando verificación de {meal_to_check}...")
    
    sessions = proactive_agent.get_active_users_for_proactive()
    print(f"Sesiones activas encontradas: {len(sessions)}")
    
    for s in sessions:
        user_id = str(s.get("user_id"))
        session_id = str(s.get("id"))
        print(f"\n=> Evaluando usuario: {user_id}")
        
        profile = proactive_agent.get_user_profile(user_id)
        if not profile:
            print("No hay perfil")
            continue
            
        health = profile.get("health_profile", {})
        diet_types = health.get("dietTypes", ["balanceada"])
        diet_type = diet_types[0] if diet_types else "balanceada"
        goals = ", ".join(health.get("goals", ["mantenerse saludable"]))
        
        print("Preparando para enviar push notification a", user_id)
        
        prompt = proactive_agent.PROACTIVE_PROMPT.format(
            missing_meal=meal_to_check,
            trigger_time=trigger_time_str,
            diet_type=diet_type,
            goals=goals
        )
        
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            chat_llm = ChatGoogleGenerativeAI(
                model="gemini-3.1-flash-lite-preview", 
                temperature=0.8,
                google_api_key=os.environ.get("GEMINI_API_KEY")
            )
            response = chat_llm.invoke(prompt)
            raw_content = response.content
            if isinstance(raw_content, list):
                content = " ".join([b.get("text", "") for b in raw_content if isinstance(b, dict) and "text" in b]).strip()
            else:
                content = str(raw_content).strip()
            
            if content:
                print(f"Mensaje generado: {content}")
                proactive_agent.save_message(session_id, "model", content)
                
                # ENVIAR PUSH
                import json
                from pywebpush import webpush, WebPushException  # type: ignore[import-untyped]
                vapid_private = os.environ.get("VAPID_PRIVATE_KEY")
                vapid_claim = os.environ.get("VAPID_CLAIM_EMAIL")
                
                subs_query = "SELECT subscription_data FROM push_subscriptions WHERE user_id = %s"
                subs = proactive_agent.execute_sql_query(subs_query, (user_id,), fetch_all=True)
                print(f"Suscripciones encontradas: {len(subs) if subs else 0}")
                
                push_payload = json.dumps({
                    "title": "Aviso de tu Nutricionista IA 👨‍⚕️",
                    "body": content,
                    "url": f"/dashboard/agent?session_id={session_id}"
                })
                
                for sub_row in subs:
                    sub_info = sub_row['subscription_data']
                    if isinstance(sub_info, str):
                        sub_info = json.loads(sub_info)
                    try:
                        webpush(
                            subscription_info=sub_info,
                            data=push_payload,
                            vapid_private_key=vapid_private,
                            vapid_claims={"sub": vapid_claim}
                        )
                        print(f"✅ Push enviado con éxito a {user_id}")
                    except WebPushException as ex:
                        print(f"❌ Error mandando a {user_id}: {repr(ex)}")
                        if ex.response is not None and ex.response.status_code in [404, 410]:
                            endpoint = sub_info.get("endpoint")
                            if endpoint:
                                proactive_agent.execute_sql_write(
                                    "DELETE FROM push_subscriptions WHERE user_id = %s AND subscription_data->>'endpoint' = %s",
                                    (user_id, endpoint)
                                )
                                print(f"🗑️ Suscripción muerta eliminada.")
                        if ex.response is not None:
                            print("Response Text:", ex.response.text)
        except Exception as e:
            print(f"Excepción general enviando: {e}")

if __name__ == '__main__':
    trigger_manual_notification("Desayuno", "11:00 AM")
