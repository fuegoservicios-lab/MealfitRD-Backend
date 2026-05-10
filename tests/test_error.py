from fastapi.testclient import TestClient
from app import app
import uuid

with TestClient(app) as client:
    print("Testing payload...")
    try:
        response = client.post(
            "/api/analyze",
            json={
                "session_id": str(uuid.uuid4()), 
                "user_id": None, 
                "mainGoal": "Pérdida", 
                "age": "30", 
                "weight": "80", 
                "height": "180", 
                "gender": "male", 
                "activityLevel": "sedentary"
            }
        )
        print("STATUS:", response.status_code)
        print("BODY:", response.json())
    except Exception as e:
        import traceback
        traceback.print_exc()
