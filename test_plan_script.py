import urllib.request
import json
import uuid

data = {
    "session_id": str(uuid.uuid4()),
    "user_id": "guest",
    "mainGoal": "Salud General",
    "mealsPerDay": 3,
    "healthProfile": "",
    "macroPreference": "Equilibrado",
    "medicalConditions": [],
    "allergies": [],
    "dislikes": [],
    "timeToCook": "Medio",
    "dietaryPreference": "Estándar",
    "skillLevel": "Intermedio",
    "budget": "Medio",
    "activityLevel": "Moderado"
}

req = urllib.request.Request(
    'http://localhost:8000/api/analyze/stream',
    data=json.dumps(data).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)

try:
    res = urllib.request.urlopen(req)
    for line in res:
        print(line.decode('utf-8').strip())
except Exception as e:
    print("Error:", e)
    if hasattr(e, 'read'):
        print(e.read().decode('utf-8'))
