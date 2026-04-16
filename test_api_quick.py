import urllib.request
import json

url = "http://127.0.0.1:8000/api/recalculate-shopping-list"
data = {"user_id": "ed4e4554-1da8-451c-9758-1676ab15b889", "householdSize": 6}
req = urllib.request.Request(url, json.dumps(data).encode(), headers={'Content-Type': 'application/json'})

try:
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode())
        if "plan_data" in result:
             aggr = result["plan_data"].get("aggregated_shopping_list_weekly", [])
             for x in aggr:
                 if "pechuga" in x.get("display_string", "").lower() or "pechuga" in x.get("name", "").lower():
                     print("Found Pechuga:", x)
except Exception as e:
    print("Error:", e)
