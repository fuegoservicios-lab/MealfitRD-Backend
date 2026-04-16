"""
Test script: Compare shopping list quantities across household sizes 1-6.
Calls the live API endpoint on port 3001.
"""
import urllib.request
import json
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = "http://127.0.0.1:3001/api/recalculate-shopping-list"

# Keywords to compare
KEYWORDS = ['pechuga', 'pavo', 'yogurt', 'lechosa', 'aguacate', 'arroz', 'pollo', 'melón', 'melon', 'cebolla', 'tomate']

for household in [1, 3, 4, 5, 6]:
    data = {
        "user_id": "ed4e4554-1da8-451c-9758-1676ab15b889",
        "householdSize": household,
        "groceryDuration": "weekly"
    }
    req = urllib.request.Request(
        url, 
        json.dumps(data).encode(), 
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            if result.get("success") and result.get("plan_data"):
                aggr = result["plan_data"].get("aggregated_shopping_list_weekly", [])
                print(f"\n{'='*60}")
                print(f"HOUSEHOLD SIZE: {household} persona(s) | Items: {len(aggr)}")
                print(f"{'='*60}")
                for item in aggr:
                    name = item.get("name", "").lower()
                    if any(kw in name for kw in KEYWORDS):
                        print(f"  {item.get('name'):30s} -> {item.get('display_qty'):25s} | {item.get('display_string')}")
            else:
                print(f"\n[{household}p] ERROR: {result}")
    except Exception as e:
        print(f"\n[{household}p] EXCEPTION: {e}")

print("\n\nDone.")
