import requests, json, sys
sys.stdout.reconfigure(encoding='utf-8')

url = "http://localhost:3001/api/shopping/auto-generate"
payload = {"user_id": "b36fe9e4-b025-4879-8ada-35e12a111a14", "force": True, "days": 7}

resp = requests.post(url, json=payload)
print(f"STATUS: {resp.status_code}")
data = resp.json()

if data.get("success"):
    items = data.get("items", [])
    print(f"\nOK: {len(items)} items")
    for item in items[:30]:
        name = item.get("display_name", item.get("name", "???"))
        qty = item.get("qty_7") or item.get("qty", "")
        cat = item.get("category", "")
        daily = item.get("qty_daily", "N/A")
        unit = item.get("unit", "")
        print(f"  [{cat}] {name}: qty_7={qty} | daily={daily} | unit={unit}")
else:
    print(f"FAIL: {json.dumps(data, indent=2, ensure_ascii=False)}")
