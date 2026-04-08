from schemas import ShoppingItemModel
import json

def _extract_fields(item):
    if hasattr(item, 'model_dump'):
        d = item.model_dump()
    elif isinstance(item, dict):
        d = item
    elif isinstance(item, str) and item.strip():
        return item.strip(), {"category": "", "display_name": item.strip(), "qty": "", "emoji": ""}
    else:
        return None, None
    item_name_json = json.dumps(d, ensure_ascii=False)
    structured = {
        "category": d.get("category", ""),
        "meal_slot": d.get("meal_slot", "Despensa General"),
        "display_name": d.get("name", ""),
        "qty": d.get("qty", ""),
        "emoji": d.get("emoji", ""),
        "is_checked": d.get("is_checked", False)
    }
    return item_name_json, structured

# Test with a Pydantic object
item = ShoppingItemModel(
    name="Test Manzana",
    category="Frutas y Verduras",
    meal_slot="Desayuno",
    emoji="🍎",
    qty_7="1 Unid",
    qty_15="2 Unid",
    qty_30="4 Unid"
)

# See what _extract_fields does
name_json, fields = _extract_fields(item)
print("EXTRACT FIELDS OUTPUT:")
print("Name JSON:", name_json)
print("Fields:", fields)

# Simulate _add_shopping_items_minimal loop
d = {}
if hasattr(item, 'model_dump'):
    print("Found model_dump()")
    d = item.model_dump()
elif hasattr(item, 'dict'):
    print("Found dict()")
    d = item.dict()
elif isinstance(item, dict):
    print("Is dict")
    d = dict(item)

print("Minimal JSON dict:", json.dumps(d, ensure_ascii=False))

