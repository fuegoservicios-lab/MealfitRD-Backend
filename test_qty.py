import os
from supabase import create_client

def main():
    import dotenv
    dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    client = create_client(url, key)
    res = client.table("custom_shopping_items").select("*").limit(5).execute()
    for item in res.data:
        print(item.get("item_name"), "7:", item.get("qty_7"), "15:", item.get("qty_15"), "30:", item.get("qty_30"))

if __name__ == "__main__":
    main()
