"""[P1-NEON-DB-MIGRATION] Inspección read-only del shopping list semanal
escalado del plan más reciente de un usuario. Reemplazo local del endpoint
`/debug-scaling/{user_id}` eliminado en P1-AUDIT-NEW-1 (vector IDOR).

SQL directo a Neon vía el pool del backend (`execute_sql_query`). Filtra por
`user_id` explícito — NO hay fallback que tome "el último plan de cualquiera".

Uso:
    conda activate mealfit
    python backend/scripts/check_scaling.py <user_id>

Requiere `NEON_DATABASE_URL` + `NEON_DATABASE_URL_POOLED` en env (mismo que el
backend; cargados desde backend/.env vía load_dotenv).
"""
import sys
sys.path.append('.')

from dotenv import load_dotenv
load_dotenv()

from db_core import execute_sql_query


def main(argv: list[str]) -> int:
    if not argv:
        print("Uso: python check_scaling.py <user_id>", file=sys.stderr)
        return 1
    user_id = argv[0]

    row = execute_sql_query(
        "SELECT id::text AS id, plan_data FROM meal_plans "
        "WHERE user_id = %s ORDER BY created_at DESC LIMIT 1",
        (user_id,),
        fetch_one=True,
    )
    if not row:
        print(f"No hay planes para user_id={user_id}")
        return 0

    data = row.get("plan_data") or {}
    scaled = data.get("aggregated_shopping_list_weekly", [])
    print([
        item.get("display_string", str(item)) if isinstance(item, dict) else str(item)
        for item in scaled
    ][:10])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
