import os
import sys
import json

# Add backend directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db_core import execute_sql_write, execute_sql_query

def run_migration():
    print("Starting GAP 4 Migration...")
    
    # 1. Update meal_plans
    res_meal_plans = execute_sql_write("""
        UPDATE meal_plans 
        SET plan_data = (plan_data - 'plan_start_date') || jsonb_build_object('_plan_start_date', plan_data->'plan_start_date')
        WHERE plan_data ? 'plan_start_date'
        RETURNING id;
    """, returning=True)
    
    updated_plans = len(res_meal_plans) if res_meal_plans else 0
    print(f"Updated {updated_plans} rows in meal_plans.")
    
    # 2. Update plan_chunk_queue
    res_chunks = execute_sql_write("""
        UPDATE plan_chunk_queue 
        SET pipeline_snapshot = (pipeline_snapshot - 'plan_start_date') || jsonb_build_object('_plan_start_date', pipeline_snapshot->'plan_start_date')
        WHERE pipeline_snapshot ? 'plan_start_date'
        RETURNING id;
    """, returning=True)
    
    updated_chunks = len(res_chunks) if res_chunks else 0
    print(f"Updated {updated_chunks} rows in plan_chunk_queue.")
    
    print("GAP 4 Migration completed.")

if __name__ == "__main__":
    run_migration()
