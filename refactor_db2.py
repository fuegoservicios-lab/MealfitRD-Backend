import ast

with open('db.py', 'r', encoding='utf-8') as f:
    original_code = f.read()

groups = {
    "db_core": ["close_connection_pool", "execute_sql_query", "execute_sql_write"],
    "db_profiles": ["upsert_user_profile", "get_user_profile", "update_user_health_profile", "log_api_usage", "get_monthly_api_usage", "migrate_guest_data"],
    "db_chat": ["delete_user_agent_sessions", "delete_single_agent_session", "update_session_title", "get_or_create_session", "get_session_owner", "get_guest_chat_sessions", "get_user_chat_sessions", "_process_and_sort_sessions", "get_session_messages", "acquire_summarizing_lock", "release_summarizing_lock", "delete_chat_session", "save_message", "save_message_feedback", "get_memory", "save_summary", "get_summaries", "archive_summaries", "search_deep_memory", "delete_summaries", "delete_old_messages", "get_recent_messages", "insert_like", "get_user_likes", "insert_rejection", "get_active_rejections"],
    "db_shopping": ["get_shopping_plan_hash", "save_shopping_plan_hash", "get_user_shopping_lock", "add_custom_shopping_items", "_add_shopping_items_minimal", "delete_auto_generated_shopping_items", "_delete_auto_shopping_items_legacy", "get_custom_shopping_items", "_get_shopping_items_minimal", "clear_all_shopping_items", "uncheck_all_shopping_items", "deduplicate_shopping_items", "_deduplicate_shopping_items_impl", "purge_old_shopping_items", "delete_custom_shopping_item", "delete_custom_shopping_items_batch", "update_custom_shopping_item", "_update_shopping_item_legacy", "update_custom_shopping_item_status", "_update_shopping_item_status_legacy"],
    "db_plans": ["check_recent_meal_plan_exists", "check_meal_plan_generated_today", "save_new_meal_plan_robust", "get_latest_meal_plan", "get_recent_meals_from_plans", "get_recent_techniques", "get_ingredient_frequencies_from_plans", "get_latest_meal_plan_with_id", "update_meal_plan_data", "track_meal_friction", "log_unknown_ingredients", "increment_ingredient_frequencies", "get_user_ingredient_frequencies", "format_qty", "deduct_inventory_items"],
    "db_facts": ["check_fact_ownership", "acquire_fact_lock", "release_fact_lock", "save_user_fact", "delete_expired_temporal_facts", "get_user_facts_by_metadata", "delete_user_facts_by_metadata", "search_user_facts", "search_user_facts_hybrid", "delete_user_fact", "enqueue_pending_fact", "dequeue_pending_facts", "delete_pending_facts", "get_all_user_facts", "save_visual_entry", "search_visual_diary", "log_consumed_meal", "get_consumed_meals_today"]
}

module_node = ast.parse(original_code)
functions_data = {}
all_funcs_in_groups = set()
for v in groups.values():
    for f_name in v:
        all_funcs_in_groups.add(f_name)

for n in module_node.body:
    if isinstance(n, ast.FunctionDef):
        if n.name not in all_funcs_in_groups:
            groups["db_core"].append(n.name)
            
        start = n.lineno - 1
        end = n.end_lineno
        options = []
        for x in [n] + n.decorator_list:
            options.append(x.lineno - 1)
        start = min(options)
            
        functions_data[n.name] = {
            'start': start,
            'end': end
        }

lines = original_code.splitlines()

first_func_start = len(lines)
for data in functions_data.values():
    if data['start'] < first_func_start:
        first_func_start = data['start']

global_setup_lines = lines[:first_func_start]

with open('db_core.py', 'w', encoding='utf-8') as f:
    f.write("\n".join(global_setup_lines))
    f.write("\n")
    for func_name in groups["db_core"]:
        if func_name in functions_data:
            data = functions_data[func_name]
            f.write("\n".join(lines[data['start']:data['end']]))
            f.write("\n\n")

for module_name, func_names in groups.items():
    if module_name == "db_core": continue
    
    with open(f'{module_name}.py', 'w', encoding='utf-8') as f:
        f.write("import logging\n")
        f.write("logger = logging.getLogger(__name__)\n")
        f.write("from db_core import supabase, connection_pool, execute_sql_query, execute_sql_write\n\n")
        
        for func_name in func_names:
            if func_name in functions_data:
                data = functions_data[func_name]
                f.write("\n".join(lines[data['start']:data['end']]))
                f.write("\n\n")

db_facade = """# Facade DB Module para mantener compatibilidad
import os
import logging

from db_core import *
from db_profiles import *
from db_chat import *
from db_shopping import *
from db_plans import *
from db_facts import *
"""

with open('db.py', 'w', encoding='utf-8') as f:
    f.write(db_facade)
    
print("✨ Split completado de forma impecable usando AST Boundaries.")
