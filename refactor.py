import codecs

with codecs.open("agent.py", "r", "utf-8") as f:
    lines = f.readlines()

new_lines = []
skip = False

i = 0
while i < len(lines):
    line = lines[i]
    
    # 1. Schemas to imports (Lines 24 to 140 approx)
    if line.startswith("# Schema Definitions for Strict"):
        # Insert imports
        new_lines.extend([
            "from schemas import MacrosModel, MealModel, DailyPlanModel, PlanModel, ShoppingListModel\n",
            "from prompts import (\n",
            "    ANALYZE_SYSTEM_PROMPT, DETERMINISTIC_VARIETY_PROMPT, SWAP_MEAL_PROMPT_TEMPLATE, \n",
            "    AUTO_SHOPPING_LIST_PROMPT, TITLE_GENERATION_PROMPT, RAG_ROUTER_PROMPT,\n",
            "    CHAT_SYSTEM_PROMPT_BASE, CHAT_STREAM_SYSTEM_PROMPT_BASE\n",
            ")\n",
            "from tools import (\n",
            "    update_form_field, generate_new_plan_from_chat,\n",
            "    log_consumed_meal, modify_single_meal,\n",
            "    add_to_shopping_list, search_deep_memory, agent_tools, analyze_preferences_agent\n",
            ")\n\n",
            "# Langchain Chat Model Initialization\n",
            "llm = ChatGoogleGenerativeAI(\n",
            "    model=\"gemini-3.1-pro-preview\",\n",
            "    temperature=0.2,\n",
            "    google_api_key=os.environ.get(\"GEMINI_API_KEY\")\n",
            ")\n\n"
        ])
        while i < len(lines) and not "--- PERFIL DE GUSTOS DEL USUARIO" in lines[i]:
            i += 1
        i += 1 # skip that return line
        continue
        
    # 2. Swap meal deterministic prompt
    if 'prompt = f"""' in line and i+1 < len(lines) and "⚠️ REGLA DE INVERSIÓN DE CONTROL DETERMINISTA" in lines[i+1]:
        new_lines.extend([
            "    prompt = DETERMINISTIC_VARIETY_PROMPT.format(\n",
            "        protein_0=chosen_proteins[0], carb_0=chosen_carbs[0],\n",
            "        protein_1=chosen_proteins[1], carb_1=chosen_carbs[1],\n",
            "        protein_2=chosen_proteins[2], carb_2=chosen_carbs[2],\n",
            "        blocked_text=blocked_text\n",
            "    )\n"
        ])
        while i < len(lines) and not '    """\n' in lines[i] and not '    """' in lines[i]:
            i += 1
        i += 1
        continue
        
    # 3. Swap meal prompt
    if 'prompt_text = f"""' in line and i+1 < len(lines) and "Eres el Chef Analítico e Inteligencia Artificial" in lines[i+1]:
        new_lines.extend([
            "    prompt_text = SWAP_MEAL_PROMPT_TEMPLATE.format(\n",
            "        rejected_meal=rejected_meal,\n",
            "        meal_type=meal_type,\n",
            "        target_calories=target_calories,\n",
            "        diet_type=diet_type,\n",
            "        context_extras=context_extras\n",
            "    )\n"
        ])
        while i < len(lines) and not '    """\n' in lines[i] and not '    """' in lines[i]:
            i += 1
        i += 1
        continue
        
    # 4. Tools section
    if line.startswith("# ============================================================"):
        if i+1 < len(lines) and lines[i+1].startswith("# TOOL: Actualizar Health Profile del usuario"):
            while i < len(lines) and not lines[i].startswith("agent_tools = ["):
                i += 1
            i += 1 # skip agent_tools line
            continue

    # 5. auto shopping prompt
    if 'prompt = f"""' in line and i+1 < len(lines) and "Eres el Asistente de Compras Inteligente de MealfitRD." in lines[i+1]:
        new_lines.append("    prompt = AUTO_SHOPPING_LIST_PROMPT.format(ingredients_json=json.dumps(ingredients))\n")
        while i < len(lines) and not '    """\n' in lines[i] and not '    """' in lines[i]:
            i += 1
        i += 1
        continue
        
    # 6. Title generation prompt
    if 'prompt = f"Genera un título muy corto' in line:
        new_lines.append("        prompt = TITLE_GENERATION_PROMPT.format(first_message=first_message)\n")
        i += 1
        continue
        
    # 7. RAG router prompt
    if 'rewrite_prompt = f"""Eres un optimizador' in line:
        new_lines.append("        rewrite_prompt = RAG_ROUTER_PROMPT.format(prompt=prompt)\n")
        while i < len(lines) and not 'Query optimizada:"""\n' in lines[i] and not 'Query optimizada:"""' in lines[i]:
            i += 1
        i += 1
        continue

    # 8. Chat system prompt base
    if 'system_prompt = """Eres el agente' in line:
        if i+3 < len(lines) and "Llámalas SIEMPRE" in lines[i+3]:
            # This is CHAT_STREAM_SYSTEM_PROMPT_BASE because it has visual format below
            if "REGLAS DE FORMATO VISUAL" in "".join(lines[i:i+10]):
                new_lines.append("    system_prompt = CHAT_STREAM_SYSTEM_PROMPT_BASE\n")
                while i < len(lines) and not 'bloque denso."""' in lines[i]:
                    i += 1
                i += 1
                continue
            else:
                new_lines.append("    system_prompt = CHAT_SYSTEM_PROMPT_BASE\n")
                while i < len(lines) and not 'usuario."""' in lines[i]:
                    i += 1
                i += 1
                continue

    new_lines.append(line)
    i += 1

with codecs.open("agent.py", "w", "utf-8") as f:
    f.writelines(new_lines)

print("done")
