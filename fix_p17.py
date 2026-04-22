with open('c:/Users/angel/OneDrive/Escritorio/MealfitRD.IA/backend/routers/plans.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    'selected_techniques = result.pop("_selected_techniques", None)',
    'selected_techniques = result.pop("_selected_techniques", None)\n                            # Evitar filtraciones de estado interno al frontend\n                            result.pop("_profile_embedding", None)\n                            result.pop("_active_learning_signals", None)'
)

# And if there's any indentation mismatch, let's fix it safely using Regex, or just do it inside the script
import re

with open('c:/Users/angel/OneDrive/Escritorio/MealfitRD.IA/backend/routers/plans.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Success')
