"""Prevent guessed packet weights on diary writes, using human evidence only."""
import re
import unicodedata


def guard(tool_name, args, messages):
    if tool_name not in ('log_consumed_meal', 'correct_consumed_meal') or not isinstance(args, dict):
        return None
    users = []
    for message in messages or []:
        role = message.get('role') if isinstance(message, dict) else getattr(message, 'type', '')
        value = message.get('content') if isinstance(message, dict) else getattr(message, 'content', '')
        if role in ('user', 'human') and isinstance(value, str):
            users.append(''.join(c for c in unicodedata.normalize('NFD', value.lower()) if not unicodedata.combining(c)))
    if not users:
        return None
    latest = users[-1]
    name = str(args.get('meal_name') or '').lower()
    if not re.search(r'\b(?:oreo|galletas?|cookies?)\b', latest + ' ' + name):
        return None
    # A short calorie correction must not manufacture an explicit packet/weight.
    if tool_name == 'correct_consumed_meal' and re.search(r'\b(?:calorias|kcal)\b', latest):
        proposed = name + ' ' + ' '.join(args.get('ingredients') or [])
        if re.search(r'\b(?:paquete|packet|pack)\b|\d+\s*g\b', proposed) and not re.search(r'\b(?:paquete|packet|pack)\b|\d+\s*g\b', latest):
            return ('NO CORREGIDO: las calorías no determinan paquete ni gramos. Corrige solo las cifras '
                    'confirmadas, sin cambiar nombre/ingredientes por una porción inventada; pregunta cuántas '
                    'galletas o gramos consumió antes de cambiar la porción.')
    if tool_name != 'log_consumed_meal':
        return None
    # Explicit piece/weight/count or a requested estimate resolves the ambiguity.
    explicit = re.search(r'\b(?:suelta|suelto|pieza|individual|estimado|aproximado)\b|\d+\s*(?:g|gr|gramos)\b|\b(?:[2-9]\d*|dos|tres|cuatro|cinco|seis)\s+galletas?\b', latest)
    if explicit:
        return None
    if re.search(r'\b(?:una?|1)\s+(?:galleta\s+)?oreo\b|\b(?:paquete|packet|pack)\b', latest):
        return ('NO REGISTRADO: falta concretar la presentación. Pregunta una sola vez: '
                '«¿Fue una galleta suelta o el paquete? ¿Cuántas galletas o gramos tenía lo que comiste?» '
                'No supongas el tamaño del paquete ni calcules su peso desde las calorías.')
    return None
