"""Clock convention and conservative defaults for meals reported as just eaten."""
from __future__ import annotations

import math
import re
import unicodedata


def normalize_live_offset(value, convention=None):
    """Convert legacy live offsets to the JS convention used throughout chat."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or not -840 <= value <= 840 or value != int(value):
        return None
    return int(value) if convention == 'utc_minus_local' else -int(value)


def infer_current_meal_slot(messages, local_hour, schedule_type=None):
    """Only a fresh, unqualified consumption statement gets the clock default.

    Explicit meal labels, earlier dates/times, clarifications, and shifted schedules
    stay with the coach's existing interpretation. Never use an assistant's question
    or a food name to determine the slot.
    """
    if local_hour is None or schedule_type in ('night_shift', 'variable'):
        return None
    text = None
    for message in reversed(messages or []):
        if getattr(message, 'type', None) in ('human', 'user'):
            content = getattr(message, 'content', '')
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = ' '.join(part.get('text', '') for part in content
                                if isinstance(part, dict) and part.get('type') == 'text')
            break
    if not text:
        return None
    text = ''.join(c for c in unicodedata.normalize('NFD', text.lower())
                   if not unicodedata.combining(c))
    if not re.search(r'\b(me (?:comi|he comido|tome|bebi)|comi|he comido|i (?:ate|had|drank)|'
                     r'comi|bebi|j.ai mange|ho mangiato)\b', text):
        return None
    # A negation, an explicitly named slot or a past/future time is not "just eaten".
    if re.search(r'\b(no (?:me )?(?:comi|he comido|tome|bebi)|didn.t|nao|ne .{0,20} pas)\b', text):
        return None
    if re.search(r'\b(desayun\w*|almorz\w*|almuerz\w*|cen(?:a|e|ar|ando)|meriend\w*|snack\w*|'
                 r'breakfast|lunch|dinner|supper|cafe da manha|almo[cç]o|jantar|lanche|'
                 r'petit.dejeuner|dejeuner|diner|gouter|colazione|pranzo)\b', text):
        return None
    if re.search(r'\b(ayer|anteayer|antier|anoche|madrugada|manana|yesterday|last night|earlier|'
                 r'ontem|hier|ieri|lunes|martes|miercoles|jueves|viernes|sabado|domingo|'
                 r'hace \w+ (?:hora|minuto)|por la|esta tarde|esta noche|'
                 r'at \d|as \d|alle \d|a \d|am|pm)\b|\d{1,2}:\d{2}', text):
        return None
    if re.search(r'\b(?:a las?|as|alle)\s+(?:\d{1,2}|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce)\b', text):
        return None
    from coach_day_context import franja_por_hora
    return franja_por_hora(local_hour)
