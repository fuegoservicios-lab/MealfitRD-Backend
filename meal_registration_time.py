"""Clock convention and conservative defaults for meals reported as just eaten."""
from __future__ import annotations

import math
import re
import unicodedata


def _latest_human_text(messages):
    for message in reversed(messages or []):
        role = message.get('role') if isinstance(message, dict) else getattr(message, 'type', None)
        if role not in ('human', 'user'):
            continue
        text = message.get('content', '') if isinstance(message, dict) else getattr(message, 'content', '')
        if isinstance(text, list):
            text = ' '.join(part.get('text', '') for part in text
                            if isinstance(part, dict) and part.get('type') == 'text')
        if isinstance(text, str):
            if 'Mensaje del usuario:' in text:
                text = text.rsplit('Mensaje del usuario:', 1)[1].strip().strip('"')
            text = re.sub(r'<dietary_context>[\s\S]*?</dietary_context>', '', text)
        return ''.join(c for c in unicodedata.normalize('NFD', text.lower())
                       if not unicodedata.combining(c)) if isinstance(text, str) else ''
    return ''


def explicit_repeat_consumption(messages):
    """A fresh statement that another serving was eaten, never an assistant's guess."""
    text = _latest_human_text(messages)
    if text.lstrip().startswith('¿') or re.search(r'\b(no|not|didn.t|nao|pas|non)\b', text):
        return False
    return bool(re.search(r'\b(?:repeti (?:el plato|la comida|la porcion|la racion|el desayuno|el almuerzo|la cena)|'
                          r'me (?:comi|tome) otr[oa]|comi de nuevo|'
                          r'i (?:ate|had) (?:another|a second)|otra (?:porcion|racion|vez).{0,20}(?:comi|tome))\b', text))


def confirmed_repeat_consumption(messages, previous_warning=False):
    """Allow a real repeat or a clear yes to the previous duplicate warning."""
    if explicit_repeat_consumption(messages):
        return True
    text = _latest_human_text(messages).strip()
    return bool(previous_warning and re.fullmatch(
        r'(?:si|yes)(?:[,\s]+(?:dos|comi dos|me lo comi otra vez|eran dos porciones))?[.!\s]*', text))


def _isolated_snack(text):
    body = re.sub(r'^(?:me (?:comi|he comido|tome)|comi|he comido|i (?:ate|had))\s+', '', text.strip())
    # Conservative whole-description match: cookies alongside eggs/plantain stay a full meal.
    return bool(re.fullmatch(r'(?:(?:una?|unos?|dos|tres|cuatro|cinco|\d+)\s+)?'
                             r'(?:(?:paquete|bolsa|porcion)\s+de\s+)?'
                             r'(?:galletas?(?:\s+oreo)?|oreos?|papitas|chips|caramelos?|chocolate|helado)'
                             r'(?:\s+(?:grande|grandes|pequeno|pequenas|suelta|sueltas|entero|completo))*[.!\s]*', body))


def normalize_live_offset(value, convention=None):
    """Convert legacy live offsets to the JS convention used throughout chat."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or not -840 <= value <= 840 or value != int(value):
        return None
    return int(value) if convention == 'utc_minus_local' else -int(value)


_RE_EXPLICIT_MEAL_SLOT = re.compile(
    r'\b(desayun\w*|almor[zc]\w*|almuer[zc]\w*|cen\w*|mer(?:i)?end\w*|snack\w*|'
    r'breakfast|lunch|dinner|supper|cafe da manha|almo[cç]o|jantar|lanch\w*|'
    r'petit.dejeuner|dejeuner|diner|gout\w*|colazione|pranzo)\b',
    re.IGNORECASE,
)


def explicit_slot_named_by_user(messages) -> bool:
    """Returns True if the user explicitly named the meal slot (e.g. 'Merendé esto', 'mi cena', 'almorcé')."""
    text = _latest_human_text(messages)
    if not text:
        return False
    return bool(_RE_EXPLICIT_MEAL_SLOT.search(text))


def infer_current_meal_slot(messages, local_hour, schedule_type=None):
    """Only a fresh, unqualified consumption statement gets the clock default.

    Explicit meal labels, earlier dates/times, clarifications, and shifted schedules
    stay with the coach's existing interpretation. An isolated snack stays a snack;
    never use an assistant's previous question to determine the slot.
    """
    if local_hour is None or schedule_type in ('night_shift', 'variable'):
        return None
    text = _latest_human_text(messages)
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
    if _RE_EXPLICIT_MEAL_SLOT.search(text):
        return None
    if re.search(r'\b(ayer|anteayer|antier|anoche|madrugada|manana|yesterday|last night|earlier|'
                 r'ontem|hier|ieri|lunes|martes|miercoles|jueves|viernes|sabado|domingo|'
                 r'hace \w+ (?:hora|minuto)|por la|esta tarde|esta noche|'
                 r'at \d|as \d|alle \d|a \d|am|pm)\b|\d{1,2}:\d{2}', text):
        return None
    if re.search(r'\b(?:a las?|as|alle)\s+(?:\d{1,2}|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce)\b', text):
        return None
    if _isolated_snack(text):
        return 'snack'
    from coach_day_context import franja_por_hora
    return franja_por_hora(local_hour)
