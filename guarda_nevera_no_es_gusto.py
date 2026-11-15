"""[P1-PLAN-LOTE-1018 · D2] «Se me acabó el pollo» es la Nevera, no un gusto ni una alergia.

Caso de la batería del 14-nov (D2): con el mensaje «se me acabó el pollo» el coach llamó
`update_form_field(field="dislikes", new_value="pollo")` y el perfil quedó diciendo que no le gusta el pollo; a partir de
ahí el generador de planes lo quita de todos sus menús. La causa inmediata (la palabra con acento escondía las
herramientas de la Nevera) se arregló en el lote 1016, pero nada impedía que el modelo volviera a hacerlo.

La guarda es determinista y estrecha: solo bloquea escribir `dislikes`/`allergies` cuando el mensaje del usuario habla de
quedarse SIN algo («se me acabó», «se me terminó», «ya no tengo», «no me queda», «se me dañó/venció») y no dice en
ninguna parte que no le guste, no lo tolere o le haga daño. Fail-open: ante cualquier duda, deja pasar.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

_CAMPOS = frozenset({"dislikes", "allergies"})


def _sin_acentos(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", t or "") if not unicodedata.combining(c)).lower()


# Quedarse sin algo (es, en, pt, fr, it), sobre el texto ya sin acentos.
_RE_SE_ACABO = re.compile(
    r"\b(se\s+me\s+(acabo|acabaron|termino|terminaron|agoto|agotaron|dano|danaron|vencio|vencieron|pudrio|pudrieron)"
    r"|se\s+(acabo|acabaron|termino|terminaron|agoto|agotaron|dano|danaron|vencio|vencieron|pudrio|pudrieron)\s+(el|la|los|las|mi|mis)"
    r"|ya\s+no\s+(tengo|me\s+queda|me\s+quedan|hay)|no\s+me\s+(queda|quedan)\s+(nada\s+de\s+)?(el|la|los|las|mas)?"
    r"|ran\s+out\s+of|out\s+of\s+\w+\s+(now|again)?|no\s+more\s+\w+\s+(left|at\s+home)|don'?t\s+have\s+any\s+\w+\s+left"
    r"|acabou|nao\s+tenho\s+mais|je\s+n'?ai\s+plus|e\s+finito|non\s+ho\s+piu)\b",
    re.IGNORECASE,
)

# Lo que SÍ autoriza escribir un gusto o una alergia aunque también diga que se le acabó.
_RE_GUSTO_O_SALUD = re.compile(
    r"\b(no\s+me\s+gusta|no\s+me\s+gustan|odio|detesto|no\s+soporto|no\s+tolero|intoleran|alergi|alergic|me\s+cae\s+mal"
    r"|me\s+hace\s+dano|me\s+da\s+asco|no\s+quiero\s+(mas|volver)|no\s+lo\s+como|no\s+como|nunca\s+como|quitalo|sacalo"
    r"|hate|dislike|don'?t\s+like|allerg|intoleran|can'?t\s+stand|nao\s+gosto|je\s+n'?aime\s+pas|non\s+mi\s+piace)",
    re.IGNORECASE,
)

MENSAJE_BLOQUEO = (
    "BLOQUEADO: el usuario dijo que se le ACABÓ / ya no tiene ese alimento; eso es su Nevera, no un gusto ni una alergia, "
    "y escribirlo en el perfil lo quitaría de todos sus planes. No cambies el perfil. Si tienes la herramienta "
    "modify_pantry_inventory, úsala para descontarlo (items_to_deplete); si no, díselo en una frase y ofrece cómo seguir "
    "(con qué sustituirlo en la próxima comida que lo lleve)."
)


def bloqueo_si_es_la_nevera(tool_name: str, tool_args, texto_usuario: str) -> Optional[str]:
    """Devuelve el texto de bloqueo para el modelo o None si la escritura puede seguir."""
    try:
        if tool_name != "update_form_field" or not isinstance(tool_args, dict):
            return None
        if str(tool_args.get("field") or "") not in _CAMPOS:
            return None
        t = _sin_acentos(str(texto_usuario or ""))
        if not t.strip() or not _RE_SE_ACABO.search(t):
            return None
        if _RE_GUSTO_O_SALUD.search(t):
            return None
        return MENSAJE_BLOQUEO
    except Exception:
        return None
