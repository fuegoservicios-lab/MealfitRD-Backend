"""[P1-PLAN-LOTE-1019 · J6] «¿Esto está bien para mi cena o me paso?» con la foto del plato es una PREGUNTA, no un registro.

Batería del 14-nov (J6, dos corridas seguidas): el usuario manda la foto de un plato y pregunta si le cabe en la cena; el
coach llamó `log_consumed_meal` y contestó «Anoté tu cena…». `respuestas_de_la_foto.foto_para_anotar` ya decía que ese
turno NO es de anotar (lleva «?»), pero el manual repite que «una foto implica consumo» y nada frenaba la escritura.

Guarda determinista y estrecha: bloquea `log_consumed_meal` solo si el último mensaje del usuario es una pregunta de
VALORACIÓN («¿está bien?», «¿me paso?», «¿puedo comer…?», «¿cuántas calorías tiene?») y no declara en ninguna parte que
ya lo comió o que lo anote («me lo comí», «cené», «anótalo», «I ate»). Fail-open: ante cualquier duda, deja pasar.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional


def _sin_acentos(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", t or "") if not unicodedata.combining(c)).lower()


_RE_VALORACION = re.compile(
    r"\b(esta\s+bien|estara\s+bien|me\s+paso|me\s+pasaria|me\s+excedo|cabe|me\s+cabe|entra\s+en|puedo\s+(comer|cenar|almorzar|desayunar|tomar)"
    r"|deberia|debo\s+comer|es\s+(bueno|buena|saludable|mucho|demasiado)|sirve|que\s+tal|vale\s+la\s+pena|me\s+conviene"
    r"|cuantas\s+calorias\s+tiene|cuanta\s+proteina\s+tiene|cuanto\s+engorda|lo\s+como|me\s+lo\s+como|lo\s+ceno"
    r"|is\s+(this|it)\s+(ok|okay|good|fine|too\s+much)|can\s+i\s+(eat|have)|should\s+i|how\s+many\s+calories"
    r"|posso\s+comer|ta\s+bom|est-ce\s+que\s+je\s+peux|c'?est\s+bon|posso\s+mangiare|va\s+bene)\b",
    re.IGNORECASE,
)

_RE_YA_LO_COMIO = re.compile(
    r"\b(me\s+(lo|la|los|las)\s+(comi|cene|almorce|desayune|tome|bebi)|ya\s+(me\s+)?(lo|la)\s+(comi|cene)"
    r"|comi|cene|almorce|desayune|merende|me\s+tome|me\s+bebi|me\s+comi|fue\s+(mi|el|la)|fueron|era\s+mi"
    r"|anota|anotalo|anotala|registra|registralo|registrala|apunta|apuntalo|agregalo|sumalo|ponlo\s+en"
    r"|i\s+(ate|had|just\s+ate|drank)|log\s+(it|this)|comeu|comi\s+isso|j'?ai\s+mange|ho\s+mangiato)\b",
    re.IGNORECASE,
)

MENSAJE_BLOQUEO = (
    "BLOQUEADO: el usuario te está PREGUNTANDO si ese plato le conviene o le cabe (todavía no dice que se lo comió ni "
    "que lo anotes): NO lo registres. Contéstale la pregunta con las cifras del análisis y lo que le queda de su meta "
    "de hoy, y al final ofrécele en una frase corta anotarlo cuando se lo coma. Nunca digas que lo anotaste."
)


def bloqueo_si_solo_pregunta(tool_name: str, texto_usuario: str) -> Optional[str]:
    """Texto de bloqueo para el modelo, o None si el registro puede seguir."""
    try:
        if tool_name != "log_consumed_meal":
            return None
        crudo = str(texto_usuario or "")
        if "?" not in crudo and "¿" not in crudo:
            return None
        t = _sin_acentos(crudo)
        if not _RE_VALORACION.search(t) or _RE_YA_LO_COMIO.search(t):
            return None
        return MENSAJE_BLOQUEO
    except Exception:
        return None
