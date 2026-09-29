# backend/diario_afirmacion.py
"""[P1-PLAN-LOTE-904 · 2026-09-29] Cuándo una frase del coach NO afirma un registro del diario.

`P1-DIARY-CLAIM-VERIFY` (agent.py) reintenta el turno si el coach dice «anoté/registrado» sin haber llamado
`log_consumed_meal`. En el modo voz del dueño (29-sep, 19:14-19:15 UTC) saltó en DOS de tres turnos seguidos, y cada
salto es otra llamada al modelo (~1 s) para escribir casi la misma frase:

  · «Ah» (ruido que el micrófono mandó)  → «Te anoté un vaso de agua, ya llevas 1 de 9»   — agua, no el diario;
  · «Hey hola»                            → «Con el desayuno anotado y un vaso de agua, lo que sigue es el almuerzo»
                                            — describe lo que YA había, no afirma un registro de este turno.

Tres excepciones estrechas; lo demás sigue igual (una afirmación de comida sin la tool se reintenta):
  1. la frase habla solo de agua (dice «agua» o «hidratación») y de ninguna comida ni otra bebida;
  2. el participio va en una construcción de ESTADO: «con el desayuno anotado», «tienes la cena registrada»;
  3. el mensaje del usuario no trae nada que registrar: charla corta («Hey hola») o ruido («Ah», «mmm»).
"""
from __future__ import annotations

import re

_RE_AGUA = re.compile(r"\b(?:agua|hidrataci[oó]n)\b", re.IGNORECASE)
# Una bebida con calorías SÍ va al diario: «te anoté un vaso de jugo» se sigue verificando.
_RE_OTRA_BEBIDA = re.compile(
    r"\b(?:jugos?|batidos?|leche|refrescos?|sodas?|caf[eé]s?|té|cervezas?|vinos?|morir\s+so[nñ]ando|avena)\b",
    re.IGNORECASE,
)

# «con el desayuno anotado», «con tus dos comidas registradas»: con + determinante + (nombre…) justo antes del
# participio, sin verbo de por medio (no cruza coma, punto y coma ni dos puntos).
_RE_ESTADO_CON = re.compile(
    r"\bcon\s+(?:el|la|los|las|tu|tus|su|sus|este|esta|estos|estas|ese|esa)\s+[^,;:.!?]{0,40}$",
    re.IGNORECASE,
)
# «ya tienes tu desayuno anotado», «llevas la cena registrada»: tener/llevar + objeto + participio.
_RE_ESTADO_TENER = re.compile(
    r"\b(?:tienes|tiene|tenemos|llevas|lleva|llevamos)\s+(?:ya\s+)?(?:el|la|los|las|tu|tus|su|sus)\s+[^,;:.!?]{0,40}$",
    re.IGNORECASE,
)

# Ruido que el reconocedor de voz manda como mensaje. Sin «sí», «no», «ok» ni «ajá»: pueden contestar «¿lo anoto?».
_RUIDO = {"ah", "aah", "eh", "ehh", "em", "emm", "mm", "mmm", "hm", "hmm", "oh", "uh", "uhm", "um", "ay"}


def frase_es_solo_de_agua(frase: str, re_comida: re.Pattern) -> bool:
    """La frase habla de agua y de ninguna comida (`re_comida` = las palabras de comida del guard)."""
    f = frase or ""
    return bool(_RE_AGUA.search(f)) and not re_comida.search(f) and not _RE_OTRA_BEBIDA.search(f)


def participio_describe_estado(antes: str) -> bool:
    """El texto de la frase ANTES del participio lo convierte en un estado, no en una acción de este turno."""
    return bool(_RE_ESTADO_CON.search(antes or "") or _RE_ESTADO_TENER.search(antes or ""))


def mensaje_sin_nada_que_registrar(texto: str) -> bool:
    """Charla corta o ruido: con eso no hay comida que el coach pudiera haber tenido que anotar."""
    from coach_voz import _normalizar, es_charla_corta
    t = _normalizar(texto)
    if not t:
        return True
    if all(p in _RUIDO for p in t.split()):
        return True
    return es_charla_corta(texto)
