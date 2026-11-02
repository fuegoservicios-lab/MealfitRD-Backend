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
# [BB-016 · 2026-11-02] Referencia a un registro previo o estado pasivo de la base de datos
# (ej. «por lo que quedó anotado», «lo que quedó anotado», «lo que mencionaste quedó anotado como...», «según lo que está registrado»).
_RE_ESTADO_REFERENCIA = re.compile(
    r"\b(?:(?:por|seg[uú]n|en|de|como)\s+(?:lo\s+)?(?:que\s+)?(?:ya\s+)?(?:qued[oó]|est[aá]|estaba|estaban|quedaron|figura|aparece)|"
    r"lo\s+que\s+(?:ya\s+)?(?:qued[oó]|est[aá]|estaba|estaban|quedaron|figura|aparece)|"
    r"que\s+(?:[^,;:.!?]{1,30}\s+)?(?:qued[oó]|est[aá]|estaba|estaban|quedaron)\s*(?:como)?)\s*$",
    re.IGNORECASE,
)

# Ruido que el reconocedor de voz manda como mensaje. Sin «sí», «no», «ok» ni «ajá»: pueden contestar «¿lo anoto?».
_RUIDO = {"ah", "aah", "eh", "ehh", "em", "emm", "mm", "mmm", "hm", "hmm", "oh", "uh", "uhm", "um", "ay"}

# [BB-016 · 2026-11-02] Preguntas o consultas informativas donde el usuario indaga sobre un ingrediente o sobre el diario
# (ej. «Sabes lo que es lo blanco que está en el pan», «¿Qué tiene el plato?», «¿Qué pusiste en mi desayuno?»),
# donde NO está declarando una nueva ingesta ni pidiendo registrar nada nuevo.
_RE_VERBOS_CONSUMO_O_REGISTRO = re.compile(
    r"\b(?:com[ií]|comiste|comimos|comieron|he\s+comido|desayun[eé]|almor[zc][eé]|cen[eé]|merend[eé]|"
    r"tom[eé]|beb[ií]|met[ií]|anota|an[oó]talo|an[oó]tame|registra|reg[ií]stralo|reg[ií]strame|apunta|ap[uú]ntalo|sumal[oa]|s[uú]mal[oa])\b",
    re.IGNORECASE,
)

_RE_PREGUNTA_CONSULTA = re.compile(
    r"^(?:[¿?¡!\s])*(?:sabes?\s+(?:lo\s+que|qu[eé])|qu[eé]\s+(?:es|tiene|viste|pusiste|anotaste|registraste|hay|era|ser[aá])|c[oó]mo\s+sabes?|por\s+qu[eé]\s+(?:pusiste|anotaste|registraste)|cu[aá]nt[ao]s?\s+(?:calor[ií]as|gramos|prote[ií]na|carbohidratos|grasas)?\s+(?:tiene|aporta)|puedo\s+comer|qu[eé]\s+me\s+recomiendas?)\b",
    re.IGNORECASE,
)


def frase_es_solo_de_agua(frase: str, re_comida: re.Pattern) -> bool:
    """La frase habla de agua y de ninguna comida (`re_comida` = las palabras de comida del guard)."""
    f = frase or ""
    return bool(_RE_AGUA.search(f)) and not re_comida.search(f) and not _RE_OTRA_BEBIDA.search(f)


def participio_describe_estado(antes: str) -> bool:
    """El texto de la frase ANTES del participio lo convierte en un estado, no en una acción de este turno."""
    return bool(
        _RE_ESTADO_CON.search(antes or "")
        or _RE_ESTADO_TENER.search(antes or "")
        or _RE_ESTADO_REFERENCIA.search(antes or "")
    )


def mensaje_sin_nada_que_registrar(texto: str) -> bool:
    """Charla corta, ruido o preguntas informativas: con eso no hay comida que el coach pudiera haber tenido que anotar."""
    from coach_voz import _normalizar, es_charla_corta
    t = _normalizar(texto)
    if not t:
        return True
    if all(p in _RUIDO for p in t.split()):
        return True
    if _RE_PREGUNTA_CONSULTA.search(texto) and not _RE_VERBOS_CONSUMO_O_REGISTRO.search(texto):
        return True
    return es_charla_corta(texto)
