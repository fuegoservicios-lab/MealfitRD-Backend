# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1047] Los gustos del perfil solo cambian si el usuario LO DIJO.

Caso real (16-nov): el coach preguntó «¿te acuerdas de la marca del yogur?», el usuario contestó «No» y el
modelo llamó a `update_form_field(dislikes="Yogur")`. Los gustos se REEMPLAZAN, así que la lista anterior se
perdió, y la limpieza borró los recuerdos de categoría «rechazo».

Dos frenos deterministas, sin IA:
  1. Sin evidencia en el último mensaje del usuario (una frase de rechazo, o el alimento nombrado junto a un
     «no»/«tampoco»), no se escribe. Nombrar el alimento solo («el yogur que te pregunté») no es rechazarlo.
  2. Si la lista nueva omite gustos previos y el usuario no pidió quitar ninguno, se conservan (unión).
"""
from __future__ import annotations

import re
import unicodedata

CAMPOS = ("dislikes",)


def _norm(t) -> str:
    t = unicodedata.normalize("NFD", str(t or "").lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


_RE_RECHAZO = re.compile(
    r"\bno me (?:gusta|gustan|cae|caen|sienta|sientan|apetece|provoca)\b|\bodio\b|\bdetesto\b|\bno soporto\b"
    r"|\bno (?:como|tolero|quiero (?:comer|mas|que me))\b|\bme (?:cae|caen|sienta|sientan) mal\b|\basco\b"
    r"|\bme disgusta|\bevit[oa]\b|\bdislike|\bhate\b|\bdon'?t (?:like|eat)\b|\bn'aime pas\b|\bdeteste\b"
)
_RE_QUITAR = re.compile(
    r"\b(?:quita|quitame|quitalo|quitala|saca|sacalo|elimina|borra|remueve)\b|\bya (?:me gusta|me gustan|como)\b"
    r"|\bahora (?:me gusta|me gustan|como)\b|\bremove\b|\bnow i (?:like|eat)\b"
)


def _items(valor) -> list[str]:
    if isinstance(valor, (list, tuple)):
        return [str(x).strip() for x in valor if str(x).strip()]
    return [x.strip() for x in str(valor or "").split(",") if x.strip()]


def _nombrado(item: str, texto_n: str) -> bool:
    palabras = [p for p in re.findall(r"\w+", _norm(item)) if len(p) >= 4]
    return any(p[:-1] in texto_n for p in palabras)   # «yogures»/«yogur», «tomates»/«tomate»


def revisar(field: str, new_value, texto_usuario: str, previos) -> tuple[bool, str]:
    """(True, valor_a_escribir) o (False, mensaje para el modelo). Campos ajenos pasan sin tocar."""
    if field not in CAMPOS:
        return True, new_value
    texto_n = _norm(texto_usuario)
    nuevos = _items(new_value)
    _negado = re.search(r"\b(?:no|tampoco|nada de|nunca|sin)\b", texto_n) is not None
    if not (_RE_RECHAZO.search(texto_n) or _RE_QUITAR.search(texto_n)
            or (_negado and any(_nombrado(i, texto_n) for i in nuevos))):
        return False, ("NO CAMBIÉ SUS GUSTOS: en su último mensaje el usuario no dijo que algo no le guste. "
                       "Si contestó «no» a una pregunta tuya, es la respuesta a ESA pregunta. No digas que "
                       "lo tendrás en cuenta; sigue con lo que estaban hablando.")
    if _RE_QUITAR.search(texto_n):
        return True, ", ".join(nuevos)
    vistos = {_norm(x) for x in nuevos}
    union = nuevos + [p for p in _items(previos) if _norm(p) not in vistos]
    return True, ", ".join(union)
