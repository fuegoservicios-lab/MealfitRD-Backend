# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1013 · A-02] Corta el razonamiento que el modelo escribe DENTRO de su respuesta.

La batería del 14-nov (caso M1, `gpt-6-luna` sin esfuerzo de razonamiento) devolvió, tras la respuesta correcta:

    …registradas. [UI_ACTION: REFRESH_INVENTORY]
    <lemma>
    ---
    We must reconcile diary context said earlier 1210 but tool says total 520 only. … Need remove.
    Anoté tu **cena**: … (la misma respuesta repetida)

El filtro de deliberación del chat (`_build_final_content_from_messages`) descarta BLOQUES enteros por longitud; esta fuga
viaja dentro de UN mismo bloque. En producción no se ha visto (0 de 940 mensajes guardados en 45 días), así que esto es una
defensa barata, no un parche de un incidente.

Reglas, deliberadamente estrechas para no comerse una respuesta legítima:
  1. una etiqueta `<lemma>` / `</lemma>` (que ninguna respuesta del coach lleva) corta ahí: se queda lo anterior;
  2. una línea `---` seguida de una frase de razonamiento EN INGLÉS («We must…», «Let's…», «The user…») corta desde la raya.
Si lo que queda antes del corte está vacío, el texto se devuelve intacto (preferimos ruido a una respuesta vacía).
"""
from __future__ import annotations

import re

_RE_ETIQUETA = re.compile(r"</?\s*lemma\s*>", re.IGNORECASE)
_RE_RAYA_CON_RAZONAMIENTO = re.compile(
    r"\n[ \t]*-{3,}[ \t]*\n+[ \t]*(?:We (?:must|need|should|have to)|Let'?s\b|The user\b|I (?:must|need|should)\b|"
    r"Need to\b|Okay,|OK,|So we\b|Actually,|Hmm|First,|Now,? (?:we|I)\b)",
)


def cortar_razonamiento_filtrado(texto):
    """Devuelve `texto` sin el razonamiento filtrado; cualquier valor que no sea `str` pasa intacto."""
    if not texto or not isinstance(texto, str):
        return texto
    corte = len(texto)
    m = _RE_ETIQUETA.search(texto)
    if m:
        corte = min(corte, m.start())
    m = _RE_RAYA_CON_RAZONAMIENTO.search(texto)
    if m:
        corte = min(corte, m.start())
    if corte >= len(texto):
        return texto
    antes = texto[:corte].rstrip()
    return antes if antes.strip() else texto
