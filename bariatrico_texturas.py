# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-266 · 2026-09-25] Bariátrico: frutos secos y semillas SIEMPRE molidos o en crema, nunca enteros.

La regla clínica bariátrica que el sistema le da al modelo (`condition_rules`: «NUECES Y SEMILLAS SIEMPRE MOLIDAS O EN
MANTEQUILLA… NUNCA enteras: riesgo de OBSTRUCCIÓN del pouch/anastomosis») era solo una instrucción; el tope de porción
(`cap_bariatric_portions`) limitaba los GRAMOS (20 g) pero no la textura. Batería final (bariátrica + SOP): «20 g de maní
sin sal» en la merienda nocturna y «20 g de almendras fileteadas» en otra merienda.

Se reescribe la FORMA —lista, pasos y nombre— sin tocar gramos ni macros (moler no cambia la composición). Lo que ya es
seguro (molido, crema/mantequilla, harina, leche o bebida vegetal, aceite, pasta) no se toca, ni la nuez moscada, que es
una especia. Corre dentro de `cap_bariatric_portions`, que ya solo actúa con la condición bariátrica.
tooltip-anchor: P1-PLAN-LOTE-266-MOLIDOS
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# (patrón del núcleo, participio que concuerda). Orden = prioridad: lo compuesto antes que lo simple.
_NUCLEOS = (
    (r"semillas de (?:ch[ií]a|calabaza|auyama|girasol|ajonjol[ií]|s[ée]samo|linaza|lino|c[áa][ñn]amo)", "molidas"),
    (r"pepitas(?: de (?:calabaza|auyama|girasol))?", "molidas"),
    (r"almendras", "molidas"), (r"almendra", "molida"),
    (r"nueces", "molidas"), (r"nuez(?! moscada)", "molida"),
    (r"man[ií]es", "molidos"), (r"man[ií]", "molido"),
    (r"cacahuates", "molidos"), (r"cacahuate", "molido"), (r"cacahuetes", "molidos"),
    (r"pistachos", "molidos"), (r"pistacho", "molido"),
    (r"avellanas", "molidas"), (r"avellana", "molida"),
    (r"mereyes", "molidos"), (r"merey", "molido"),
    (r"mara[ñn]ones", "molidos"), (r"mara[ñn][óo]n", "molido"),
    (r"linaza", "molida"), (r"ch[ií]a", "molida"),
    (r"ajonjol[ií]", "molido"), (r"s[ée]samo", "molido"),
)
_NUCLEO_FULL = tuple((re.compile(r"(?i)" + p + r"\Z"), part) for p, part in _NUCLEOS)
# Texturas que pasan a «molido»; «tostado» también (tostado y molido no chocan, pero entero sí).
_TEXTURA = (r"(?:enter[oa]s?|filetead[oa]s?|laminad[oa]s?|picad[oa]s?|trocead[oa]s?|tostad[oa]s?|en trozos|"
            r"en mitades|en l[áa]minas)")
_RX = re.compile(r"(?i)\b(" + "|".join(p for p, _ in _NUCLEOS) + r")\b(\s+" + _TEXTURA + r")?")
_ANTES_SEGURO = re.compile(r"(?i)(?:mantequilla|crema|harina|leche|bebida|aceite|pasta)\s+de\s+(?:las?\s+|los?\s+)?\Z")
_DESPUES_SEGURO = re.compile(r"(?i)\A\s+molid")


def _participio(nucleo: str) -> str:
    return next((p for rx, p in _NUCLEO_FULL if rx.match(nucleo)), "molidos")


def moler_texto(texto: str, *, solo_con_textura: bool = False) -> str:
    """Los frutos secos y semillas del texto, en su forma molida. `solo_con_textura`: sólo los que traen una textura
    («almendras fileteadas» → «almendras molidas»), para pasos y nombres, donde «el maní» a secas se deja."""
    if not isinstance(texto, str) or not texto:
        return texto

    def _uno(m):
        if _ANTES_SEGURO.search(texto[:m.start()]) or _DESPUES_SEGURO.match(texto[m.end(1):]):
            return m.group(0)
        part = _participio(m.group(1))
        if m.group(2):
            textura = m.group(2).strip()
            return m.group(1) + " " + (part[:1].upper() + part[1:] if textura[:1].isupper() else part)
        return m.group(0) if solo_con_textura else m.group(1) + " " + part

    return _RX.sub(_uno, texto)


def moler_frutos_secos(days) -> int:
    """Reescribe a su forma molida los frutos secos y semillas de cada comida (lista, lista cruda, pasos y nombre).
    Devuelve cuántas líneas de ingredientes cambió. Nunca lanza."""
    cambios = 0
    try:
        for d in days or []:
            for m in (d.get("meals") or []) if isinstance(d, dict) else []:
                if not isinstance(m, dict):
                    continue
                tocado = False
                for clave in ("ingredients", "ingredients_raw"):
                    lista = m.get(clave)
                    if not isinstance(lista, list):
                        continue
                    for i, linea in enumerate(lista):
                        nueva = moler_texto(linea)
                        if nueva != linea:
                            lista[i] = nueva
                            cambios += clave == "ingredients"
                            tocado = True
                if not tocado:
                    continue
                if isinstance(m.get("recipe"), list):
                    m["recipe"] = [moler_texto(p, solo_con_textura=True) for p in m["recipe"]]
                if isinstance(m.get("name"), str):
                    m["name"] = moler_texto(m["name"], solo_con_textura=True)
        if cambios:
            logger.info(f"🥜 [P1-PLAN-LOTE-266] bariátrico: {cambios} fruto(s) seco(s)/semilla(s) a su forma molida.")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-266] moler frutos secos falló (no bloquea): {type(e).__name__}: {e}")
    return cambios
