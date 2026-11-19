# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1102 · 2026-11-18] Lo que ya se comió ANTES de la foto suma también sus kcal y macros.

La regla del lote 1087 («4 panecillos en la foto + "ya me había comido uno" = 5») la siguió el coach en el NOMBRE pero
no en las cifras: batería J7 en real (gpt-6-luna, 18-nov), tres corridas: «5 panecillos…» con 560 kcal, la estimación
de la foto para los 4 que se ven; y una vez hasta volvió a «4 panecillos». Un ajuste del prompt no lo movió (560 en las
dos corridas siguientes). Este guard lo hace sin modelo, justo antes de registrar:

  · el mensaje del usuario dice que ya se había comido N antes («ya me había comido uno», «me comí dos antes»);
  · el turno trae UNA estimación de foto («(Estimación: Calorías: 560, …)», la del analizador);
  · el registro conserva esa estimación (las kcal quedan a ±4 % de las de la foto: el modelo no las tocó);
  · y el nombre (o el primer ingrediente) cuenta el total («5 panecillos»), mayor que N.

Entonces escala kcal y macros por total / (total − N): 4 → 5 = ×1,25. Si algo no cuadra, no toca nada (fail-open).
«Eran 4 y ya me comí uno» (lo comido está DENTRO de lo que se ve) no se escala.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Optional

_NUM = {"un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6}
_RE_YA_COMIDO = re.compile(
    r"\bya\s+me\s+(?:habia\s+)?comi(?:do)?\s+(un[oa]?|dos|tres|cuatro|cinco|seis|\d+)\b"
    r"|\bme\s+comi\s+(un[oa]?|dos|tres|cuatro|cinco|seis|\d+)\b[^.?!]{0,30}\bantes\b"
)
# lo comido ya está contado en lo que se ve: «eran 4 y ya me comí uno», «de los 4, ya me comí uno»
_RE_INCLUIDO = re.compile(r"\b(?:eran|habia|tenia|de\s+(?:los|las))\s+\d+\b")
_RE_ESTIMACION = re.compile(
    r"\(Estimación: Calorías: (?P<calories>-?\d+(?:\.\d+)?), Proteína: (?P<protein>-?\d+(?:\.\d+)?)g, "
    r"Carbohidratos: (?P<carbs>-?\d+(?:\.\d+)?)g, Grasas Saludables: (?P<healthy_fats>-?\d+(?:\.\d+)?)g\)"
)
_RE_TOTAL = re.compile(r"^\s*(\d{1,2})\s+[a-záéíóúñ]")
TOLERANCIA = 0.04
FACTOR_MAX = 3.0


def _plano(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", t or "") if not unicodedata.combining(c)).lower()


def _texto(m: Any) -> str:
    c = getattr(m, "content", m if isinstance(m, str) else "")
    if isinstance(c, list):
        return " ".join(str(p.get("text") or "") if isinstance(p, dict) else str(p) for p in c)
    return str(c or "")


def extra_ya_comido(texto: str) -> Optional[int]:
    """Cuántos dice que ya se había comido antes de la foto, o None (o si lo comido ya está en lo que se ve)."""
    p = _plano(texto)
    if _RE_INCLUIDO.search(p):
        return None
    m = _RE_YA_COMIDO.search(p)
    if not m:
        return None
    w = m.group(1) or m.group(2)
    n = _NUM.get(w) if not w.isdigit() else int(w)
    return n if n and 0 < n <= 10 else None


_RE_PIEZAS = re.compile(r"\b(\d{1,2})\s+(?:[a-z]+\s+)?[a-z]{4,}")


def piezas_en_la_foto(texto: str) -> Optional[int]:
    """El número de piezas que dice la descripción de la foto (lo que va antes de su «(Estimación: …)»), o None."""
    try:
        i = texto.find("(Estimación:")
        if i < 0:
            return None
        inicio = max(texto.rfind("\n", 0, i), 0)
        m = _RE_PIEZAS.search(_plano(texto[inicio:i]))
        return int(m.group(1)) if m else None
    except Exception:
        return None


def _ultimo_del_usuario(messages: list) -> str:
    for m in reversed(messages or []):
        if getattr(m, "type", None) == "human" or (isinstance(m, dict) and m.get("role") == "user"):
            return _texto(m) if not isinstance(m, dict) else str(m.get("content") or "")
    return ""


def ajustar_args(tool_name: str, args: Any, messages: list, sys_prompt: str = "") -> Any:
    """Los args de `log_consumed_meal`, escalados si se cumple TODO lo del docstring; si no, los mismos."""
    try:
        if tool_name != "log_consumed_meal" or not isinstance(args, dict):
            return args
        n = extra_ya_comido(_ultimo_del_usuario(messages))
        if not n:
            return args
        ests = _RE_ESTIMACION.findall(str(sys_prompt or "")) or [
            e for m in (messages or []) for e in _RE_ESTIMACION.findall(_texto(m))]
        if len(ests) != 1:
            return args
        foto = dict(zip(("calories", "protein", "carbs", "healthy_fats"), (float(x) for x in ests[0])))
        cal = float(args.get("calories") or 0)
        if foto["calories"] <= 0 or abs(cal - foto["calories"]) > TOLERANCIA * foto["calories"]:
            return args       # el modelo ya movió las cifras: no se toca
        ings = args.get("ingredients") or []
        m = _RE_TOTAL.match(_plano(str(args.get("meal_name") or ""))) or (
            _RE_TOTAL.match(_plano(str(ings[0]))) if ings else None)
        if not m:
            return args
        total = int(m.group(1))
        if total <= n:
            return args
        # cuántas piezas dice la foto («Plato con 4 panecillos…»): si se sabe, el total tiene que ser ESAS más las ya
        # comidas; si el modelo anotó solo las de la foto (se olvidó del extra), no se escala a medias
        en_foto = piezas_en_la_foto(str(sys_prompt or "") or " ".join(_texto(x) for x in (messages or [])))
        if en_foto is not None and total != en_foto + n:
            return args
        factor = total / (total - n)
        if factor > FACTOR_MAX:
            return args
        nuevo = dict(args)
        for k in ("calories", "protein", "carbs", "healthy_fats"):
            if args.get(k) is not None:
                try:
                    nuevo[k] = int(round(float(args[k]) * factor))
                except (TypeError, ValueError):
                    pass
        return nuevo
    except Exception:
        return args


def nota_para_el_coach(antes: Any, despues: Any) -> str:
    """La nota que va con el resultado del registro cuando el guard escaló: sin ella el modelo ve otras kcal de las que
    mandó y las «corrige» (batería J7, 18-nov: cinco `correct_consumed_meal` seguidos)."""
    try:
        if not (isinstance(antes, dict) and isinstance(despues, dict)) or antes is despues:
            return ""
        a, d = antes.get("calories"), despues.get("calories")
        if a == d:
            return ""
        return (f" (Para el asistente: sumé lo que ya se había comido antes de la foto — {int(round(float(a)))} → "
                f"{int(round(float(d)))} kcal, y los macros en la misma proporción —; usa ESTAS cifras y no lo corrijas.)")
    except Exception:
        return ""


def correccion_que_deshace(tool_name: str, args: Any, messages: list, sys_prompt: str = "") -> str:
    """Si el modelo intenta DESHACER lo comido antes de la foto (vuelve al conteo o a las kcal de la foto), el texto con
    el que se le contesta en vez de ejecutar la corrección; "" si la corrección es otra cosa y debe ejecutarse.

    Batería J7 (18-nov): tras registrar «5 panecillos» a 700 kcal, una corrida corrigió el nombre a «4 panecillos»
    (con los ingredientes aún en 5) y otra lo «corrigió» cinco veces."""
    try:
        if tool_name != "correct_consumed_meal" or not isinstance(args, dict) or args.get("delete"):
            return ""
        n = extra_ya_comido(_ultimo_del_usuario(messages))
        if not n:
            return ""
        ests = _RE_ESTIMACION.findall(str(sys_prompt or "")) or [
            e for m in (messages or []) for e in _RE_ESTIMACION.findall(_texto(m))]
        if len(ests) != 1:
            return ""
        kcal_foto = float(ests[0][0])
        cal = args.get("calories")
        vuelve_a_la_foto = cal is not None and kcal_foto > 0 and abs(float(cal) - kcal_foto) <= TOLERANCIA * kcal_foto
        ings = args.get("ingredients") or []
        m_nombre = _RE_TOTAL.match(_plano(str(args.get("meal_name") or "")))
        m_ing = _RE_TOTAL.match(_plano(str(ings[0]))) if ings else None
        baja_el_conteo = bool(m_nombre and m_ing and int(m_nombre.group(1)) < int(m_ing.group(1)))
        if not (vuelve_a_la_foto or baja_el_conteo):
            return ""
        return ("Sin cambios: la comida ya quedó bien registrada sumando lo que se comió antes de la foto (no se ve en "
                "ella). No hace falta corregirla; confirma al usuario lo registrado.")
    except Exception:
        return ""
