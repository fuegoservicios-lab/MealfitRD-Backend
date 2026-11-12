# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-988 · 2026-11-12] «Dos paquetes de las de ayer»: el SISTEMA busca ese registro y hace la cuenta.

Caso vivo del dueño (12-nov, 02:11): ayer anotó «3 galletas Crakeñas CLUB (24 g)» = 106 kcal; hoy «me metí dos paquete de
galletas de las de ayer» y el coach preguntó «¿individuales o grandes? ¿cuántas galletas?». El lote 987 lo dijo en el
prompt; esto lo hace determinista en DOS puntos:

1. `bloque_para_chat(...)`: si el mensaje se refiere a un registro anterior, busca la fila en su diario (días
   biológicos anteriores, hasta 3) y le pasa al modelo la porción y la cuenta hecha (n × porción), con la orden de
   registrar sin preguntar.
2. `rechazo(...)` en `execute_tools`: si `log_consumed_meal` llega con calorías a más del 15 % de n × porción, rechaza
   UNA vez (usuario/120 s) con las cifras. Con cambios declarados («pero sin el atún», «con más arroz») NO rechaza: la
   cuenta ya no es n × porción.

Knob `MEALFIT_PRIOR_LOG_REFERENCE` (True). tooltip-anchor: P1-PLAN-LOTE-988
"""
from __future__ import annotations

import logging
import re
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_VENTANA_S = 120.0
_TOLERANCIA = 0.15
_YA_AVISADO: dict = {}

# «de las de ayer», «como la de anoche», «las mismas de ayer», «igual que el de antier»
_REFERENCIA = re.compile(
    r"\b(?:de|como|igual(?:es)?\s+(?:que|a))\s+(?:las|los|la|el|lo)\s+(?:de|del)\s+(ayer|anoche|antier|anteayer)\b"
    r"|\b(?:las|los|la|el|lo)\s+mism[oa]s?\s+(?:([a-z]+(?:\s+[a-z]+)?)\s+)?(?:de|del|que)\s+(?:ayer|anoche|antier|anteayer)\b")
_CAMBIOS = re.compile(r"\b(?:pero|sin|menos|mas|extra|ademas|en vez|en lugar|cambi\w*|mitad)\b")
_NUMEROS = {"un": 1, "una": 1, "uno": 1, "otra": 1, "otro": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
            "seis": 6, "media": 0.5, "medio": 0.5}
_RE_NUM = re.compile(r"\b(\d+(?:[.,]\d+)?|un|una|uno|otra|otro|dos|tres|cuatro|cinco|seis|media|medio)\b")
_VACIAS = {"me", "meti", "comi", "tome", "de", "las", "los", "la", "el", "lo", "del", "como", "ayer", "anoche",
           "antier", "anteayer", "paquete", "paquetes", "mismo", "misma", "mismos", "mismas", "igual", "iguales",
           "que", "una", "uno", "dos", "tres", "otra", "otro", "con", "y", "pero", "hoy", "tambien", "solo"}


def _norm(t) -> str:
    t = unicodedata.normalize("NFKD", str(t or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", t).strip()


def _sing(w: str) -> str:
    if len(w) > 4 and w.endswith("es") and w[-3] not in "aeiou":
        return w[:-2]
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


_VERBO_FRANJA = {"desayune": "desayuno", "almorce": "almuerzo", "cene": "cena", "merende": "merienda"}


def _tokens(t: str) -> set:
    out = set()
    for w in re.findall(r"[a-z]{4,}", _norm(t)):
        if w in _VERBO_FRANJA:
            out.add(_VERBO_FRANJA[w])
        elif w not in _VACIAS:
            out.add(_sing(w))
    return out


def activo() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_PRIOR_LOG_REFERENCE", True)
    except Exception:                                                          # noqa: BLE001
        return True


def detectar(texto: str) -> Optional[dict]:
    """`{"dias": 1|2, "n": float, "cambios": bool}` si el mensaje se refiere a un registro anterior; si no, None."""
    t = _norm(texto)
    m = _REFERENCIA.search(t)
    if not m:
        return None
    palabra = re.findall(r"ayer|anoche|antier|anteayer", m.group(0))[-1]
    dias = 2 if palabra in ("antier", "anteayer") else 1
    antes = t[:m.start()] + " " + (m.group(2) or "")   # «el mismo DESAYUNO de ayer»: lo que nombra va dentro
    n = 1.0
    num = _RE_NUM.search(antes)
    if num:
        v = num.group(1)
        n = float(v.replace(",", ".")) if v[0].isdigit() else float(_NUMEROS[v])
    return {"dias": dias, "n": n, "cambios": bool(_CAMBIOS.search(t[m.end():]) or _CAMBIOS.search(antes)),
            "tokens": _tokens(antes)}


def _inicio_dia_biologico(ahora_utc: datetime, tz_offset_min: Optional[int]) -> datetime:
    off = 240 if tz_offset_min is None else int(tz_offset_min)
    local = ahora_utc - timedelta(minutes=off)
    corte = (local - timedelta(hours=4)).replace(hour=4, minute=0, second=0, microsecond=0)
    return corte + timedelta(minutes=off)   # vuelve a UTC


def _dt(v) -> Optional[datetime]:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    try:
        d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:                                                          # noqa: BLE001
        return None


def elegir(ref: dict, filas: list, ahora_utc: datetime, tz_offset_min: Optional[int]) -> Optional[dict]:
    """La fila de ese día (biológico) que comparte un alimento con lo que nombra; la más reciente si hay varias. Sin
    alimento nombrado («lo mismo de ayer») no se elige nada: sería adivinar cuál."""
    if not ref or not ref.get("tokens"):
        return None
    hoy = _inicio_dia_biologico(ahora_utc, tz_offset_min)
    desde, hasta = hoy - timedelta(days=ref["dias"]), hoy - timedelta(days=ref["dias"] - 1)
    del_dia = [(f, _dt(f.get("consumed_at") or f.get("created_at"))) for f in filas or []]
    del_dia = [(f, at) for f, at in del_dia if at and desde <= at < hasta]
    mejores = []
    for f, at in del_dia:
        comun = ref["tokens"] & _tokens(f.get("meal_name"))
        if comun:
            mejores.append((len(comun), at, f))
    if mejores:
        mejores.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return mejores[0][2]
    # Nombra la FRANJA («el mismo desayuno de ayer», «desayuné lo mismo de ayer»): la franja ENTERA de ese día, sumada
    # — elegir una de sus filas daba el yogurt suelto en vez del desayuno.
    franja = [f for f, _at in del_dia if _norm(f.get("meal_type")) in ref["tokens"]]
    if not franja:
        return None
    if len(franja) == 1:
        return franja[0]
    suma = {k: sum(float(f.get(k) or 0) for f in franja) for k in ("calories", "protein", "carbs", "healthy_fats")}
    return {**suma, "meal_name": " + ".join(str(f.get("meal_name") or "") for f in franja),
            "meal_type": franja[0].get("meal_type"),
            "ingredients": [i for f in franja for i in (f.get("ingredients") or [])]}


def _filas(user_id: str, ahora_utc: datetime) -> list:
    from db import get_consumed_meals_since
    return get_consumed_meals_since(user_id, (ahora_utc - timedelta(days=4)).date().isoformat(),
                                    include_ingredients=True) or []


def resolver(user_id, texto: str, tz_offset_min: Optional[int] = None, ahora_utc: Optional[datetime] = None,
             filas: Optional[list] = None) -> Optional[tuple]:
    """`(ref, fila)` o None. Nunca lanza."""
    try:
        if not user_id or user_id == "guest" or not activo():
            return None
        ref = detectar(texto)
        if not ref:
            return None
        ahora = ahora_utc or datetime.now(timezone.utc)
        fila = elegir(ref, filas if filas is not None else _filas(user_id, ahora), ahora, tz_offset_min)
        return (ref, fila) if fila else None
    except Exception as e:                                                     # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-988] referencia a un registro no resuelta: {e!r}")
        return None


def _cifras(fila: dict, n: float) -> dict:
    g = lambda k: float(fila.get(k) or 0)                                      # noqa: E731
    return {k: int(round(g(k) * n)) for k in ("calories", "protein", "carbs", "healthy_fats")}


def _fmt_n(n: float) -> str:
    return str(int(n)) if float(n).is_integer() else str(n).replace(".", ",")


def bloque_para_chat(user_id, texto: str, tz_offset_min: Optional[int] = None, **kw) -> str:
    r = resolver(user_id, texto, tz_offset_min, **kw)
    if not r:
        return ""
    ref, fila = r
    c1, cn = _cifras(fila, 1), _cifras(fila, ref["n"])
    ings = "; ".join(str(i) for i in (fila.get("ingredients") or [])[:6] if i)
    cuando = "ayer" if ref["dias"] == 1 else "antier"
    out = (f"\n\n🔁 SE REFIERE A UN REGISTRO SUYO [P1-PLAN-LOTE-988] (buscado por el sistema en su diario): "
           f"«{fila.get('meal_name')}» de {cuando}: {c1['calories']} kcal, {c1['protein']} g proteína, "
           f"{c1['carbs']} g carbohidratos, {c1['healthy_fats']} g grasa" + (f" ({ings})" if ings else "") + ". ")
    if ref["cambios"]:
        out += ("Parte de esa porción y aplica SOLO los cambios que dice; no preguntes el tamaño de lo que ya registró.\n")
    else:
        out += (f"Cada unidad que nombra es esa porción: {_fmt_n(ref['n'])} × = {cn['calories']} kcal, {cn['protein']} g "
                f"proteína, {cn['carbs']} g carbohidratos, {cn['healthy_fats']} g grasa. Regístralo YA con "
                "`log_consumed_meal` con esas cifras y ese nombre, SIN preguntar el tamaño, y di en una frase corta lo "
                "que asumiste para que te corrija si no era así.\n")
    return out


def rechazo(user_id, texto: str, calorias, tz_offset_min: Optional[int] = None, *, ahora: float = None, **kw):
    """Texto del rechazo o None. Una vez por usuario en 120 s; nunca con cambios declarados."""
    try:
        cal = float(calorias or 0)
    except (TypeError, ValueError):
        return None
    r = resolver(user_id, texto, tz_offset_min, **kw)
    if not r or r[0]["cambios"] or cal <= 0:
        return None
    ref, fila = r
    cn = _cifras(fila, ref["n"])
    if cn["calories"] <= 0 or abs(cal - cn["calories"]) <= _TOLERANCIA * cn["calories"]:
        return None
    t = time.time() if ahora is None else ahora
    for k, ts in list(_YA_AVISADO.items()):
        if t - ts > _VENTANA_S:
            _YA_AVISADO.pop(k, None)
    if str(user_id) in _YA_AVISADO:
        return None
    _YA_AVISADO[str(user_id)] = t
    return (f"⚠️ NO REGISTRADO TODAVÍA: el usuario se refiere a «{fila.get('meal_name')}», que ya está en su diario "
            f"({int(round(float(fila.get('calories') or 0)))} kcal la porción). {_fmt_n(ref['n'])} × esa porción = "
            f"{cn['calories']} kcal, {cn['protein']} g proteína, {cn['carbs']} g carbohidratos, {cn['healthy_fats']} g "
            f"grasa, y tú pusiste {int(round(cal))} kcal. Vuelve a llamar a `log_consumed_meal` con esas cifras. No le "
            "preguntes nada al usuario por esto ni le digas que hubo un rechazo.")
