# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1047] «¿Recuerdas el último yogur que registré?» — buscar un alimento en el diario por nombre.

Caso real (16-nov): sin esta búsqueda el coach consultó día por día (9, 10, 11… 14-nov), agotó las rondas
del turno (`max_rondas`) y antes contestó «no encuentro ningún yogur» mirando solo hoy y ayer.
Una sola consulta SQL sobre lo ya guardado: sin IA.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timedelta, timezone

DIAS = 60
MAX = 5


def _norm(t) -> str:
    t = unicodedata.normalize("NFD", str(t or "").lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


_VACIAS = {"de", "del", "la", "el", "los", "las", "un", "una", "con", "y", "mi", "que", "en", "al"}


def _raices(consulta: str) -> list[str]:
    """«yogures» y «yogurt» → «yogur»; «atún» → «atun». La raíz quita la última letra de las palabras ≥ 5."""
    out = []
    for p in re.findall(r"\w+", _norm(consulta)):
        if p in _VACIAS or len(p) < 3:
            continue
        out.append(p[:-1] if len(p) >= 5 else p)
    return out


def coincide(fila: dict, raices: list[str]) -> bool:
    ings = fila.get("ingredients")
    texto = _norm(" ".join([str(fila.get("meal_name") or "")]
                           + ([str(x) for x in ings] if isinstance(ings, list) else [])))
    return bool(raices) and all(r in texto for r in raices)


def _fecha_local(consumed_at, tz_offset_min: int):
    if isinstance(consumed_at, str):
        try:
            consumed_at = datetime.fromisoformat(consumed_at.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(consumed_at, datetime):
        return None
    if consumed_at.tzinfo is None:
        consumed_at = consumed_at.replace(tzinfo=timezone.utc)
    return (consumed_at - timedelta(minutes=tz_offset_min)).date()


def buscar(user_id: str, consulta: str, tz_offset_min: int = 0, hoy=None, filas=None) -> str:
    """Texto para el modelo con las últimas coincidencias (la más reciente primero). `filas` es para tests."""
    raices = _raices(consulta)
    if not raices:
        return "Dime qué alimento busco en el diario (por ejemplo «yogur» o «atún Genova»)."
    hoy = hoy or (datetime.now(timezone.utc) - timedelta(minutes=tz_offset_min)).date()
    if filas is None:
        from db_facts import get_consumed_meals_since
        filas = get_consumed_meals_since(user_id, (hoy - timedelta(days=DIAS + 1)).isoformat(),
                                         include_ingredients=True) or []
    hits = [f for f in filas if isinstance(f, dict) and coincide(f, raices)]
    hits.sort(key=lambda f: str(f.get("consumed_at") or ""), reverse=True)
    if not hits:
        return (f"DIARIO (últimos {DIAS} días): ningún registro con «{consulta}». Dile la verdad: no lo encuentras "
                f"en lo que ha registrado; no busques día por día.")
    out = [f"DIARIO — últimos registros con «{consulta}» (más reciente primero, de {len(hits)} en {DIAS} días):"]
    for f in hits[:MAX]:
        d = _fecha_local(f.get("consumed_at"), tz_offset_min)
        cuando = "hoy" if d == hoy else ("ayer" if d and d == hoy - timedelta(days=1) else (d.isoformat() if d else "?"))
        out.append(f"▸ {cuando} · {f.get('meal_type') or 'comida'}: {f.get('meal_name') or '(sin nombre)'} · "
                   f"{int(round(float(f.get('calories') or 0)))} kcal · P:{int(round(float(f.get('protein') or 0)))}g "
                   f"C:{int(round(float(f.get('carbs') or 0)))}g G:{int(round(float(f.get('healthy_fats') or 0)))}g")
    return "\n".join(out)
