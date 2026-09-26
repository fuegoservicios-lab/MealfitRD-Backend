# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-323 · 2026-09-25] El tiempo de un plato lo dicen sus PASOS, no solo el `prep_time` que declara.

El detector de tiempo del corrector (`graph_orchestrator._detect_prep_time_issues`, lote 220) leía solo el `prep_time`
DECLARADO, y la IA declara el tope aunque sus pasos lo doblen: batería de cierre del 25-sep, perfil del dueño («Nada» de
tiempo, 10 min por comida): «Arepitas salteadas al wok con filete de pescado» declara 10 min y sus pasos sellan el
pescado 4-5 min por lado, saltean 3 min y doran las arepitas 2 min por lado en el mismo wok (~20 min de fuego). Con un
estimador que tiene en cuenta el paralelismo, 20 de 48 comidas «Nada» pasaban de 12 min.

`minutos_de_fuego(pasos)` es CONSERVADOR a propósito (un falso positivo cuesta una corrección de la IA): cada cláusula
aporta su duración con el extremo BAJO de los rangos («4-5 min» = 4), «por lado» cuenta ×2, lo que va «mientras / a la
vez / aparte / en paralelo» corre en paralelo con lo anterior (cuenta el máximo), y los tiempos PASIVOS (reposar,
enfriar, marinar, remojar, refrigerar) no cuentan. Sin mise en place. Las notas (⚠/💡/🌱/⚕) no son pasos.
tooltip-anchor: P1-PLAN-LOTE-323
"""
from __future__ import annotations

import re
import unicodedata

_DUR = re.compile(r"(\d+(?:[.,]\d+)?)(?:\s*(?:-|–|a)\s*(\d+(?:[.,]\d+)?))?\s*(?:min\b|minutos?\b)(\s+(?:por|de cada|en cada)\s+(?:lado|cara))?")
_PARALELO = re.compile(r"\b(mientras|mientras tanto|aparte|a la vez|en paralelo|al mismo tiempo|simultaneamente|"
                       r"en otra (?:sarten|olla|hornilla|cazuela))\b")
_PASIVO = re.compile(r"\b(repos\w*|enfri\w*|marin\w*|remoj\w*|refriger\w*|nevera|congel\w*|deja(?:la|lo|las|los)?\s+"
                     r"(?:templar|entibiar|asentar))\b")
_CORTE = re.compile(r";|(?<!\d)\.|\.(?!\d)")   # el punto ENTRE cifras es un decimal («2.5 min»), no un corte (lote 52)
_NOTA = ("⚠", "💡", "🌱", "⚕", "🤰")


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def _num(txt: str) -> float:
    return float(str(txt).replace(",", "."))


def minutos_de_fuego(pasos) -> float:
    """Minutos de cocción ACTIVA que suman los pasos (0 si no hay pasos o ninguno trae duración). Nunca lanza."""
    try:
        total = 0.0
        bloque = 0.0
        for p in pasos or []:
            if not isinstance(p, str) or any(e in p for e in _NOTA):
                continue
            for cl in _CORTE.split(_sa(p)):
                ds = []
                for m in _DUR.finditer(cl):
                    if _PASIVO.search(cl[max(0, m.start() - 40):m.start()]):
                        continue
                    v = _num(m.group(1))
                    if m.group(3):
                        v *= 2
                    ds.append(v)
                if not ds:
                    continue
                d = sum(ds)
                if bloque and _PARALELO.search(cl):
                    bloque = max(bloque, d)
                else:
                    total += bloque
                    bloque = d
        return round(total + bloque, 1)
    except Exception:
        return 0.0


def declara(declarados, fuego) -> str:
    """Lo que el mensaje al corrector dice tras «declara»: «15 min» (manda lo declarado, texto de siempre) o «10 min pero
    sus pasos suman ~18 min de fuego» (mandan los pasos)."""
    try:
        d = int(declarados) if declarados is not None else None
        f = float(fuego or 0)
        if d is not None and d >= f:
            return f"{d} min"
        return f"{d if d is not None else '?'} min pero sus pasos suman ~{f:.0f} min de fuego"
    except Exception:
        return f"{declarados} min"


def minutos_del_plato(meal, declarados) -> float:
    """El mayor de lo declarado (`prep_time`) y lo que suman los pasos. `declarados` puede ser None."""
    try:
        pasos = meal.get("recipe") if isinstance(meal, dict) else None
        return max(float(declarados or 0), minutos_de_fuego(pasos))
    except Exception:
        return float(declarados or 0)
