# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-287 · 2026-09-25] Lo que el usuario LEE tras un cambio por presupuesto.

El plan del dueño («Económico», batería del 25-sep) mostraba tres restos del ajuste de presupuesto:
  · el aviso «Para cuidar tu bolsillo ajustamos: …» repetía la misma sustitución (el yogur griego estaba en tres
    comidas: «yogurt griego → Yogurt natural» ×3 entre los 6 que caben) — las dos listas del aviso concatenaban la nota
    de cada comida;
  · el título quedaba con el nombre del catálogo en MAYÚSCULA a media frase: «Arroz frío con Filete de pescado blanco
    estilo ceviche»;
  · el corte del premium sobrevivía al cambio: «almendras fileteadas» → «maní fileteado», que no se vende.
Los dos pases que abaratan (el estático del formulario y el que ataca los ítems caros de la lista) reescriben con
`sustituir`; el aviso sale de `sustituciones_unicas`. tooltip-anchor: P1-PLAN-LOTE-287-PRESUPUESTO-TEXTO"""
from __future__ import annotations

import re

#: cortes de un fruto seco laminable que el maní no tiene: «maní fileteado» → «maní picado»
_CORTE_MANI_RX = re.compile(r"(?i)\b(man[ií])\s+(fileteado|fileteada|fileteados|fileteadas|laminado|laminada|laminados|"
                            r"laminadas|en l[aá]minas|en lascas)\b")


def sustituir(texto, rx, candidato, *, titulo: bool = False) -> str:
    """El primer `rx` de `texto` por `candidato`, sin el corte que era del premium y con el colapso de unidad del pase
    (`_dedup_unit_noun_collision`: «½ filete de Filete de…»). En un TÍTULO copia la mayúscula de la palabra que
    reemplaza: a media frase en minúscula, «filete de pescado blanco» (el catálogo escribe «Filete…»); en un título en
    mayúsculas («Yuca con Habas Guisadas») se queda «Habichuelas rojas». Las líneas conservan el nombre del catálogo (el
    pulido final del lote 181 las pasa a minúscula)."""
    t = str(texto or "")
    try:
        m = re.search(rf"\b(?:{rx})\b", t, re.IGNORECASE)
        if not m:
            return t
        cand = str(candidato or "")
        if titulo and m.start() > 0 and t[m.start()].islower() and cand[:1].isupper() and not cand[1:2].isupper():
            cand = cand[:1].lower() + cand[1:]
        nuevo = t[:m.start()] + cand + t[m.end():]
        nuevo = _CORTE_MANI_RX.sub(lambda mm: f"{mm.group(1)} picado", nuevo)
        try:
            import graph_orchestrator as go
            nuevo = go._dedup_unit_noun_collision(nuevo)
        except Exception:
            pass
        return nuevo
    except Exception:
        return re.sub(rf"\b(?:{rx})\b", str(candidato or ""), t, count=1, flags=re.IGNORECASE)


def sustituciones_unicas(days, limite: int = 6) -> list:
    """Las notas `_budget_substitutions` de todas las comidas, sin repetir (sin distinguir mayúsculas) y en orden."""
    out, vistos = [], set()
    try:
        for d in days or []:
            for m in ((d.get("meals") or []) if isinstance(d, dict) else []):
                if not isinstance(m, dict):
                    continue
                for s in (m.get("_budget_substitutions") or []):
                    k = str(s).strip().lower()
                    if k and k not in vistos:
                        vistos.add(k)
                        out.append(s)
    except Exception:
        pass
    return out[:limite]
