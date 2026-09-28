# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-565 · 2026-09-27] El paso no pide yema líquida cuando la nota del plato exige yema firme.

Batería real (perfil tipo dueño, día 3): «plancha 2 huevos 2-3 min hasta que la clara cuaje (yema líquida)» y, en el
mismo plato, «⚠️ Seguridad alimentaria: cocina el huevo por completo (≥71°C, yema y clara firmes…)». En el corpus
(4.710 comidas) 13 pasos piden yema líquida o huevo poché; en 6 la nota del plato dice lo contrario («cocina 3 minutos
para una yema líquida» en una avena con huevo poché). Una receta que se contradice obliga al usuario a elegir; la nota es
la política (embarazo y lactancia incluidos), así que el paso se alinea: «(yema líquida)» → «(yema firme)», «hasta que
la clara cuaje» → «hasta que la clara y la yema cuajen» y «cocina 3 minutos para una yema líquida» → «cocina 4-5
minutos, hasta que la yema esté firme». Sin esa nota en el plato no se toca nada. tooltip-anchor: P1-PLAN-LOTE-565
"""
from __future__ import annotations

import re

_NOTA_RE = re.compile(r"yema\s+y\s+clara\s+firmes|yema\s+y\s+clara\s+firme|cocina\s+el\s+huevo\s+por\s+completo", re.IGNORECASE)
_NOTAS = ("⚠", "🤰", "⚕", "💡", "🌱")
# [P1-PLAN-LOTE-631 · 2026-09-28] La yema «cremosa», «blanda» o «suave» es la misma yema líquida con otra palabra. Validación
# del 592 (celíaco, día 2): «cuaja el huevo entero con 5 claras hasta que la clara esté firme y la yema siga cremosa» con
# «⚠️ … cocina el huevo por completo (≥71°C, yema y clara firmes…)» en el mismo plato; corpus: 10 pasos así (3 en las
# corridas recientes), la forma que el 565 no veía. Con la nota, el paso pide la yema firme. tooltip-anchor: P1-PLAN-LOTE-631
# [P1-PLAN-LOTE-730 · 2026-09-28] + «al punto», «a tu/su gusto», «como te guste»: batería real sobre el 659 (adulto mayor
# con HTA): «cocínalos 3-4 minutos hasta que la clara cuaje y la yema quede al punto» y «…la yema quede a tu gusto», los
# dos con «⚠️ … yema y clara firmes» en el plato. Dejar la yema al gusto de quien cocina es permitirla líquida. En
# plural también («hasta que las claras estén cuajadas y las yemas a tu punto»: 3 platos con la nota en el corpus).
# tooltip-anchor: P1-PLAN-LOTE-730
_BLANDA_631 = (r"(?:cremosa|blanda|suave|tierna|jugosa|semil[ií]quida|melosa|fluida|a\s+punto|al\s+punto(?:\s+deseado)?|"
               r"a\s+(?:(?:tu|su)\s+)?(?:gusto|punto)|como\s+(?:te|le)\s+guste)")
_BLANDAS_730 = (r"(?:cremosas|blandas|suaves|tiernas|jugosas|semil[ií]quidas|melosas|fluidas|a\s+punto|"
                r"al\s+punto(?:\s+deseado)?|a\s+(?:(?:tu|su)\s+)?(?:gusto|punto)|como\s+(?:te|le)\s+gusten?)")
_YEMAS_730 = re.compile(r"\b(?P<las>las\s+yemas\s+)(?:(?P<v>queden|sigan|est[eé]n)\s+)?(?:ligeramente\s+|un\s+poco\s+)?"
                        + _BLANDAS_730 + r"\b", re.IGNORECASE)


def _firmes_730(m) -> str:
    v = (m.group("v") or "").lower()
    return m.group("las") + ("queden firmes" if v == "queden" else "estén firmes" if v else "firmes")
_CLARA_Y_YEMA_631 = re.compile(
    r"hasta\s+que\s+la\s+clara\s+(?P<v>cuaje|est[eé]\s+(?:firme|cuajada))(?:\s+por\s+completo)?\s+y\s+la\s+yema\s+"
    r"(?:(?:quede|siga|est[eé]|a[uú]n|todav[ií]a)\s+)?(?:ligeramente\s+|un\s+poco\s+)?" + _BLANDA_631 + r"\b", re.IGNORECASE)
_YEMA_631 = re.compile(r"\b(?P<la>la\s+yema\s+)(?P<v>quede|siga|est[eé]|a[uú]n|todav[ií]a)\s+(?:ligeramente\s+|un\s+poco\s+)?"
                       + _BLANDA_631 + r"\b", re.IGNORECASE)


def _cambia(p: str) -> str:
    q = re.sub(r"(\b\d+)\s*(?:[-–]\s*\d+\s*)?min(?:utos)?\s+para\s+una\s+yema\s+l[ií]quida",
               "4-5 minutos, hasta que la yema esté firme", p, flags=re.IGNORECASE)
    q = re.sub(r"para\s+una\s+yema\s+l[ií]quida", "hasta que la yema esté firme", q, flags=re.IGNORECASE)
    q = re.sub(r"hasta\s+que\s+la\s+clara\s+cuaje(\s*\(yema\s+l[ií]quida\))",
               "hasta que la clara y la yema cuajen", q, flags=re.IGNORECASE)
    q = re.sub(r"\(yema\s+l[ií]quida\)", "(yema firme)", q, flags=re.IGNORECASE)
    q = re.sub(r"\byema\s+l[ií]quida\b(?!\s*,\s*pero)", "yema firme", q, flags=re.IGNORECASE)
    q = _CLARA_Y_YEMA_631.sub(lambda m: "hasta que la clara y la yema " + ("cuajen" if m.group("v").lower() == "cuaje"
                                                                          else "estén firmes"), q)   # [P1-PLAN-LOTE-631]
    q = _YEMA_631.sub(lambda m: m.group("la") + {"quede": "quede firme", "siga": "esté firme"}.get(
        m.group("v").lower(), "esté firme" if m.group("v").lower().startswith("est") else "firme"), q)
    q = _YEMAS_730.sub(_firmes_730, q)                     # [P1-PLAN-LOTE-730] también en plural
    return q


def ajustar(meal) -> int:
    """Nº de pasos alineados; 0 ante cualquier error o sin la nota de yema firme."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not any(isinstance(p, str) and any(e in p for e in _NOTAS) and _NOTA_RE.search(p)
                                                for p in rec):
            return 0
        n = 0
        for i, p in enumerate(rec):
            if not isinstance(p, str) or any(e in p for e in _NOTAS):
                continue
            if re.search(r"sin\s+yema\s+l[ií]quida", p, re.IGNORECASE):
                continue                                   # «revuelve… sin yema líquida» ya dice lo correcto
            q = _cambia(p)
            if q != p:
                rec[i] = q
                n += 1
        if n:
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


__all__ = ["ajustar"]
