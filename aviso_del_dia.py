# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-413 · 2026-09-27] Lo que la persona lleva HOY, para que el aviso de comida hable de su día.

Los avisos que escribe la IA sonaban a plantilla («Es el momento perfecto para desayunar: arma un plato balanceado…
para apoyar tu objetivo de ganar músculo») porque el prompt no sabía nada del día. Este bloque se AÑADE al prompt
del aviso (no es un hueco de la plantilla: los llamadores que la formatean con la lista fija de siempre siguen
valiendo): qué registró hoy y, cuando el perfil alcanza para calcular la meta —la misma función pura que
`/api/nutrition/targets`—, calorías y proteína contra ella. Sin perfil suficiente, sin metas: no se inventan.
"""
from __future__ import annotations

# Los campos sin los que /api/nutrition/targets tampoco da metas (mismo contrato, fail-closed)
_REQUERIDOS = ("gender", "age", "height", "weight", "weightUnit", "activityLevel", "mainGoal", "medicalConditions")


def _n(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    return x if x == x else 0.0


def _metas(health) -> tuple[int, int] | None:
    hp = health if isinstance(health, dict) else {}
    if any(not hp.get(c) for c in _REQUERIDOS):
        return None
    try:
        from nutrition_calculator import get_nutrition_targets
        t = get_nutrition_targets(hp) or {}
        kcal = int(_n(t.get("target_calories")))
        prot = int(_n((t.get("macros") or {}).get("protein_g")))
        return (kcal, prot) if kcal > 0 else None
    except Exception:
        return None


def bloque_del_dia(consumed, health) -> str:
    """El bloque «lo que lleva hoy» para el prompt del aviso. Nunca lanza: sin datos, lo dice."""
    comidas = [m for m in (consumed or []) if isinstance(m, dict)]
    lineas = ["", "Lo que lleva hoy (datos reales; úsalos solo si ayudan a lo que le dices, no los recites todos):"]
    if not comidas:
        lineas.append("- Todavía no registró nada hoy.")
    else:
        partes = []
        for m in comidas[:6]:
            tipo = str(m.get("meal_type") or m.get("meal_name") or "comida").strip().lower()
            partes.append(f"{tipo} ({int(round(_n(m.get('calories'))))} kcal)")
        lineas.append("- Registró: " + ", ".join(partes) + ".")
        kcal = int(round(sum(_n(m.get("calories")) for m in comidas)))
        prot = int(round(sum(_n(m.get("protein")) for m in comidas)))
        metas = _metas(health)
        if metas:
            meta_kcal, meta_prot = metas
            quedan = meta_kcal - kcal
            lineas.append(
                f"- Calorías: {kcal} de {meta_kcal} "
                + (f"(le quedan {quedan})." if quedan >= 0 else f"(ya pasó su meta por {-quedan}).")
            )
            if meta_prot:
                falta = meta_prot - prot
                lineas.append(
                    f"- Proteína: {prot} de {meta_prot} g "
                    + (f"(le faltan {falta} g)." if falta > 0 else "(ya la cubrió).")
                )
    return "\n".join(lineas) + "\n"
