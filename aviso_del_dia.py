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


# [P1-PLAN-LOTE-1070 · 2026-11-17] El dueño, sin haber anotado nada en el día, recibió a las 19:00 «Una arepita mediana
# con media taza de habichuelas y aguacate… para cenar y sumar proteína» (~450 kcal, ~13 g de proteína). Sin
# registros el bloque solo decía «Todavía no registró nada hoy»: ni su meta ni que una comida principal no puede ser
# ligera si el día va en blanco. Ahora lleva la meta y, en almuerzo y cena, esa pauta.
_PRINCIPALES_SIN_NADA = ("almuerzo", "cena")


def _regla_de_dia_en_blanco(comida, metas) -> list[str]:
    c = str(comida or "").strip().lower()
    if c not in _PRINCIPALES_SIN_NADA:
        return []
    prot = f"{max(30, min(45, round(metas[1] * 0.3)))} g" if metas and metas[1] else "30-40 g"
    return [
        "- DÍA EN BLANCO: puede que no haya comido nada hoy (o que no lo anotara). NO propongas algo ligero ni una "
        f"porción pequeña: propón un plato COMPLETO con ~{prot} de proteína (pescado, res, pollo, huevos, atún, pavo, "
        "yogur…) y su carbohidrato. Sin querer recuperar el día entero en una comida, sin regañar y sin preguntar si "
        "se le olvidó anotar.",
    ]


_REGLA_PROTEINA = ("- Di que algo «suma proteína» o «aporta proteína» SOLO si el plato lleva una fuente clara (pescado, "
                   "res, pollo, huevos, atún, pavo, queso, yogur, whey); arepa, habichuelas o aguacate solas no lo son.")


def bloque_del_dia(consumed, health, comida=None) -> str:
    """El bloque «lo que lleva hoy» para el prompt del aviso. Nunca lanza: sin datos, lo dice.

    `comida` (opcional): la comida del aviso («Almuerzo», «Cena»…), para la pauta del día en blanco."""
    comidas = [m for m in (consumed or []) if isinstance(m, dict)]
    lineas = ["", "Lo que lleva hoy (datos reales; úsalos solo si ayudan a lo que le dices, no los recites todos):"]
    if not comidas:
        lineas.append("- Todavía no registró nada hoy.")
        metas = _metas(health)
        if metas:
            lineas.append(f"- Su meta de hoy: {metas[0]} kcal" + (f" y {metas[1]} g de proteína." if metas[1] else "."))
        lineas += _regla_de_dia_en_blanco(comida, metas)
    else:
        partes = []
        alimentos_detallados = []
        for m in comidas[:6]:
            tipo = str(m.get("meal_type") or m.get("meal_name") or "comida").strip().lower()
            kcal_m = int(round(_n(m.get('calories'))))
            partes.append(f"{tipo} ({kcal_m} kcal)")
            nombre = str(m.get("meal_name") or "").strip()
            if nombre and nombre.lower() != tipo:
                alimentos_detallados.append(f"{tipo}: {nombre}")
            elif nombre:
                alimentos_detallados.append(nombre)
        if alimentos_detallados:
            lineas.append(
                "- Consumido hoy (ofrece variedad culinaria, NO repitas estos alimentos ni sus fuentes proteicas): "
                + "; ".join(alimentos_detallados)
                + "."
            )
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
            # [P1-PLAN-LOTE-1079] la cena con mucho margen: el plato del tamaño de lo que queda (no ~500 kcal)
            _cena = __import__("cena_a_la_medida").regla(comida, quedan, (health or {}).get("mainGoal") if isinstance(health, dict) else None)
            if _cena:
                lineas.append(_cena)
            if meta_prot and meta_kcal > 0:
                falta = meta_prot - prot
                lineas.append(
                    f"- Proteína: {prot} de {meta_prot} g "
                    + (f"(le faltan {falta} g)." if falta > 0 else "(ya la cubrió).")
                )
                pct_k = kcal / meta_kcal
                pct_p = prot / meta_prot
                if (pct_k >= 0.92 or quedan <= 120) and pct_p <= 0.65:
                    lineas.append(
                        "- REGLA DE PRESUPUESTO [BB-029 / MICRO-RESCATE PROTEICO]: Calorías casi al límite (>= 92%) con proteína muy deficitaria (<= 65%). "
                        "PROHIBIDO sugerir cenas pesadas o carbohidratos. Sugiere exclusivamente un micro-rescate ultra-aislado de ~100-120 kcal "
                        "(25 g de proteína limpia en claras o batido whey en agua) para proteger el músculo sin romper la barra verde (BB-024)."
                    )
                elif quedan <= 100 or pct_k >= 0.95:
                    lineas.append(
                        "- REGLA DE PRESUPUESTO: Margen calórico prácticamente agotado (>= 95%). "
                        "PROHIBIDO sugerir platos calóricos; enfócate en cierre de jornada, hidratación sin calorías o descanso."
                    )
                elif quedan < 350 and falta >= 25 and quedan > 0:
                    lineas.append(
                        "- REGLA DE PRESUPUESTO: Margen calórico estrecho con proteína pendiente alta. "
                        "Sugiere únicamente proteína magra pura (claras, atún al agua, batido whey, pechuga limpia)."
                    )
            elif quedan <= 100 or (meta_kcal > 0 and kcal / meta_kcal >= 0.95):
                lineas.append(
                    "- REGLA DE PRESUPUESTO: Margen calórico prácticamente agotado (>= 95%). "
                    "PROHIBIDO sugerir platos calóricos; enfócate en cierre de jornada, hidratación sin calorías o descanso."
                )
    lineas.append(_REGLA_PROTEINA)
    return "\n".join(lineas) + "\n"
