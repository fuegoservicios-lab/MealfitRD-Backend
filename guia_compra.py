# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-283 · 2026-09-25] La guía de compra quincenal/mensual no contradice el tiempo de cocina ni el congelador.

`build_grocery_duration_context` (prompts/plan_generator.py) le decía al generador del día, con la compra de 30 días y a
TODOS: «granos secos, arroz, tubérculos (yuca, batata, plátano verde), proteínas congelables … Para cualquier perecedero,
incluye instrucciones de congelación en la receta». Quien marcó «sin congelador» leía en el mismo prompt, en el bloque
📐, lo contrario (fresco solo los primeros días); quien tiene «Nada» de tiempo (≤10 min) leía «nada de víveres hervidos»
y a la vez «tubérculos y granos secos». El modelo concilia las dos órdenes con «ya cocido»: en el perfil del dueño (30
días de una vez, sin congelador, «Nada», nunca cocina por tandas) casi cada plan de 3 días pedía «arroz blanco cocido y
refrigerado», «auyama ya cocida», «pescado ya cocido», «garbanzos secos» — alguien tiene que cocinarlos otro día, y el
formulario dice que nadie lo hace.

Aquí las mismas líneas según lo contestado. Sin «Nada» y con congelador (o sin la respuesta), el texto de siempre, byte a
byte. tooltip-anchor: P1-PLAN-LOTE-283-GUIA-DE-COMPRA"""
from __future__ import annotations

_MENSUAL = ("Usa predominantemente: granos secos, arroz, avena, tubérculos (yuca, batata, plátano verde),\n"
            "proteínas congelables (pollo, carne, pescado empacado al vacío), leche en polvo o UHT, huevos.\n"
            "Para cualquier perecedero, incluye instrucciones de congelación en la receta.\n")
_QUINCENAL = ("Equilibra entre frescos e ingredientes duraderos. Los vegetales de hoja y frutas muy maduras\n"
              "deben usarse en los primeros días del plan. Planifica congelación para proteínas frescas.\n")
_LISTO = ("legumbres y maíz EN LATA (escurridos; nunca legumbres secas), atún y sardinas en lata, avena, casabe, harina "
          "de maíz precocida, huevos, leche en polvo o UHT; batata, auyama y plátano solo cocidos en el microondas "
          "dentro de la receta. Nada de arroz ni de nada «ya cocido» que haya que cocinar otro día.\n")


def _dias_frescos() -> int:
    try:
        return int(__import__("pantry_durability")._dias_libres_sin_congelador())
    except Exception:
        return 3


def _sin_congelador_linea() -> str:
    return (f"SIN CONGELADOR: no escribas «congela» ni «descongela»; pollo, carne y pescado frescos solo en los primeros "
            f"{_dias_frescos()} días (en la nevera no aguantan más); después huevo, lata, queso o legumbres.\n")


def lineas_duracion(form_data, duracion) -> str:
    """Las líneas de qué comprar para `duracion` ∈ monthly|biweekly según `cookingTime` y `freezerMode`."""
    try:
        fd = form_data if isinstance(form_data, dict) else {}
        sin_tiempo = str(fd.get("cookingTime") or "").strip().lower() == "none"
        sin_congelador = str(fd.get("freezerMode") or "").strip().lower() == "none"
        if duracion == "monthly":
            if not sin_tiempo and not sin_congelador:
                return _MENSUAL
            if sin_tiempo:
                base = "Usa predominantemente lo duradero que se sirve LISTO o en minutos: " + _LISTO
            else:
                base = ("Usa predominantemente: granos secos, arroz, avena, tubérculos (yuca, batata, plátano verde),\n"
                        "leche en polvo o UHT, huevos, enlatados (atún, sardinas, habichuelas).\n")
            return base + (_sin_congelador_linea() if sin_congelador
                           else "Para cualquier perecedero, incluye instrucciones de congelación en la receta.\n")
        if duracion == "biweekly":
            if not sin_tiempo and not sin_congelador:
                return _QUINCENAL
            txt = ("Equilibra entre frescos e ingredientes duraderos. Los vegetales de hoja y frutas muy maduras\n"
                   "deben usarse en los primeros días del plan. ")
            txt += _sin_congelador_linea() if sin_congelador else "Planifica congelación para proteínas frescas.\n"
            if sin_tiempo:
                txt += "Con «Nada» de tiempo, lo duradero va listo: " + _LISTO
            return txt
    except Exception:
        pass
    return {"monthly": _MENSUAL, "biweekly": _QUINCENAL}.get(duracion, "")
