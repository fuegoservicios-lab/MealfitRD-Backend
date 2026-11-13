# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-998] Recorte SIN PÉRDIDA del manual estático del coach.

Dos recortes deterministas que no tocan ninguna regla:
  1. Etiquetas de procedencia («[BB-031 / P1-CORRECT-MEAL-TARGET-DISAMBIGUATION · 2026-11-04]», «[P1-PLAN-LOTE-168]»):
     48 en el manual (1,9 k de 63 k). Sirven para nosotros (grep, tests); el modelo no las necesita.
  2. El bullet «- Usa `<herramienta>` …» de las herramientas que ESTE turno no tiene ligadas (el gating por turno ya las
     quita del esquema): compras, Nevera, nutrientes y perfil clínico. Si el turno las necesita, el gating las liga
     Y el bullet se queda (misma función `_select_tools_for_turn` decide ambas cosas).

Knob `MEALFIT_CHAT_PROMPT_LEAN` (True). Con foto, modo voz o sin texto, no se poda nada (igual que el gating).
tooltip-anchor: P1-PLAN-LOTE-998-LEAN
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

#: Bullets de herramienta que se pueden omitir cuando la herramienta no está ligada en el turno.
HERRAMIENTAS_PODABLES = (
    "check_shopping_list",
    "modify_pantry_inventory",
    "suggest_foods_for_nutrient",
    "check_clinical_profile",
)

_ETIQUETA = re.compile(
    r"[ \t]*[\[(]\s*(?:(?:P\d|BB|AI)-[A-Z0-9\-]+|Q)"
    r"(?:\s*[/·,+]\s*[A-Z0-9\-]+)*(?:\s*·\s*\d{4}-\d{2}-\d{2})?(?:\s*[/·,+]\s*[A-Z0-9\-]+)*\s*[\])]"
)


def lean_activo() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_CHAT_PROMPT_LEAN", True)
    except Exception:  # noqa: BLE001
        return True


def quitar_etiquetas(texto: str) -> str:
    return _ETIQUETA.sub("", texto or "")


def _linea_de_bullet(nombre: str) -> "re.Pattern[str]":
    return re.compile(r"(?m)^- Usa `%s`[^\n]*\n?" % re.escape(nombre))


def podar_herramientas_no_ligadas(texto: str, ligadas: Optional[Iterable[str]]) -> str:
    """Quita el bullet de cada herramienta podable que NO está en `ligadas`. `ligadas=None` ⇒ no poda nada."""
    if ligadas is None:
        return texto
    ligadas = set(ligadas)
    for nombre in HERRAMIENTAS_PODABLES:
        if nombre not in ligadas:
            texto = _linea_de_bullet(nombre).sub("", texto, count=1)
    return texto


def adelgazar(texto: str, ligadas: Optional[Iterable[str]] = None) -> str:
    """Aplica los dos recortes si el knob está activo."""
    if not lean_activo() or not texto:
        return texto
    return podar_herramientas_no_ligadas(quitar_etiquetas(texto), ligadas)
