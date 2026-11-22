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


_PATRON_BLOQUE_SUPLEMENTOS = re.compile(
    r"[ \t]*Para vitaminas/minerales \(magnesio, multivitaminas, B12, etc\.\), llama analizar_suplemento"
    r".*?datos no confiables\.\n?",
    re.DOTALL,
)

# [P1-PLAN-LOTE-1005 · G-03] Copias 2x/3x o repeticiones literales del docstring de herramientas
# SIEMPRE ligadas. La copia principal / autoritativa permanece íntegra en `texto` o en el schema.
_PATRONES_DUPLICADOS_UNIVERSALES = (
    # BB-034 copia 2 en A. (copia completa permanece en Regla 8)
    re.compile(r"PLANTILLAS COMPARATIVAS DE AYER EN HOY:.*?no suma a las macros de hoy»\.\s*", re.DOTALL),
    # BB-028 copia 2 en A. (copia completa permanece en `correct_consumed_meal`)
    re.compile(r"Al corregir un ingrediente o cantidad de un plato registrado.*?macros coherentes\.\s*", re.DOTALL),
    # BB-066 copia 1 en R2. (protocolo 3 puntos permanece en `_REGLA_UNA_LLAMADA_POR_COMIDA`)
    re.compile(r"(?m)^R2\.\s*(?:\[[^\]]+\]\s*)?CERO DESPEDIDAS NOCTURNAS REPETIDAS:[^\n]*\n?"),
    # Suplementos copia 1 en `_REGLA_UNA_LLAMADA_POR_COMIDA` (permanece en `correct_consumed_meal`)
    re.compile(r"EXCEPCIÓN CRÍTICA CON SUPLEMENTOS:.*?para meter una comida\.\s*", re.DOTALL),
    # BB-067 copia 2 (permanece en Q4)
    re.compile(r"RESPUESTA DIRECTA ANTE NOMBRE DE FRANJA AISLADO:.*?te lo anoto\.'\)\.\s*", re.DOTALL),
    # BB-058 copia 2 (permanece en Q2 y `proponer_comida`)
    re.compile(r"DESAMBIGUACIÓN DE ALIMENTO AISLADO SIN VERBO:.*?diario\.'\)\.\s*", re.DOTALL),
    # BB-059 copia 2 (permanece en Q3)
    re.compile(r"FIDELIDAD ESTRICTA DEL INGREDIENTE EN PROPUESTAS:.*?al ingrediente del usuario\.\s*", re.DOTALL),
    # BB-058/BB-059 cola repetida en el bullet de `proponer_comida` (ya en Q2/Q3)
    re.compile(r"PROHIBIDO usarla ante alimentos o productos aislados.*?si tiene atún\)\.", re.DOTALL),
    # Copias 2x en bullet `log_consumed_meal` ya presentes en Reglas 3, 6-bis, 7, A y Q
    re.compile(r"Tampoco le preguntes la FRANJA si no la dice:.*?(?:solo si suma\.|usa `correct_consumed_meal`\.)\s*", re.DOTALL),
    re.compile(r"AL CONFIRMAR(?:\s+EL REGISTRO DE UNA COMIDA)?:.*?(?:si el estimado no cuadra\.|en cada comida\)\.)\s*", re.DOTALL),
    re.compile(r"NUNCA digas 'lo registro'.*?(?:explícitamente|guardado)\.\s*", re.DOTALL),
    re.compile(r"BEBIDA LIGERA NO CIERRA LA FRANJA:.*?desayunado\?»\)\.\s*", re.DOTALL),
)

_PATRONES_POR_HERRAMIENTA = (
    (
        "log_water_glass",
        re.compile(
            r"2\. EQUIVALENCIAS CLÍNICAS Y COTIDIANAS \(1 vaso = 250 ml / 8 oz\):.*?count_delta=0\.5\)\.\s*",
            re.DOTALL,
        ),
    ),
    (
        "log_consumed_meal",
        re.compile(r"Pasa SIEMPRE `meal_type`.*?pregunta ANTES de llamar la herramienta\.\s*", re.DOTALL),
    ),
    (
        "correct_consumed_meal",
        re.compile(r"Pásale el `meal_id` EXACTO del ID_REGISTRO_DIARIO.*?no llamaste la herramienta en ese turno\.\s*", re.DOTALL),
    ),
    (
        "proponer_comida",
        re.compile(r"devuelve hasta 3 platos del catálogo.*?(?:dio esta herramienta\.|cuánto tiempo tiene\.)\s*", re.DOTALL),
    ),
    (
        "proponer_comida",
        re.compile(r"Cuando presentes la propuesta, PRIORIZA.*?sugerencia es para mañana\.\s*", re.DOTALL),
    ),
    (
        "proponer_comida",
        re.compile(r"Al recomendar comidas para hoy o para mañana, apóyate primero en.*?«te quedarían ~X kcal»\)\.\s*", re.DOTALL),
    ),
)


_PATRON_REGLA_RECUPERAR = re.compile(r"(?m)^S2\. RECUPERAR LO BORRADO[^\n]*\n?")


def dedup_activo() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_CHAT_PROMPT_DEDUP_MANUAL", True)
    except Exception:  # noqa: BLE001
        return True


def podar_extras_no_ligados(texto: str, ligadas: Optional[Iterable[str]]) -> str:
    """[P1-PLAN-LOTE-1005 · G-03] Poda bullets e instrucciones extensas de herramientas
    no ligadas en el turno (suplementos, ajustes/pantallas de la app, plan-day, propuestas)."""
    if ligadas is None or not texto:
        return texto
    ligadas_set = set(ligadas)
    if "cambiar_ajuste_de_la_app" not in ligadas_set and "abrir_pantalla_de_la_app" not in ligadas_set:
        texto = _linea_de_bullet("cambiar_ajuste_de_la_app").sub("", texto, count=1)
    if "consultar_dia_del_plan" not in ligadas_set:
        texto = _linea_de_bullet("consultar_dia_del_plan").sub("", texto, count=1)
    if "proponer_comida" not in ligadas_set:
        texto = _linea_de_bullet("proponer_comida").sub("", texto, count=1)
    if not (ligadas_set & {"analizar_suplemento", "guardar_suplemento", "buscar_etiqueta_en_internet"}):
        texto = _PATRON_BLOQUE_SUPLEMENTOS.sub("", texto, count=1)
    if "recuperar_de_la_nevera" not in ligadas_set:   # [P1-PLAN-LOTE-1160] la regla viaja con su herramienta
        texto = _PATRON_REGLA_RECUPERAR.sub("", texto, count=1)
    return texto


def deduplicar_manual(texto: str, ligadas: Optional[Iterable[str]]) -> str:
    """[P1-PLAN-LOTE-1005 · G-03] Elimina copias duplicadas 2x/3x cuyo texto autoritativo
    permanece una vez en `texto` o en el esquema de una herramienta ligada."""
    if ligadas is None or not texto or not dedup_activo():
        return texto
    ligadas_set = set(ligadas)
    for pat in _PATRONES_DUPLICADOS_UNIVERSALES:
        texto = pat.sub("", texto, count=1)
    for tool_name, pat in _PATRONES_POR_HERRAMIENTA:
        if tool_name in ligadas_set:
            texto = pat.sub("", texto, count=1)
    return texto


def adelgazar(texto: str, ligadas: Optional[Iterable[str]] = None) -> str:
    """Aplica los recortes sin pérdida si el knob está activo."""
    if not lean_activo() or not texto:
        return texto
    out = podar_herramientas_no_ligadas(quitar_etiquetas(texto), ligadas)
    if ligadas is not None:
        out = deduplicar_manual(podar_extras_no_ligados(out, ligadas), ligadas)
    return out
