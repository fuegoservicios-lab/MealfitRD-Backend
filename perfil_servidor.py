# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-717 · 2026-09-28] Quién es DUEÑO de cada clave de `user_profiles.health_profile` — SSOT.

`health_profile` es un JSONB con muchos escritores, y el frontend es el más peligroso: hidrata en su formulario TODA
clave del perfil que encuentre vacía (`AssessmentContext`, «solo se hidrata lo vacío») y la devuelve ENTERA por dos
puertas — `PATCH /api/profile` (el Dashboard manda el formulario completo en cada carga, `safeUpdateHealthProfile`) y la
reescritura que hace la generación al terminar (`_postprocess_pipeline_result` vuelca el formulario de `/analyze` al
perfil). Esa copia se hidrató UNA vez y no se refresca: devuelta al servidor, pisaba con valores viejos lo que su dueño
había cambiado después — el historial de peso, lo aprendido por el motor, el perfil clínico del panel (el TFG que pone
el tope renal), los básicos. Un merge `||` superficial no distingue «el usuario lo cambió» de «el formulario lo trae».

Tres familias:

  · SERVIDOR (`CLAVES_DEL_SERVIDOR`): las escribe SOLO el backend — el historial de peso (`POST /api/diary/progress`,
    FOR UPDATE), el aprendizaje del motor (patrones de rechazo, fricciones, ciclo de compras, reflexiones, puntuaciones
    y calidad del pipeline, EMAs de adherencia, experimentos) y el plan de emergencia del cron. Ningún cliente las edita
    (verificado: cero apariciones en `frontend/src`). Se DESCARTAN siempre, en PATCH y en la reescritura tras generar.
    Las claves internas `_*` también: el frontend ya las filtra (`stripInternalFlags`) y el backend no se fía.
  · PANEL (`CLAVES_SOLO_DEL_PANEL`): `clinical_profile` y `super_personalization`. Su dueño es su panel de
    Configuración (`PUT /api/user/preferences/{clinical-profile,super-personalization}`, FOR UPDATE); el asistente NO
    las pregunta. Se descartan en PATCH, NUNCA se reescriben tras generar, y para GENERAR manda la del perfil.
  · BÁSICOS (`CLAVES_DE_BASICOS`): `staple_foods` (la canónica: la escribe su panel, `PUT
    /api/user/preferences/staple-foods`, y la reescritura del asistente), su espejo `stapleFoods` (la clave del
    formulario) y `stapleAnchors` (las anclas de cada básico, paso «Mis básicos» del asistente). Se descartan en
    PATCH; al generar siguen el patrón P1-COUNTRY-RENEWAL-PROFILE-WINS: en la RENOVACIÓN (`update_reason`) manda el
    perfil y no se reescriben; en el asistente completo manda lo que se acaba de contestar y se reescriben con las dos
    claves IGUALES (y sin anclas de básicos que ya no están), para que ningún lector —unos leen una, otros la otra—
    vuelva a ver dos listas distintas.

Todo lo demás (alergias, condiciones, medicación, peso, edad, sexo, altura, país, dieta, avisos…) se edita
legítimamente por `PATCH /api/profile` y se aplica como hasta hoy. El contrato opcional `health_profile_keys` (lista
de claves que el cliente QUIERE escribir) limita el PATCH a esas claves: un cliente que manda el formulario entero
puede declarar qué cambió de verdad y el resto de la copia no toca nada. Sin la lista ⇒ conducta de siempre.

tooltip-anchor: CLAVES_DEL_SERVIDOR, CLAVES_SOLO_DEL_PANEL, CLAVES_DE_BASICOS, filtrar_parche_health_profile,
reescritura_tras_generar, perfil_manda_al_generar, sincronizar_basicos (tests/test_p1_plan_lote_717_memory_sync.py)
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

# Escritores (grep 2026-09-28): diario/progreso, ai_helpers, db_plans, graph_orchestrator y cron_tasks. Si un cliente
# empieza a editar una de estas, sale de aquí y gana su propio endpoint: esta lista es de claves SIN escritor cliente.
CLAVES_DEL_SERVIDOR = frozenset({
    "weight_history",                                   # POST /api/diary/progress (FOR UPDATE)
    "rejection_patterns", "frictions", "grocery_cycle", "reflection_history",
    "pipeline_score_history", "last_pipeline_score", "last_pipeline_attempts",
    "emergency_backup_plan", "tuning_metrics", "last_fatigued_ingredients", "judge_calibration",
    "attribution_tracker", "counterfactual_pending", "active_experiment_id", "last_plan_quality",
    "quality_history_chunks", "last_zero_log_nudge_at",
    "meal_adherence_weekday", "meal_adherence_weekend", "meal_adherence_weekday_long", "meal_adherence_weekend_long",
})

CLAVES_SOLO_DEL_PANEL = frozenset({"clinical_profile", "super_personalization"})

CLAVE_BASICOS = "staple_foods"          # la canónica (la que escribe su panel)
CLAVE_BASICOS_ESPEJO = "stapleFoods"    # la del formulario; se escribe SIEMPRE igual que la canónica
CLAVE_ANCLAS = "stapleAnchors"
CLAVES_DE_BASICOS = frozenset({CLAVE_BASICOS, CLAVE_BASICOS_ESPEJO, CLAVE_ANCLAS})

# El ruteo del formulario y la identidad no son datos de salud (P1-APPMODE-REQUIRED).
_CLAVES_DE_LA_PETICION = frozenset({"session_id", "user_id", "appMode"})


def es_clave_con_dueno(clave: Any) -> bool:
    """True si la clave NO se escribe por PATCH /api/profile: interna (`_*`), del servidor, de un panel o de los
    básicos."""
    if not isinstance(clave, str):
        return True
    return (clave.startswith("_") or clave in CLAVES_DEL_SERVIDOR or clave in CLAVES_SOLO_DEL_PANEL
            or clave in CLAVES_DE_BASICOS)


def filtrar_parche_health_profile(parche: Optional[dict], claves_declaradas: Optional[Iterable[str]] = None):
    """El filtro de `PATCH /api/profile`. Devuelve `(parche_a_aplicar, ignoradas_por_dueno)`.

    - `claves_declaradas is None` (clientes de siempre): se aplica todo el parche menos las claves con dueño.
    - Con `claves_declaradas`: SOLO las claves declaradas que vengan en el parche (menos las que tienen dueño). Lo no
      declarado es la copia que el cliente trae pero no quiere escribir: no se toca y no es un aviso.
    - `ignoradas_por_dueno`: las claves con dueño que se descartan (ordenadas). Con lista declarada, solo las que el
      cliente DECLARÓ — las demás no pretendía escribirlas.
    """
    if not isinstance(parche, dict) or not parche:
        return {}, []
    declaradas = None if claves_declaradas is None else {c for c in claves_declaradas if isinstance(c, str)}
    aplicar: dict = {}
    ignoradas = []
    for clave, valor in parche.items():
        if declaradas is not None and clave not in declaradas:
            continue
        if es_clave_con_dueno(clave):
            ignoradas.append(str(clave))
            continue
        aplicar[clave] = valor
    return aplicar, sorted(ignoradas)


def _nombres(lista: Any) -> list:
    return [str(x).strip() for x in lista if str(x or "").strip()] if isinstance(lista, list) else []


def sincronizar_basicos(hp: dict, basicos: Optional[list] = None) -> None:
    """Deja `staple_foods` y `stapleFoods` IGUALES en `hp` (IN-PLACE) y quita las anclas de básicos que ya no están.

    `basicos` = la lista que se está escribiendo; si falta, la canónica que ya tenga `hp`. Las anclas se podan por
    nombre (sin distinguir mayúsculas): `plan_policy` y `culinary_context` leen los NOMBRES de las anclas como básicos,
    así que un ancla huérfana resucitaba en la generación el básico que el usuario quitó en Configuración."""
    if not isinstance(hp, dict):
        return
    if basicos is None:
        basicos = hp.get(CLAVE_BASICOS)
    if not isinstance(basicos, list):
        return
    lista = _nombres(basicos)
    hp[CLAVE_BASICOS] = list(lista)
    hp[CLAVE_BASICOS_ESPEJO] = list(lista)
    anclas = hp.get(CLAVE_ANCLAS)
    if isinstance(anclas, list):
        vivos = {n.lower() for n in lista}
        hp[CLAVE_ANCLAS] = [a for a in anclas
                            if isinstance(a, dict) and str(a.get("name") or "").strip().lower() in vivos]


def reescritura_tras_generar(hp_data: dict, *, renovacion: bool):
    """Lo que la generación puede volcar al perfil al terminar. Devuelve `(hp_data_filtrado, descartadas)`.

    Fuera siempre: identidad/ruteo, internas, las del servidor y las de los paneles (el valor fresco del perfil, bajo
    el FOR UPDATE de la escritura, se queda como está). Básicos: en la renovación fuera (manda el perfil); en el
    asistente completo dentro, con las dos claves iguales y las anclas podadas."""
    if not isinstance(hp_data, dict):
        return {}, []
    salida: dict = {}
    descartadas = []
    for clave, valor in hp_data.items():
        if clave in _CLAVES_DE_LA_PETICION:
            continue
        basico = clave in CLAVES_DE_BASICOS
        if (basico and renovacion) or (not basico and es_clave_con_dueno(clave)):
            descartadas.append(str(clave))
            continue
        salida[clave] = valor
    if not renovacion and (CLAVE_BASICOS in salida or CLAVE_BASICOS_ESPEJO in salida):
        canonica = salida.get(CLAVE_BASICOS)
        sincronizar_basicos(salida, canonica if isinstance(canonica, list) else salida.get(CLAVE_BASICOS_ESPEJO))
    return salida, sorted(descartadas)


def basicos_del_perfil(hp: Any) -> Optional[list]:
    """La lista de básicos del PERFIL: la canónica; la del formulario solo si la canónica no existe (perfiles de antes
    de que se escribieran iguales). None = el perfil no dice nada (≠ `[]`, que es «ninguno»)."""
    if not isinstance(hp, dict):
        return None
    for clave in (CLAVE_BASICOS, CLAVE_BASICOS_ESPEJO):
        if isinstance(hp.get(clave), list):
            return _nombres(hp[clave])
    return None


def perfil_manda_al_generar(data: dict, hp: Any) -> list:
    """El formulario de una petición de GENERACIÓN (`/analyze`, `/analyze/stream`, `/generation-runs`) toma del perfil
    las claves que tienen dueño en Configuración. Muta `data` IN-PLACE, ANTES del pipeline y de la reescritura.
    Devuelve las claves que tomó.

    - Paneles (`clinical_profile`, `super_personalization`): si el perfil la tiene, SIEMPRE manda — el asistente no las
      pregunta, así que la del formulario solo puede ser una copia vieja (el TFG 45 del teléfono que el ordenador no
      veía y el plan salía sin tope renal).
    - Básicos y anclas: patrón P1-COUNTRY-RENEWAL-PROFILE-WINS. Con `update_reason` (Renovar/Actualizar) el formulario
      no trae una elección nueva, trae la copia del dispositivo ⇒ manda el perfil. Sin él (asistente completo) manda lo
      contestado; si el formulario no trae básicos, se rellenan del perfil.
    Sin perfil legible (`hp` vacío/None) no hay nada que imponer: el formulario queda como vino."""
    if not isinstance(data, dict) or not isinstance(hp, dict) or not hp:
        return []
    tomadas = []
    for clave in sorted(CLAVES_SOLO_DEL_PANEL):
        valor = hp.get(clave)
        if isinstance(valor, dict) and data.get(clave) != valor:
            data[clave] = valor
            tomadas.append(clave)
    renovacion = bool(data.get("update_reason"))
    basicos = basicos_del_perfil(hp)
    anclas = hp.get(CLAVE_ANCLAS) if isinstance(hp.get(CLAVE_ANCLAS), list) else None
    trae_basicos = isinstance(data.get(CLAVE_BASICOS), list) or isinstance(data.get(CLAVE_BASICOS_ESPEJO), list)
    if basicos is not None and (renovacion or not trae_basicos):
        if data.get(CLAVE_BASICOS) != basicos or data.get(CLAVE_BASICOS_ESPEJO) != basicos:
            tomadas.append(CLAVE_BASICOS)
        data[CLAVE_BASICOS] = list(basicos)
        data[CLAVE_BASICOS_ESPEJO] = list(basicos)
    if anclas is not None and (renovacion or not isinstance(data.get(CLAVE_ANCLAS), list)):
        if data.get(CLAVE_ANCLAS) != anclas:
            tomadas.append(CLAVE_ANCLAS)
        data[CLAVE_ANCLAS] = list(anclas)
    if renovacion and basicos is not None:
        sincronizar_basicos(data, basicos)   # sin anclas de básicos que el perfil ya no tiene
    if tomadas:
        logger.info(f"[P1-PLAN-LOTE-717] generación: el perfil manda en {tomadas} "
                    f"({'renovación' if renovacion else 'asistente'})")
    return tomadas
