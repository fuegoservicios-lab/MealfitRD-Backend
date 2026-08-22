"""[P1-PLAN-QUALITY-INDEX · 2026-07-31] Medidor de VARIEDAD y COHERENCIA por plan.

POR QUÉ EXISTE
El owner preguntó si cambiar el generador (deepseek-flash → gpt-5.6-luna) subiría
la calidad. La respuesta honesta era "no se puede saber": el sistema producía
muchas señales sueltas —reporte de variedad, guard de coherencia recetas↔lista,
panel de micros, banda de macros— pero **ninguna se guardaba junta ni comparable**,
así que dos modelos sólo se podían comparar por impresión. Esto es el instrumento
que faltaba: un número por plan, con sus componentes, persistido.

QUÉ ES Y QUÉ NO ES
  · ES un índice 0-100 DETERMINISTA, derivado de señales que el pipeline ya
    calcula. Mismo plan ⇒ mismo número, siempre. No llama a ningún LLM.
  · NO es un juicio de si el plan "está rico". Mide lo que se puede contar:
    repeticiones, incoherencias detectables, huecos de micros, banda de macros.
  · NO bloquea nada. Es telemetría: se persiste y se lee. Un medidor que además
    rechaza planes deja de ser un medidor y se vuelve otro gate.

CÓMO SE LEE
Cada componente empieza en 100 y se le restan penalizaciones por defecto
CONTADO (no por opinión). El total es la media ponderada. Los pesos están
declarados abajo y son el único sitio donde se decide qué importa más.

SESGO CONOCIDO (importante al comparar)
El índice sólo ve lo que el plan CONTIENE, no lo que costó producirlo. Un plan
que necesitó 3 intentos y otro que salió al primero puntúan igual si el
resultado es igual — para eso está `attempts` en el payload, al lado.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from knobs import _env_bool, _env_float

# Pesos de cada componente en el total. Suman 1.0. Cambiar esto cambia lo que
# el producto declara importante — no es un detalle de implementación.
_PESOS = {
    "variedad": _env_float("MEALFIT_PQI_PESO_VARIEDAD", 0.35, validator=lambda v: 0 <= v <= 1),
    "coherencia": _env_float("MEALFIT_PQI_PESO_COHERENCIA", 0.35, validator=lambda v: 0 <= v <= 1),
    "nutricion": _env_float("MEALFIT_PQI_PESO_NUTRICION", 0.30, validator=lambda v: 0 <= v <= 1),
}

PQI_ENABLED = _env_bool("MEALFIT_PLAN_QUALITY_INDEX", True)

# Penalización por cada defecto contado. Calibradas para que un plan con 1-2
# defectos menores siga en los 80 (bueno con peros) y uno con media docena baje
# de 60 (malo de verdad). Son knobs porque la calibración se ajusta con datos,
# no con intuición.
_PEN_VARIEDAD = {
    "same_day_protein_repeats": 12,   # misma proteína 2 comidas el mismo día
    "same_day_formula_repeats": 8,    # [P1-SAME-DAY-FORMULA-REPEAT] misma base+formato+≥2 acompañantes
    "same_day_repeats": 8,            # plato-base repetido el mismo día
    "fruit_repeats": 8,               # misma fruta dulce en 2 comidas del día
    "cross_day_dishes": 6,            # el mismo plato en días distintos
    "cross_day_preps": 4,             # la misma preparación repartida
    "sweet_savory_clash": 6,          # fruta dulce dominante + base salada
    "sweet_fish_pairings": 8,         # pareo fruta+pescado
    "light_name_heavy_meals": 5,      # nombre "ligero" con plato pesado
}

# [P2-QUALITY-INDEX-COHERENCE · 2026-08-21] Hipótesis del guard de coherencia que NO son un
# defecto DEL PLAN: la receta no cuantificó el alimento, el modelo de yield no lo cubre, la unidad
# de la receta y la de la lista no son la misma, o la nevera dedujo. Todas apuntan a una carencia
# sistémica del instrumento o del modelo — cobrárselas al plan mide otra cosa.
#
# `unknown`, `cap_swallowed_modifier` y `magnitude_undersupply` NO están aquí a propósito: los dos
# últimos son defectos reales del motor, y el primero es como se ve un defecto real antes de tener
# nombre. Ampliar esta lista sin evidencia convierte el índice en un medidor complaciente.
# tooltip-anchor: P2-QUALITY-INDEX-COHERENCE
_COHERENCIA_NO_ATRIBUIBLE = frozenset({
    "recipe_unquantified", "unit_mismatch", "yield_uncovered", "pantry_overdeduct",
})

_PEN_COHERENCIA = {
    "recipe_coherence": 7,     # ingrediente listado que la receta no menciona
    "shopping_presence": 15,   # ingrediente de receta AUSENTE en la lista
    "shopping_magnitude": 5,   # cantidad divergente entre receta y lista
    "dish_quality": 10,        # plato placeholder / ingrediente no-real
}

_PEN_NUTRICION = {
    "micro_piso": 6,           # micro por debajo del piso (status 'bajo')
    "micro_techo": 12,         # micro por encima del techo (status 'alto': sodio/azúcar)
}
# La banda de macros llega como ratio 0..1 (`clinical_band_score.score`): se
# convierte a puntos en vez de penalizarse por evento, porque ya ES una medida
# continua y bien calibrada del pipeline.
_PESO_BANDA_EN_NUTRICION = _env_float(
    "MEALFIT_PQI_PESO_BANDA", 0.4, validator=lambda v: 0 <= v <= 1)


def _clamp(v: float) -> float:
    return max(0.0, min(100.0, round(float(v), 1)))


def _sub_variedad(rep: Dict[str, Any]) -> Dict[str, Any]:
    puntos, detalle = 100.0, {}
    for clave, pen in _PEN_VARIEDAD.items():
        n = rep.get(clave)
        # `cross_day_*` vienen como dict {token: n_dias}: cuenta las entradas.
        if isinstance(n, dict):
            n = len(n)
        try:
            n = int(n or 0)
        except (TypeError, ValueError):
            n = 0
        if n > 0:
            puntos -= pen * n
            detalle[clave] = n
    return {"score": _clamp(puntos), "defectos": detalle}


def _sub_coherencia(plan: Dict[str, Any]) -> Dict[str, Any]:
    """⚠️ Las claves de aquí se VERIFICARON contra planes reales antes de
    escribirlas (2026-07-31, 6 planes). El primer intento leía
    `_shopping_coherence_block` y `_low_quality_dishes`, que NO existen en
    `plan_data` — este componente devolvía 100 en los 6 planes y parecía sano.
    Un medidor inerte es peor que ninguno: da confianza falsa."""
    puntos, detalle = 100.0, {}

    # Ingredientes listados que ninguna instrucción menciona (la clase
    # "huérfano"). Lista de strings, una por defecto encontrado.
    errores = plan.get("_recipe_coherence_errors")
    if isinstance(errores, list) and errores:
        puntos -= _PEN_COHERENCIA["recipe_coherence"] * len(errores)
        detalle["recipe_coherence"] = len(errores)

    # Coherencia recetas↔lista de compras: se lee la ÚLTIMA entrada del
    # historial (el estado con el que el plan se entregó; las anteriores son
    # intentos ya superados).
    hist = plan.get("_shopping_coherence_block_history")
    if isinstance(hist, list) and hist:
        ultimo = hist[-1] if isinstance(hist[-1], dict) else {}
        pres = int(ultimo.get("presence_count") or 0)
        magn = int(ultimo.get("magnitude_count") or 0)
        # [P2-QUALITY-INDEX-COHERENCE · 2026-08-21] Descontar lo que el guard ya atribuyó a una
        # causa que NO es este plan.
        #
        # Al lado de estos dos contadores viaja `hypotheses`, donde `_classify_divergence_hypothesis`
        # clasificó CADA divergencia por su causa — y ese bucketing existe precisamente para separar
        # señal de artefacto: su propio comentario dice que separar `recipe_unquantified` es «el
        # prerequisito para que los umbrales del cron midan señal en vez de RUIDO CONOCIDO». El cron
        # ya lo usa; este índice lo ignoraba y cobraba 15 puntos por cada una.
        #
        # Medido sobre los planes vivos: `d2f2dbc6` perdía 15 puntos con
        # `{'recipe_unquantified': 3}` — o sea por cómo estaban REDACTADAS sus propias recetas — y
        # los dos planes beta eran los peores de 27 por −21,0 puntos exactos de artefacto.
        #
        # EL PRINCIPIO: este índice mide EL PLAN. Una divergencia atribuida a una causa SISTÉMICA
        # (la receta no cuantificó, el modelo de yield no cubre ese alimento, la nevera dedujo) no
        # es un defecto de ese plan concreto. Una atribuida al motor (el cap se comió un
        # modificador, se compró menos de la mitad) sí lo es.
        #
        # `unknown` SIGUE COBRANDO, y es la mitad importante de la decisión: «no sé por qué diverge»
        # es exactamente como se ve un defecto real antes de tener nombre. Perdonarlo dejaría al
        # índice viendo sólo lo que ya sabemos buscar — el medidor inerte contra el que avisa el
        # docstring de esta misma función.
        #
        # Sin `hypotheses` (entradas viejas del historial) no hay nada que discriminar y se cobra
        # como siempre: inventar una atribución que el guard no hizo sería peor que cobrar de más.
        _no_atribuible = 0
        _hyp = ultimo.get("hypotheses")
        if isinstance(_hyp, dict):
            for _k, _v in _hyp.items():
                if str(_k) in _COHERENCIA_NO_ATRIBUIBLE:
                    try:
                        _no_atribuible += max(0, int(_v))
                    except (TypeError, ValueError):
                        continue
        if _no_atribuible:
            # Se descuenta primero de magnitud y luego de presencia: la clasificación no dice a
            # cuál de los dos contadores pertenece cada hipótesis, y magnitud es el bucket barato
            # (5 vs 15), así que este orden es el CONSERVADOR — perdona menos puntos.
            _quita = min(_no_atribuible, magn)
            magn -= _quita
            pres = max(0, pres - (_no_atribuible - _quita))
            detalle["coherence_no_atribuible"] = _no_atribuible
        if pres:
            puntos -= _PEN_COHERENCIA["shopping_presence"] * pres
            detalle["shopping_presence"] = pres
        if magn:
            puntos -= _PEN_COHERENCIA["shopping_magnitude"] * magn
            detalle["shopping_magnitude"] = magn

    # Platos placeholder / ingredientes no-reales.
    dq = plan.get("dish_quality_report")
    if isinstance(dq, dict):
        n_low = dq.get("low_quality_meals")
        if n_low is None:
            issues = dq.get("issues")
            n_low = len(issues) if isinstance(issues, list) else 0
        try:
            n_low = int(n_low or 0)
        except (TypeError, ValueError):
            n_low = 0
        if n_low:
            puntos -= _PEN_COHERENCIA["dish_quality"] * n_low
            detalle["dish_quality"] = n_low

    return {"score": _clamp(puntos), "defectos": detalle}


def _sub_nutricion(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Piso y techo se cuentan por SEPARADO: `gaps` mezcla los dos y
    distinguirlos es lo que evita cobrar dos veces el mismo sodio (el primer
    intento penalizaba el sodio como hueco Y otra vez como techo)."""
    puntos, detalle = 100.0, {}
    micro = plan.get("micronutrient_report") or {}
    gaps = micro.get("gaps") if isinstance(micro, dict) else None
    if isinstance(gaps, list):
        n_alto = sum(1 for g in gaps if isinstance(g, dict) and g.get("status") == "alto")
        n_bajo = sum(1 for g in gaps if isinstance(g, dict) and g.get("status") == "bajo")
        if n_bajo:
            puntos -= _PEN_NUTRICION["micro_piso"] * n_bajo
            detalle["micro_piso"] = n_bajo
        if n_alto:
            puntos -= _PEN_NUTRICION["micro_techo"] * n_alto
            detalle["micro_techo"] = n_alto

    # Banda de macros: el pipeline ya publica un ratio 0..1 medido por día.
    banda = plan.get("clinical_band_score")
    if isinstance(banda, dict) and banda.get("score") is not None:
        try:
            ratio = max(0.0, min(1.0, float(banda["score"])))
        except (TypeError, ValueError):
            ratio = None
        if ratio is not None:
            # Mezcla: el ratio aporta su parte y el resto viene de los micros.
            puntos = puntos * (1 - _PESO_BANDA_EN_NUTRICION) + (ratio * 100.0) * _PESO_BANDA_EN_NUTRICION
            if ratio < 1.0:
                detalle["banda_macros"] = round(ratio, 3)

    return {"score": _clamp(puntos), "defectos": detalle}


def compute_plan_quality_index(plan: Dict[str, Any],
                               variety_report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Índice 0-100 + desglose. Puro y sin efectos: no toca DB ni LLM.

    `variety_report` se acepta inyectado para no recalcularlo cuando el caller
    ya lo tiene (`plan['variety_report']`); si falta, se toma del plan.
    """
    if not isinstance(plan, dict):
        return {"score": None, "error": "plan_no_dict"}

    rep = variety_report or plan.get("variety_report") or {}
    if not isinstance(rep, dict):
        rep = {}

    comp = {
        "variedad": _sub_variedad(rep),
        "coherencia": _sub_coherencia(plan),
        "nutricion": _sub_nutricion(plan),
    }
    total = sum(comp[k]["score"] * _PESOS[k] for k in comp) / (sum(_PESOS.values()) or 1)

    dias = plan.get("days") or []
    return {
        "score": _clamp(total),
        "componentes": {k: comp[k]["score"] for k in comp},
        "defectos": {k: comp[k]["defectos"] for k in comp if comp[k]["defectos"]},
        "pesos": dict(_PESOS),
        # Contexto para comparar dos planes de forma justa (un plan de 7 días
        # tiene más oportunidades de repetir que uno de 3).
        "dias": len(dias) if isinstance(dias, list) else 0,
        "comidas": int(rep.get("total_meals") or 0),
        "intentos": plan.get("_quality_degraded_attempts"),
        "degradado": bool(plan.get("_quality_degraded")),
        "version": 1,
    }
