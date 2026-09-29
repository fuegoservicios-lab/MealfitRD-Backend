# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-227 · 2026-09-25 · fuera del grafo en P1-PLAN-LOTE-240] El revisor no quema intentos con no-defectos.

Batería real del 25-sep (16 perfiles): el revisor LLM rechazó 4 veces un plan SIN defectos — «Severidad: none» y textos
que se niegan solos («no es violación», «El plan es seguro», «los rechazos declarados son Pescado y Berenjena, y ninguno
aparece en el plan») — y el perfil que no come pescado quemó los 3 intentos y se entregó marcado como degradado. Un
rechazo sin severidad no es un rechazo, y una observación que dice que no viola nada no es un defecto: pasan a aviso. Una
observación que además sigue («sin embargo, se detecta…») se queda, salvo que su ÚLTIMA oración concluya que se cumple.
Los guards DETERMINISTAS (alérgeno/dieta/rechazo/piso) corren después y conservan la última palabra.

El knob `MEALFIT_REVIEWER_NON_ISSUES_ADVISORY` vive en `graph_orchestrator` (registro de knobs) y se lee al llamar.
[P1-PLAN-LOTE-746 · 2026-09-28] Los partes de cumplimiento («No se detectan alérgenos declarados», «Dieta 'balanced'
respetada») los decide `revisor_confirmaciones` (knob propio `MEALFIT_REVIEWER_CONFIRMATIONS_DISCARD`). Esa regla vive
DENTRO de este bucle: con `MEALFIT_REVIEWER_NON_ISSUES_ADVISORY` apagado, la función sale antes y la del 746 no corre.
tooltip-anchor: P1-PLAN-LOTE-227-NO-DEFECTOS
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger("graph_orchestrator")

_SELF_NEGATING_ISSUE_RX = re.compile(
    r"no (?:es|constituye|representa|supone|implica) (?:una |ninguna )?violaci[oó]n|no hay (?:ninguna )?violaci[oó]n|"
    r"el plan es seguro|respetando (?:los|las|sus) (?:rechazos|restricciones|preferencias)|"
    r"ninguno aparece en el plan|no (?:aparece|figura)n? en el plan",
    re.IGNORECASE)
# La conclusión manda: «…sin embargo, se detecta que… no se encontró pescado ni berenjena, por lo que este punto se
# cumple» (batería real, 25-sep) termina negándose. Solo se mira la ÚLTIMA oración.
_SELF_NEGATING_TAIL_RX = re.compile(
    r"(?:por lo que|as[ií] que|de modo que|con lo que)\s+(?:este|el|dicho) (?:punto|requisito|criterio)\s+(?:s[ií]\s+)?se cumple|"
    r"(?:por lo que|as[ií] que|de modo que)\s+no (?:hay|existe|es|constituye) (?:una |ninguna )?violaci[oó]n",
    re.IGNORECASE)
_ISSUE_CONTINUES_RX = re.compile(
    r"sin embargo|no obstante|pero (?:adem[aá]s|se detecta|el plan)|se detecta(?:n)? (?:una|un|que)|aunque (?:el plan|se)|"
    r"\bexcepto\b|\bsalvo\b|a excepci[oó]n de",   # [P1-PLAN-LOTE-746] «el plan es seguro excepto por…» afirma un defecto
    re.IGNORECASE)
# [P1-PLAN-LOTE-746 · 2026-09-28] La negación suelta se buscaba en CUALQUIER parte: «El Día 3 incluye hígado
# encebollado, que el paciente rechazó. La berenjena no aparece en el plan.» salía APROBADA con severidad `critical`
# (también con el knob del 746 apagado). La revisión 1 lo cerró con una lista de verbos («incluye/contiene/aporta…») y
# se le escapaban «aparece en», «hay», «está presente» y las frases sin verbo («Día 3 con 4200 mg de potasio para
# paciente renal. El plátano no aparece en el plan.»). [revisión 2] Invertida, sin listas de verbos
# (`revisor_confirmaciones.oraciones_sin_hallazgo`): absuelve solo si CADA oración trae su propio veredicto («…, no es
# violación», con lo que le sigue cerrado) o CADA cláusula suya es confirmación, declaración de rechazo/alergia del
# paciente o la AUSENCIA DE LO DECLARADO. «El plan es seguro» ya no absuelve otra cláusula de su oración, y la ausencia
# de algo que nadie declaró rechazar («el hierro hemo no aparece en el plan») puede ser algo bueno que falta: se queda.
# El caso del 227 («contiene queso mozzarella — sin alergia a lácteos declarada, no es violación») no cambia, ni la regla
# de la conclusión (`_SELF_NEGATING_TAIL_RX`, la del 25-sep). No depende del knob del 746: el hueco estaba en main.
# [P1-PLAN-LOTE-746 · 2026-09-28 · revisión 3] Y lo que SIGUE al veredicto en su oración ya no queda absuelto: «No hay
# violación de la alergia, pero el Día 3 incluye maní» o «…, por lo que no es una violación, pero el Día 3 aporta
# 4200 mg de potasio» salían aprobadas con `critical` (`_ISSUE_CONTINUES_RX` solo ve «pero además / pero se detecta /
# pero el plan»). No se añade un «pero» suelto aquí: el de la n.º 20 va ANTES del veredicto y es el mismo objeto. La
# regla vive en `revisor_confirmaciones._pendientes` (veredicto local) y `.conclusion_cerrada` (la conclusión).
# [revisión 4] Y lo que lo PRECEDE en su cláusula propia: «…prohibido temporalmente, pero no es violación de alergia»,
# «…para paciente renal, el plátano no constituye violación» (`_antes_del_veredicto`); a la conclusión se le pasa lo que
# absuelve para que una declaración en su cola no nombre lo que aparece ahí («…; el paciente rechazó el hígado»).
# [P1-PLAN-LOTE-255 · 2026-09-25] «Posible reactividad cruzada» con un alimento que el usuario NO declaró no es un
# defecto: batería rd252 (maní + sésamo + «piña» escrita a mano) — el revisor rechazó como CRÍTICO la linaza «por el
# sésamo», el edamame «por el maní» y la lechosa y el guineo «por la piña», dos veces, y el usuario recibió el PLAN DE
# EMERGENCIA. Como el texto dice «alergia», G18 lo contaba como agudo. Lo DECLARADO (y toda su clase) lo sigue parando la
# guarda determinista de alérgenos, que corre después y conserva la última palabra; esto pasa a aviso para el usuario.
# tooltip-anchor: P1-PLAN-LOTE-255-REACTIVIDAD-CRUZADA
_CROSS_REACTIVITY_RX = re.compile(
    r"reactividad(?:es)? cruzadas?|reacci[oó]n(?:es)? cruzadas?|reactiv[oa]s? cruzad[oa]s?|sensibilizaci[oó]n cruzada|"
    r"cross[- ]?reactiv",
    re.IGNORECASE)


def _downgrade_reviewer_non_issues(approved, issues, severity):
    """[P1-PLAN-LOTE-227] Rechazo con severidad «none» ⇒ aprobado; observaciones que se niegan solas ⇒ aviso.
    Devuelve (approved, issues_reales, severity, avisos). Puro; nunca lanza."""
    try:
        import graph_orchestrator as _go
        if not _go.REVIEWER_NON_ISSUES_ADVISORY or approved or not issues:
            return approved, list(issues or []), severity, []
        if str(severity or "").strip().lower() in ("none", "ninguna", "ninguno"):
            logger.warning(f"📋 [P1-PLAN-LOTE-227] el revisor rechazó con severidad «none» → aprobado; sus "
                           f"{len(issues)} observación(es) pasan a aviso (los guards deterministas validan aparte).")
            return True, [], "low", list(issues)
        real, avisos = [], []
        for it in issues:
            t = str(it)
            _ultima = [s for s in re.split(r"(?<=[.;!?])\s+", t.strip()) if s.strip()][-1:] or [""]
            _fin = _SELF_NEGATING_TAIL_RX.search(_ultima[0])
            _niega = ((_SELF_NEGATING_ISSUE_RX.search(t) and not _ISSUE_CONTINUES_RX.search(t)
                       and __import__("revisor_confirmaciones").oraciones_sin_hallazgo(t))  # [P1-PLAN-LOTE-746 · rev. 2]
                      or bool(_fin and __import__("revisor_confirmaciones").conclusion_cerrada(  # [P1-PLAN-LOTE-746 · rev. 3-4]
                          _ultima[0][_fin.end():], _ultima[0][:_fin.start()]))
                      or bool(_CROSS_REACTIVITY_RX.search(t))   # [P1-PLAN-LOTE-255]
                      or __import__("revisor_confirmaciones").es_confirmacion_sin_defecto(t))  # [P1-PLAN-LOTE-746]
            (avisos if _niega else real).append(it)
        if not avisos:
            return approved, real, severity, []
        if real:
            logger.info(f"📋 [P1-PLAN-LOTE-227] {len(avisos)} observación(es) que se niegan solas → aviso; "
                        f"{len(real)} issue(s) reales se quedan.")
            return approved, real, severity, avisos
        logger.warning(f"📋 [P1-PLAN-LOTE-227] TODAS las {len(avisos)} observaciones del revisor se negaban solas → "
                       f"aprobado (los guards deterministas validan aparte).")
        return True, [], "low", avisos
    except Exception:
        return approved, list(issues or []), severity, []
