# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-227 · 2026-09-25 · fuera del grafo en P1-PLAN-LOTE-240] El revisor no quema intentos con no-defectos.

Batería real del 25-sep (16 perfiles): el revisor LLM rechazó 4 veces un plan SIN defectos — «Severidad: none» y textos
que se niegan solos («no es violación», «El plan es seguro», «los rechazos declarados son Pescado y Berenjena, y ninguno
aparece en el plan») — y el perfil que no come pescado quemó los 3 intentos y se entregó marcado como degradado. Un
rechazo sin severidad no es un rechazo, y una observación que dice que no viola nada no es un defecto: pasan a aviso. Una
observación que además sigue («sin embargo, se detecta…») se queda, salvo que su ÚLTIMA oración concluya que se cumple.
Los guards DETERMINISTAS (alérgeno/dieta/rechazo/piso) corren después y conservan la última palabra.

El knob `MEALFIT_REVIEWER_NON_ISSUES_ADVISORY` vive en `graph_orchestrator` (registro de knobs) y se lee al llamar.
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
    r"sin embargo|no obstante|pero (?:adem[aá]s|se detecta|el plan)|se detecta(?:n)? (?:una|un|que)|aunque (?:el plan|se)",
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
            _niega = ((_SELF_NEGATING_ISSUE_RX.search(t) and not _ISSUE_CONTINUES_RX.search(t))
                      or bool(_SELF_NEGATING_TAIL_RX.search(_ultima[0])))
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
