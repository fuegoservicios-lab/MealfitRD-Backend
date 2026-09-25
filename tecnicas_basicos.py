# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-240 · 2026-09-25] Técnica canónica por alimento básico. Movido VERBATIM desde `graph_orchestrator.py`
(aire en el god-file); el grafo lo re-exporta con el mismo nombre, así que `go.<nombre>` sigue siendo el mismo objeto.
Solo datos: ningún import, ninguna lógica."""

_STAPLE_TECHNIQUE_CANONICAL = {
    # -- compartido con culinary_coherence.VERB_TO_METHOD (mismos nombres canónicos) --
    "hervid": "hervir", "hervir": "hervir",
    "plancha": "plancha",
    "frito": "freir", "frita": "freir", "freir": "freir",
    "horneado": "hornear", "horno": "hornear", "airfryer": "hornear",
    "guisad": "guisar",
    "salteado": "saltear", "saltear": "saltear",
    "licuado": "licuar", "licuada": "licuar",
    # NOTA: "tostar"/"tostado"/"tostada" (también en VERB_TO_METHOD) se dejó FUERA a propósito —
    # "tostada" colisiona por substring con menciones de PAN TOSTADO como acompañante ("con
    # Tostadas") que no describen la técnica de la proteína principal, produciendo una firma falsa
    # (detectado en test: "Huevo Revuelto con Tostadas" resolvía a 'tostar' en vez de 'revuelto').
    # Añadir tostar exigiría word-boundary + desambiguación de rol (¿tostada es el plato o un
    # acompañante?) que no está en el alcance de este fix — mismo criterio conservador que el
    # resto del módulo: mejor NO mapear una técnica que mapearla mal.
    # -- propios (fuera del alcance de VERB_TO_METHOD, que solo valida verbos de cocción) --
    "asado": "asado", "asar": "asado",
    "empaniz": "empanizado",
    "revoltillo": "revuelto", "revuelto": "revuelto",
    "majado": "majado", "majar": "majado",
    "batido": "batido", "batida": "batido",
    "sopa": "sopa",
    "crema": "crema",
    "ensalada": "ensalada",
    "vapor": "vapor",
    "mechada": "estofado", "mechado": "estofado", "estofado": "estofado",
    "croqueta": "croqueta",
    "tortitas": "tortilla", "tortilla": "tortilla",
    "sarten": "sarten",
    "crudo": "crudo", "cruda": "crudo",
    "duro": "duro",
    "pochado": "pochado", "pochada": "pochado", "escalfado": "pochado", "cocid": "hervir",  # [P1-PLAN-LOTE-196] «huevo cocido» del cerrador (último: prioridad mínima)
}
