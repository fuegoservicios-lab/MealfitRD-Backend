# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-250 · 2026-09-25] Pescados y mariscos que el escáner determinista no reconocía.

Auditoría del formulario (alergias, dieta, rechazos): con «Pescado» marcado, «150 g de corvina» pasaba limpio por el
backstop de alérgenos, por el guard de dieta de un vegetariano y por el de rechazos, porque los tres leen el mismo
vocabulario y en él faltaban los pescados de mercado más comunes fuera del catálogo dominicano (corvina, pargo, lubina,
caballa, jurel, lenguado…) y mariscos regionales (jaiba, chipirones, berberechos, ostiones, carrucho…). Varios ya eran
ALIAS de filas del catálogo («mojarra» → Tilapia, «jaiba» → Cangrejo, «chipiron» → Calamar): la IA podía escribirlos.

Los consume `graph_orchestrator` en `_ALLERGEN_SYNONYMS['pescado'/'mariscos']` y en `_DIET_SEAFOOD_TERMS` (la paridad
entre ambos la exige `test_paridad_dieta_alergeno_bidireccional`). Formas sin tilde y en singular: el escáner compara
sin acentos, con frontera de palabra y plural español. Fuera a propósito los que también son adjetivos o palabras de
cocina («dorada», «sierra», «lisa», «gallo», «concha» sola, «choco»): una alerta que grita de más se deja de leer.
tooltip-anchor: P1-PLAN-LOTE-250-VOCABULARIO-MAR
"""

PESCADOS_EXTRA = (
    "corvina", "pargo", "lubina", "robalo", "caballa", "jurel", "lenguado", "bagre", "mojarra", "pez espada",
    "emperador", "tiburon", "cazon", "rape", "abadejo", "besugo", "rodaballo", "sargo", "salmonete", "palometa",
    "bonito", "melva", "mojama", "huachinango", "guachinango", "pampano", "colirrubia", "bocachico", "mahi mahi",
    "mahi-mahi", "pangasius", "perca", "anguila", "angula", "sabalo", "barracuda", "picua", "pescadilla",
    "bacaladilla", "carpa", "congrio", "marlin", "hueva", "fumet",
    "cesar",  # [P1-PLAN-LOTE-270] el aderezo/ensalada César lleva anchoas (como la salsa inglesa)
)

MARISCOS_EXTRA = (
    "jaiba", "centolla", "centollo", "necora", "bogavante", "cigala", "quisquilla", "berberecho", "navaja", "chipiron",
    "zamburina", "ostion", "choro", "macha", "erizo", "carrucho", "pulpito", "camaroncito",
)
# Fuera a propósito los COMPUESTOS cuya palabra suelta se declara para otra cosa: la resolución declaración→clase es
# bidireccional por palabra completa, así que «buey de mar» haría que quien declara «buey» pierda los mariscos, «callo de
# hacha» a quien rechaza los «callos», y «concha de abanico / concha negra» a quien no come «conchas» (el pan).
# [P1-PLAN-LOTE-254] Fuera también «caracol»: «1 taza de pasta caracol» (la forma de pasta) salía como marisco.
