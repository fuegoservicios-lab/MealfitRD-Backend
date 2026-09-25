# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-269 · 2026-09-25] Productos animales «escondidos» que el escáner de dieta no reconocía.

Sonda contra `_scan_diet_violations` (vegana, vegetariana, pescetariana): «1 cdta de miel» pasaba limpia para una
vegana —el prompt del revisor dice «Vegano: CERO productos animales (incluyendo huevos, lácteos, miel)», pero eso no es
determinista— y «gelatina», «grenetina», «colágeno» (hueso y piel de animal) y la «sopita» (caldo de pollo en cubito)
pasaban para una vegetariana. El filtro de catálogo (`constants._get_fast_filtered_catalogs`) le ofrecía además la fila
«Miel» a la vegana.

`CARNE_OCULTA` se suma a la carne de tierra (vegana, vegetariana y pescetariana); `SOLO_VEGANO`, solo a la vegana. Las
alternativas vegetales («gelatina de agar», «miel de agave/caña», «sopita de vegetales», «colágeno vegano») las excusa
`excusas_vegetales.excusa_contextual`, ACOTADAS al término: «tocino de maple» sigue siendo cerdo.
tooltip-anchor: P1-PLAN-LOTE-269-DIETA-OCULTA
"""
CARNE_OCULTA = ("gelatina", "grenetina", "colageno", "sopita")
SOLO_VEGANO = ("miel",)
