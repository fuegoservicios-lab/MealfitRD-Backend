# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-651 · 2026-09-27] Las palancas de vuelta atrás del sistema de países se ven desde el arranque (G89) y
CLAUDE.md deja de anunciar una que no gobierna nada (G88).

Los knobs se registran en `_KNOBS_REGISTRY` cuando se LEEN; seis de los siete de rollback del flip viven en funciones
que solo corren al armar una lista o un presupuesto, así que tras arrancar el registro mostraba UNO. Un operador en
incidente consulta el registro para saber qué puede revertir sin redeploy. Ahora `shopping_calculator` y
`nutrition_calculator` los leen una vez al importarse (bloque al FINAL del módulo: arriba, los accesores aún no existen).

tooltip-anchor: P1-PLAN-LOTE-651
"""
import subprocess
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]

PALANCAS = (
    "MEALFIT_COUNTRY_CATALOG_UNPRICED_KEEP", "MEALFIT_COUNTRY_KEEP_RESPECT_RECIPE_QTY", "MEALFIT_UNIT_SYSTEM_BY_COUNTRY",
    "MEALFIT_BUDGET_FLOOR_ENABLED", "MEALFIT_BAKING_STAPLES_KEEP", "MEALFIT_SEASONING_CATALOG_KEEP",
)


def test_las_palancas_estan_en_el_registro_nada_mas_importar():
    # proceso limpio: en la suite, otro test ya pudo haberlas leído y el registro las tendría por casualidad
    codigo = ("import shopping_calculator, nutrition_calculator\n"
              "from knobs import get_knobs_registry_snapshot\n"
              "r = get_knobs_registry_snapshot()\n"
              f"print(','.join(k for k in {PALANCAS!r} if k not in r))\n")
    out = subprocess.run([sys.executable, "-c", codigo], cwd=_BACKEND, capture_output=True, text=True, timeout=240)
    assert out.returncode == 0, out.stderr[-2000:]
    faltan = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    assert faltan == "", f"palancas de rollback invisibles tras importar: {faltan}"


def test_claude_md_no_anuncia_el_knob_que_no_gobierna_nada():
    # la copia del repo del backend (la de la raíz se corrige en el mismo lote, en su repo)
    txt = (_BACKEND / "CLAUDE.md").read_text(encoding="utf-8")
    assert "`MEALFIT_COUNTRY_COLDSTART_SEGMENT` (False)" not in txt
