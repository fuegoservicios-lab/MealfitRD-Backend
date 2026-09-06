# -*- coding: utf-8 -*-
"""[P1-QUALITY-SWEEP-STATUS · 2026-09-06] El barrido que nunca pudo barrer.

`_resolve_stale_plan_quality_alerts` cierra una alerta `plan_quality_degraded:<user>:<plan>`
cuando el mismo usuario obtiene DESPUÉS un plan entregado y sin marcas de degradación. Su
predicado exigía `generation_status = 'complete'`.

En la base entera no hay un solo plan en ese estado. Medido el 06-sep sobre las 95 filas de
`meal_plans`:

    complete_partial 71 · partial 21 · generating_next 1 · paused_by_user 1 · failed 1
    complete .............................................................. 0

El estado terminal de un plan por bloques es `complete_partial`; `complete` lo estampa la vía
no-troceada de `services.py`, que hoy no se recorre. Consecuencia medida: de las 180 alertas
abiertas, 132 eran `plan_quality_degraded` que su propio auto-resolve no podía tocar. No era un
backlog sin revisar — era un barrido inalcanzable, corriendo cada tick y reportando cero.

Este test ancla las dos mitades: que el SQL acepte los DOS terminales, y que esa pareja siga
siendo la misma que usa `generation_lifecycle` para decidir «este plan ya está entregado». Si
alguien añade un tercer estado terminal allí, aquí se entera.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

_CRON = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
_LIFECYCLE = (_BACKEND / "generation_lifecycle.py").read_text(encoding="utf-8")


def _cuerpo_del_sweep() -> str:
    """El cuerpo de `_resolve_stale_plan_quality_alerts`, de su `def` al siguiente top-level."""
    i = _CRON.find("def _resolve_stale_plan_quality_alerts(")
    assert i > 0, "renombraron el sweep: actualiza este test junto al renombre"
    j = re.search(r"\n(?:async )?def ", _CRON[i + 10:])
    return _CRON[i:i + 10 + (j.start() if j else len(_CRON))]


def test_el_sweep_acepta_los_dos_estados_terminales():
    cuerpo = _cuerpo_del_sweep()
    assert "generation_status" in cuerpo, "el sweep dejó de mirar el estado del plan"
    assert "complete_partial" in cuerpo, (
        "el sweep volvió a exigir solo 'complete' — un estado que ningún plan troceado alcanza, "
        "así que no cerraría ni una alerta (132 abiertas por esto el 06-sep)")


def test_el_predicado_no_es_una_igualdad_a_complete_a_secas():
    """La forma importa: `= 'complete'` es justo el bug. Debe ser una pertenencia a la pareja."""
    cuerpo = _cuerpo_del_sweep()
    igualdades = re.findall(r"generation_status'\s*=\s*'complete'", cuerpo)
    assert not igualdades, f"quedó una igualdad exacta a 'complete': {igualdades}"


def test_la_pareja_terminal_sigue_siendo_la_misma_que_en_generation_lifecycle():
    """Cross-link: `generation_lifecycle.plan_availability` decide con esta misma pareja si un
    plan está ENTREGADO. Son dos sitios que deben decir lo mismo; si allí crece el conjunto, este
    test obliga a mirar el sweep antes de que vuelva a quedarse ciego."""
    m = re.search(r"status in \(\s*((?:\"[a-z_]+\"\s*,?\s*)+)\)", _LIFECYCLE)
    assert m, "cambió la forma del check de estados en generation_lifecycle.py"
    terminales = set(re.findall(r'"([a-z_]+)"', m.group(1)))
    assert terminales == {"complete", "complete_partial"}, (
        f"generation_lifecycle ahora considera entregados {sorted(terminales)}; el sweep de "
        f"`plan_quality_degraded` en cron_tasks.py debe aceptar exactamente los mismos")
    cuerpo = _cuerpo_del_sweep()
    for est in terminales:
        assert f"'{est}'" in cuerpo, f"el sweep no contempla el estado terminal {est!r}"


def test_partial_sigue_fuera():
    """Un plan `partial` tiene bloques en vuelo: su calidad todavía no es definitiva y cerrar la
    alerta con él sería cerrarla antes de tiempo."""
    cuerpo = _cuerpo_del_sweep()
    sql = cuerpo[cuerpo.find("plan_quality_degraded:"):]
    linea = next((l for l in sql.splitlines() if "IN ('complete'" in l or "IN ('complete\"" in l), None)
    if linea is None:
        linea = next(l for l in sql.splitlines() if "complete_partial" in l)
    assert "'partial'" not in linea, f"`partial` entró en el predicado del sweep: {linea.strip()}"


@pytest.mark.parametrize("script", ["scripts/backfill_veg_lines_v7.py",
                                    "scripts/measure_cooked_protein_lines.py"])
def test_los_scripts_de_medicion_no_miran_el_estado_fantasma(script):
    """Misma trampa, distinto sitio: estos dos scripts seleccionan «planes activos». Con
    `= 'complete'` devolvían el conjunto vacío y reportaban «nada que hacer» — una medición que
    no puede encontrar nada es peor que no medir, porque parece un resultado."""
    txt = (_BACKEND / script).read_text(encoding="utf-8")
    assert "generation_status' = 'complete'" not in txt, (
        f"{script} volvió a filtrar por un estado que ningún plan tiene")
    assert "complete_partial" in txt, f"{script} no contempla el estado terminal real"
