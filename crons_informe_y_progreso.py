# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1058/1059 · 2026-11-17] Los crons del informe para el médico y del juego de semanas.

Viven aquí y no en `cron_tasks.py` porque ese fichero tiene tope de líneas (`test_p1_plan_lote_746`,
`test_p3_shopping_projection_pkg`): solo recibe la llamada a `registrar`.

- `purge_health_reports` (24 h): borra los informes con más de 365 días (`informe_servicio.purgar_informes_antiguos`).
- `progreso_aviso_domingo` (1 h): el aviso del domingo a quien lo encendió (`aviso_semanal.enviar_avisos_del_domingo`).
"""
from __future__ import annotations

JOBS = (
    ("purge_health_reports", 24, "informe_servicio", "purgar_informes_antiguos"),
    ("progreso_aviso_domingo", 1, "aviso_semanal", "enviar_avisos_del_domingo"),
)


def registrar(scheduler, add_job_jittered, misfire_grace_time) -> list[str]:
    """Registra los jobs que falten. Devuelve los ids registrados ahora."""
    nuevos = []
    for job_id, horas, modulo, funcion in JOBS:
        if scheduler.get_job(job_id):
            continue
        fn = getattr(__import__(modulo), funcion)
        add_job_jittered(scheduler, fn, "interval", hours=horas, id=job_id, max_instances=1, coalesce=True,
                         replace_existing=True, misfire_grace_time=misfire_grace_time)
        nuevos.append(job_id)
    return nuevos
