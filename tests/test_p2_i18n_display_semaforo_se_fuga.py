"""[P2-I18N-DISPLAY-SEMAFORO-SE-FUGA-SI-EL-HILO-NO-ARRANCA · 2026-08-23] Si `Thread.start()`
fallaba, el permiso del semáforo no se devolvía nunca.

`schedule_plan_display_enrichment` hace `acquire(blocking=False)` y el `release` vive en el
`finally` de `_run` — el cuerpo del hilo. Correcto para todo lo que pase DENTRO del hilo.
Pero entre el `acquire` y el arranque hay una llamada que puede fallar y que no está dentro
de ningún `finally`: `threading.Thread(target=_run, daemon=True).start()`. Si lanza
(`RuntimeError: can't start new thread` — límite de hilos del proceso, memoria), `_run` no
corre, nadie suelta el permiso, y el `except` exterior sólo escribe un `warning`.

Con `MAX_INFLIGHT=4`: cuatro fallos de arranque y la traducción del plan queda APAGADA para
todo el proceso, en silencio, hasta el próximo redeploy. Y el síntoma que se vería —«reason:
inflight_cap» en cada intento— apunta a un techo alcanzado, no a un techo fugado.

Es exactamente la clase de defecto que el comentario del `finally` describe para el hilo
(«una excepción por la que nadie suelta el permiso convierte el techo en un candado
permanente») aplicada al tramo que ese `finally` no cubre.

tooltip-anchor: P2-I18N-DISPLAY-SEMAFORO-SE-FUGA-SI-EL-HILO-NO-ARRANCA
"""
from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

import plan_display_i18n as pdi

_MARKER = "P2-I18N-DISPLAY-SEMAFORO-SE-FUGA-SI-EL-HILO-NO-ARRANCA"


@pytest.fixture()
def semaforo_limpio():
    """Un semáforo propio y conocido, para contar permisos sin depender del estado global
    que otros tests hayan dejado."""
    original = pdi._INFLIGHT_SEMAPHORE
    pdi._INFLIGHT_SEMAPHORE = threading.BoundedSemaphore(2)
    try:
        yield pdi._INFLIGHT_SEMAPHORE
    finally:
        pdi._INFLIGHT_SEMAPHORE = original


def _permisos_libres(sem: threading.BoundedSemaphore) -> int:
    """Cuántos permisos quedan, sin consumirlos."""
    return sem._value  # noqa: SLF001 — es lo único que mide la fuga


def test_si_el_hilo_no_arranca_el_permiso_se_devuelve(semaforo_limpio) -> None:
    sem = semaforo_limpio
    antes = _permisos_libres(sem)

    class _HiloQueNoArranca:
        def __init__(self, *a, **k): pass
        def start(self):
            raise RuntimeError("can't start new thread")

    with patch.object(pdi.threading, "Thread", _HiloQueNoArranca), \
         patch.object(pdi, "_plan_display_i18n_enabled", return_value=True), \
         patch.object(pdi, "_emit_result_telemetry"):
        pdi.schedule_plan_display_enrichment("plan-1", "user-1", "fr-FR")

    assert _permisos_libres(sem) == antes, (
        f"el semáforo pasó de {antes} a {_permisos_libres(sem)} permisos: `Thread.start()` "
        f"falló y nadie devolvió el que se había cogido. Con MAX_INFLIGHT fallos así, la "
        f"traducción del plan queda apagada para todo el proceso. [{_MARKER}]"
    )


def test_el_camino_normal_sigue_devolviendo_el_permiso(semaforo_limpio) -> None:
    """Control: cuando el hilo SÍ arranca, el `finally` de `_run` lo suelta, y el arreglo
    no puede soltarlo DOS veces (BoundedSemaphore lanzaría `ValueError`)."""
    sem = semaforo_limpio
    antes = _permisos_libres(sem)
    terminado = threading.Event()

    def _enrich_falso(*a, **k):
        terminado.set()
        return {"enriched_meals": 0}

    with patch.object(pdi, "enrich_plan_display", _enrich_falso), \
         patch.object(pdi, "_plan_display_i18n_enabled", return_value=True), \
         patch.object(pdi, "_emit_result_telemetry"):
        pdi.schedule_plan_display_enrichment("plan-2", "user-2", "fr-FR")
        assert terminado.wait(5), "el hilo no corrió"

    # El hilo puede tardar un instante en llegar a su `finally`.
    for _ in range(50):
        if _permisos_libres(sem) == antes:
            break
        threading.Event().wait(0.02)
    assert _permisos_libres(sem) == antes, (
        f"tras un enriquecimiento normal el semáforo quedó en {_permisos_libres(sem)} "
        f"(antes {antes}): o no se soltó, o se soltó de más. [{_MARKER}]"
    )
