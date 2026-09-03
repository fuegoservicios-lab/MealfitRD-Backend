"""[P2-HEALTH-LIMITER · 2026-08-15] Los cinco health públicos, con freno.

QUÉ FALTABA. `/atomic-pool-health`, `/chunk-queue-health`,
`/pantry-tolerance-health`, `/tz-fallback-health` y `/health/plan-graph` son
públicos a propósito (P3-HEALTH-AGGREGATES-DISCLOSURE-DEFERRED: un poller externo
tiene que poder leerlos sin credenciales) — pero no tenían ningún límite de
frecuencia.

El daño no es el que parece. No es que se caigan ellos: los cinco son `def` y no
`async def`, así que FastAPI los ejecuta en el threadpool de anyio, que tiene 40
tokens para TODO el proceso. `chunk-queue-health` además dispara tres
`execute_sql_query`. Una inundación sobre un endpoint sin auth degrada la latencia
de **todos los handlers síncronos de la aplicación**, incluidos los que sí
importan. El coste se paga en otra parte, que es justo por lo que no se ve.

LA LÍNEA QUE NO SE CRUZA. El freno es un `RateLimiter` per-IP genérico, NO
`_check_admin_rate_limit`. Ese helper pertenece al gate admin y su aplicación aquí
revertiría la decisión de divulgación, que sigue siendo correcta. Público y con
freno no son opuestos: lo contrario de público es autenticado, no ilimitado.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_SYSTEM_PY = Path(__file__).resolve().parent.parent / "routers" / "system.py"

_HANDLERS = [
    "get_atomic_pool_health",
    "get_chunk_queue_health",
    "get_pantry_tolerance_health",
    "get_tz_fallback_health",
    "get_plan_graph_health",
]


@pytest.fixture(scope="module")
def src() -> str:
    return _SYSTEM_PY.read_text(encoding="utf-8")


def test_los_cinco_llevan_limitador(src: str) -> None:
    sin_freno = []
    for fn in _HANDLERS:
        m = re.search(rf"def {fn}\(([^)]*)\)", src)
        if not m:
            sin_freno.append(f"{fn}: handler NO encontrado — ¿renombrado?")
            continue
        if "_HEALTH_LIMITER" not in m.group(1):
            sin_freno.append(f"{fn}: sin `Depends(_HEALTH_LIMITER)`")
    assert not sin_freno, (
        "Health públicos sin freno:\n  " + "\n  ".join(sin_freno)
        + "\n\nSon `def`, así que corren en el threadpool de anyio (40 tokens para "
        "todo el proceso): una inundación degrada la latencia de TODOS los handlers "
        "síncronos, no sólo la de estos."
    )


def test_el_freno_no_es_el_gate_admin(src: str) -> None:
    """La decisión de divulgación no se revierte de rebote.

    Es la mitad que hace legítimo el cambio: si el freno fuera
    `_check_admin_rate_limit`, habría convertido cinco endpoints públicos en
    endpoints de admin sin que nadie tomara esa decisión.
    """
    m = re.search(r"_HEALTH_LIMITER\s*=\s*RateLimiter\(([^)]*)\)", src)
    assert m, "No encuentro la declaración de `_HEALTH_LIMITER = RateLimiter(...)`."

    for fn in _HANDLERS:
        # `.*?` y no `[^)]*`: la firma ya contiene un `)` propio —el de
        # `Depends(_HEALTH_LIMITER)`— así que una clase negada se corta ahí y el
        # `\):` nunca casa. Mismo tropiezo que este fichero vigila en otros.
        cuerpo = re.search(rf"def {fn}\(.*?\):(.*?)(?=\n@router\.|\ndef |\Z)", src, re.S)
        assert cuerpo, f"No pude extraer el cuerpo de {fn}."
        assert "_check_admin_rate_limit(" not in cuerpo.group(1), (
            f"{fn} usa `_check_admin_rate_limit`. Ese es el limitador del GATE "
            "ADMIN: aplicarlo aquí revierte P3-HEALTH-AGGREGATES-DISCLOSURE-DEFERRED. "
            "El freno correcto es un `RateLimiter` per-IP genérico."
        )
        assert "_verify_admin_token(" not in cuerpo.group(1), (
            f"{fn} pasó a exigir token de admin. Estos cinco son públicos por "
            "decisión documentada."
        )


def test_el_umbral_deja_pasar_a_un_poller_legitimo(src: str) -> None:
    """60/60s no es un número al azar: es el que NO rompe el caso de uso.

    La razón de que existan es que Grafana/UptimeRobot los consulten. Un límite
    que estrangulara a un poller de 1/min habría cambiado un problema de
    saturación por uno de observabilidad — y el segundo se descubre en un
    incidente, que es el peor momento.
    """
    m = re.search(r"_HEALTH_LIMITER\s*=\s*RateLimiter\(\s*max_calls\s*=\s*(\d+)\s*,\s*period_seconds\s*=\s*(\d+)", src)
    assert m, "La declaración de `_HEALTH_LIMITER` cambió de forma; reapuntá este test."
    llamadas, periodo = int(m.group(1)), int(m.group(2))
    por_minuto = llamadas * 60 / periodo
    assert por_minuto >= 30, (
        f"El límite deja {por_minuto:.0f} peticiones/min. Un poller externo (1/min) "
        "necesita margen de sobra: apretarlo hasta rozar su cadencia convierte un "
        "problema de saturación en uno de observabilidad, y ese se descubre durante "
        "un incidente."
    )
