"""[P2-LOADING-ETA-57 · 2026-07-06] Página de carga: el ETA y la auditoría de producción.

Pedidos del owner sobre /plan ("Diseñando tu plan"):
1. "Puedes salir si quieres" debe ser VERDAD: cerrar y volver → si sigue
   generando, de vuelta a la pantalla de carga; si terminó, al dashboard.
   VEREDICTO AUDIT: ya implementado end-to-end (PendingPipelineRecovery en
   App.jsx: boot-check contra el KV backend aun sin flag local — cross-device;
   polling 10s consciente de visibilidad; ack idempotente; exit tras 6 fallos).
2. Gap de producción encontrado: en modo recovery el contador "Transcurrido"
   reiniciaba en 0:00 (mentía). Ahora arranca del started_at real del flag.

EL RANGO SE MUEVE, Y ESO ESTÁ BIEN. Historial: 4-5 (07-06) → 5-7 → 9-10 (07-09,
renovaciones de 7,5-8,4 min con 2-3 intentos del reviewer) → **3-6** (commit
ded7c6f, "el 9-10 quedó impreciso"). Cada cambio es una decisión del owner sobre
lo que la pantalla promete, así que el valor vivo se ancla explícitamente: si
alguien lo toca, este test lo pregunta.

[reapuntado 2026-08-18] Lo que este fichero NO hacía y ahora sí. Traía el rango
9-10 escrito a mano en TRES sitios (la copia inicial, la del contador y los dos
umbrales), y el commit que bajó el rango a 3-6 movió la copia y dejó los umbrales
—y este test— hablando del rango anterior. Estuvo rojo desde entonces, escondido
detrás de otro fallo por el `-x` de pytest.

La lección es que había UN dato (el techo del estimado) copiado a mano en cuatro
sitios, y por eso el arreglo no es escribir 3-6 cuatro veces: el techo se LEE de
la copia y los umbrales se comprueban CONTRA él. Así el próximo cambio de rango
sólo tiene que tocar una línea de este fichero —la del valor que el owner
decidió— y la coherencia interna se verifica sola.
"""

import pytest
import os
import re

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FRONTEND = os.path.join(os.path.dirname(_BACKEND), "frontend")


def _read(*parts) -> str:
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module", autouse=True)
def _load_frontend_sibling_sources(frontend_repo_path):
    # La fixture compartida salta el módulo antes de cualquier I/O si falta el hermano.
    _ = frontend_repo_path
    global _APP, _PLAN, _REC
    _PLAN = _read(_FRONTEND, "src", "pages", "Plan.jsx")
    _APP = _read(_FRONTEND, "src", "App.jsx")
    _REC = _read(_FRONTEND, "src", "components", "PendingPipelineRecovery.jsx")


# [P2-LOADING-ETA-HONEST · 2026-09-03] El rango ya NO es una cifra fija: lo sirve el backend
# (`GET /api/plans/generation-eta`, p50/p90 reales de 14 días) y el copy es adaptativo. Todo rango
# fijo que la pantalla prometió alguna vez (incluido el «3 y 6» que esta versión anclaba) queda
# retirado: dos copias del mismo dato en la misma pantalla es la forma en que el usuario lee una
# y el contador le enseña otra.
_RANGOS_RETIRADOS = ("4 y 5 minutos", "4-5 minutos", "5 y 7 minutos", "5-7 minutos",
                     "9 y 10 minutos", "9-10 minutos", "3 y 6 minutos", "3-6 minutos")
_COPY_P50 = "Normalmente tarda unos {p50} minutos; 9 de cada 10 planes están listos antes de {p90}."
_COPY_P90 = "Ya pasamos la marca habitual; casi todos los planes terminan antes de {p90} minutos."


def test_el_rango_vivo_viene_del_backend_no_de_una_cifra_fija():
    assert "/api/plans/generation-eta" in _PLAN, "el ETA lo sirve el backend (p50/p90 reales)"
    assert _COPY_P50 in _PLAN and _COPY_P90 in _PLAN, "el copy es adaptativo (p50 / p90)"
    # Solo CÓDIGO: los comentarios de Plan.jsx cuentan la historia de los rangos retirados.
    codigo = re.sub(r"//[^\n]*", "", _PLAN)
    for viejo in _RANGOS_RETIRADOS:
        assert viejo not in codigo, (
            f"El rango fijo `{viejo}` volvió a Plan.jsx. Una cifra fija envejece en semanas "
            "(P2-LOADING-ETA-HONEST): el tiempo lo pone el backend."
        )


def test_los_umbrales_no_contradicen_al_estimado():
    """«Ya pasamos la marca habitual» arranca en el p90 REAL, no en un literal en minutos.
    Antes aquí había umbrales `elapsedSec < N * 60` que sobrevivieron a dos cambios de rango."""
    i = _PLAN.index("const timeMessage")
    win = _PLAN[max(0, i - 600):i + 1200]
    assert "const pastP90 = !!etaMin && elapsedSec >= etaMin.p90 * 60;" in win
    assert "elapsedSec < etaMin.p50 * 60" in win
    assert re.search(r"elapsedSec [<>]=? \d+ \* 60", win) is None, (
        "umbral literal en minutos dentro de timeMessage: el estimado y el aviso volverían a "
        "poder contradecirse"
    )


def test_elapsed_continuity_across_reentry():
    assert "P2-LOADING-ETA-57" in _PLAN
    # [reapuntado 2026-07-28] startTimeRef → useState lazy-init (P2-LINT-ZERO).
    i = _PLAN.index("const [startTime] = useState(")
    win = _PLAN[i:i + 600]
    assert "mealfit_plan_in_progress" in win and "started_at" in win, (
        "en modo recovery el contador arranca del inicio REAL del pipeline (no 0:00)"
    )
    assert "6 * 3600 * 1000" in win, "sanity: started_at más viejo que el cap del recovery → hoy"


def test_reentry_redirects_already_wired():
    """El 'Puedes salir si quieres' está respaldado end-to-end (audit)."""
    assert "<PendingPipelineRecovery />" in _APP, "recovery montado global"
    assert "status.status === 'generating'" in _REC and "navigate('/plan'" in _REC, (
        "generando + fuera de /plan → de vuelta a la pantalla de carga"
    )
    assert "status.status === 'complete'" in _REC and "navigate('/dashboard'" in _REC, (
        "completo → dashboard con el plan nuevo"
    )
    assert re.search(r"pending-status/ack", _REC), "ack idempotente (sin loops de redirect)"
    assert "visibilitychange" in _REC, "resume inmediato al volver de suspend"
