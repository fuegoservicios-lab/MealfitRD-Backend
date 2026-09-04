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


# El rango que la pantalla promete HOY. Es lo único que hay que editar cuando el
# owner vuelva a moverlo; todo lo demás de este fichero se deriva de aquí.
# [P2-LOADING-ETA-HONEST · 2026-09-03] El rango FIJO desapareció: la pantalla lee el p50/p90
# real del bloque 1 y solo conserva UN literal, el fallback prudente cuando el ETA no llega
# («Suele tardar entre 5 y 15 minutos»). Es el único rango que la copia promete a mano.
_ETA_MIN, _ETA_MAX = 5, 15

# Rangos que la pantalla prometió antes. Ninguno puede seguir vivo: dos copias
# distintas del mismo dato en la misma pantalla es la forma en que el usuario lee
# una y el contador le enseña otra.
_RANGOS_RETIRADOS = ("4 y 5 minutos", "4-5 minutos", "5 y 7 minutos", "5-7 minutos",
                     "9 y 10 minutos", "9-10 minutos", "3 y 6 minutos", "3-6 minutos")


def test_el_rango_vivo_es_el_que_el_owner_decidio():
    assert f"entre {_ETA_MIN} y {_ETA_MAX} minutos" in _PLAN, (
        f"La copia inicial ya no dice «entre {_ETA_MIN} y {_ETA_MAX} minutos». Si el owner "
        "movió el rango, actualiza `_ETA_MIN`/`_ETA_MAX` arriba y añade el rango "
        "anterior a `_RANGOS_RETIRADOS`."
    )
    # [P2-LOADING-ETA-HONEST] el contador ya NO repite un rango a mano: cuando llega el ETA
    # real, las frases interpolan `{p50}`/`{p90}` — un solo dato, cero copias.
    assert "estimado " not in _PLAN.split("const timeMessage")[1][:1200], (
        "El contador volvió a anunciar un rango escrito a mano junto a la copia inicial. "
        "Es el mismo dato en dos frases: el usuario lee una y luego ve la otra."
    )
    assert "{p50}" in _PLAN and "{p90}" in _PLAN, (
        "Las frases con ETA real deben interpolar p50/p90 del backend, no cifras fijas."
    )
    # [P0-CI-VERDICT] se mira el CÓDIGO: el comentario de P2-LOADING-ETA-HONEST cita «estimado 3-6
    # minutos» precisamente para contar por qué se retiró, y eso no es una promesa en pantalla.
    _codigo = re.sub(r"/\*.*?\*/", "", _PLAN, flags=re.S)
    _codigo = "\n".join(l for l in _codigo.splitlines() if not l.lstrip().startswith("//"))
    for viejo in _RANGOS_RETIRADOS:
        assert viejo not in _codigo, (
            f"El rango retirado `{viejo}` sigue en Plan.jsx. Un rango viejo superviviente "
            "es una promesa que la pantalla ya no cumple."
        )


def test_los_umbrales_no_contradicen_al_estimado():
    """«Ya casi terminamos» tiene que empezar DESPUÉS del estimado, no dentro.

    Este es el invariante de verdad, y por eso se comprueba CONTRA el rango en vez
    de contra un número escrito a mano: decir «ya casi» a los 6 minutos mientras la
    pantalla promete «3 a 9» es contradecirse a sí misma en la misma vista. Antes
    aquí había un `10 * 60` literal que sobrevivió a dos cambios de rango.
    """
    i = _PLAN.index("const timeMessage")
    win = _PLAN[i:i + 1200]

    # [P2-LOADING-ETA-HONEST · 2026-09-03] Los umbrales ya no son literales: se DERIVAN del
    # mismo ETA que la copia enseña (p50 y p90 del bloque 1), así que no pueden contradecirla
    # por construcción. Lo que se ancla ahora es exactamente eso: ninguna cifra a mano en las
    # bandas, y las dos marcas presentes (primero el p50, después el p90 vía `pastP90`).
    literales = [int(m) for m in re.findall(r"elapsedSec < (\d+) \* 60", win)]
    assert literales == [], (
        f"Volvió un umbral escrito a mano ({literales}) en `timeMessage`. Las bandas se "
        "derivan de `etaMin.p50`/`etaMin.p90`; una cifra fija se desincroniza del ETA real."
    )
    assert "elapsedSec < etaMin.p50 * 60" in win, (
        "Esperaba la banda derivada del p50 (`elapsedSec < etaMin.p50 * 60`) en `timeMessage`. "
        "Si cambió la forma, enséñasela a este test."
    )
    assert "elapsedSec >= etaMin.p90 * 60" in _PLAN and "pastP90" in win, (
        "La banda del p90 (`pastP90 = elapsedSec >= etaMin.p90 * 60`) debe existir y usarse "
        "en `timeMessage` después de la del p50."
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
