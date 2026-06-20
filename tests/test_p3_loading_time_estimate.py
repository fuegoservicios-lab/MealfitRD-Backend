"""[P3-LOADING-TIME-ESTIMATE · 2026-05-16] LoadingScreen muestra tiempo
transcurrido + estimado adaptativo.

Antes: el usuario veía solo "Diseñando tu plan" sin idea de duración →
ansiedad + cancelación prematura (especialmente cuando el plan tardaba
4-5 min normal). Ahora el copy evoluciona:
  - <30s: estimación pura ("3 a 5 minutos")
  - 30s-5min: tracking ("Transcurrido X:XX · estimado 3-5 min")
  - 5-10min: warning suave ("tardando un poco más de lo normal")
  - >10min: mensaje de paciencia ("ya casi · gracias por tu paciencia")

El contador usa `useRef` para que el start time NO se reinicie en re-renders
del componente (que ocurren a cada cambio de streamPhase/daysCompleted).
"""
from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PLAN_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Plan.jsx"


def test_start_time_uses_useref_for_stable_anchor():
    """`startTimeRef = useRef(Date.now())` debe estar dentro de LoadingScreen.
    `useRef` es crítico aquí: re-renders por cambios de status/streamPhase
    NO deben reiniciar el contador."""
    src = _PLAN_JSX.read_text(encoding="utf-8")
    # Aislar LoadingScreen
    loading_block = re.search(
        r"const LoadingScreen = \(\{.*?\n\};",
        src,
        re.DOTALL,
    )
    assert loading_block, "LoadingScreen component no encontrado."
    body = loading_block.group(0)
    assert "startTimeRef = useRef(Date.now())" in body, (
        "startTimeRef debe usar `useRef(Date.now())` — useState reiniciaría "
        "el anchor en cada re-render por cambios de props."
    )


def test_elapsed_state_and_timer_present():
    """Estado `elapsedSec` + setInterval(1000ms) para actualizar el contador."""
    src = _PLAN_JSX.read_text(encoding="utf-8")
    assert "const [elapsedSec, setElapsedSec] = useState(0);" in src
    # setInterval con 1000ms (1 update/seg) — más rápido sería ruido,
    # más lento se siente "stuck".
    assert re.search(
        r"setInterval\(\s*\(\s*\)\s*=>\s*\{\s*setElapsedSec",
        src,
    ), "Timer setInterval para elapsedSec no encontrado."
    # Cleanup del interval (return clearInterval) — sin esto leak de timers
    # en cada re-render de LoadingScreen.
    loading_block = re.search(
        r"const LoadingScreen = \(\{.*?\n\};",
        src,
        re.DOTALL,
    )
    body = loading_block.group(0)
    assert body.count("clearInterval") >= 2, (
        "LoadingScreen debe limpiar TODOS los intervals (timer del elapsed "
        "+ progress timer + tips timer). Al menos 2 clearInterval esperados."
    )


def test_timer_pauses_when_status_ready():
    """El timer del elapsed counter NO debe seguir corriendo cuando el plan
    terminó (`status === 'ready'`). Sin el guard, el contador sigue ticando
    aunque ya estés en preview/dashboard."""
    src = _PLAN_JSX.read_text(encoding="utf-8")
    # Buscar el effect con el guard y setEelapsedSec
    effect_block = re.search(
        r"useEffect\(\(\)\s*=>\s*\{\s*if\s*\(status\s*===\s*['\"]ready['\"]\)\s*return\s*undefined;\s*const\s+t\s*=\s*setInterval[^}]+setElapsedSec",
        src,
        re.DOTALL,
    )
    assert effect_block, (
        "El useEffect del timer debe tener guard `if (status === 'ready') "
        "return undefined` al inicio para pausar al terminar el plan."
    )


def test_adaptive_copy_four_phases():
    """El copy debe tener 4 fases según `elapsedSec`. Calibrado a la duración
    REAL de prod en la era DeepSeek V4 (medido 2026-06-17): ~3-5 min pipeline
    completo (skeleton ~22s + day_gen paralelo ~30s + self-critique ~2min +
    reviewer + assembly). Rango honesto = 4-5 min.

    Verificamos los 4 strings literales esperados."""
    src = _PLAN_JSX.read_text(encoding="utf-8")
    expected_phrases = [
        "Esto suele tomar entre 4 y 5 minutos",                   # <30s
        "estimado 4-5 minutos",                                   # 30s-6min
        "ya casi terminamos, espera un poco más",                # 6-10min
        "gracias por tu paciencia",                              # >10min
    ]
    for phrase in expected_phrases:
        assert phrase in src, (
            f"Copy adaptativo incompleto — falta la frase de transición: "
            f"`{phrase!r}`."
        )


def test_estimate_calibrated_to_realistic_duration():
    """Regression guard (invertido 2026-06-17): el estimate NO debe regresar al
    rango STALE de la era Gemini ("10 y 15 minutos" / "10-15 minutos"). Con
    DeepSeek V4 el pipeline tarda ~3-5 min (medido en prod 2026-06-17, logs de
    generaciones reales). Mostrar 10-15 min hacía esperar 2-3x lo real → el
    usuario asumía que se colgó o cancelaba innecesariamente."""
    src = _PLAN_JSX.read_text(encoding="utf-8")
    forbidden_phrases = [
        "entre 10 y 15 minutos",
        "estimado 10-15 minutos",
        "5-10 minutos",
    ]
    for phrase in forbidden_phrases:
        assert phrase not in src, (
            f"Calibración stale (era Gemini) detectada: `{phrase!r}` en Plan.jsx. "
            f"La duración real con DeepSeek es ~4-5 min. No revertir al rango 10-15."
        )


def test_format_elapsed_supports_minutes_and_hours():
    """`formatElapsed` debe soportar tanto MM:SS (default) como H:MM:SS
    para casos de planes >1h (retries múltiples + Pro escalation)."""
    src = _PLAN_JSX.read_text(encoding="utf-8")
    fn_match = re.search(
        r"const formatElapsed = \(sec\) => \{.*?\};",
        src,
        re.DOTALL,
    )
    assert fn_match, "Helper `formatElapsed` no encontrado."
    body = fn_match.group(0)
    # Debe contemplar el caso h > 0 (hora)
    assert "h > 0" in body, (
        "formatElapsed no maneja el caso de horas — planes que se atascan "
        ">1h verán MM:SS overflow (e.g. 90:00 en vez de 1:30:00)."
    )


def test_time_message_rendered_in_jsx():
    """El estado computado `timeMessage` debe estar referenciado en el JSX
    (no quedar como dead variable)."""
    src = _PLAN_JSX.read_text(encoding="utf-8")
    # Una mención en const + otra en JSX expr `{timeMessage}`
    assert "{timeMessage}" in src, (
        "`timeMessage` computado pero no renderizado en JSX — dead code."
    )


def test_div_uses_tabular_nums_and_aria_live():
    """Defense in depth UX:
    - `fontVariantNumeric: 'tabular-nums'` evita jitter visual cuando los
      dígitos cambian (1:09 → 1:10 con ancho variable).
    - `aria-live=polite` para screen readers (accesibilidad)."""
    src = _PLAN_JSX.read_text(encoding="utf-8")
    # Localizar el div del timeMessage
    timemsg_block = re.search(
        r"<div[^>]*aria-live=\"polite\"[^>]*>\s*\{timeMessage\}",
        src,
        re.DOTALL,
    )
    assert timemsg_block, (
        "El div que renderiza timeMessage debe tener `aria-live=\"polite\"` "
        "(accesibilidad)."
    )
    assert "fontVariantNumeric: 'tabular-nums'" in src, (
        "tabular-nums ausente — los dígitos del contador jitterean al "
        "cambiar de ancho (e.g. 0:09 → 0:10)."
    )
