"""[P1-LANDING-OBS-PAPER · 2026-08-14] La política de observabilidad del landing
tiene que estar CONECTADA, y la Política de Privacidad tiene que decir la verdad.

DOS COSAS QUE ESTE TEST VIGILA Y EL DE VITEST NO PUEDE.

1. EL CABLEADO. `utils/observabilityScope.js` puede ser correcto y estar sin usar
   — es el modo de fallo «feature inerte» que este repo ya pagó más de una vez
   (un wire-format que nadie leía, un knob que no ahorraba bytes). El contrato de
   Vitest prueba que la política DECIDE bien; esto prueba que alguien le
   pregunta. Los tres consumidores son `main.jsx` (Sentry), `posthogClient.js`
   (autocapture) y `Settings.jsx` (escritura del opt-out en los dos soportes).

2. EL TEXTO LEGAL. `LegalPages.jsx` afirmaba categóricamente «No utilizamos
   cookies de publicidad, marketing **ni rastreadores de terceros**» mientras
   PostHog corría con `persistence: 'localStorage+cookie'` sobre el visitante
   anónimo. Y faltaba de las tres listas donde un encargado de tratamiento debe
   aparecer: proveedores subcontratados (§8), transferencias internacionales
   (§12) y almacenamientos (§13). Una afirmación falsificable en diez segundos
   con las DevTools, en la página que un usuario prudente lee ANTES de entregar
   su perfil clínico.

   Ojo con el matiz que hay que preservar: la frase de §7 sobre «Google
   Analytics, Mixpanel, Facebook Pixel ni rastreador publicitario» ES cierta y no
   debe borrarse. Lo que era falso es negar la categoría entera de terceros.

Tooltip-anchor: P1-LANDING-OBS-PAPER
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _REPO_ROOT / "frontend" / "src"
_MAIN_JSX = _SRC / "main.jsx"
_POSTHOG = _SRC / "utils" / "posthogClient.js"
_SETTINGS = _SRC / "pages" / "Settings.jsx"
_SCOPE = _SRC / "utils" / "observabilityScope.js"
_LEGAL = _SRC / "pages" / "legal" / "LegalPages.jsx"


def _read(path: Path) -> str:
    if not path.exists():
        pytest.fail(f"[P1-LANDING-OBS-PAPER] No existe {path.relative_to(_REPO_ROOT)}")
    return path.read_text(encoding="utf-8")


def _privacy_section(texto: str) -> str:
    """El cuerpo del componente `Privacy`, para no medir contra los Términos."""
    inicio = texto.find('title="Política de Privacidad"')
    if inicio == -1:
        pytest.fail("[P1-LANDING-OBS-PAPER] No se encontró el componente `Privacy`.")
    fin = texto.find('title="Términos de Servicio"', inicio)
    return texto[inicio: fin if fin != -1 else len(texto)]


# ---------------------------------------------------------------------------
# 1. La política existe y está CONECTADA a sus tres consumidores
# ---------------------------------------------------------------------------

def test_la_politica_de_alcance_existe():
    contenido = _read(_SCOPE)
    for exportado in ("isMarketingVisit", "shouldAttachSentryReplay", "posthogCaptureOptions"):
        assert f"export const {exportado}" in contenido, (
            f"[P1-LANDING-OBS-PAPER] `observabilityScope.js` ya no exporta `{exportado}`."
        )


def test_sentry_replay_gateado_por_la_politica():
    """El chunk de replay (~118 kB gz) no debe descargarse en el landing."""
    contenido = _read(_MAIN_JSX)
    assert "shouldAttachSentryReplay" in contenido, (
        "[P1-LANDING-OBS-PAPER] `main.jsx` no consulta `shouldAttachSentryReplay`.\n"
        "Sin ese gate, `_attachSentryIntegrations` vuelve a hacer "
        "`await import('@sentry/react')` en la portada y se descargan los 357.767 B "
        "del namespace completo (browserTracing + replay + feedback + replay-canvas) "
        "para grabar la sesión de alguien que sólo está leyendo marketing.\n"
        "⚠️ Bajar `VITE_SENTRY_REPLAYS_SESSION_RATE` a 0 NO sustituye a esto: regula "
        "la ingesta, no los bytes — el chunk se descarga igual."
    )


def test_posthog_toma_sus_opciones_de_la_politica():
    contenido = _read(_POSTHOG)
    assert "posthogCaptureOptions" in contenido, (
        "[P1-LANDING-OBS-PAPER] `posthogClient.js` no usa `posthogCaptureOptions`."
    )
    assert not re.search(r"autocapture\s*:\s*true", contenido), (
        "[P1-LANDING-OBS-PAPER] `posthogClient.js` volvió a fijar `autocapture: true` "
        "a mano. En el apex eso captura cada click y cada campo de un visitante sin "
        "cuenta. La decisión pertenece a `posthogCaptureOptions(hostname)`, que la "
        "conserva encendida dentro de la app."
    )


def test_el_opt_out_se_escribe_por_el_ssot():
    """Escribir sólo en localStorage es el bug: el apex no puede leerlo."""
    contenido = _read(_SETTINGS)
    assert "persistAnalyticsOptOut" in contenido, (
        "[P1-LANDING-OBS-PAPER] Configuración no usa `persistAnalyticsOptOut`.\n"
        "Escribir el opt-out sólo en `localStorage` lo deja invisible para el apex "
        "(el almacenamiento es POR ORIGEN), que es exactamente el fallo que este "
        "P-fix cierra: el usuario apaga la analítica y la portada lo sigue rastreando."
    )
    assert not re.search(
        r"safeLocalStorageSet\(\s*ANALYTICS_OPT_OUT_KEY", contenido
    ), (
        "[P1-LANDING-OBS-PAPER] Configuración vuelve a escribir la clave del opt-out "
        "a pelo. Tiene que pasar por `persistAnalyticsOptOut`, que deja coherentes "
        "los dos soportes (localStorage + cookie de dominio)."
    )


# ---------------------------------------------------------------------------
# 2. La Política de Privacidad declara PostHog donde toca
# ---------------------------------------------------------------------------

def test_privacidad_no_niega_los_rastreadores_de_terceros():
    privacidad = _privacy_section(_read(_LEGAL))
    assert "ni rastreadores de terceros" not in privacidad, (
        "[P1-LANDING-OBS-PAPER] La Política de Privacidad vuelve a negar "
        "categóricamente los «rastreadores de terceros» mientras PostHog corre con "
        "`persistence: 'localStorage+cookie'`.\n"
        "Es falsificable en 10 segundos abriendo las DevTools, en el documento que "
        "un usuario lee antes de entregar su perfil de salud.\n"
        "NO borres a cambio la frase de §7 sobre Google Analytics / Mixpanel / "
        "Facebook Pixel: esa sí es cierta."
    )


@pytest.mark.parametrize(
    "seccion,ancla",
    [
        ("§8 Proveedores Subcontratados", "Encargados de Tratamiento"),
        ("§12 Transferencias Internacionales", "Transferencias Internacionales"),
        ("§13 Cookies y Almacenamiento Local", "Cookies y Almacenamiento Local"),
    ],
)
def test_posthog_declarado_en_las_tres_listas(seccion, ancla):
    privacidad = _privacy_section(_read(_LEGAL))
    inicio = privacidad.find(ancla)
    assert inicio != -1, f"[P1-LANDING-OBS-PAPER] No se encontró la sección {seccion}."
    # Hasta el siguiente <h3> — el alcance de la sección.
    siguiente = privacidad.find("<h3>", inicio)
    cuerpo = privacidad[inicio: siguiente if siguiente != -1 else len(privacidad)]
    assert "PostHog" in cuerpo, (
        f"[P1-LANDING-OBS-PAPER] PostHog no aparece en {seccion}.\n"
        "Es un encargado de tratamiento que recibe datos de uso y almacena una "
        "cookie en el navegador del visitante: tiene que estar en la lista de "
        "proveedores, en la de transferencias fuera de RD (opera en EE. UU.) y en "
        "la tabla de almacenamientos."
    )


def test_la_fecha_de_privacidad_se_bumpeo():
    """Cambiar el fondo de una política sin mover su fecha la vuelve poco creíble."""
    texto = _read(_LEGAL)
    m = re.search(r'title="Política de Privacidad"\s+lastUpdated="([^"]+)"', texto)
    assert m, "[P1-LANDING-OBS-PAPER] No se pudo leer el `lastUpdated` de Privacidad."
    assert m.group(1) != "12 de Julio, 2026", (
        "[P1-LANDING-OBS-PAPER] La Política de Privacidad cambió de contenido "
        "(se declaró un encargado de tratamiento nuevo) pero conserva la fecha "
        f"«{m.group(1)}». §14 promete publicar los cambios con su nueva fecha."
    )
