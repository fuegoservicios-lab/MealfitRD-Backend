"""[P2-DEGRADED-HEADLINE-TRUTH · 2026-07-31] El banner de calidad degradada
dejaba de acusar a la IA cuando la IA no falló.

CASO REAL (plan d476023a, cuenta gratis del owner): el revisor devolvió
"APROBADO" en el intento #1, la calidad holística fue 0.925 con `retry=1.00` y
`review=1.00`, y aun así el banner decía "La IA no logró un plan óptimo tras 1
intento". Lo que había pasado era otra cosa: una auditoría POSTERIOR al visto
bueno (`P2-PANEL-SOFT-REJECT`) marcó el plan por `micro_worst_day_ceiling` —
el Día 1 se pasó del techo de sodio (gouda en desayuno + almuerzo, camarones de
cena). Acusar al motor de un fallo que no cometió confunde el diagnóstico (yo
mismo leí el banner y busqué el fallo en el sitio equivocado) y erosiona la
confianza en el producto sin motivo.

Los motivos son de dos familias:
  A) La IA no convergió de verdad → agotó intentos / presupuesto / contexto.
  B) La IA entregó y el revisor APROBÓ → una auditoría marcó un detalle.
Sólo la familia A merece el titular de agotamiento.

Parser-based sobre el JSX (no hay runner de JS en CI): ancla la REGLA — que el
titular se derive del motivo y no esté hardcodeado en la vista.

[P1-I18N-DASHBOARD · 2026-08-15] El dashboard pasó a multiidioma y el copy quedó
envuelto en `t()`/`tn()`. Nada de lo que este test vigila cambió de CONDUCTA —
verificado ejecutando los helpers con stubs es-DO: los 14 motivos siguen en el
mapa, la familia B sigue saliendo "Plan listo, con un aviso" con 3 intentos, y
`motivo_desconocido` con 3 intentos sigue reconociendo el agotamiento. Lo que
cambió es la GRAFÍA, en dos sitios:

  1. `const Q_DEGRADED_REASON_MAP = {…}` → `const getQDegradedReasonMap = () => ({…})`.
     Obligado: un `t()` en ámbito de módulo se evalúa ANTES de que el catálogo
     cargue y congela el copy en español para siempre. El mapa tenía que ser
     función. Los tests que citaban el nombre de la constante se reanclan a
     AMBAS formas — y, mejor, al CONTENIDO (las 14 claves de motivo).
  2. El titular plural pasó de un template con ternario (`${n} intento${…}`) a
     `tn(n, '…{n} intento', '…{n} intentos')`, que necesita las DOS formas
     escritas. Por eso la frase acusatoria aparece 3 veces donde antes aparecía
     2. Contar ocurrencias globales medía la grafía; lo que la regla dice de
     verdad es «esa frase no se escribe en la VISTA», así que ahora se cuenta
     FUERA del helper y el umbral es cero — más estricto y ciego al pluralizador.
"""
from __future__ import annotations

import pytest

import re
from pathlib import Path

@pytest.fixture(scope="module", autouse=True)
def _load_frontend_sibling_sources(frontend_repo_path):
    # La fixture compartida salta el módulo antes de cualquier I/O si falta el hermano.
    _ = frontend_repo_path
    global _DASH
    _DASH = (
        Path(__file__).resolve().parent.parent.parent
        / "frontend" / "src" / "pages" / "Dashboard.jsx"
    ).read_text(encoding="utf-8")


_FRASE_ACUSATORIA = "La IA no logró un plan óptimo"

# Los 14 motivos que el backend emite en `_quality_degraded_reason` y que el mapa
# traduce a copy accionable. Se anclan por NOMBRE porque son el contrato con
# `graph_orchestrator.py` / `_maybe_mark_*_degraded`: perder uno devuelve al
# usuario al genérico "Calidad por debajo del óptimo" (el bug de
# P3-BANNER-REASON-COPY), y además lo saca de `conocido` → un motivo de auditoría
# con 3 intentos volvería a acusar a la IA.
_MOTIVOS_CANONICOS = (
    # familia A — la IA no convergió
    "high_contextual", "max_attempts", "invalid_pipeline_start", "budget_exhausted",
    # familia B — el revisor aprobó y una auditoría posterior marcó un detalle
    "low_band_score", "condition_panel_gap", "low_micros", "high_sodium_sugar",
    "shopping_list_incomplete", "clinical_layer_incomplete",
    "composite_dish_unresolved", "slot_coherence_unresolved",
    "micro_worst_day_ceiling", "micro_worst_day",
)


def _sin_comentarios(src: str) -> str:
    """Se audita lo que EJECUTA, no la prosa de al lado: los comentarios de este
    arreglo citan la frase acusatoria y hacían fallar al test contra su propio fix."""
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    return re.sub(r"^\s*//.*$", "", src, flags=re.MULTILINE)


def _cuerpo_del_mapa_de_motivos() -> str:
    """Devuelve el cuerpo del mapa motivo→copy, sea constante o getter.

    [P1-I18N-DASHBOARD] Acepta las dos declaraciones porque la propiedad vigilada
    es el CONTENIDO (qué motivos tienen copy propio), no cómo se declara el mapa.
    """
    m = re.search(
        r"const\s+(?:"
        r"Q_DEGRADED_REASON_MAP\s*=\s*\{"            # pre-i18n: constante
        r"|getQDegradedReasonMap\s*=\s*\([^)]*\)\s*=>\s*\(\{"   # post-i18n: getter
        r")",
        _DASH,
    )
    assert m, (
        "no se encuentra el mapa de motivos degradados (ni `Q_DEGRADED_REASON_MAP` "
        "ni `getQDegradedReasonMap`) — sin él el banner no puede decir POR QUÉ"
    )
    fin = re.search(r"^\}\)?;", _DASH[m.end():], re.MULTILINE)
    assert fin, "el mapa de motivos no cierra en columna 0 — parser desorientado"
    return _DASH[m.end(): m.end() + fin.start()]


def test_retry_exhaustion_family_is_explicit():
    m = re.search(r"Q_DEGRADED_RETRY_EXHAUSTION = new Set\(\[(.*?)\]\)", _DASH, re.DOTALL)
    assert m, "falta el set de motivos de agotamiento de intentos"
    cuerpo = m.group(1)
    for motivo in ("high_contextual", "max_attempts", "invalid_pipeline_start",
                   "budget_exhausted"):
        assert motivo in cuerpo, (
            f"'{motivo}' SÍ es agotamiento de la IA y debe llevar ese titular"
        )
    # Los de auditoría post-aprobación NO pueden estar aquí: el plan fue aprobado.
    for motivo in ("micro_worst_day_ceiling", "micro_worst_day", "low_band_score",
                   "high_sodium_sugar", "low_micros"):
        assert motivo not in cuerpo, (
            f"'{motivo}' lo marca una auditoría DESPUÉS de aprobar — no es un "
            f"fallo de convergencia de la IA"
        )


def test_headline_is_derived_not_hardcoded():
    assert "export function resolveQualityDegradedHeadline" in _DASH
    # La frase acusatoria vive SOLO dentro del helper (que decide cuándo aplica).
    codigo = _sin_comentarios(_DASH)
    # [P1-I18N-DASHBOARD · 2026-08-15] Se cuenta FUERA del helper, con umbral CERO,
    # en vez de acotar el total. El total dependía del pluralizador: `tn()` exige
    # las dos formas escritas ("…tras {n} intento" / "…tras {n} intentos") donde el
    # template anterior interpolaba una sola. Ese conteo medía la grafía; la regla
    # de P2-DEGRADED-HEADLINE-TRUTH es «el titular se DERIVA del motivo y nunca se
    # escribe en la vista», y eso es exactamente lo que mide la ventana de fuera.
    i = codigo.index("export function resolveQualityDegradedHeadline")
    j = codigo.index("export function resolveQualityDegradedLabel", i)
    dentro, fuera = codigo[i:j], codigo[:i] + codigo[j:]

    assert fuera.count(_FRASE_ACUSATORIA) == 0, (
        f"'{_FRASE_ACUSATORIA}' aparece {fuera.count(_FRASE_ACUSATORIA)} vez/veces "
        f"en código FUERA de resolveQualityDegradedHeadline — el titular acusatorio "
        f"sólo puede nacer donde se decide si aplica; escrito en la vista vuelve a "
        f"culpar a la IA de un fallo que el revisor ya había aprobado"
    )
    assert dentro.count(_FRASE_ACUSATORIA) >= 1, (
        "el helper ya no contiene el titular de agotamiento — si de verdad se "
        "eliminó el encuadre, este test y su motivo (familia A) hay que rehacerlos"
    )
    # Y la vista debe llamar al helper, no componer el texto.
    assert "resolveQualityDegradedHeadline(" in _DASH.split("export function resolveQualityDegradedHeadline")[1], (
        "el banner debe consumir el helper"
    )


def test_unknown_reason_net_is_not_inert():
    """La red para motivos NUEVOS (sin clasificar) se mide contra el MAPA.
    Primer intento usaba `resolveQualityDegradedLabel`, que jamás devuelve null
    para un motivo con texto (cae a un genérico) — la red quedaba INERTE y sólo
    se supo EJECUTÁNDOLA: `motivo_desconocido` con 3 intentos salía como "Plan
    listo, con un aviso" en vez de reconocer el agotamiento."""
    # Ventana anclada a la ESTRUCTURA (hasta la siguiente función exportada), no a
    # un conteo de bytes: con `i + 1800` la ventana se derramaba dentro de
    # `resolveQualityDegradedLabel`, cuya propia firma contiene la cadena que este
    # test prohíbe — el test se acusaba a sí mismo.
    i = _DASH.index("export function resolveQualityDegradedHeadline")
    j = _DASH.index("export function resolveQualityDegradedLabel", i)
    win = _DASH[i:j]
    # [P1-I18N-DASHBOARD · 2026-08-15] Dos grafías del MISMO lookup: el mapa dejó de
    # ser constante para ser getter (un `t()` en ámbito de módulo se congelaría en
    # español). Lo vigilado no es el nombre sino CONTRA QUÉ se mide la pertenencia:
    # el mapa explícito, jamás el resolver.
    assert re.search(
        r"hasOwnProperty\.call\(\s*(?:Q_DEGRADED_REASON_MAP|getQDegradedReasonMap\(\))\s*,\s*reason\s*\)",
        win,
    ), (
        "la pertenencia debe consultarse contra el mapa explícito "
        "(`Q_DEGRADED_REASON_MAP` o `getQDegradedReasonMap()`)"
    )
    assert "resolveQualityDegradedLabel(reason)" not in win, (
        "usar el resolver aquí deja la red inerte: nunca devuelve null"
    )
    # El prefijo dinámico también cuenta como "conocido" (P3-BANNER-REASON-COPY):
    # sin esta rama, `low_band_macro:carbs` con 3 intentos volvería a acusar a la IA.
    assert "low_band_macro:" in win, (
        "el prefijo dinámico `low_band_macro:` debe seguir contando como motivo "
        "conocido — es exact-match miss en el mapa por construcción"
    )


def test_reason_map_keeps_every_reason_after_the_i18n_wrapper():
    """[P1-I18N-DASHBOARD · 2026-08-15] El mapa es la MEDIDA de «motivo conocido».

    Se ancla al CONTENIDO (las 14 claves y que su valor siga siendo copy español),
    no a la declaración: la migración lo convirtió de constante en getter y un test
    atado a `const Q_DEGRADED_REASON_MAP = {` habría bloqueado un cambio obligado.

    Perder una clave tiene DOS efectos, y el segundo es el que este archivo existe
    para impedir: (a) el usuario vuelve al genérico "Calidad por debajo del óptimo"
    sin saber qué pasó, y (b) ese motivo deja de ser `conocido`, así que con ≥2
    intentos el banner vuelve a acusar a la IA de no converger — justo el bug del
    plan d476023a.
    """
    cuerpo = _cuerpo_del_mapa_de_motivos()
    claves = set(re.findall(r"^\s+(\w+):", cuerpo, re.MULTILINE))
    faltan = [m for m in _MOTIVOS_CANONICOS if m not in claves]
    assert not faltan, (
        f"motivos sin copy propio en el mapa: {faltan} — caen al genérico Y dejan "
        f"de ser 'conocidos', así que con ≥2 intentos el banner los acusa de "
        f"agotamiento de la IA"
    )
    # Y el valor sigue siendo la FRASE española, envuelta o no en `t()`. La clave del
    # motor de i18n ES el texto español (no hay catálogo es-DO): sustituirla por un
    # identificador tipo `t('banner.high_contextual')` dejaría al usuario dominicano
    # leyendo la clave en crudo.
    for motivo in _MOTIVOS_CANONICOS:
        m = re.search(
            rf"^\s+{motivo}:\s*(?:t\(\s*)?'([^']+)'",
            cuerpo,
            re.MULTILINE,
        )
        assert m, f"el copy de '{motivo}' no es un literal español (¿clave opaca?)"
        copy = m.group(1)
        assert len(copy) >= 20 and " " in copy, (
            f"'{motivo}' → {copy!r}: el mapa debe guardar la frase española "
            f"completa, no un identificador"
        )


def test_notification_shares_the_same_source():
    """La notificación decía 'Plan no óptimo (1 intento)' con el mismo defecto:
    dos superficies contando la misma historia deben leer del mismo sitio.

    [P1-I18N-DASHBOARD · 2026-08-15] Dos aprietes, ambos descubiertos MUTANDO:

      · La ventana era `i + 2000` bytes. La envoltura `t()`/`tn()` alarga el
        bloque sin cambiar su conducta, así que un conteo de bytes es un anclaje
        que caduca solo (es la misma trampa que el test de la red documenta).
        Ahora la ventana llega hasta el cierre del `useCallback`.
      · `"_head.exhausted" in win` era PASSABLE: al quitarle el gate al TÍTULO
        el test seguía verde, porque `_head.exhausted` sobrevive en `guidance`
        unas líneas más abajo. O sea que vigilaba la presencia de un símbolo, no
        la propiedad — y la propiedad es justo la del bug: el TÍTULO de la
        notificación no puede afirmar agotamiento sin que el helper lo confirme.
    """
    i = _DASH.index("const buildQualityNotification")
    fin = _DASH.index("\n    }, [", i)
    win = _DASH[i:fin]
    assert "resolveQualityDegradedHeadline(" in win, (
        "la notificación debe derivar su título del mismo helper que el banner"
    )
    assert re.search(r"title:\s*_head\.exhausted\s*\?", win), (
        "el título de agotamiento sólo aplica si el helper lo confirma: "
        "`title: _head.exhausted ? <agotamiento> : <aviso>`. Un título "
        "incondicional vuelve a decir 'Plan no óptimo (1 intento)' con el "
        "revisor habiendo APROBADO"
    )
