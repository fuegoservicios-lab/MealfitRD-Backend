"""[P3-HIST-ACTIVE-NO-REACTIVATE · 2026-05-18] Ocultar "Reactivar este
Plan" en el modal del Historial cuando el plan abierto es el plan ACTIVO.

Síntoma del usuario:
> "en los planes activos el boton de reactivar plan no tiene sentido,
>  ese boton solo tiene sentido en planes que ya pasaron"

El botón "Reactivar este Plan" abre el flujo `/api/plans/restore` que
copia el plan archivado como el nuevo plan activo. Para un plan que YA
es el activo (today ∈ [grocery_start_date, +totalDays)), reactivarse
sobre sí mismo no tiene sentido conceptual — el usuario está literalmente
comiendo de ese plan ahora mismo.

Diseño:
  - El modal computa `getTemporalStatus(selectedPlan)` (helper
    introducido en P3-HIST-ACTIVE-CHIP · 2026-05-18) y aplica
    `_hideRestore = bucket in {'active', 'future'}`. Solo bucket
    `past` (o null para planes legacy sin start) muestra el CTA.
  - Modificador CSS `.modalFooterSingle` centra el botón "Cerrar"
    cuando se queda solo (sin el flex:2 vecino, quedaría pegado al
    borde izquierdo).
  - Copy de banners action_required y missingDaysReason pivota a
    "Vuelve al Dashboard…" cuando el botón se oculta — antes asumía
    "Pulsa 'Reactivar este Plan' abajo…", roto si el botón no existe.

Si alguien reintroduce el botón sin condicional (o quita el copy
condicional de los banners), este test falla con apuntador al marker.
"""
from __future__ import annotations

from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_HISTORY_JSX = (
    _BACKEND_ROOT.parent / "frontend" / "src" / "pages" / "History.jsx"
).read_text(encoding="utf-8")
_HISTORY_CSS = (
    _BACKEND_ROOT.parent / "frontend" / "src" / "pages" / "History.module.css"
).read_text(encoding="utf-8")
_APP_PY = (_BACKEND_ROOT / "app.py").read_text(encoding="utf-8")


def test_marker_present_in_source():
    """El marker `P3-HIST-ACTIVE-NO-REACTIVATE` DEBE permanecer en el
    código fuente del fix (History.jsx + History.module.css) como
    anchor de regresión. NO miramos `_LAST_KNOWN_PFIX` en app.py
    porque ese campo rota al siguiente P-fix mergeado en HEAD."""
    assert "P3-HIST-ACTIVE-NO-REACTIVATE" in _HISTORY_JSX, (
        "Marker P3-HIST-ACTIVE-NO-REACTIVATE ausente en History.jsx — "
        "un refactor podría reintroducir el botón en planes activos "
        "sin dejar trazo."
    )
    assert "P3-HIST-ACTIVE-NO-REACTIVATE" in _HISTORY_CSS, (
        "Marker P3-HIST-ACTIVE-NO-REACTIVATE ausente en "
        "History.module.css — el CSS .modalFooterSingle perdió su "
        "anchor."
    )


def test_modal_footer_hides_restore_button_for_active():
    """El motion.div del modal footer DEBE evaluar `getTemporalStatus`
    y suprimir el botón cuando bucket ∈ {active, future}."""
    # Anchor: el comentario marker en el modal footer.
    assert "P3-HIST-ACTIVE-NO-REACTIVATE" in _HISTORY_JSX, (
        "Marker P3-HIST-ACTIVE-NO-REACTIVATE ausente en History.jsx — "
        "un refactor podría borrar el fix y reintroducir el botón en "
        "planes activos."
    )
    # El bucket que oculta el botón debe cubrir 'active' Y 'future'
    # (no solo active — futures tampoco aplican).
    idx = _HISTORY_JSX.find("P3-HIST-ACTIVE-NO-REACTIVATE")
    assert idx > 0
    # Localizamos el bloque del modalFooter (cerca de los 1500 chars
    # siguientes al primer marker, donde está el footer).
    # Buscamos directamente la expresión `_hideRestore = !!(_t &&`.
    expr_idx = _HISTORY_JSX.find("_hideRestore = !!(_t && (_t.bucket === 'active' || _t.bucket === 'future'))")
    assert expr_idx > 0, (
        "Expresión `_hideRestore` ausente o con buckets distintos. "
        "Si solo cubre 'active', los planes futuros (raros, generados "
        "para empezar después) tampoco deben mostrar reactivar."
    )


def test_footer_returns_null_when_hidden():
    """[P3-HIST-NO-CERRAR-BTN · 2026-05-18] Cuando `_hideRestore=true`
    el footer se renderiza como `null` — la X del header cierra y el
    "Cerrar" del footer ya no existe. Sin esto, el modal mostraría
    una banda blanca vacía con padding 1.5rem en su pie (UX rara).

    Nota: hay 2 callsites de `_hideRestore = !!(_t && …)` en
    History.jsx — uno en el IIFE del banner action_required (línea
    ~2294) y otro en el IIFE del modal footer (línea ~4927). Solo el
    SEGUNDO debe tener el early-return null. Iteramos para localizar
    el del footer específicamente."""
    import re
    matches = list(re.finditer(
        r"_hideRestore = !!\(_t && \(_t\.bucket === 'active' \|\| _t\.bucket === 'future'\)\);",
        _HISTORY_JSX,
    ))
    assert len(matches) >= 2, (
        "Esperaba al menos 2 callsites de `_hideRestore = !!(_t && …)`: "
        "1 en el banner action_required (copy) + 1 en el modal footer "
        "(render del CTA). Encontrados: %d" % len(matches)
    )
    # El último match es el del modal footer (más cerca del final del
    # archivo). Verificamos el early-return null en su bloque.
    footer_idx = matches[-1].start()
    block = _HISTORY_JSX[footer_idx:footer_idx + 800]
    assert "if (_hideRestore)" in block and "return null" in block, (
        "El IIFE del modal footer NO retorna null cuando "
        "`_hideRestore=true`. Resultado: banda blanca vacía con "
        "padding visible en planes activos/futuros."
    )


def test_restore_button_present_when_not_hidden():
    """Para planes pasados el botón "Reactivar este Plan" sigue
    presente. Defensa: callsite único de `handleRestoreRequest` y
    SIN gate `!_hideRestore &&` (porque ahora el early-return `null`
    hace ese trabajo)."""
    handler_idx = _HISTORY_JSX.find("onClick={handleRestoreRequest}")
    assert handler_idx > 0, (
        "Botón Reactivar removido completamente — planes pasados "
        "perdieron el CTA de restore."
    )
    second = _HISTORY_JSX.find("onClick={handleRestoreRequest}", handler_idx + 1)
    assert second < 0, (
        "Hay más de un callsite `onClick={handleRestoreRequest}` — "
        "puede haber un botón duplicado."
    )


def test_no_cerrar_button_in_footer():
    """[P3-HIST-NO-CERRAR-BTN · 2026-05-18] El botón "Cerrar" se
    eliminó del modalFooter — la X del modalHeader hace lo mismo
    (mismo handler `setSelectedPlan(null)`). Mantener ambos era
    duplicación sin valor."""
    # El header tiene `.closeButton` con setSelectedPlan(null) +
    # icono X (lucide). NO debe haber un segundo botón "Cerrar"
    # textual en el footer.
    assert "P3-HIST-NO-CERRAR-BTN" in _HISTORY_JSX, (
        "Marker P3-HIST-NO-CERRAR-BTN ausente — un refactor podría "
        "reintroducir el botón Cerrar sin dejar trazo."
    )
    # `styles.modalCloseBtn` ya no se usa en JSX.
    assert "styles.modalCloseBtn" not in _HISTORY_JSX, (
        "`styles.modalCloseBtn` sigue referenciado en JSX — el botón "
        "Cerrar fue reintroducido en el footer (debe estar solo en la "
        "X del header)."
    )
    # El callsite del closeButton (la X) SÍ debe seguir existiendo.
    assert "styles.closeButton" in _HISTORY_JSX, (
        "La X del header (`styles.closeButton`) desapareció. Sin ella "
        "Y sin el botón Cerrar del footer, el modal solo se cierra "
        "haciendo click fuera — UX inaceptable."
    )


def test_modal_footer_single_modifier_applied_for_past_plans():
    """Cuando se renderiza el footer (planes pasados o legacy), debe
    tener la clase `modalFooterSingle` para centrar el CTA "Reactivar"
    como botón solitario. Sin esto, el `flex: 2` heredado lo dejaría
    estirado al 100% del modal."""
    assert "styles.modalFooterSingle" in _HISTORY_JSX, (
        "Clase `modalFooterSingle` no se aplica en el footer — el "
        "CTA Reactivar quedará estirado al 100% del modal (manda "
        "señal de 'acción obligatoria' que no es)."
    )


def test_modal_footer_single_css_defined():
    """CSS `.modalFooterSingle` Y override `.modalActionBtn` (NO
    `.modalCloseBtn` que ya no existe) presentes en History.module.css."""
    assert ".modalFooterSingle {" in _HISTORY_CSS, (
        ".modalFooterSingle ausente de History.module.css — la clase "
        "JSX no tiene efecto visual sin la regla CSS."
    )
    assert ".modalFooterSingle .modalActionBtn" in _HISTORY_CSS, (
        "Override de `.modalActionBtn` dentro de `.modalFooterSingle` "
        "ausente — sin esto el flex:2 default estira el botón al 100% "
        "del modal."
    )


def test_banner_action_required_copy_pivots_to_dashboard():
    """El banner action_required tenía hardcoded el texto
    'Pulsa Reactivar este Plan abajo…'. Si el botón ahora se oculta
    para planes activos, ese copy queda roto. Verificamos que existe
    el branching condicional."""
    # Texto del nuevo branch (plan activo/futuro).
    assert "Vuelve al" in _HISTORY_JSX and "Dashboard" in _HISTORY_JSX, (
        "Copy alternativo 'Vuelve al Dashboard…' ausente — el banner "
        "action_required dirigiría al usuario a un botón que ya no "
        "está visible."
    )
    # El texto legacy debe seguir presente para planes past.
    assert "Reactivar este Plan" in _HISTORY_JSX, (
        "Mención a 'Reactivar este Plan' eliminada por completo — "
        "para planes past el botón sigue existiendo y el copy debe "
        "apuntarlo. Si lo cambiaste a algo más genérico, actualiza "
        "este test también."
    )


def test_missing_days_reason_copy_branches_by_temporal_bucket():
    """Las 3 ramas de `_reason` (_exhaustedCount > 0, _puac > 0,
    _failedC > 0) tenían hardcoded 'Pulsa "Reactivar este Plan"
    abajo…'. Verificamos que ahora usan helpers `_ctaText`,
    `_ctaRetry`, `_ctaRetryWithInfo` que branching internamente."""
    # Helpers de copy.
    assert "_ctaText" in _HISTORY_JSX, (
        "Helper `_ctaText` ausente — las 3 ramas siguen con copy "
        "hardcoded apuntando a un botón que puede no estar visible."
    )
    assert "_ctaRetry" in _HISTORY_JSX, (
        "Helper `_ctaRetry` ausente."
    )
    assert "_ctaRetryWithInfo" in _HISTORY_JSX, (
        "Helper `_ctaRetryWithInfo` ausente."
    )
    # Las ramas deben USARLOS (interpolación con template literal).
    assert "${_ctaText}" in _HISTORY_JSX, (
        "La rama _puac > 0 no interpola `_ctaText` — sigue con "
        "string literal hardcoded."
    )
    assert "${_ctaRetry}" in _HISTORY_JSX, (
        "La rama _failedC > 0 no interpola `_ctaRetry`."
    )
    assert "${_ctaRetryWithInfo}" in _HISTORY_JSX, (
        "La rama _exhaustedCount > 0 no interpola `_ctaRetryWithInfo`."
    )
