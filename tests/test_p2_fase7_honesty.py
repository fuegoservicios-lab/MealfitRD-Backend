"""[P2-FASE7-HONESTY · 2026-06-21] Honestidad user-facing (Fase 7, última del build "todo terreno").

Cada modo de degradación tiene un mensaje claro + accionable para que el usuario SIEMPRE sepa POR QUÉ:
- Banner de degradación del Dashboard (`_quality_degraded_reason` → Q_DEGRADED_REASON_MAP): copy
  mejorado para los motivos que fluyen (max_attempts, low_band_score, high_contextual, ...) +
  `shopping_list_incomplete` (preocupación #1 del owner: la lista de compras).
- `_maybe_mark_shopping_incomplete_degraded` hace que la lista vacía emita un motivo ESPECÍFICO
  (sobrescribiendo el genérico max_attempts), en vez de "el revisor no aprobó".
- Las otras honestidades se surfacean en SU superficie (no se duplican en el banner): presupuesto →
  bloqueo + toast pre-gen (Fase 3); piso de proteína → disclaimer del fallback en Plan.jsx (Fase 2);
  nevera baja → banner de Mi Nevera (Fase 4).
"""
import os

import graph_orchestrator as go


def _plan(is_empty, reason=None, marked=False):
    p = {"_shopping_completeness": {"is_empty": is_empty}}
    if marked:
        p["_quality_degraded"] = True
        p["_quality_degraded_reason"] = reason
    return p


# ---------------------------------------------------------------------------
# _maybe_mark_shopping_incomplete_degraded
# ---------------------------------------------------------------------------
def test_lista_vacia_sobrescribe_max_attempts():
    p = _plan(True, reason="max_attempts", marked=True)
    assert go._maybe_mark_shopping_incomplete_degraded(p, False, 3) is True
    assert p["_quality_degraded_reason"] == "shopping_list_incomplete"
    assert p["_quality_degraded_severity"] == "high"


def test_lista_vacia_respeta_razon_clinica_mas_especifica():
    p = _plan(True, reason="condition_panel_gap", marked=True)
    assert go._maybe_mark_shopping_incomplete_degraded(p, False, 3) is False
    assert p["_quality_degraded_reason"] == "condition_panel_gap", "Una razón clínica más específica gana."


def test_lista_no_vacia_no_marca():
    p = _plan(False)
    assert go._maybe_mark_shopping_incomplete_degraded(p, False, 3) is False
    assert not p.get("_quality_degraded")


def test_fallback_no_se_toca():
    # Un plan fallback ya lleva su propio disclaimer (Plan.jsx) → este marker no lo pisa.
    p = _plan(True)
    assert go._maybe_mark_shopping_incomplete_degraded(p, True, 3) is False


def test_marca_cuando_no_habia_marca_previa():
    p = _plan(True)  # is_empty pero sin _quality_degraded previo
    assert go._maybe_mark_shopping_incomplete_degraded(p, False, 2) is True
    assert p["_quality_degraded_reason"] == "shopping_list_incomplete"


# ---------------------------------------------------------------------------
# Callsite + frontend map (parser)
# ---------------------------------------------------------------------------
def test_marker_invocado_post_scoring():
    src = open(go.__file__, encoding="utf-8").read()
    assert "_maybe_mark_shopping_incomplete_degraded(plan" in src, "El marker debe invocarse en el bloque post-scoring."
    assert "P2-FASE7-HONESTY" in src


def test_frontend_map_tiene_copy_honesto():
    here = os.path.dirname(__file__)
    dash = os.path.normpath(os.path.join(here, "..", "..", "frontend", "src", "pages", "Dashboard.jsx"))
    if not os.path.exists(dash):
        import pytest
        pytest.skip("Dashboard.jsx no disponible en este entorno")
    js = open(dash, encoding="utf-8").read()
    assert "shopping_list_incomplete:" in js, "El mapa debe tener copy para la lista incompleta."
    # Copy accionable (no solo 'no aprobado'): el de max_attempts debe guiar al usuario.
    idx = js.find("max_attempts:")
    assert idx > -1
    assert "Cambiar Plato" in js[idx: idx + 200], "El copy de max_attempts debe ser accionable."
