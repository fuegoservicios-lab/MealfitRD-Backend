# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-169 · 2026-09-23] Dos capturas del dueño; el cambio vive en el frontend (este test es el contrato).

1. iPhone: «le di a activar generación del plan y no hace nada, solo me redirige al inicio del dashboard». nginx lo
   fecha (Configuración 03:26:40 UTC, el panel recargando sus datos 03:26:44, ni un paso del formulario). La causa es de
   React Router 7: todo lo que cuelga de un `<Routes location={…}>` ve `navigationType === "POP"` FIJO
   (`useRoutesImpl`), y `ModalAwareRoutes` —la ventana de Configuración, 10-ago— le pasaba la ubicación SIEMPRE. Cada
   `navigate()` parecía un arranque en frío para las guardas POP de `ProtectedRoute`: /assessment y /plan devolvían al
   contador. Aun llegando al formulario, «Finalizar y Generar» habría rebotado igual. Sin ventana, sin `location`.
2. Chat del coach: con foto + texto, la burbuja gris tomaba el ancho del texto y dejaba un hueco gris junto a la foto
   (y en el teléfono enmarcaba también la foto sola, por la clase con !important). La foto va FUERA de la burbuja.

Los tests de cliente (`frontend/src/__tests__/lote169.test.jsx`) montan el `ModalAwareRoutes` REAL: la receta de rutas
modales se probaba con un `<Routes>` pelado, y por eso nadie vio el POP fijo.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def _front(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip("sin el repo del frontend al lado")
    return p.read_text(encoding="utf-8")


def test_las_rutas_modales_no_fijan_pop_sin_ventana():
    app = _front("src/App.jsx")
    assert "<Routes location={backgroundLocation || undefined}>{children}</Routes>" in app
    assert "<Routes location={backgroundLocation || location}>" not in app
    assert "export function ModalAwareRoutes(" in app   # lo monta el test de cliente


def test_las_guardas_pop_siguen_en_su_sitio():
    """El arreglo es de la ENVOLTURA, no de las guardas: una llegada fría a /assessment o /plan sigue volviendo al
    contador (lo comprueba también el test de cliente)."""
    pr = _front("src/components/layout/ProtectedRoute.jsx")
    assert "if (isOnAssessment && navigationType === 'POP') {" in pr
    assert "if (isOnPlan && navigationType === 'POP' && !_hasPendingPlanRecovery) {" in pr


def test_la_foto_del_chat_va_fuera_de_la_burbuja():
    mb = _front("src/components/agent/MessageBubble.jsx")
    assert "const fotoAparte = msg.role === 'user' && media.length > 0;" in mb
    assert "className={fotoAparte ? 'msg-user-grupo'" in mb
    assert "soloFoto" not in mb
    ap = _front("src/pages/AgentPage.jsx")
    assert ".msg-user-grupo > .msg-bubble-user {" in ap


def test_el_cliente_monta_las_rutas_reales():
    t = _front("src/__tests__/lote169.test.jsx")
    assert "import { ModalAwareRoutes } from '../App';" in t


def test_marcador_del_lote():
    import re
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', app)
    assert m and int(m.group(1)) >= 169 and m.group(2) >= "2026-09-23"   # la serie sigue; el marker nunca baja
    assert "[P1-PLAN-LOTE-169 · 2026-09-23]" in app
