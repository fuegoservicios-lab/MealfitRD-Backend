"""[P2-SHOPPING-2 · 2026-05-14] Banner "plan vencido" en el HTML del PDF
de lista de compras.

Bug pre-fix:
    `Dashboard.jsx::handleDownloadShoppingList` permite descarga aunque
    `isPlanExpired === true` (botón `disabled={isRecalculating}` solo).
    El PDF NO mostraba ningún disclaimer al usuario sobre que su plan
    venció — usuario podía comprar ingredientes para un ciclo cuyas
    fechas ya pasaron, sin señal de que algo está mal.

Fix:
    Banner rojo inyectado en el HTML del PDF cuando `isPlanExpired ===
    true`. Color rojo (#dc2626) para diferenciarlo del banner ámbar
    de inventario stale (P1-PDF-1) — uno es "acción requerida", el
    otro es "información de contexto". Decisión UX: NO bloquear la
    descarga (el usuario puede legítimamente querer el PDF histórico
    o estar en transición a un nuevo plan); solo educar.

Cobertura del test:
    1. El banner se inyecta condicionalmente con `isPlanExpired`.
    2. Contiene "Plan vencido" + "Regenera tu plan" para que el copy
       sobreviva refactors.
    3. Usa color rojo (#dc2626 o #fef2f2 background) — bloquea
       degradación accidental a "ámbar" (que mezclaría señal con
       el stale-inventory banner).
    4. Aparece después del banner de stale-inventory (orden de
       prominencia: vencido > stale).
    5. Marker anchor `[P2-SHOPPING-2 · 2026-05-14]` presente.

Tooltip-anchor: P2-SHOPPING-2.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DASH_FP = _REPO_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx"


@pytest.fixture(scope="module")
def dash_src() -> str:
    return _DASH_FP.read_text(encoding="utf-8")


def _extract_handler_block(src: str) -> str:
    start = src.find("const handleDownloadShoppingList")
    assert start > 0
    after = src[start + 50:]
    next_match = re.search(r"\n    const handle[A-Z]", after)
    end = (start + 50 + next_match.start()) if next_match else (start + 50 + 6000)
    return src[start:end]


# ---------------------------------------------------------------------------
# Banner presence + conditional
# ---------------------------------------------------------------------------

def test_banner_block_is_conditional_on_is_plan_expired(dash_src: str):
    """El banner debe estar gateado por `isPlanExpired` para no aparecer
    en planes vigentes (que serían ruido visual injustificado).
    """
    body = _extract_handler_block(dash_src)
    # Pattern: template literal con ternario `${isPlanExpired ? ... : ''}`
    # alrededor del bloque del banner. Tolerante a whitespace y line breaks.
    assert re.search(
        r"\$\{\s*isPlanExpired\s*\?\s*`",
        body,
    ), (
        "P2-SHOPPING-2 regresión: el banner ya NO está gateado por "
        "`isPlanExpired`. Sin el ternario, aparecería en TODOS los PDFs "
        "(falso positivo). Restaurar `${isPlanExpired ? \\`<div>...\\` : ''}`."
    )


def test_banner_copy_contains_plan_vencido(dash_src: str):
    """El copy del banner debe contener 'Plan vencido' (título) y
    'Regenera tu plan' (call-to-action). Sin esos textos, el usuario
    no entiende qué acción tomar.
    """
    body = _extract_handler_block(dash_src)
    assert "Plan vencido" in body, (
        "P2-SHOPPING-2 regresión: el banner ya no contiene 'Plan vencido' "
        "como título. El usuario no recibe la señal principal."
    )
    assert "Regenera tu plan" in body, (
        "P2-SHOPPING-2 regresión: el banner ya no contiene 'Regenera tu plan' "
        "como CTA. El usuario no sabe qué hacer."
    )


def test_banner_uses_red_color_palette(dash_src: str):
    """Color rojo (#dc2626 para borde, #fef2f2 background) — diferencia
    semántica frente al banner ámbar de stale inventory (P1-PDF-1).
    Una degradación a ámbar mezclaría señales.
    """
    body = _extract_handler_block(dash_src)
    # Buscar la ventana del banner vencido específicamente (no el banner
    # global rojo de "Prioridad Alta" perecederos).
    expired_block_match = re.search(
        r"isPlanExpired\s*\?\s*`([^`]+)`",
        body,
        re.DOTALL,
    )
    assert expired_block_match, (
        "P2-SHOPPING-2 regresión: bloque condicional `isPlanExpired ? \\`...\\``"
        " no extraíble — posible refactor cosmético."
    )
    block = expired_block_match.group(1)
    has_red_bg = "#fef2f2" in block
    has_red_border = "#dc2626" in block or "#fca5a5" in block
    assert has_red_bg, (
        "P2-SHOPPING-2 regresión: el background del banner ya NO usa "
        "`#fef2f2` (red-50). Si lo cambiaste a ámbar/amarillo, mezclaste "
        "la señal 'plan vencido' con 'inventario stale'. Restaurar el rojo."
    )
    assert has_red_border, (
        "P2-SHOPPING-2 regresión: el borde del banner ya NO usa rojo "
        "(#dc2626 o #fca5a5). Sin contraste el banner se camufla con el "
        "resto del PDF."
    )


def test_banner_appears_after_stale_inventory_banner(dash_src: str):
    """Orden de prominencia: stale-inventory (P1-PDF-1) viene primero
    (información de contexto), luego plan-vencido (acción requerida).
    Esto garantiza que si AMBOS están activos, el plan-vencido es lo
    último que el usuario ve antes de la lista — máxima retención.
    """
    body = _extract_handler_block(dash_src)
    stale_idx = body.find("freshInventoryStale ?")
    expired_idx = body.find("isPlanExpired ?")
    assert stale_idx > 0, "Banner stale-inventory no encontrado."
    assert expired_idx > 0, "Banner plan-vencido no encontrado."
    assert stale_idx < expired_idx, (
        "P2-SHOPPING-2 regresión: el orden de los banners cambió. "
        "stale-inventory (P1-PDF-1) debe aparecer ANTES de plan-vencido "
        "(P2-SHOPPING-2): el primero es contexto, el segundo es acción "
        "requerida — el usuario debe terminar con el call-to-action "
        "fresco en su pantalla."
    )


def test_banner_marker_anchored(dash_src: str):
    """Anchor `[P2-SHOPPING-2 · 2026-05-14]` en comment del banner para
    que un refactor cosmético no remueva el bloque sin entender la razón.
    """
    body = _extract_handler_block(dash_src)
    assert "P2-SHOPPING-2" in body, (
        "P2-SHOPPING-2 regresión: marker `[P2-SHOPPING-2 · ...]` "
        "desapareció del handler PDF. Restaurar el comment que documenta "
        "por qué el banner se inyecta acá (decisión UX: educar, no "
        "bloquear)."
    )
