"""[P3-CAPS-COHERENCE-RECONCILE-3 · 2026-05-30] Cierre de la clase ENTERA de
caps cap-aware del coherence guard.

Contexto: el fix original P1-CAPS-COHERENCE-RECONCILE (2026-05-16) instrumentó
solo 5 caps (y de varios, solo UNA de sus ramas de magnitud). P2-CAPS-COHERENCE-
RECONCILE-2 (2026-05-30) añadió 3 perecederos + la rama 'mazo' de HERB. Quedaban
diferidos 7 caps (OLIVE, CITRUS, SAUCE, OIL, CARBS, SWEETENER, BROTHS) bajo el
scoping "cerrar solo FPs observados" + ramas parciales sin registrar en los caps
viejos (VEG unit, SPICE sobre, LEGUMES paquete, CANNED weight, EGGS unidad,
LACTEOS lb+pote).

Esta pasada cierra la clase entera: los 16 caps registran `_record_cap_applied`
en TODAS sus ramas de magnitud. Razón del cierre total: registrar es puramente
ADITIVO y dirección-SEGURA (los caps solo reducen over-buy) → cero riesgo de
under-buy/corrupción, y el guard nunca ve un FP de magnitud de un cap sin importar
qué unidad nativa emitió el LLM. CITRUS era el FP genuino (perecedero no-staple
no-líquido que llega a la capa de magnitud); los líquidos/condimentos
(OIL/SAUCE/SWEETENER) son belt-and-suspenders sobre la tolerancia de líquidos +
filtro is_staple.

Este test es el ANCLA de la invariante "16/16 caps instrumentados, cero diferidos".
Complementa la verificación de presencia parametrizada de
`test_p1_caps_coherence_reconcile.py::test_cap_callsite_records_metadata` (que
chequea proximidad log↔record) con un chequeo de COBERTURA TOTAL: cada marker
canónico DEBE aparecer como `reason` de al menos un `_record_cap_applied(...)`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_SHOPPING_PATH = _BACKEND_ROOT / "shopping_calculator.py"


def _read_shopping() -> str:
    return _SHOPPING_PATH.read_text(encoding="utf-8")


# Lista CANÓNICA de los 16 caps de magnitud del aggregator. Si añades un cap
# nuevo, agrégalo aquí Y registra `_record_cap_applied` en cada una de sus ramas.
_CANONICAL_CAP_MARKERS = [
    "P3-HERB-CAP",
    "P5-VEG-CAP",
    "P5-OLIVE-CAP",
    "P6-CITRUS-CAP",
    "P6-SPICE-CAP",
    "P6-LEGUMES-DRY-CAP",
    "P6-CANNED-PROTEIN-CAP",
    "P6-EGGS-AGGREGATE-CAP",
    "P6-LACTEOS-PERISHABLE-CAP",
    "P6-FRUITS-LARGE-CAP",
    "P6-FRUITS-PERISHABLE-CAP",
    "P6-CARBS-CAP",
    "P6-SAUCE-CAP",
    "P6-OIL-CAP",
    "P6-SWEETENER-CAP",
    "P6-BROTHS-CAP",
]


def _record_reasons(text: str) -> list[str]:
    """Extrae el último argumento string (`reason`) de cada
    `_record_cap_applied(...)` callsite. Tolerante a argumentos con paréntesis
    anidados (`float(_old)`) porque ancla en la comilla del reason, no en el
    cierre del paréntesis."""
    # _record_cap_applied( ... , "<REASON>" )
    return re.findall(
        r'_record_cap_applied\([^\n]*?["\']([A-Z0-9][A-Z0-9\-]*-CAP)["\']\s*\)',
        text,
    )


@pytest.mark.parametrize("cap_marker", _CANONICAL_CAP_MARKERS)
def test_every_cap_is_registered(cap_marker: str):
    """Cada cap canónico DEBE aparecer como `reason` de al menos un
    `_record_cap_applied(...)`. Sin esto, el coherence guard (default block en
    prod) trata el recorte de magnitud del cap como divergencia crítica y fuerza
    un retry innecesario (2-10s latencia)."""
    reasons = _record_reasons(_read_shopping())
    assert cap_marker in reasons, (
        f"Cap `{cap_marker}` no tiene NINGÚN `_record_cap_applied(..., "
        f"\"{cap_marker}\")`. Sin registro, el guard reporta su recorte como FP "
        f"y dispara retry en mode=block. Registra el cap en CADA rama que "
        f"modifique `_units[...]`."
    )


def test_no_deferred_caps_remain():
    """Invariante de cierre: TODOS los markers de log `[Pn-...-CAP]` que
    correspondan a un cap de magnitud canónico deben estar registrados. Cero
    diferidos. Si alguien añade un cap nuevo sin registrarlo, este test lo caza."""
    reasons = set(_record_reasons(_read_shopping()))
    missing = [m for m in _CANONICAL_CAP_MARKERS if m not in reasons]
    assert not missing, (
        f"Caps de magnitud SIN registrar (clase no cerrada): {missing}. "
        f"La invariante P3-CAPS-COHERENCE-RECONCILE-3 exige 16/16 instrumentados."
    )
    # Sanity: confirmamos que el set registrado cubre al menos los 16 canónicos.
    canonical = set(_CANONICAL_CAP_MARKERS)
    assert canonical.issubset(reasons), (
        f"Faltan caps canónicos en los registros: {canonical - reasons}"
    )


def test_olive_cap_all_four_branches_registered():
    """OLIVE es el cap con más ramas (unit/weight/count/volumétric). Las 4
    modifican `_units` → las 4 deben registrar. Regresión específica porque el
    finder original solo veía la rama 'mazo' de HERB y asumía OLIVE cubierto."""
    reasons = _record_reasons(_read_shopping())
    olive_count = sum(1 for r in reasons if r == "P5-OLIVE-CAP")
    assert olive_count >= 4, (
        f"P5-OLIVE-CAP tiene {olive_count} registros; esperaba >=4 (una por rama: "
        f"unit/weight/count/volumétric)."
    )
