"""[P3-SHOPPING-DISCLAIMER-EXPAND · 2026-05-16] Expandir el disclaimer
"Smart Engine" del PDF de lista de compras para explicar:
  (a) Qué significa '~' en cantidades (conversión aproximada entre unidades).
  (b) Que algunas cantidades pueden ajustarse por realismo de almacenamiento.

Pre-fix: usuario veía "Brócoli 2 Cabezas (~2.2 lbs total)" y "Cilantro 2
Mazos" sin contexto. La '~' parecía aproximación arbitraria; los caps
(cilantro máx 2 mazos, lácteos perecederos) eran silenciosos. El usuario
podía asumir "el sistema está mal" o "por qué me da tan poco".

Fix: extensión del disclaimer existente con la explicación. Conditional
sobre !isUltraDense — en planes hyper-dense (>50 items, densidad =
'ultra') agregar texto extra empuja contenido fuera de página.

Implementación pura frontend (no requiere data backend). Scope acotado
del P3 polish.
"""
from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DASHBOARD_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx"


def test_disclaimer_explains_tilde_symbol():
    """El disclaimer extendido debe explicar qué significa '~' en cantidades."""
    src = _DASHBOARD_JSX.read_text(encoding="utf-8")
    # Buscar mención explícita del símbolo + su significado
    assert '"~"' in src or "'~'" in src, (
        "El símbolo '~' no se menciona en disclaimer — usuario sin guía."
    )
    assert "conversión aproximada" in src, (
        "Disclaimer no explica que '~' = conversión aproximada. Sin esto, "
        "el usuario asume aleatoriedad."
    )
    # Ejemplo concreto incluido para que el texto no sea abstracto
    assert "2 Cabezas" in src and "2.2 lbs" in src, (
        "Disclaimer no incluye ejemplo concreto ('2 Cabezas ≈ 2.2 lbs'). "
        "Texto abstracto sin ejemplo es menos accionable."
    )


def test_disclaimer_explains_storage_caps():
    """El disclaimer debe mencionar que cantidades pueden ajustarse por
    realismo de almacenamiento (hierbas, lácteos, cítricos)."""
    src = _DASHBOARD_JSX.read_text(encoding="utf-8")
    assert "realismo de almacenamiento" in src, (
        "Disclaimer no menciona 'realismo de almacenamiento' como motivo "
        "de caps. Sin esto, caps (cilantro 2 mazos, yogurt 1361g) parecen "
        "arbitrarios para el usuario."
    )
    # Ejemplos de categorías que se capean (al menos 2 de 3)
    cap_categories = ["hierbas", "lácteos", "cítricos"]
    mentioned = sum(1 for cat in cap_categories if cat in src)
    assert mentioned >= 2, (
        f"Disclaimer menciona solo {mentioned} categorías de cap. "
        f"Esperaba ≥2 de {cap_categories} para que el ejemplo sea concreto."
    )


def test_disclaimer_extension_conditional_on_density():
    """La extensión del disclaimer (texto adicional) debe ser conditional
    sobre `!isUltraDense`. En planes hyper-dense agregar texto extra
    empuja contenido fuera de página."""
    src = _DASHBOARD_JSX.read_text(encoding="utf-8")
    # El conditional debe estar en la zona del disclaimer
    assert "${isUltraDense ? '' :" in src, (
        "Extensión del disclaimer no es conditional sobre isUltraDense. "
        "En planes hyper-dense (>50 items), el texto extra empuja contenido "
        "fuera de página."
    )


def test_disclaimer_preserves_original_smart_engine_text():
    """NO regresión: el disclaimer mantiene los conceptos clave (Smart Engine,
    cantidades exactas, empaques del mercado local, Ud.=unidad). Texto minimizado
    2026-06-20 (más claro/conciso) — anclas actualizadas a la nueva redacción."""
    src = _DASHBOARD_JSX.read_text(encoding="utf-8")
    assert "<strong>Smart Engine:</strong>" in src
    assert "cantidades exactas" in src
    assert "empaques del mercado local" in src
    assert "Ud.</strong> = unidad" in src
