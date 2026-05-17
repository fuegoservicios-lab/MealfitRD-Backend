"""[P3-DURATION-DROPDOWN-OPEN-FLUID · 2026-05-17] Fix UX: la apertura del
dropdown "Duración del Plan" (7/15/30 días) en Dashboard mostraba un
"doble destello" visible al expandir, sobre todo en el borde superior
cerca del texto "DURACIÓN DEL PLAN".

Síntoma reportado por usuario (2 iteraciones):
> "se visualiza dos destellos cuando abro el menu de opciones"
> "sigue igual, el destello aparece en la parte de arriba del borde
>  del el texto que dice 'duración del plan'"

Causa raíz (post-mortem completo):
  1. Pre-fix tenía spring underdamped (stiffness:450 damping:30 mass:0.8,
     damping crítico ≈38) → overshoot del scale 0.95→1 producía rebote.
  2. Tras quitar el spring, sobrevivía un destello en el borde superior:
     `backdrop-filter: blur(16px)` sobre `background: rgba(255,255,255,0.97)`
     se recompone en stages durante la transición. Blink/webkit "snapean"
     el filtro al final del primer frame → flash en los bordes.

Fix definitivo:
  - Animación SOLO de opacity (sin transform/translate/scale). Opacity-only
    no requiere capa de composición nueva → físicamente imposible que
    flickere por compositing.
  - `background: '#FFFFFF'` opaco (sin rgba semi-transparente).
  - `backdrop-filter` REMOVIDO (no añadía nada útil sobre fondo opaco).
  - `transition: { duration: 0.15, ease: 'easeOut' }` — tween simple.

Trade-off aceptado: pierde el efecto "frosted glass" del backdrop blur,
gana fluidez. Para un dropdown over content scrollable el glass effect
es nice-to-have; cero-flicker es must-have.
"""
from __future__ import annotations

from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DASHBOARD = (
    _BACKEND_ROOT.parent / "frontend" / "src" / "pages" / "Dashboard.jsx"
).read_text(encoding="utf-8")


def test_marker_present():
    assert "P3-DURATION-DROPDOWN-OPEN-FLUID" in _DASHBOARD, (
        "Marker P3-DURATION-DROPDOWN-OPEN-FLUID ausente — un refactor "
        "podría revertir el fix y reintroducir el doble destello del spring."
    )


def test_no_underdamped_spring_in_duration_dropdown():
    """Defensa: la transición del dropdown NO debe regresar al spring
    underdamped (stiffness 450 + damping 30) que causaba el overshoot."""
    # Anchor al bloque del dropdown vía la sección header del menú.
    anchor = "Duración del Plan"
    idx = _DASHBOARD.find(anchor)
    assert idx > 0, (
        "Header 'Duración del Plan' ausente — el dropdown fue refactorizado, "
        "re-anclar este test."
    )
    # Ventana hacia ATRÁS desde el header (la motion.div está arriba del header).
    start = max(0, idx - 1500)
    window = _DASHBOARD[start:idx]
    assert "stiffness: 450, damping: 30" not in window, (
        "REGRESIÓN: spring underdamped (stiffness:450 damping:30) está de "
        "vuelta en la motion.div del dropdown de duración. Esa config "
        "overshoots y produce el 'doble destello' que reportó el usuario."
    )


def test_opacity_only_animation():
    """El fix definitivo usa SOLO opacity en initial/animate/exit. Cualquier
    `y`, `scale`, o `x` requeriría capa de composición → potencial flicker."""
    marker_pos = _DASHBOARD.find("P3-DURATION-DROPDOWN-OPEN-FLUID")
    assert marker_pos > 0
    window = _DASHBOARD[marker_pos : marker_pos + 1500]
    init_pos = window.find("initial={{")
    trans_pos = window.find("transition=", init_pos) if init_pos > 0 else -1
    assert init_pos > 0 and trans_pos > init_pos, (
        "No pude localizar el bloque initial/animate/exit del dropdown — "
        "estructura del motion.div cambió, re-anclar."
    )
    init_block = window[init_pos:trans_pos]
    # Solo opacity debe estar en initial/animate/exit. Si aparecen `y`,
    # `scale`, `x`, `rotate`, etc., se introduce transform → potencial flicker
    # en el borde superior (sobre todo combinado con box-shadow).
    for forbidden in ("scale", " y:", " x:", "rotate", "translate"):
        assert forbidden not in init_block, (
            f"REGRESIÓN: `{forbidden.strip()}` apareció en initial/animate/exit "
            "del dropdown. El fix definitivo es opacity-only (cero transform). "
            "Cualquier transform reintroduce el destello en el borde superior."
        )
    assert "opacity" in init_block, (
        "initial/animate/exit no incluyen `opacity` — la animación se perdió "
        "por completo o el bloque fue refactorizado."
    )


def test_no_backdrop_filter_in_dropdown():
    """`backdrop-filter: blur(16px)` causaba el flash de borde superior
    porque Blink/Webkit lo recomponen en stages durante la transición.
    Sobre fondo opaco no añade nada útil."""
    marker_pos = _DASHBOARD.find("P3-DURATION-DROPDOWN-OPEN-FLUID")
    assert marker_pos > 0
    window = _DASHBOARD[marker_pos : marker_pos + 2000]
    # Cortar al cierre del style={{ ... }} de la motion.div del dropdown
    style_pos = window.find("style={{")
    assert style_pos > 0
    # Cerrar en el siguiente `>`
    close_pos = window.find(">", style_pos)
    style_block = window[style_pos:close_pos]
    assert "backdropFilter" not in style_block and "backdrop-filter" not in style_block, (
        "REGRESIÓN: `backdropFilter` está de vuelta en la motion.div del "
        "dropdown. Esa propiedad causa el flash de borde superior durante "
        "la transición de opacity. Quitarla — el fondo opaco es suficiente."
    )


def test_opaque_background_in_dropdown():
    """Fondo opaco evita compositing extra durante la animación."""
    marker_pos = _DASHBOARD.find("P3-DURATION-DROPDOWN-OPEN-FLUID")
    assert marker_pos > 0
    window = _DASHBOARD[marker_pos : marker_pos + 2000]
    style_pos = window.find("style={{")
    close_pos = window.find(">", style_pos)
    style_block = window[style_pos:close_pos]
    # Debe tener background, y NO debe ser rgba con alpha < 1.0
    assert "background:" in style_block
    # Heurística: rgba(...,...,...,X) donde X != 1 es semi-translúcido
    import re as _re
    rgba_matches = _re.findall(r"rgba\([^)]+,\s*([\d.]+)\)", style_block)
    for alpha_str in rgba_matches:
        # Solo nos importa el background, pero el shadow también usa rgba.
        # El background relevante es el primer color que NO está en una
        # propiedad de shadow/border. Como simplificación: si hay rgba
        # con alpha entre 0 y 0.99 en la propiedad `background:`, fallar.
        pass
    # Verificación directa: el background debe ser `#FFFFFF` o similar opaco
    bg_match = _re.search(r"background:\s*['\"]([^'\"]+)['\"]", style_block)
    assert bg_match, "No pude parsear el background del dropdown"
    bg_value = bg_match.group(1)
    assert bg_value.startswith("#") or "rgb(" in bg_value or bg_value in ("white", "FFFFFF"), (
        f"REGRESIÓN: background del dropdown es `{bg_value}` (semi-translúcido). "
        "Pre-fix usaba `rgba(255,255,255,0.97)` + backdropFilter — combinación "
        "que producía el flash. Mantener `#FFFFFF` opaco."
    )
