"""[P1-PAPER-HERO-FIG00 · 2026-08-01] El orbe-vídeo del hero murió.

Se sustituye por una figura diagramática SVG inline (~2 KB) — el despiece de
un plato dominicano acotado. Se borran 5 assets (5.550.239 bytes medidos) y
toda la maquinaria de autoplay.

Los 6 casos de `frontend/src/__tests__/Hero.p1_orb_autoplay_mobile.test.jsx`
se borran CON la feature, no se «arreglan»: codificaban un bug real de
autoplay en Chrome Android e iOS Low Power Mode que deja de existir cuando
no hay vídeo. Borrarlos es una decisión, no limpieza.

Tooltip-anchor: P1-PAPER-HERO-FIG00
"""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FRONTEND = _REPO_ROOT / "frontend"
_HERO_JSX = _FRONTEND / "src" / "components" / "home" / "Hero.jsx"
_PUBLIC = _FRONTEND / "public"

_ORB_ASSETS = ["orb.mp4", "orb.webm", "orb-sm.mp4", "orb-sm.webm", "orb-poster.jpg"]


def test_orb_assets_are_gone():
    """Un asset huérfano en public/ se sigue sirviendo y sigue pesando en el
    bundle de despliegue, aunque ningún componente lo referencie."""
    survivors = [name for name in _ORB_ASSETS if (_PUBLIC / name).exists()]
    assert not survivors, (
        f"P1-PAPER-HERO-FIG00: siguen en frontend/public/: {survivors}. "
        "Son 5.550.239 bytes que ya no sirven a nadie."
    )


def test_hero_has_no_video_machinery():
    text = _HERO_JSX.read_text(encoding="utf-8")
    forbidden = ["<video", "videoRef", "orbBreath", "orb-poster", "saveData", "NotAllowedError"]
    found = [f for f in forbidden if f in text]
    assert not found, (
        f"P1-PAPER-HERO-FIG00: queda maquinaria de vídeo en Hero.jsx: {found}. "
        "El e2e exige 0 pageerror en `/`: un useEffect apuntando a videoRef.current "
        "que ya no existe tumba golden_path.spec.js."
    )


def test_orb_test_file_deleted_not_repaired():
    orb_test = _FRONTEND / "src" / "__tests__" / "Hero.p1_orb_autoplay_mobile.test.jsx"
    assert not orb_test.exists(), (
        "P1-PAPER-HERO-FIG00: `Hero.p1_orb_autoplay_mobile.test.jsx` sigue ahí. "
        "Sus 6 casos prueban una feature que ya no existe — se borran CON ella, "
        "no se adaptan para que pasen."
    )


def test_hero_renders_the_figure():
    text = _HERO_JSX.read_text(encoding="utf-8")
    assert "PlateExploded" in text, (
        "P1-PAPER-HERO-FIG00: el hero debe montar la Fig. 00 (`PlateExploded`)."
    )
    assert "Fig. 00" in text, (
        "P1-PAPER-HERO-FIG00: falta el pie de figura «Fig. 00 — …»."
    )
