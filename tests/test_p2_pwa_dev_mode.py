"""[P2-PWA-DEV-MODE · 2026-05-12] `vite.config.js` no debe registrar el
Service Worker en `npm run dev`.

Pre-fix: `VitePWA({ devOptions: { enabled: true, ... } })` registraba el
SW en dev. Riesgos:

  (a) Browsers que abrieron tanto `localhost:5173` como `mealfitrd.com`
      en el mismo dispositivo pueden cachear bundles dev/stale en el SW
      y servirlos en sesiones futuras (depende del scope del SW por
      origen).
  (b) Rompe HMR — cualquier cambio de source dispara invalidación parcial,
      dejando el module graph mitad nuevo / mitad cacheado.
  (c) Deja artefactos en `.vite/` que confunden bug reports
      ("¿por qué mi cambio no aparece?" cuando el SW lo intercepta).

Para testear PWA localmente: `npm run build && npm run preview`
(production-like sin tocar el binary corriendo).

Test enforza:
  - `devOptions.enabled` es literal `false` (NO `true`, NO derivado de
    variable que pueda flippear sin code review).
  - El comment anchor `P2-PWA-DEV-MODE` permanece (sino futuros readers
    pueden "arreglar" la flag pensando que era cosmética).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_VITE_CONFIG = _REPO_ROOT / "frontend" / "vite.config.js"


@pytest.fixture(scope="module")
def vite_config_text() -> str:
    assert _VITE_CONFIG.exists(), f"vite.config.js no encontrado en {_VITE_CONFIG}"
    return _VITE_CONFIG.read_text(encoding="utf-8")


def test_anchor_present(vite_config_text):
    """Anchor `P2-PWA-DEV-MODE` debe estar presente para que un futuro
    reader entienda por qué `devOptions.enabled` es `false`."""
    assert "P2-PWA-DEV-MODE" in vite_config_text, (
        "Anchor `P2-PWA-DEV-MODE` removido — el comment explicativo del "
        "rationale es load-bearing. Sin él, alguien puede flippear "
        "`enabled` a `true` para 'testear PWA local' sin entender el "
        "blast radius (scope SW cross-origin + HMR roto + artefactos "
        "cache stale)."
    )


def test_devoptions_enabled_is_false(vite_config_text):
    """`devOptions.enabled` debe ser literal `false`."""
    # Localizar el bloque devOptions { ... }.
    m = re.search(
        r"devOptions\s*:\s*\{[^}]*\}",
        vite_config_text,
        re.DOTALL,
    )
    assert m, "Bloque `devOptions: { ... }` no encontrado en vite.config.js."
    block = m.group(0)
    # Buscar `enabled: <valor>` dentro del bloque.
    enabled_m = re.search(r"enabled\s*:\s*(\S+?)\s*[,}]", block)
    assert enabled_m, "Key `enabled` no encontrada dentro de `devOptions`."
    enabled_val = enabled_m.group(1).strip()
    assert enabled_val == "false", (
        f"`devOptions.enabled` es `{enabled_val}` — debe ser literal `false`. "
        f"Habilitar SW en dev rompe HMR + cachea bundles stale + interfere "
        f"con bug reports. Para testear PWA localmente: "
        f"`npm run build && npm run preview`."
    )


def test_no_devoptions_enabled_true_anywhere(vite_config_text):
    """Defensa adicional: ninguna línea NO-comentada debe contener
    `enabled: true` dentro del scope de `devOptions`. Cubre el caso
    de alguien añadiendo un segundo bloque devOptions o usando shorthand."""
    # Filtrar líneas que son comentario (// o *).
    lines = vite_config_text.split("\n")
    code_lines = [
        ln for ln in lines
        if not ln.strip().startswith("//")
        and not ln.strip().startswith("*")
    ]
    code = "\n".join(code_lines)
    # Buscar `devOptions:` seguido por `enabled: true` en ventana local.
    matches = re.findall(
        r"devOptions\s*:\s*\{[^}]*?enabled\s*:\s*true",
        code,
        re.DOTALL,
    )
    assert not matches, (
        f"Detectado `devOptions: { '{...enabled: true...}' }` en código "
        f"activo de vite.config.js. Esto rompe HMR + cachea bundles "
        f"stale. Cambiar a `enabled: false` y usar `npm run build && "
        f"npm run preview` para testear PWA localmente."
    )
