"""[P3-AUDIT-3 · 2026-05-15] Test parser-based: `frontend/public/manifest.json`
declara solo entries de icons con sizes que el archivo REAL tiene (mínimo
viable: 192 + 512).

Por qué este test:
    Pre-fix tenía 6 entries (72/96/128/144/192/512) todos apuntando al
    mismo `/favicon.png`. Lighthouse PWA audit penaliza cuando un icon
    declara `sizes="72x72"` pero el archivo es 512x512 — el browser
    intenta escalar y deja artifacts visibles en el launcher.

Fix esperado:
    - Reducir `icons` a 2 entries: 192x192 (Chrome Add-to-Home-Screen
      mínimo) + 512x512 (iOS Safari + maskable).
    - Mantener el resto del manifest (name, short_name, lang, shortcuts).
    - La entry 512 debe tener `purpose: "any maskable"` para que Android
      adapte el icono a cada launcher shape sin distorsión.

Drift detection:
    - `icons[]` length == 2.
    - Sizes presentes: {"192x192", "512x512"} exactos (no faltan ni
      sobran).
    - 512x512 entry tiene `"any maskable"` en `purpose`.
    - `shortcuts[]` preservados (3 entries: Chat, Plan, Súper).

Cross-link convention (P2-HIST-AUDIT-14): slug `p3_audit_3`.

Tooltip-anchor: P3-AUDIT-3-START | gap audit 2026-05-15
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MANIFEST = _REPO_ROOT / "frontend" / "public" / "manifest.json"


@pytest.fixture(scope="module")
def manifest_obj() -> dict:
    return json.loads(_MANIFEST.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Icons reducidos a 2 entries
# ---------------------------------------------------------------------------
def test_icons_has_exactly_two_entries(manifest_obj: dict):
    icons = manifest_obj.get("icons", [])
    assert len(icons) == 2, (
        f"P3-AUDIT-3 regresión: `icons[]` tiene {len(icons)} entries "
        f"(esperado 2: 192x192 + 512x512). Pre-fix tenía 6 entries "
        f"falsas — si añadiste un sizes nuevo, asegúrate de que el archivo "
        f"REAL tiene esa resolución (Lighthouse audita la coincidencia)."
    )


def test_icons_sizes_are_192_and_512(manifest_obj: dict):
    icons = manifest_obj.get("icons", [])
    sizes = sorted([icon.get("sizes") for icon in icons])
    assert sizes == ["192x192", "512x512"], (
        f"P3-AUDIT-3 regresión: sizes presentes {sizes} (esperado "
        f"['192x192', '512x512']). 192x192 es mínimo Chrome Add-to-Home-Screen, "
        f"512x512 es el de iOS Safari + maskable Android."
    )


def test_icons_512_has_maskable_purpose(manifest_obj: dict):
    icons = manifest_obj.get("icons", [])
    icon_512 = next((i for i in icons if i.get("sizes") == "512x512"), None)
    assert icon_512 is not None, "P3-AUDIT-3 regresión: entry 512x512 ausente."
    purpose = icon_512.get("purpose", "")
    assert "maskable" in purpose, (
        f"P3-AUDIT-3 regresión: la entry 512x512 no tiene `maskable` en "
        f"`purpose` (actual: {purpose!r}). Sin maskable, Android no puede "
        f"adaptar el icono al shape del launcher (circle, squircle, etc.) "
        f"sin distorsión."
    )


# ---------------------------------------------------------------------------
# 2. Estructura del manifest preservada
# ---------------------------------------------------------------------------
def test_manifest_preserves_core_metadata(manifest_obj: dict):
    """Cleanup de icons NO debe haber tocado otros campos críticos."""
    for required_key in ("name", "short_name", "lang", "start_url", "display"):
        assert required_key in manifest_obj, (
            f"P3-AUDIT-3 regresión colateral: campo `{required_key}` "
            f"removido del manifest. El cleanup debía afectar solo `icons`."
        )


def test_manifest_lang_es_do(manifest_obj: dict):
    """es-DO es invariante del producto (P3-I18N-DEFERRED)."""
    assert manifest_obj.get("lang") == "es-DO", (
        "P3-AUDIT-3 regresión colateral: `lang` cambió de `es-DO`. "
        "Convención P3-I18N-DEFERRED documenta es-DO como permanente."
    )


def test_manifest_preserves_shortcuts(manifest_obj: dict):
    """`shortcuts` permite quick actions desde el launcher long-press.
    Pre-fix tenía 3 (Chat, Plan, Súper) — NO deben perderse."""
    shortcuts = manifest_obj.get("shortcuts", [])
    assert len(shortcuts) >= 3, (
        f"P3-AUDIT-3 regresión colateral: `shortcuts[]` tiene "
        f"{len(shortcuts)} entries (esperado ≥3: Chat, Plan, Súper). El "
        f"cleanup de icons NO debía afectar shortcuts."
    )


# ---------------------------------------------------------------------------
# 3. Sanity: ningún sizes "falso" superviviente (72, 96, 128, 144)
# ---------------------------------------------------------------------------
def test_no_invalid_legacy_sizes(manifest_obj: dict):
    """Los 4 sizes del pre-fix (72/96/128/144) NO deben aparecer — el
    favicon real es 512x512, declararlos sería incoherencia documental."""
    icons = manifest_obj.get("icons", [])
    sizes_present = {icon.get("sizes") for icon in icons}
    forbidden = {"72x72", "96x96", "128x128", "144x144"}
    leaked = sizes_present & forbidden
    assert not leaked, (
        f"P3-AUDIT-3 regresión: sizes legacy {sorted(leaked)} reaparecen "
        f"en el manifest. El archivo real es 512x512 — declarar sizes más "
        f"pequeños es incoherencia documental que Lighthouse penaliza."
    )
