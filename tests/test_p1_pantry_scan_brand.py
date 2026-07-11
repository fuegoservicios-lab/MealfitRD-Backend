"""[P1-PANTRY-SCAN-BRAND · 2026-07-11] El escáner lee la MARCA del empaque.

Owner (antes de su prueba con 3 alimentos de marca): "debería detectarlo
correctamente, ¿no?" — el schema no tenía campo brand y el prompt pedía nombres
genéricos: las marcas se perdían. Ahora el modelo devuelve 'brand' (null si no
es legible), viaja como detected_brand, se muestra en el checklist y ETIQUETA
el item al confirmar.

Invariante de seguridad: el scan NUNCA escribe `user_brand_preferences` — la
preferencia "para siempre" es SOLO elección manual (un OCR equivocado no debe
contaminar las marcas preferidas globales que usan lista y planes).

tooltip-anchor: P1-PANTRY-SCAN-BRAND
"""
from __future__ import annotations

from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_UD_SRC = (_BACKEND / "routers" / "user_data.py").read_text(encoding="utf-8")
_QPB_SRC = (_BACKEND.parent / "frontend" / "src" / "components" / "assessment"
            / "questions" / "QPantryBuilder.jsx").read_text(encoding="utf-8")


def test_schema_and_prompt_capture_brand():
    assert '"brand": {"type": ["string", "null"]}' in _UD_SRC, "campo brand en el schema de visión"
    assert "la marca NO" in _UD_SRC and "ponla en 'brand'" in _UD_SRC, (
        "el prompt debe separar nombre genérico (matchea catálogo) de marca (etiqueta)"
    )


def test_brand_passthrough_sanitized():
    assert '"detected_brand": _brand[:40] if _brand else None' in _UD_SRC


def test_scan_never_touches_brand_preferences():
    i = _UD_SRC.find('@router.post("/inventory/photo-scan")')
    body = _UD_SRC[i:i + 6000]
    # La invariante es sobre ESCRITURAS reales (el comentario del bloque sí
    # menciona la tabla al documentar justamente esta prohibición).
    for write in ("INSERT INTO public.user_brand_preferences",
                  "UPDATE public.user_brand_preferences",
                  "api_put_brand_preference"):
        assert write not in body, (
            "el scan NO escribe preferencias globales — solo la elección manual "
            "del usuario las toca (OCR equivocado ≠ marca preferida para siempre)"
        )


def test_frontend_shows_and_applies_detected_brand():
    assert "it.detected_brand && (" in _QPB_SRC, "el checklist muestra la marca leída"
    assert "brand: it.detected_brand || null" in _QPB_SRC, (
        "confirmar debe etiquetar el item con la marca del empaque"
    )
    # Y el confirm NO debe llamar al PUT de preferencias (solo changeBrand manual
    # lo hace — changeBrand vive DESPUÉS de confirmScanItems en el archivo, por
    # eso el slice termina en commitQty, no en removeItem).
    i = _QPB_SRC.find("const confirmScanItems")
    j = _QPB_SRC.find("const commitQty")
    assert 0 < i < j, "orden de funciones cambió — re-anclar el slice"
    assert "'/api/supermarket/preferences'" not in _QPB_SRC[i:j]


def test_marker_anchored_in_source():
    assert _UD_SRC.count("P1-PANTRY-SCAN-BRAND") >= 2
    assert _QPB_SRC.count("P1-PANTRY-SCAN-BRAND") >= 2
