"""[P1-VISION-LUNA · 2026-07-28] Test ancla.

Doc: backend/docs/vision_luna.md
"""
import io
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from image_prep import prepare_image_for_vision  # noqa: E402

_BACKEND = os.path.join(os.path.dirname(__file__), "..")


def _src(rel):
    with open(os.path.join(_BACKEND, rel), encoding="utf-8") as fh:
        return fh.read()


def _png(w, h, mode="RGB"):
    from PIL import Image
    buf = io.BytesIO()
    Image.new(mode, (w, h), "white").save(buf, format="PNG")
    return buf.getvalue()


def _wh(raw):
    from PIL import Image
    return Image.open(io.BytesIO(raw)).size


def test_foto_de_movil_se_reduce_al_cap():
    """3024x4032 son 12.406 tokens de entrada ($0,0130); a 1024 son 1.374
    ($0,0020). 6,6x, medido contra la API real el 2026-07-28."""
    raw = _png(3024, 4032)
    out, info = prepare_image_for_vision(raw, max_side=1024)
    assert max(_wh(out)) == 1024
    assert info["resized"] is True
    assert len(out) < len(raw)


def test_mantiene_el_aspecto():
    out, _ = prepare_image_for_vision(_png(4000, 2000), max_side=1024)
    w, h = _wh(out)
    assert (w, h) == (1024, 512), (w, h)


def test_imagen_pequena_pasa_intacta():
    """Re-encodear algo que ya cabe solo pierde calidad sin ahorrar nada."""
    raw = _png(800, 600)
    out, info = prepare_image_for_vision(raw, max_side=1024)
    assert out == raw
    assert info["resized"] is False


def test_png_con_alpha_no_revienta_el_encoder_jpeg():
    """RGBA -> JPEG lanza OSError si no se convierte a RGB antes."""
    out, info = prepare_image_for_vision(_png(2000, 2000, mode="RGBA"), max_side=1024)
    assert max(_wh(out)) == 1024
    assert info["resized"] is True


@pytest.mark.parametrize("basura", [b"", b"no soy una imagen", b"\x00" * 100, None])
def test_fail_open_devuelve_los_bytes_originales(basura):
    """Un escaneo NUNCA debe morir porque el resize fallo."""
    out, info = prepare_image_for_vision(basura, max_side=1024)
    assert out == basura
    assert info["resized"] is False
    assert info["skipped_reason"]


def test_knob_max_side_con_clamp():
    from image_prep import vision_max_side_px
    assert vision_max_side_px() == 1024
    os.environ["MEALFIT_VISION_MAX_SIDE_PX"] = "99999"
    try:
        assert vision_max_side_px() == 1024, "fuera de [256,4096] debe caer al default"
    finally:
        os.environ.pop("MEALFIT_VISION_MAX_SIDE_PX", None)
