"""[P1-VISION-GEMINI-FLASH · 2026-09-04] Verifica el provider de visión configurado con
una foto REAL, leyendo la key del entorno (nunca por argumento).

    python scripts/check_vision_model.py --image foto.jpg [--env /ruta/.env]

Imprime el resultado estructurado del escáner (`photo_kind`, ítems, macros) y, si el
proveedor rechaza la llamada, el error HTTP tal cual — es lo que distingue «la capa
compatible con OpenAI de Gemini no acepta este `response_format`» de «key mal puesta».
"""
# [P2-LOGGER-EXEMPT: script CLI de verificación, salida a stdout a propósito]
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--env", default=str(Path(__file__).resolve().parents[1] / ".env"))
    args = ap.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(args.env)
    except Exception:
        pass
    os.environ.setdefault("MEALFIT_VISION_PROVIDER", "openai_compatible")
    import vision_agent as va

    model = va._vision_model_name()
    base = va._vision_base_url() or ("<default Gemini>" if va.is_google_model(model) else "")
    key_src = "VISION_API_KEY" if os.environ.get("VISION_API_KEY") else (
        "GEMINI_API_KEY" if os.environ.get("GEMINI_API_KEY") else (  # [P1-VISION-GEMINI-FLASH]
            "GOOGLE_API_KEY" if os.environ.get("GOOGLE_API_KEY") else (  # [P1-VISION-GEMINI-FLASH]
                "OPENAI_API_KEY" if os.environ.get("OPENAI_API_KEY") else "(ninguna)")))
    print(f"modelo={model!r} base_url={base!r} key={key_src} enabled={va.is_vision_enabled()}")
    if not va.is_vision_enabled():
        print("visión apagada: revisa MEALFIT_VISION_PROVIDER / MEALFIT_VISION_MODEL")
        return 2
    data = Path(args.image).read_bytes()
    t0 = time.time()
    try:
        out = asyncio.run(va.process_image_with_vision(data))
    except Exception as e:
        print(f"FALLO {type(e).__name__}: {e}")
        return 1
    dt = time.time() - t0
    print(f"{dt:.1f}s")
    print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
    if out.get("analysis_failed"):
        print("→ el escáner devolvió analysis_failed: mira el journal/log para el HTTP real")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
