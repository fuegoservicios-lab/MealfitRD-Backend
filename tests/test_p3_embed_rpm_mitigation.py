"""[P3-EMBED-RPM-MITIGATION · 2026-05-16] Regression guard del tuning de batch
size + delay del init de embeddings para no saturar el RPM cap del Tier 1
gemini-embedding-2.

Bug observado en test E2E del 2026-05-15 21:51:30-21:51:43 (plan_id=ae29c7a9):
  - `[GEMINI/QUOTA] embed_documents (master_ingredients cache init) batch 7/11
    429 (intento 1/3); backoff 2.4s.`
  - `batch 7/11 agotó 3 intentos por 429 — upstream sin quota; el caller cae
    al fast-path.`
  - `🟡 Caché semántico no disponible; usando Regex Fast-Path.`

Diagnóstico previo (en comentario .env): "cuota DIARIA del modelo" — INCORRECTO.
Verificación 2026-05-16 en AI Studio Dashboard (proyecto Mealfitt) reveló:
  - gemini-embedding-2: peak 3.9K / 3K RPM (en rojo, +30% sobre cap Tier 1).
  - TPM: 2.46K / 1M (0.25%).
  - RPD: 118.2K / Ilimitado.

Root cause: el cap saturado es RPM por minuto, NO daily. Pre-fix con
batch_size=10 + delay=1.5s la tasa efectiva era 10/1.5 = 400 RPM mientras corre
el init. Combinado con picos transitorios + ingredient lookups runtime, se
acumula y dispara 429 reproducible en batch 7/11.

Fix: BATCH_SIZE 10→5 + DELAY 1.5→3.0s = 100 RPM sostenido (debajo de 3K cap).
Trade-off: init pasa de ~16s a ~30-40s. SOLO afecta primer plan tras restart.

Si migras a Tier 2 (RPM cap 10K), restaurar batch=10 + delay=1.0s.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _BACKEND_ROOT / ".env"


def _read_env() -> str:
    return _ENV_PATH.read_text(encoding="utf-8")


def test_embed_init_batch_size_conservative():
    """`MEALFIT_EMBED_INIT_BATCH_SIZE` <= 7 para no superar RPM cap Tier 1.
    Combinado con DELAY >= 2.5s, la tasa sostenida es <= 168 RPM."""
    text = _read_env()
    m = re.search(
        r"^MEALFIT_EMBED_INIT_BATCH_SIZE\s*=\s*(\d+)",
        text,
        re.MULTILINE,
    )
    assert m, "Falta `MEALFIT_EMBED_INIT_BATCH_SIZE` en .env."
    val = int(m.group(1))
    assert val <= 7, (
        f"P3-EMBED-RPM-MITIGATION: BATCH_SIZE={val} > 7. Con DELAY=3.0s la "
        f"tasa efectiva es {val * 60 / 3.0:.0f} RPM. Si migraste a Tier 2 "
        f"(cap 10K RPM) y quieres subirlo, actualizar este threshold."
    )


def test_embed_init_batch_delay_generous():
    """`MEALFIT_EMBED_INIT_BATCH_DELAY_S` >= 2.5s para mantener RPM bajo cap."""
    text = _read_env()
    m = re.search(
        r"^MEALFIT_EMBED_INIT_BATCH_DELAY_S\s*=\s*([\d.]+)",
        text,
        re.MULTILINE,
    )
    assert m, "Falta `MEALFIT_EMBED_INIT_BATCH_DELAY_S` en .env."
    val = float(m.group(1))
    assert val >= 2.5, (
        f"P3-EMBED-RPM-MITIGATION: DELAY={val}s < 2.5s. Bajo eso, la tasa "
        f"efectiva supera 200 RPM y puede saturar bajo carga concurrente. "
        f"Si migraste a Tier 2 (cap 10K RPM), actualizar este threshold."
    )


def test_embed_init_sustained_rpm_below_cap():
    """Tasa efectiva sostenida (batch_size * 60 / delay) debe ser <= 200 RPM
    para dejar headroom 15× contra el cap 3K Tier 1. Si necesitas más
    throughput, sube ambos juntos manteniendo este test pasando."""
    text = _read_env()
    bs_match = re.search(
        r"^MEALFIT_EMBED_INIT_BATCH_SIZE\s*=\s*(\d+)",
        text,
        re.MULTILINE,
    )
    dl_match = re.search(
        r"^MEALFIT_EMBED_INIT_BATCH_DELAY_S\s*=\s*([\d.]+)",
        text,
        re.MULTILINE,
    )
    assert bs_match and dl_match, "Faltan knobs del init de embeddings."
    bs = int(bs_match.group(1))
    dl = float(dl_match.group(1))
    sustained_rpm = bs * 60.0 / dl
    assert sustained_rpm <= 200.0, (
        f"Tasa sostenida {sustained_rpm:.0f} RPM > 200 RPM (15× headroom contra "
        f"cap 3K Tier 1). Reduce BATCH_SIZE o sube DELAY."
    )
