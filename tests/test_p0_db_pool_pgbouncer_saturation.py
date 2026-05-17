"""[P0-DB-POOL-PGBOUNCER-SATURATION · 2026-05-15] Regression guard del
.env tuning del DB pool para alinearlo con pgBouncer Transaction Mode.

Bug observado en test E2E del 2026-05-15 20:14-20:24 (plan_id=e7582150):
  - 10-15 warnings "DB async cache/CB error: couldn't get a connection after 10.00 sec"
    durante la generación del plan.
  - El plan NO se persistió en `meal_plans` pese a que el log decía
    "💾 [CHUNK] Plan parcial guardado".
  - `pending_pipeline:*` KV también vacío (depende del mismo pool).

Root cause: `MEALFIT_DB_POOL_MAX_SIZE=60` excede el cap del Supabase free
tier pgBouncer en Transaction Mode (~15-30 client connections por proyecto).
El pool local intentaba mantener 60 conexiones pero pgBouncer rechazaba al
pasar de su límite interno → checkouts bloqueados → timeout 10s → query falla.

Fix: bajar a `max=25` (alineado con pgBouncer) + raise timeout a 20s + bajar
min a 5 (mantener menos idle, liberar slots para otros consumers).

Si migras a Supabase Pro / dedicated, restaurar max=60.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _BACKEND_ROOT / ".env"


def _read_env() -> str:
    return _ENV_PATH.read_text(encoding="utf-8")


def test_pool_max_size_aligned_with_pgbouncer():
    """`MEALFIT_DB_POOL_MAX_SIZE` debe ser ≤25 para no saturar pgBouncer
    Transaction Mode de Supabase free tier. Subir a 60 reintroduce el bug."""
    text = _read_env()
    m = re.search(
        r"^MEALFIT_DB_POOL_MAX_SIZE\s*=\s*(\d+)",
        text,
        re.MULTILINE,
    )
    assert m, "Falta `MEALFIT_DB_POOL_MAX_SIZE` en .env."
    val = int(m.group(1))
    assert val <= 25, (
        f"P0-DB-POOL-PGBOUNCER-SATURATION: MAX_SIZE={val} > 25. "
        f"Si migraste a Supabase Pro / dedicated y quieres subirlo, "
        f"actualiza este threshold y el comentario en .env."
    )


def test_pool_min_size_modest():
    """`MEALFIT_DB_POOL_MIN_SIZE` debe ser ≤10. Mantener muchas conexiones
    idle consume slots de pgBouncer que otros clientes podrían usar."""
    text = _read_env()
    m = re.search(
        r"^MEALFIT_DB_POOL_MIN_SIZE\s*=\s*(\d+)",
        text,
        re.MULTILINE,
    )
    assert m, "Falta `MEALFIT_DB_POOL_MIN_SIZE` en .env."
    val = int(m.group(1))
    assert val <= 10, f"MIN_SIZE={val} > 10 — excesivo para pgBouncer free tier."


def test_pool_timeout_generous_enough():
    """`MEALFIT_DB_POOL_TIMEOUT_S` debe ser ≥15s. Default 10s era marginal
    bajo carga concurrente (pipeline + crons + chunk workers)."""
    text = _read_env()
    m = re.search(
        r"^MEALFIT_DB_POOL_TIMEOUT_S\s*=\s*(\d+)",
        text,
        re.MULTILINE,
    )
    assert m, "Falta `MEALFIT_DB_POOL_TIMEOUT_S` en .env."
    val = int(m.group(1))
    assert val >= 15, (
        f"TIMEOUT_S={val} < 15s — bajo carga, los checkouts pueden necesitar "
        f"esperar más para que un async query termine."
    )
