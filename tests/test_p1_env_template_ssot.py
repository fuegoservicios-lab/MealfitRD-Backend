"""[P1-POOL-DEFAULTS-SSOT · 2026-09-04] `.env.example` es el espejo VERSIONADO del `.env` afinado.

Lo que había: once ficheros de test leían `backend/.env` —gitignored— para anclar los valores
afinados de producción (pool del pooler, hedging, timeouts del critique y del pipeline, lotes de
embeddings). Un guard sobre un fichero que ningún checkout limpio tiene sólo pasa en la máquina
del dueño y nunca corre en CI; y el template versionado —lo que copia un deploy nuevo o un
`.env` reseteado— llevaba desde mayo los valores que esos mismos guards prohíben
(min=10/max=60/timeout=10: la configuración que SATURÓ el pooler). Nadie lo veía.

Lo que hay: los guards leen `.env.example`; los defaults del CÓDIGO (db_core.py) llevan los
valores afinados; y este fichero vigila las dos costuras que quedan:
  1. El template declara TODOS los knobs guardados (si alguien borra una línea, aquí falla).
  2. Donde exista un `.env` local, sus valores coinciden con los del template para esos knobs
     — es decir, el operador y el repo no divergen en silencio. Sin `.env` se salta, con razón.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_TEMPLATE = _BACKEND / ".env.example"
_LOCAL = _BACKEND / ".env"

# Knobs con guard propio (los ficheros de test que los anclan) — la lista es el inventario de lo
# que el template DEBE declarar explícitamente.
KNOBS_GUARDADOS = (
    "MEALFIT_DB_POOL_MIN_SIZE",            # test_p0_db_pool_pgbouncer_saturation
    "MEALFIT_DB_POOL_MAX_SIZE",            # idem + test_p0_pool_freetier_retune
    "MEALFIT_DB_POOL_TIMEOUT_S",           # test_p0_db_pool_pgbouncer_saturation
    "MEALFIT_DB_ASYNC_POOL_MIN_SIZE",      # test_p1_besteffort_db_cb
    "MEALFIT_DB_ASYNC_POOL_MAX_SIZE",      # test_p0_pool_freetier_retune + besteffort
    "MEALFIT_DB_ASYNC_POOL_TIMEOUT_S",     # idem
    "MEALFIT_GLOBAL_PIPELINE_TIMEOUT_S",   # test_p2_pipeline_timeout_raise
    "MEALFIT_HEDGE_AFTER_BASE_S",          # test_p1_cost_instrumentation_fix + besteffort
    "MEALFIT_HEDGE_MAX_CONCURRENT",        # test_p2_hedge_limiter_raise
    "MEALFIT_SHOPPING_COHERENCE_GUARD",    # test_p1_cost_instrumentation_fix
    "MEALFIT_CRITIQUE_FIX_TIMEOUT_S",      # test_p1_critique_timeout_raise(_v2)
    "MEALFIT_CRITIQUE_PRO_FALLBACK_ENABLED",
    "MEALFIT_EMBED_INIT_BATCH_SIZE",       # test_p3_embed_rpm_mitigation + warm_deadline
    "MEALFIT_EMBED_INIT_BATCH_DELAY_S",
)


def _valores(texto: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in re.finditer(r"^([A-Z][A-Z0-9_]+)\s*=\s*(.*?)\s*$", texto, re.MULTILINE):
        out[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return out


def test_el_template_declara_todos_los_knobs_guardados():
    valores = _valores(_TEMPLATE.read_text(encoding="utf-8"))
    faltan = [k for k in KNOBS_GUARDADOS if k not in valores]
    assert not faltan, (
        f"`.env.example` ya no declara {faltan}. Es el espejo versionado del `.env` afinado: "
        "un deploy nuevo lo copia tal cual, y sin la línea vuelve al default del código."
    )


def test_los_guards_leen_el_template_y_no_el_env_local():
    """Ningún test debe volver a anclar valores sobre `backend/.env` (gitignored)."""
    culpables = []
    for f in sorted((_BACKEND / "tests").glob("test_*.py")):
        if f.name == Path(__file__).name:
            continue
        src = f.read_text(encoding="utf-8", errors="replace")
        if re.search(r"""_ENV_PATH\s*=\s*\w+\s*/\s*["']\.env["']""", src):
            culpables.append(f.name)
    assert not culpables, (
        f"{culpables} vuelven a leer `backend/.env` (gitignored): ese guard no corre en CI. "
        "Apunta a `.env.example` y deja que este fichero vigile la paridad con el `.env` local."
    )


@pytest.mark.skipif(not _LOCAL.is_file(), reason="sin `.env` local (gitignored): nada que comparar")
def test_el_env_local_no_diverge_del_template_en_los_knobs_guardados():
    """En la máquina del operador, `.env` y `.env.example` deben decir lo mismo para los knobs
    con guard. Si el dueño retoca un valor en producción, el commit que lo acompaña actualiza
    el template (y el guard correspondiente); si no, esto lo pide."""
    local = _valores(_LOCAL.read_text(encoding="utf-8"))
    template = _valores(_TEMPLATE.read_text(encoding="utf-8"))
    divergen = {
        k: (local.get(k), template.get(k))
        for k in KNOBS_GUARDADOS
        if k in local and local[k].lower() != template.get(k, "").lower()
    }
    assert not divergen, (
        "`.env` local y `.env.example` divergen en knobs guardados (local, template): "
        f"{divergen}. Actualiza el template en el mismo commit que el ajuste operativo."
    )
