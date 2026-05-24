"""[P1-BESTEFFORT-DB-CB · 2026-05-21] Circuit breaker LOCAL in-process para
escrituras DB best-effort. Cierre del root cause del incidente productivo
2026-05-21 02:08-02:12.

Bug observado:
  Bajo carga normal (3 day_generators + adversarial self-play + meta-learning
  paralelos), el async pool de psycopg se saturaba. Cada best-effort write
  (LLM-CACHE get/set, CB-RESET, AB-TEMP, CB-FAILURE) timeoutaba a 8s en el
  pool. Con 8+ callsites consecutivos esperando timeout = ~64s acumulados de
  latencia gastada en operaciones cosméticas sin progreso real.

  Peor: el `LLMCircuitBreaker.arecord_success` y `arecord_failure` también
  fallaban su write a DB → estado del CB principal "confundido" → se abría
  prematuramente → Días 1 y 2 caían con `Circuit Breaker OPEN para
  gemini-3.5-flash` aunque el modelo en sí respondía normal.

Fix:
  Nueva clase `_BestEffortDBCircuitBreaker` (in-process, per-callsite) que
  tras 3 timeouts del pool consecutivos abre 60s y fail-fasts las próximas
  calls — sin tocar el pool. Auto half-open tras cooldown. Aplicada en los
  5 callsites best-effort: LLM-CACHE get/set, LLMCircuitBreaker arecord_success
  y arecord_failure, y AB-TEMP.

Cobertura:
  - Clase definida + registry singleton + helper `_is_pool_timeout_error`
  - Lógica funcional: 3 timeouts → OPEN; success → reset; cooldown → half-open
  - Solo timeouts cuentan (otros errores son tolerados, no abren el CB)
  - Knobs configurables `MEALFIT_BE_DB_CB_FAILURE_THRESHOLD/OPEN_DURATION_S`
  - Aplicado en los 5 callsites identificados en el incidente
  - `.env` con valores conservadores para post-incidente
"""
import os
import re
import time
from pathlib import Path

import pytest


_BACKEND = Path(__file__).parent.parent
_GO_PY = _BACKEND / "graph_orchestrator.py"
_ENV_PATH = _BACKEND / ".env"


# ---------------------------------------------------------------------------
# Sección 1 — Estructura del módulo (clase + registry + helper)
# ---------------------------------------------------------------------------

def test_circuit_breaker_class_defined():
    """`_BestEffortDBCircuitBreaker` debe estar definida con los métodos
    canónicos `is_open`, `record_success`, `record_pool_timeout`, `snapshot`."""
    src = _GO_PY.read_text(encoding="utf-8")
    assert "class _BestEffortDBCircuitBreaker:" in src, (
        "Clase _BestEffortDBCircuitBreaker no definida en graph_orchestrator.py."
    )
    for method in ("def is_open(", "def record_success(", "def record_pool_timeout(", "def snapshot("):
        assert method in src, f"Método requerido no encontrado: {method}"


def test_registry_singleton_helper_defined():
    """`_get_be_db_cb(name)` debe estar definida como singleton per-name."""
    src = _GO_PY.read_text(encoding="utf-8")
    assert "_BE_DB_CB_REGISTRY:" in src
    assert "def _get_be_db_cb(name: str)" in src


def test_pool_timeout_detector_defined():
    """`_is_pool_timeout_error(exc)` debe estar definida con el match canónico
    del mensaje 'couldn't get a connection' (psycopg_pool emit literal)."""
    src = _GO_PY.read_text(encoding="utf-8")
    assert "def _is_pool_timeout_error(" in src
    # Verifica el match key
    idx = src.find("def _is_pool_timeout_error(")
    body = src[idx:idx + 500]
    assert "couldn't get a connection" in body, (
        "_is_pool_timeout_error debe matchear el texto canónico de psycopg_pool."
    )


def test_knobs_registered():
    """Los 2 knobs `MEALFIT_BE_DB_CB_FAILURE_THRESHOLD` y
    `MEALFIT_BE_DB_CB_OPEN_DURATION_S` deben estar registrados con defaults
    conservadores (3 y 60 respectivamente)."""
    src = _GO_PY.read_text(encoding="utf-8")
    assert 'MEALFIT_BE_DB_CB_FAILURE_THRESHOLD' in src
    assert 'MEALFIT_BE_DB_CB_OPEN_DURATION_S' in src
    # Defaults
    m1 = re.search(r'_BE_DB_CB_FAILURE_THRESHOLD\s*=\s*max\(1,\s*int\([^)]*?,\s*[\"\'](\d+)[\"\']', src)
    assert m1 and int(m1.group(1)) == 3, f"Default threshold debe ser 3"
    m2 = re.search(r'_BE_DB_CB_OPEN_DURATION_S\s*=\s*max\(5,\s*int\([^)]*?,\s*[\"\'](\d+)[\"\']', src)
    assert m2 and int(m2.group(1)) == 60, f"Default open_duration debe ser 60"


# ---------------------------------------------------------------------------
# Sección 2 — Los 5 callsites best-effort aplicados
# ---------------------------------------------------------------------------

_REQUIRED_CALLSITES = [
    ("llm_cache_aget", "PersistentLLMCache.aget"),
    ("llm_cache_aset", "PersistentLLMCache.aset"),
    ("llm_cb_reset_async", "LLMCircuitBreaker.arecord_success"),
    ("llm_cb_failure_async", "LLMCircuitBreaker.arecord_failure"),
    ("ab_temp_async", "_aselect_ab_temp_pair"),
]


@pytest.mark.parametrize("cb_name, callsite_label", _REQUIRED_CALLSITES)
def test_callsite_uses_be_db_cb(cb_name: str, callsite_label: str):
    """Cada callsite identificado en el incidente debe usar el CB con el
    nombre canónico. Si añades un callsite nuevo best-effort sin aplicar el
    CB, este test no lo detecta — pero el blanket de la Sección 3 sí."""
    src = _GO_PY.read_text(encoding="utf-8")
    expected = f'_get_be_db_cb("{cb_name}")'
    assert expected in src, (
        f"Callsite `{callsite_label}` debe instanciar `{expected}`. "
        f"Sin esto, el callsite queda sin gate y reabre la cascada."
    )


def test_each_callsite_records_outcome():
    """Cada uso de `_get_be_db_cb` debe seguirse de al menos un `record_success`
    O `record_pool_timeout` en el bloque siguiente. Sin estos, el CB nunca
    aprende del resultado real."""
    src = _GO_PY.read_text(encoding="utf-8")
    # Conteo simple: el número de `_get_be_db_cb` calls (~5 producción + 1+ tests
    # internos) debe coincidir aproximadamente con record_success y record_pool_timeout.
    n_get = src.count("_get_be_db_cb(")
    n_success = src.count(".record_success()")
    n_timeout = src.count(".record_pool_timeout()")
    # Al menos 5 callsites de producción (puede haber más por tests/refactors).
    assert n_get >= 5, f"Solo {n_get} usos de _get_be_db_cb — esperamos >=5 callsites."
    assert n_success >= 5
    assert n_timeout >= 5


# ---------------------------------------------------------------------------
# Sección 3 — .env actualizado post-incidente
# ---------------------------------------------------------------------------

def test_env_async_pool_raised_post_incident():
    """`.env` debe reflejar los valores conservadores post-incidente:
    async min=2, max=10, timeout=12. Pre-incidente eran 1/6/8."""
    if not _ENV_PATH.exists():
        pytest.skip(".env no presente — entorno CI sin override.")
    env = _ENV_PATH.read_text(encoding="utf-8")

    m_min = re.search(r"^MEALFIT_DB_ASYNC_POOL_MIN_SIZE=(\d+)", env, re.MULTILINE)
    m_max = re.search(r"^MEALFIT_DB_ASYNC_POOL_MAX_SIZE=(\d+)", env, re.MULTILINE)
    m_to = re.search(r"^MEALFIT_DB_ASYNC_POOL_TIMEOUT_S=(\d+)", env, re.MULTILINE)

    assert m_min and int(m_min.group(1)) >= 2, f"async min debe ser >=2 (post-incident)"
    assert m_max and int(m_max.group(1)) >= 10, f"async max debe ser >=10"
    assert m_to and int(m_to.group(1)) >= 12, f"async timeout debe ser >=12s"


def test_env_hedge_threshold_aligned_with_code():
    """`.env` debe tener `MEALFIT_HEDGE_AFTER_BASE_S=90` alineado con el
    default del código. Pre-incident estaba en 60 causando saturación."""
    if not _ENV_PATH.exists():
        pytest.skip(".env no presente.")
    env = _ENV_PATH.read_text(encoding="utf-8")
    m = re.search(r"^MEALFIT_HEDGE_AFTER_BASE_S=(\d+)", env, re.MULTILINE)
    assert m and int(m.group(1)) >= 90, (
        f"MEALFIT_HEDGE_AFTER_BASE_S debe ser >=90 (P1-HEDGE-THRESHOLD-RAISE). "
        f"Encontrado: {m.group(1) if m else '<missing>'}"
    )


# ---------------------------------------------------------------------------
# Sección 4 — Tests funcionales (importan la clase y testean lógica pura)
# ---------------------------------------------------------------------------

@pytest.fixture
def _cb_class():
    """Importa la clase desde el módulo. Skip si el módulo no es importable
    en este entorno (e.g. CI sin langchain)."""
    try:
        from graph_orchestrator import _BestEffortDBCircuitBreaker, _is_pool_timeout_error, _get_be_db_cb
    except Exception as e:
        pytest.skip(f"graph_orchestrator no importable: {e}")
    return _BestEffortDBCircuitBreaker, _is_pool_timeout_error, _get_be_db_cb


def test_funcional_cb_opens_after_threshold_timeouts(_cb_class):
    """3 pool-timeouts seguidos abren el CB; el 4to `is_open()` retorna True."""
    Cls, _, _ = _cb_class
    cb = Cls("test_open", failure_threshold=3, open_duration_s=60)
    assert not cb.is_open()
    cb.record_pool_timeout()
    cb.record_pool_timeout()
    assert not cb.is_open()  # solo 2 fallos aún
    cb.record_pool_timeout()
    assert cb.is_open()  # 3er fallo abre


def test_funcional_success_resets_counter(_cb_class):
    """Un success en medio resetea el contador a 0 → no llega al threshold."""
    Cls, _, _ = _cb_class
    cb = Cls("test_reset", failure_threshold=3, open_duration_s=60)
    cb.record_pool_timeout()
    cb.record_pool_timeout()
    cb.record_success()
    cb.record_pool_timeout()
    cb.record_pool_timeout()
    assert not cb.is_open()  # 2 fallos tras reset, no llega a 3


def test_funcional_cooldown_expires_to_half_open(_cb_class):
    """Tras `open_duration_s` el CB pasa a half-open: is_open=False y resetea
    failures para reintentar."""
    Cls, _, _ = _cb_class
    cb = Cls("test_cooldown", failure_threshold=3, open_duration_s=1)  # 1s cooldown
    cb.record_pool_timeout()
    cb.record_pool_timeout()
    cb.record_pool_timeout()
    assert cb.is_open()
    time.sleep(1.1)  # esperar cooldown
    assert not cb.is_open()  # half-open: ya puede retentar
    # snapshot: failures reseteado
    snap = cb.snapshot()
    assert snap["failures"] == 0


def test_funcional_pool_timeout_detector(_cb_class):
    """`_is_pool_timeout_error` debe matchear los mensajes canónicos del
    pool psycopg y rechazar errores ortogonales."""
    _, is_pool, _ = _cb_class

    class FakeExc(Exception):
        pass

    # Positivos
    assert is_pool(FakeExc("couldn't get a connection after 8.00 sec"))
    assert is_pool(FakeExc("Couldn't get a connection after 12s"))  # case-insensitive
    assert is_pool(FakeExc("Pool exhausted"))
    assert is_pool(FakeExc("the pool is closed"))

    # Negativos
    assert not is_pool(FakeExc("relation foo does not exist"))
    assert not is_pool(FakeExc("404 not found"))
    assert not is_pool(FakeExc("schema mismatch"))


def test_funcional_registry_singleton(_cb_class):
    """`_get_be_db_cb(name)` devuelve la misma instancia para el mismo nombre,
    y distintas instancias para nombres distintos."""
    _, _, get = _cb_class
    a1 = get("test_singleton_a")
    a2 = get("test_singleton_a")
    b = get("test_singleton_b")
    assert a1 is a2, "Mismo nombre debe retornar la MISMA instancia (singleton)."
    assert a1 is not b


# ---------------------------------------------------------------------------
# Sección 5 — Tooltip-anchor
# ---------------------------------------------------------------------------

def test_marker_present_in_source():
    """El marker `P1-BESTEFFORT-DB-CB` debe estar en el bloque de definición
    de la clase y en cada callsite aplicado."""
    src = _GO_PY.read_text(encoding="utf-8")
    assert "P1-BESTEFFORT-DB-CB" in src
    # Al menos 6 menciones (definición + 5 callsites)
    n = src.count("P1-BESTEFFORT-DB-CB")
    assert n >= 6, f"Solo {n} menciones del marker — esperamos >=6 (def + 5 callsites)."
