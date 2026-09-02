"""[P1-DREAMING-CB-KWARG · 2026-07-24] Un circuit breaker que no abría nunca.

`dreaming.py` construía el breaker en sus tres call sites así:

    LLMCircuitBreaker(model_id).can_proceed()

pero el primer parámetro posicional de `LLMCircuitBreaker` es `failure_threshold`, no el
modelo. Resultado: `self.threshold = "glm-5.3-flash"` (str) y `model_name = None`.

Dos consecuencias, ambas silenciosas:

  1. `failures >= self.threshold` → `TypeError: '>=' not supported between 'int' and 'str'`,
     que el `except Exception: pass` de cada call site se tragaba. **El breaker no abría
     jamás**, así que el cron seguía golpeando un proveedor caído.
  2. Sin `model_name`, las keys perdían el sufijo por modelo (P1-Q3) y este cron escribía
     sobre el breaker GLOBAL legacy (`cb:llm:failures`), contaminando el estado compartido.

En producción se veía como 17 errores de escritura del CB sin un solo breaker abierto.

Un breaker roto no protege de nada, así que degradar a "no protege" en silencio es el peor
resultado posible: además del fix en los call sites, el constructor ahora corrige la
intención evidente y grita.
"""
from pathlib import Path

import pytest

import dreaming
import graph_orchestrator as go


# ───────────── 1. los call sites ─────────────

def test_dreaming_usa_kwarg_en_los_tres_call_sites():
    src = Path(dreaming.__file__).read_text(encoding="utf-8")
    llamadas = [l.strip() for l in src.splitlines() if "LLMCircuitBreaker(" in l and "import" not in l]
    assert len(llamadas) == 3, llamadas
    for l in llamadas:
        assert "LLMCircuitBreaker(model_name=" in l, (
            f"el primer posicional es failure_threshold, no el modelo: {l}"
        )


# ───────────── 2. el constructor se defiende ─────────────

def test_posicional_con_str_se_corrige_y_grita(caplog):
    with caplog.at_level("ERROR"):
        cb = go.LLMCircuitBreaker("glm-5.3-flash")
    assert cb.threshold == 3, "el threshold vuelve al default numérico"
    assert cb.model_name == "glm-5.3-flash", "el modelo se interpreta como lo que era"
    assert "P1-DREAMING-CB-KWARG" in caplog.text, "silencioso = el peor resultado posible"


def test_comparacion_de_umbral_ya_no_revienta():
    """El síntoma exacto: la comparación que decide si el breaker abre."""
    cb = go.LLMCircuitBreaker("glm-5.3-flash")
    assert (3 >= cb.threshold) is True   # antes: TypeError, tragado por `except Exception: pass`


def test_keys_recuperan_el_sufijo_por_modelo():
    """Sin sufijo, el cron del dreaming escribía sobre el breaker global legacy."""
    cb = go.LLMCircuitBreaker("glm-5.3-flash")
    assert cb._failures_key == "cb:llm:failures:glm-5.3-flash"
    assert cb._open_key == "cb:llm:open:glm-5.3-flash"


def test_el_kwarg_explicito_gana_sobre_el_posicional():
    cb = go.LLMCircuitBreaker("glm-5.3-flash", model_name="glm-5.3")
    assert cb.model_name == "glm-5.3"


@pytest.mark.parametrize("thr", [1, 5, 10])
def test_uso_normal_intacto(thr):
    cb = go.LLMCircuitBreaker(thr, model_name="m")
    assert cb.threshold == thr and cb.model_name == "m"


def test_sin_argumentos_conserva_las_keys_legacy():
    """Compatibilidad P1-Q3: sin modelo, las keys quedan como antes."""
    cb = go.LLMCircuitBreaker()
    assert cb.threshold == 3 and cb.model_name is None
    assert cb._failures_key == "cb:llm:failures"
