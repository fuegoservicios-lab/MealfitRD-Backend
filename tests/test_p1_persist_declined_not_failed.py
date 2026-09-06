# -*- coding: utf-8 -*-
"""[P1-PERSIST-DECLINED-NOT-FAILED · 2026-09-06] Una escritura RECHAZADA no es una escritura FALLIDA.

`fill_placeholder_meal_plan_atomic` devuelve `None` por cinco motivos distintos y el caller los
trataba a todos igual: «INSERT de meal_plans fallido». Dos de esos cinco son el fence de
`P0-FILL-FENCED` haciendo exactamente su trabajo:

  · `fence_declined` — la fila de la cola dice que el trabajo ya no es nuestro.
  · `already_filled` — el placeholder ya lo rellenó otro worker.

En los dos, el plan del usuario **existe o va a existir**, escrito por el worker que sí tiene el
claim. Y sin embargo el caller levantaba tres señales falsas por el evento que el fence existe para
producir:

  1. `logger.error` con 🛑 diciendo «INSERT meal_plans fallido».
  2. una `system_alert` operacional que nadie podía cerrar, porque no había nada roto.
  3. `_persist_failed` → `_fail("persist_failed")` en el ciclo de vida del chunk.

Visto en producción el 2026-09-06 a las 09:51, en el mismo evento que destapó el bug de `attempts=0`
(`int(x or -1)`: cero es falso en Python). Aquel se corrigió el mismo día; el mensaje que lo
acompañaba sigue mintiendo, y es el que un operador lee primero.

**Lo que NO cambia:** en SSE y sync el `_persist_failed` se conserva aunque el motivo sea un rechazo.
Solo la cola tiene un worker (`run_initial_chunk`) que sabe retirarse en silencio; en los otros dos
transportes hay un usuario esperando que necesita una respuesta definitiva.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import db_plans  # noqa: E402

from test_p0_fill_fenced import _insert_data, _preparar  # noqa: E402

_ROUTERS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_LIFECYCLE = (_BACKEND / "generation_lifecycle.py").read_text(encoding="utf-8")
_SERVICES = (_BACKEND / "services.py").read_text(encoding="utf-8")


# ── el motivo llega al caller ─────────────────────────────────────────────────────────────────
def test_el_fence_dice_que_rechazo(monkeypatch):
    """Desplazado: la cola va por attempts=4 y nosotros creíamos ser el 3."""
    _preparar(monkeypatch, {"status": "processing", "attempts": 4})
    out: dict = {}
    assert db_plans.fill_placeholder_meal_plan_atomic(
        "plan-1", "user-1", _insert_data(attempts=3), outcome=out) is None
    assert out["reason"] == "fence_declined"


def test_el_chunk_cancelado_tambien_es_rechazo(monkeypatch):
    _preparar(monkeypatch, {"status": "cancelled", "attempts": 3})
    out: dict = {}
    assert db_plans.fill_placeholder_meal_plan_atomic(
        "plan-1", "user-1", _insert_data(attempts=3), outcome=out) is None
    assert out["reason"] == "fence_declined"


def test_el_placeholder_ya_relleno_es_rechazo_no_fallo(monkeypatch):
    """`generation_status != 'generating'`: otro worker ya escribió el plan. El usuario LO TIENE."""
    cur = _preparar(monkeypatch, {"status": "processing", "attempts": 3})
    cur.plan_row = {"plan_data": {"generation_status": "partial"}}
    out: dict = {}
    assert db_plans.fill_placeholder_meal_plan_atomic(
        "plan-1", "user-1", _insert_data(attempts=3), outcome=out) is None
    assert out["reason"] == "already_filled"


def test_el_placeholder_ausente_si_es_un_fallo(monkeypatch):
    """Aquí no hay nadie más escribiendo: la fila no existe o no es de este usuario."""
    cur = _preparar(monkeypatch, {"status": "processing", "attempts": 3})
    cur.plan_row = None
    out: dict = {}
    assert db_plans.fill_placeholder_meal_plan_atomic(
        "plan-1", "user-1", _insert_data(attempts=3), outcome=out) is None
    assert out["reason"] == "placeholder_missing"


def test_el_camino_bueno_tambien_se_declara(monkeypatch):
    """Si solo se marcara el fracaso, un `reason` viejo sobreviviría a un éxito posterior."""
    _preparar(monkeypatch, {"status": "processing", "attempts": 3})
    out: dict = {"reason": "fence_declined"}
    assert db_plans.fill_placeholder_meal_plan_atomic(
        "plan-1", "user-1", _insert_data(attempts=3), outcome=out) == "plan-1"
    assert out["reason"] == "ok"


def test_sin_outcome_nadie_nota_el_cambio(monkeypatch):
    """El parámetro es opcional: los callers que no lo pasan se comportan igual que ayer."""
    _preparar(monkeypatch, {"status": "processing", "attempts": 4})
    assert db_plans.fill_placeholder_meal_plan_atomic(
        "plan-1", "user-1", _insert_data(attempts=3)) is None


@pytest.mark.parametrize("motivo", ["fence_declined", "already_filled",
                                    "placeholder_missing", "invalid_plan_data",
                                    "update_no_rows", "ok"])
def test_los_seis_motivos_estan_declarados_en_el_fuente(motivo):
    """Si alguien añade un `return None` nuevo sin motivo, el caller lo leerá como 'unknown' y
    alertará — que es el fail-safe correcto, pero conviene que la lista esté escrita."""
    src = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
    i = src.index("def fill_placeholder_meal_plan_atomic(")
    fin = src.index("\ndef ", i + 10)      # hasta el final de ESA función, no una ventana a ojo
    assert f'"{motivo}"' in src[i:fin], motivo


# ── el caller no alerta por un rechazo ────────────────────────────────────────────────────────
def _bloque_persist() -> str:
    i = _ROUTERS.index("_persist_reason = str(")
    return _ROUTERS[i:i + 2500]


def test_la_alerta_operacional_solo_se_emite_si_fallo():
    """`_persist_plan_persist_failed_alert` levanta una fila en `system_alerts` de modelo Manual: un
    operador tiene que investigarla. Emitirla por un rechazo del fence la vuelve ruido que no se
    puede cerrar, porque no hay nada roto que arreglar."""
    b = _bloque_persist()
    i_guard = b.index("if not _declinada:")
    i_alert = b.index("_persist_plan_persist_failed_alert(")
    assert i_guard < i_alert, "la alerta debe quedar DENTRO de la rama de fallo real"


def test_el_error_rojo_tambien_queda_dentro_de_la_rama_de_fallo():
    b = _bloque_persist()
    assert b.index("if not _declinada:") < b.index("[P2-PLAN-PERSIST-FAILED] save_partial_plan_get_id")


def test_solo_la_cola_se_libra_de_persist_failed():
    """SSE y sync conservan `_persist_failed` aunque el motivo sea un rechazo: allí hay un usuario
    esperando y necesita una respuesta definitiva. Solo `run_initial_chunk` sabe retirarse."""
    assert 'if not _declinada or transport_label != "queue":' in _ROUTERS
    assert 'result["_persist_failed"] = True' in _ROUTERS


def test_los_dos_motivos_de_rechazo_son_exactamente_esos_dos():
    assert '_persist_reason in ("fence_declined", "already_filled")' in _ROUTERS


def test_el_motivo_desconocido_alerta():
    """Fail-safe: un `return None` futuro sin motivo cae en la rama ruidosa, no en la silenciosa."""
    assert '(_persist_outcome or {}).get("reason") or "unknown"' in _ROUTERS


def test_el_outcome_se_pasa_de_verdad():
    """Sin esto el motivo siempre sería 'unknown' y el arreglo quedaría inerte — el modo de fallo
    que este repo ya conoce: una defensa cableada que nadie invoca."""
    assert "outcome=_persist_outcome," in _ROUTERS
    assert "outcome=outcome)" in _SERVICES


# ── el worker desplazado se retira en silencio ────────────────────────────────────────────────
def test_el_worker_desplazado_no_llama_a_fail():
    """`_fail` marca el chunk (pending con reintento, o failed al agotarse) por un evento que no es
    un fallo. La rama del CAS de más abajo ya se retiraba así; ésta hace lo mismo."""
    i_dec = _LIFECYCLE.index('if result.get("_persist_declined"):')
    i_fail = _LIFECYCLE.index('if result.get("_persist_failed"):')
    assert i_dec < i_fail, "el rechazo debe comprobarse ANTES del fallo"
    entre = _LIFECYCLE[i_dec:i_fail]
    assert "_fail(" not in entre, entre
    assert "return" in entre


def test_el_rechazo_emite_la_misma_metrica_que_el_cas():
    """`fencing_rejected` es la métrica que ya cuenta los desplazamientos; un sitio nuevo con nombre
    nuevo partiría la serie en dos."""
    i = _LIFECYCLE.index('if result.get("_persist_declined"):')
    bloque = _LIFECYCLE[i:i + 900]
    assert '_emit_lifecycle_metric("fencing_rejected"' in bloque
    assert '"site": "persist"' in bloque
