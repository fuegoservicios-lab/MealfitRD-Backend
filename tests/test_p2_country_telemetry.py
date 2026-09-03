"""[P2-COUNTRY-TELEMETRY · 2026-08-21] «¿Los planes de España salen peor que los dominicanos?» no
se podía responder sin abrir cada plan a mano.

La telemetría del pipeline es estructuralmente ciega al país. `slot_drift` —la métrica que mide si
el reparto fisiológico por slot se respeta— se persiste a `pipeline_metrics` con `user_id=None`
cableado y una metadata que sólo lleva el propio dict y el número de días. Sin país no hay forma de
agrupar, y sin agrupar no hay forma de saber si el flip empeoró algo.

Es la pregunta que ordena el resto de la ola: cada P-fix de aquí ha tenido que MEDIRSE abriendo la
base de datos a mano porque no existía el eje.

QUÉ SE CIERRA Y QUÉ NO:

  · **El país sí entra.** Sale de `country_for_form_data(state['form_data'])`, la única puerta —
    no de una segunda derivación. Con el knob apagado da 'DO' para todos, que es la conducta de
    siempre, así que la fila no cambia de forma para nadie hasta el flip.

  · **El `user_id` NO.** Está cableado a `None` porque `PlanState` no lo lleva: el nodo recibe
    `form_data`, no la identidad. Meterlo exige cambiar el estado del grafo, que es bastante más
    que la «S» que la auditoría le puso a este gap — y correlacionar por usuario es una pregunta
    distinta de la que aquí se contesta. Se deja escrito para que el siguiente no lo lea como un
    olvido.

Un `alert_key` por país tampoco se añade, y esa omisión es deliberada: las alertas se emiten cuando
hay que ACTUAR, y hoy no existe una acción distinta para «el slot_drift de España es peor». Primero
la métrica agrupable; la alerta, cuando haya umbral que defender.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


@pytest.fixture
def knob_on(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")


@pytest.fixture
def capturado(monkeypatch, go):
    """Intercepta el INSERT y devuelve la metadata parseada."""
    caja = {}

    def _fake_write(sql, params=None, *a, **k):
        caja["sql"] = sql
        caja["params"] = params
        return True

    import db_core
    monkeypatch.setattr(db_core, "execute_sql_write", _fake_write)

    def _emitir(slot_drift, plan, form_data=None):
        caja.clear()
        go._emit_slot_drift_metric_best_effort(slot_drift, plan, form_data)
        if not caja.get("params"):
            return None
        return json.loads(caja["params"][-1])
    return _emitir


_DRIFT = {"score": 0.42, "per_macro": {"protein": 0.9}}
_PLAN = {"days": [{"day": 1, "meals": []}, {"day": 2, "meals": []}]}


# ── El país entra en la métrica ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cc", ["ES", "MX", "CO", "PR", "US", "DO"])
def test_la_metrica_lleva_el_pais(capturado, knob_on, cc):
    """Sin esto la pregunta «¿los planes ES salen peor?» sólo se contesta abriendo planes a mano —
    que es exactamente como se ha medido cada gap de esta ola."""
    meta = capturado(_DRIFT, _PLAN, {"country": cc})
    assert meta is not None, "no se emitió la fila"
    assert meta.get("country") == cc, f"la metadata no lleva el país: {meta}"


def test_el_pais_sale_de_la_unica_puerta(go):
    """`country_for_form_data` es el SSOT. Una segunda derivación aquí sería la tabla que
    P1-DIET-CANON-SSOT ya pagó una vez."""
    import inspect
    src = inspect.getsource(go._emit_slot_drift_metric_best_effort)
    assert "country_for_form_data" in src
    assert "canonicalize_country" not in src, (
        "el emisor canonicaliza por su cuenta en vez de pasar por la única puerta"
    )


def _argumentos_del_call_site(src: str) -> str:
    r"""Los argumentos de la llamada a `_emit_slot_drift_metric_best_effort`, balanceando paréntesis.

    Un `\(([^)]*)\)` no vale aquí: el argumento es `state.get("form_data")` y lleva un `)` DENTRO,
    así que la clase negada se corta en el paréntesis equivocado y el test falla contra código
    correcto. Me pasó al primer intento — y un guard que falla contra lo bueno se acaba borrando,
    que es la peor forma de perder una defensa."""
    marca = "_emit_slot_drift_metric_best_effort("
    i = src.find(marca)
    assert i > 0, "`review_plan_node` ya no emite `slot_drift` (¿se movió el call site?)"
    j = i + len(marca)
    nivel = 1
    while j < len(src) and nivel:
        if src[j] == "(":
            nivel += 1
        elif src[j] == ")":
            nivel -= 1
        j += 1
    return src[i + len(marca):j - 1]


def test_el_call_site_le_pasa_el_form_data(go):
    """Lo pidió la mutación: quitar `state.get("form_data")` del call site NO rompía ningún test,
    porque todos los de arriba llaman al emisor DIRECTAMENTE. Sin este, el país saldría 'DO' para
    todo el mundo en producción y los tests seguirían en verde — la función correcta a la que nadie
    llama bien, que es el modo de fallo que esta ola lleva encontrando desde el primer día."""
    import inspect
    args = _argumentos_del_call_site(inspect.getsource(go.review_plan_node))
    assert "form_data" in args, (
        f"el call site no le pasa `form_data` ({args!r}): el país saldría 'DO' para todos y la "
        f"métrica volvería a ser ciega, con los tests en verde"
    )


def test_sin_form_data_no_revienta_ni_inventa_pais(capturado, knob_on):
    """Es telemetría best-effort: no puede tumbar una generación, y tampoco puede afirmar un país
    que nadie declaró."""
    meta = capturado(_DRIFT, _PLAN, None)
    assert meta is not None
    assert meta.get("country") == "DO", "sin form_data debería caer al fail-safe de siempre"


def test_con_el_knob_apagado_la_fila_no_cambia_de_forma(capturado, monkeypatch):
    """Contrato de rollback: apagado, `country_for_form_data` devuelve 'DO' para todos, así que la
    fila es la de siempre más una clave constante — nada que reinterpretar."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "false")
    for cc in ("ES", "MX", "DO"):
        assert capturado(_DRIFT, _PLAN, {"country": cc}).get("country") == "DO"


# ── Lo que ya medía sigue midiendo ──────────────────────────────────────────────────────────────

def test_no_se_pierde_lo_que_la_metrica_ya_llevaba(capturado, knob_on):
    """El error caro de añadir un campo: pisar los que ya estaban. `slot_drift` y `days` son la
    razón por la que esta fila existe."""
    meta = capturado(_DRIFT, _PLAN, {"country": "ES"})
    assert meta.get("slot_drift") == _DRIFT
    assert meta.get("days") == 2


def test_un_slot_drift_vacio_sigue_sin_emitir(capturado, knob_on):
    """La guarda de entrada no cambia: sin drift no hay fila que escribir."""
    assert capturado({}, _PLAN, {"country": "ES"}) is None
    assert capturado(None, _PLAN, {"country": "ES"}) is None


def test_sigue_siendo_best_effort(go, monkeypatch, knob_on):
    """Telemetría que tumba una generación es peor que no tenerla."""
    import db_core

    def _boom(*a, **k):
        raise RuntimeError("DB caída")
    monkeypatch.setattr(db_core, "execute_sql_write", _boom)
    go._emit_slot_drift_metric_best_effort(_DRIFT, _PLAN, {"country": "ES"})


# ── Lo que NO se cierra, escrito ────────────────────────────────────────────────────────────────

def test_el_user_id_sigue_siendo_none_y_esta_documentado(go):
    """`PlanState` no lleva la identidad: el nodo recibe `form_data`, no el usuario. Meterlo exige
    cambiar el estado del grafo — bastante más que la «S» de este gap, y una pregunta distinta.
    Se ancla para que el siguiente que lo lea no lo tome por un olvido."""
    import inspect
    src = inspect.getsource(go._emit_slot_drift_metric_best_effort)
    assert "user_id" in src and "PlanState" in src, (
        "desapareció la explicación de por qué el user_id sigue en None"
    )
