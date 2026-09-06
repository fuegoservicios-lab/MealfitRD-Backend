# -*- coding: utf-8 -*-
"""[P1-QUALITY-ALERT-PLAN-ID · 2026-09-06] La alerta que no nombra ninguna fila no la cierra nadie.

`_emit_plan_quality_degraded_alert` corre en `should_retry`, o sea **antes** del INSERT: en el
camino inicial no hay plan que nombrar y la clave cae al centinela `:no_plan_id`. Medido el
06-sep: **112 de las 118** alertas abiertas de esta familia terminan así, todas con
`caller_context=initial_generate`. Los dos barridos que se añadieron hoy funcionan y aun así el
backlog no baja, porque una alerta sin referente no es cerrable por construcción.

El canje reutiliza el idioma que ya existía para el costo (`attach_plan_id_to_usage_events`: «el
emisor sólo pudo estampar el id de correlación; aquí se canjea»), en el mismo punto del código y
con el mismo id. Se re-clava la alerta en vez de añadir el plan_id a la metadata: así queda
indistinguible de una emitida con plan_id y los barridos existentes la tratan sin cambiar una
línea — la alternativa era repartir un `COALESCE` por cada consumidor, que es como nacen los
drifts.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import services  # noqa: E402

_GO_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_SVC_SRC = (_BACKEND / "services.py").read_text(encoding="utf-8")


def _sql_del_canje() -> str:
    i = _SVC_SRC.find("def _promote_quality_alert_to_plan_id")
    assert i > 0, "renombraron el canje: actualiza este test junto al renombre"
    j = _SVC_SRC.find("\ndef ", i + 10)
    return _SVC_SRC[i:j if j > 0 else len(_SVC_SRC)]


# ── el emisor deja el asa ────────────────────────────────────────────────────────────────────
def test_el_emisor_estampa_correlacion_solo_cuando_falta_el_plan():
    """Estamparla siempre sería ruido; el asa solo hace falta cuando hay algo que canjear."""
    i = _GO_SRC.find('alert_key = f"plan_quality_degraded:{user_id}:{plan_id}"')
    assert i > 0, "cambió la construcción del alert_key"
    trozo = _GO_SRC[i:i + 2200]
    assert 'if plan_id == "no_plan_id":' in trozo, (
        "el id de correlación dejó de estamparse condicionalmente")
    assert "get_correlation_id" in trozo


def test_la_correlacion_llega_a_la_metadata():
    i = _GO_SRC.find('"caller_context": caller_context,  # P1-NEW-9')
    assert i > 0
    assert '"correlation_id": _corr_id' in _GO_SRC[i:i + 400], (
        "la metadata no lleva el id de correlación: sin él no hay canje posible")


# ── las tres guardas del canje ───────────────────────────────────────────────────────────────
def test_el_canje_exige_la_misma_corrida():
    """Sin esto, una alerta vieja del mismo usuario se llevaría el plan_id de un plan ajeno."""
    assert "metadata->>'correlation_id' = %s" in _sql_del_canje()


def test_el_canje_no_reabre_una_alerta_cerrada():
    assert "a.resolved_at IS NULL" in _sql_del_canje()


def test_el_canje_respeta_la_unicidad_de_alert_key():
    """`system_alerts_alert_key_key` es UNIQUE. Sin el `NOT EXISTS`, re-clavar sobre una clave ya
    existente reventaría el UPDATE — y este canje corre justo después de persistir un plan, o sea
    en el peor sitio posible para lanzar una excepción."""
    sql = _sql_del_canje()
    assert "NOT EXISTS" in sql
    i, j = sql.find("NOT EXISTS"), sql.find("RETURNING")
    assert 0 < i < j and "system_alerts b" in sql[i:j], (
        "el NOT EXISTS no comprueba la clave destino en la propia tabla")


def test_el_canje_solo_toca_la_familia_correcta():
    sql = _sql_del_canje()
    assert "'plan_quality_degraded:%%:no_plan_id'" in sql or \
           "'plan_quality_degraded:%:no_plan_id'" in sql, \
        "el filtro dejó de acotarse a las claves centinela de esta familia"


# ── comportamiento ───────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("plan_id, corr", [("", "c1"), ("p1", None), ("", None), (None, "c1")])
def test_sin_las_dos_piezas_es_no_op(plan_id, corr, monkeypatch):
    """No basta con no fallar: no debe ni tocar la base. Un canje sin id de correlación
    re-clavaría una alerta al azar del mismo usuario."""
    llamadas = []
    import db_core
    monkeypatch.setattr(db_core, "execute_sql_write",
                        lambda *a, **k: llamadas.append(a) or [])
    assert services._promote_quality_alert_to_plan_id(plan_id, corr) == 0
    assert not llamadas, "tocó la base sin tener las dos piezas"


def test_un_fallo_de_base_no_propaga(monkeypatch):
    """Corre inmediatamente después de persistir el plan. Una alerta mal etiquetada jamás puede
    costar un plan."""
    import db_core

    def _revienta(*a, **k):
        raise RuntimeError("la base dijo que no")

    monkeypatch.setattr(db_core, "execute_sql_write", _revienta)
    assert services._promote_quality_alert_to_plan_id("plan-1", "corr-1") == 0


def test_cuenta_las_filas_reclavadas(monkeypatch):
    import db_core
    monkeypatch.setattr(db_core, "execute_sql_write",
                        lambda *a, **k: [{"alert_key": "plan_quality_degraded:u:plan-1"}])
    assert services._promote_quality_alert_to_plan_id("plan-1", "corr-1") == 1


def test_los_parametros_van_en_el_orden_del_sql(monkeypatch):
    """Cuatro `%s` y cuatro parámetros: plan_id, plan_id, correlación, plan_id. Un orden invertido
    buscaría por plan_id y escribiría la correlación en la clave."""
    capturado = {}

    import db_core
    monkeypatch.setattr(db_core, "execute_sql_write",
                        lambda sql, params=None, **k: capturado.update(sql=sql, params=params) or [])
    services._promote_quality_alert_to_plan_id("plan-1", "corr-1")
    assert capturado["params"] == ("plan-1", "plan-1", "corr-1", "plan-1")
    assert capturado["sql"].count("%s") == len(capturado["params"])


# ── el enganche ──────────────────────────────────────────────────────────────────────────────
def test_se_invoca_donde_se_atribuye_el_costo():
    """Mismo momento y mismo id que `attach_plan_id_to_usage_events` — el primer instante en que
    el plan_id existe (invariante I1). Si el canje se separara de ahí, volvería a haber dos
    verdades sobre cuándo nace el plan_id."""
    i = _SVC_SRC.find("attach_plan_id_to_usage_events(plan_id")
    assert i > 0, "cambió la atribución de costo; revisa que el canje siga a su lado"
    trozo = _SVC_SRC[i:i + 1200]
    assert "_promote_quality_alert_to_plan_id(plan_id, _corr)" in trozo, (
        "el canje ya no corre junto a la atribución de costo")
    assert re.search(r"_corr\s*=\s*get_correlation_id\(\)", _SVC_SRC), (
        "los dos canjes deben compartir el MISMO id de correlación, no pedirlo dos veces")
