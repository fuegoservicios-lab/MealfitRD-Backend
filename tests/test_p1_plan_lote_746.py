# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-746 · 2026-09-28] El revisor no rechaza con confirmaciones, y la alerta cuenta ENTREGAS.

Producción, 29-ago → 28-sep (journal de `mealfit-backend`): 140 rechazos del revisor médico con 210 razones. 172 las
añaden los guards deterministas (nevera, piso de proteína, horario, repetición…) y son defectos por construcción. De las
38 que escribió el revisor LLM, 7 no señalan ningún defecto; cinco de ellas llegaron JUNTAS en una sola revisión
(25-sep 04:33:03, bloque 92328ff7 semana 2), con severidad «minor», y quemaron dos intentos:

    · «No se detectan alérgenos declarados (paciente sin alergias).»
    · «No se detectan violaciones de condiciones médicas (paciente sin condiciones ni medicamentos).»
    · «Dieta 'balanced' respetada; no hay restricciones vegetarianas/veganas/sin gluten declaradas.»
    · «El plan contiene 7 días (Día 4 a Día 7) pero el plan solicitado es de 3 días; esto es una inconsistencia
      estructural, no un riesgo médico.»   ← el bloque era de 4 días (4-7): el revisor no recibe el número pedido.
    · «El plan incluye Hígado? No. […] ninguno aparece en el plan, por lo que no hay violación por rechazos.»

La última ya la rebaja el lote 227 (su conclusión se niega sola); las otras cuatro no las veía ninguna regla. Además
viajaban a la directiva del reintento como «RESTRICCIONES ACUMULADAS». Las 29 clínicas/reales (dieta vegetariana
violada, prohibiciones temporales, anemia) y las 2 ambiguas («verificar que sea plátano verde») se quedan.

Y la alerta `review_failed_delivered_rate_high` contaba FILAS `clinical_band`, que se emiten una por CORRIDA del pipeline:
el 27-sep el bloque 9 del plan 3957a669 corrió 4 veces (reintentos de nevera y de pickup) y se entregó UNA; la alerta
leyó 2 fallidas de 6 «entregas» cuando hubo 1 de 3.
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import entregas_revisadas as er  # noqa: E402
import revisor_confirmaciones as rc  # noqa: E402
import revisor_no_defectos as rnd  # noqa: E402

_FIX = json.loads((_BACKEND / "tests" / "fixtures" / "revisor_razones_prod_2026_09_28.json").read_text(encoding="utf-8"))
_LLM = _FIX["llm"]


# ─────────────────────────── (1) el filtro: confirmaciones sin defecto ───────────────────────────

def test_las_cuatro_confirmaciones_de_produccion_se_descartan():
    for n in (35, 36, 37, 38):
        texto = _LLM[n - 1]["texto"]
        assert rc.es_confirmacion_sin_defecto(texto), (n, texto)


def test_la_revision_del_25_sep_pasa_a_aprobada():
    """Las cinco razones juntas: cuatro por esta regla y la del hígado por la conclusión del lote 227."""
    ok, issues, sev, avisos = rnd._downgrade_reviewer_non_issues(False, list(_FIX["revision_0925"]), "minor")
    assert (ok, issues, sev) == (True, [], "low") and len(avisos) == 5, (ok, issues, sev)


def test_ninguna_razon_real_ni_ambigua_de_produccion_se_descarta():
    """Las 31 que no son «no_problema» (29 reales + 2 ambiguas) jamás las toca esta regla."""
    quedan = [x for x in _LLM if x["clase"] != "no_problema"]
    assert len(quedan) == 31
    for x in quedan:
        assert not rc.es_confirmacion_sin_defecto(x["texto"]), (x["n"], x["texto"])


def test_ninguna_razon_de_los_guards_deterministas_se_descarta():
    for texto in _FIX["deterministas"]:
        assert not rc.es_confirmacion_sin_defecto(texto), texto


def test_las_reales_de_produccion_siguen_rechazando_por_el_camino_completo():
    """Cada razón real, sola, sigue siendo un rechazo tras la rebaja completa del lote 227 + este lote."""
    for x in _LLM:
        if x["clase"] != "real":
            continue
        ok, issues, sev, _ = rnd._downgrade_reviewer_non_issues(False, [x["texto"]], "high")
        assert ok is False and issues == [x["texto"]], (x["n"], x["texto"])


NO_SE_DESCARTAN = [
    # negación de algo BUENO = defecto
    "No se detectan fuentes de hierro hemo en el plan; para la anemia debe incluirse carne roja.",
    "No hay suficiente proteína en la cena del Día 2.",
    "El plan no incluye fuentes de calcio; no se detectan alérgenos.",
    # la negación sigue con un defecto
    "No hay violación de la alergia, pero el Día 3 incluye maní en la merienda.",
    "No se detectan alérgenos declarados, pero el sodio del Día 2 supera los 2.300 mg.",
    "No hay alérgenos declarados salvo el maní, que aparece en el Día 2.",
    "Sin violaciones de alergia. El Día 3 incluye pollo en una dieta vegetariana.",
    "No hay riesgo de hipoglucemia si se respetan los horarios de las comidas.",
    # matiz que insinúa algo menor
    "No se detectan violaciones graves.",
    "No hay contraindicaciones evidentes.",
    # cumplimiento negado o parcial
    "Dieta vegetariana no respetada: la cena del Día 2 incluye pollo.",
    "No se respeta la dieta vegetariana: el Día 4 incluye atún.",
    "El plan no respeta la dieta vegetariana.",
    "El plan cumple parcialmente con la dieta DASH.",
    "El plan respeta la dieta vegetariana excepto en la cena del Día 3.",
    "El plan respeta la dieta vegetariana y no alcanza la proteína diaria.",
    # número de días sin la conclusión «no es riesgo médico», o con matiz
    "El plan contiene 7 días pero el plan solicitado es de 3 días.",
    "El plan contiene 7 días (Día 4 a Día 7) pero el plan solicitado es de 3 días; esto es una inconsistencia "
    "estructural, no un riesgo médico inmediato.",
    # «no médica» no basta: la prohibición del perfil es un defecto real (razón n.º 24 de producción)
    "Día 2, Cena contiene '1 tortilla de trigo'. Violación regenerable, no médica.",
    # pregunta y respuesta sin conclusión: ambigua (depende de qué se pregunta) → se queda
    "¿Incluye hígado? No.",
    # «sin problema» con giro, o que no es la conclusión del punto
    "El Día 3 incluye berenjena, que el paciente rechazó; debe reemplazarse (sin problema en este punto).",
    "El Día 2 incluye 300 g de piña, pero la porción es alta (sin problema en este punto).",
    "Día 2: casabe en dos comidas (sin problema de sodio).",
]


def test_lo_que_no_es_una_confirmacion_se_queda():
    for texto in NO_SE_DESCARTAN:
        assert not rc.es_confirmacion_sin_defecto(texto), texto


SE_DESCARTAN = [
    "No se detectan alérgenos declarados en el plan.",
    "No hay violaciones de las restricciones declaradas.",
    "Ningún alimento rechazado aparece en el plan.",
    "No se encontraron ingredientes prohibidos.",
    "La dieta vegetariana se respeta en todas las comidas.",
    "El plan respeta las restricciones declaradas.",
    "Sin alérgenos declarados (paciente sin alergias); dieta 'vegetarian' respetada.",
    # corpus de baterías guardado en el VPS (919 planes, 135 textos del revisor): la que ninguna regla veía
    "El paciente declaró que NO le gusta la berenjena; no aparece berenjena en el plan (sin problema en este punto).",
]


def test_otras_confirmaciones_de_la_misma_forma_tambien():
    for texto in SE_DESCARTAN:
        assert rc.es_confirmacion_sin_defecto(texto), texto


def test_una_confirmacion_junto_a_un_defecto_real_solo_se_va_ella():
    real = _LLM[1]["texto"]            # carne en dieta vegetariana
    conf = _LLM[35]["texto"]           # «No se detectan alérgenos declarados…»
    ok, issues, sev, avisos = rnd._downgrade_reviewer_non_issues(False, [real, conf], "critical")
    assert ok is False and issues == [real] and sev == "critical" and avisos == [conf]


def test_excepto_o_salvo_ya_no_lo_rebaja_la_regla_del_227():
    """Endurecimiento del 227: «El plan es seguro…» seguido de una excepción afirma un defecto."""
    t = "El plan es seguro para el paciente excepto por el exceso de sodio del Día 2."
    ok, issues, _, avisos = rnd._downgrade_reviewer_non_issues(False, [t], "minor")
    assert ok is False and issues == [t] and avisos == []


def test_el_knob_apaga_solo_esta_regla(monkeypatch):
    monkeypatch.setenv("MEALFIT_REVIEWER_CONFIRMATIONS_DISCARD", "false")
    assert not rc.es_confirmacion_sin_defecto(_LLM[35]["texto"])
    # la del hígado la sigue rebajando el lote 227
    ok, issues, _, _ = rnd._downgrade_reviewer_non_issues(False, [_LLM[33]["texto"]], "minor")
    assert ok is True and issues == []


def test_nunca_lanza():
    for raro in (None, "", 123, "   ", ";;;", "(" * 50):
        assert rc.es_confirmacion_sin_defecto(raro) is False


def test_cableado_en_revisor_no_defectos():
    src = (_BACKEND / "revisor_no_defectos.py").read_text(encoding="utf-8")
    assert '__import__("revisor_confirmaciones").es_confirmacion_sin_defecto(t)' in src
    assert "[P1-PLAN-LOTE-746]" in src
    mod = (_BACKEND / "revisor_confirmaciones.py").read_text(encoding="utf-8")
    assert "tooltip-anchor: P1-PLAN-LOTE-746-CONFIRMACIONES" in mod
    assert "MEALFIT_REVIEWER_CONFIRMATIONS_DISCARD" in mod


# ─────────────────────────── (2) la alerta cuenta entregas ───────────────────────────

def test_clave_de_entrega_por_bloque():
    pid = uuid.UUID("3957a669-c28a-40e2-9f4e-c1afffaf4e36")
    k = er.clave_de_entrega({"_caller_target_plan_id": pid, "_caller_context": "chunk_worker:week_9"})
    assert k == {"clave": f"{pid}:chunk_worker:week_9", "plan_id": str(pid),
                 "contexto": "chunk_worker:week_9", "semana": 9}
    json.dumps(k)                       # va a `pipeline_metrics.metadata`
    k1 = er.clave_de_entrega({"_caller_target_plan_id": "p1", "_caller_context": "chunk_worker:initial"})
    assert k1["semana"] == 1 and k1["clave"] == "p1:chunk_worker:initial"
    kj = er.clave_de_entrega({"_caller_target_plan_id": "p2", "_caller_context": "jit_week2"})
    assert kj["semana"] is None and kj["clave"] == "p2:jit_week2"


def test_clave_de_entrega_sin_plan_usa_la_correlacion(monkeypatch):
    import correlation
    monkeypatch.setattr(correlation, "get_correlation_id", lambda: "abc123")
    assert er.clave_de_entrega({})["clave"] == "corr:abc123:initial_generate"
    for vacio in (None, "-", ""):          # «-» es el valor por defecto del ContextVar (sin petición)
        monkeypatch.setattr(correlation, "get_correlation_id", lambda v=vacio: v)
        assert er.clave_de_entrega({}) == {}
    assert er.clave_de_entrega(None) == {}


def _t(h, m):
    return datetime(2026, 9, 27, h, m, tzinfo=timezone.utc)


def _run(ts, passed, entrega=None, fb=False):
    return {"created_at": ts, "review_passed": "true" if passed else "false",
            "fallback": "true" if fb else "false", "entrega": entrega}


_W9 = {"clave": "3957a669:chunk_worker:week_9", "plan_id": "3957a669", "contexto": "chunk_worker:week_9", "semana": 9}


def test_el_caso_del_27_sep_cuenta_una_entrega():
    corridas = [_run(_t(4, 37), True, _W9), _run(_t(4, 43), True, _W9),
                _run(_t(4, 51), False, _W9), _run(_t(16, 59), False, _W9)]
    completados = [{"plan_id": "3957a669", "semana": 9, "updated_at": _t(17, 2)}]
    r = er.contar_entregas(corridas, completados)
    assert (r["entregas"], r["fallidas"], r["corridas"]) == (1, 1, 4), r
    # con el knob apagado, el conteo viejo (una por corrida)
    r0 = er.contar_entregas(corridas, completados, por_entrega=False)
    assert (r0["entregas"], r0["fallidas"]) == (4, 2), r0


def test_bloque_sin_completar_no_es_entrega():
    corridas = [_run(_t(4, 37), False, _W9)]
    assert er.contar_entregas(corridas, [])["entregas"] == 0
    # completado ANTES de la corrida (otra vuelta anterior): tampoco
    assert er.contar_entregas(corridas, [{"plan_id": "3957a669", "semana": 9, "updated_at": _t(4, 0)}])["entregas"] == 0


def test_la_entrega_es_la_ultima_corrida_antes_de_completar():
    corridas = [_run(_t(4, 37), False, _W9), _run(_t(4, 43), True, _W9), _run(_t(18, 0), False, _W9)]
    r = er.contar_entregas(corridas, [{"plan_id": "3957a669", "semana": 9, "updated_at": _t(4, 45)}])
    assert (r["entregas"], r["fallidas"]) == (1, 0), r


def test_fallback_se_mira_en_la_corrida_entregada():
    corridas = [_run(_t(4, 37), False, _W9), _run(_t(4, 43), False, _W9, fb=True)]
    r = er.contar_entregas(corridas, [{"plan_id": "3957a669", "semana": 9, "updated_at": _t(4, 45)}])
    assert r["entregas"] == 0, r        # lo entregado fue un plan de contingencia: fuera del denominador


def test_sin_cola_cuenta_la_ultima_corrida_de_la_clave_y_el_legado_una_por_fila():
    jit = {"clave": "p2:jit_week2", "plan_id": "p2", "contexto": "jit_week2", "semana": None}
    corridas = [_run(_t(4, 0), False, jit), _run(_t(4, 5), True, jit),
                _run(_t(5, 0), False, None), _run(_t(5, 1), False, {})]
    r = er.contar_entregas(corridas, [])
    assert (r["entregas"], r["fallidas"], r["corridas"]) == (3, 2, 4), r


def test_contar_entregas_revisadas_solo_lee(monkeypatch):
    llamadas = []

    def _q(sql, params=None, fetch_all=False, **kw):
        llamadas.append(sql)
        if "FROM pipeline_metrics" in sql:
            return [{"created_at": _t(4, 37), "review_passed": "false", "fallback": "false", "entrega": _W9},
                    {"created_at": _t(4, 51), "review_passed": "true", "fallback": "false", "entrega": _W9}]
        return [{"plan_id": "3957a669", "semana": 9, "updated_at": _t(5, 0)}]

    import db_core
    monkeypatch.setattr(db_core, "execute_sql_query", _q)
    assert er.contar_entregas_revisadas(72) == (1, 0, 2)
    assert len(llamadas) == 2 and all(s.lstrip().upper().startswith("SELECT") for s in llamadas)
    assert "node = 'clinical_band'" in llamadas[0] and "status = 'completed'" in llamadas[1]


def test_contar_entregas_revisadas_falla_a_cero(monkeypatch):
    import db_core

    def _boom(*a, **k):
        raise RuntimeError("db caída")

    monkeypatch.setattr(db_core, "execute_sql_query", _boom)
    assert er.contar_entregas_revisadas(72) == (0, 0, 0)


def _cron_con(monkeypatch, corridas, completados):
    import cron_tasks
    import db_core
    escrituras = []
    monkeypatch.setattr(cron_tasks, "execute_sql_write",
                        lambda sql, params=None: escrituras.append((str(sql), params)), raising=False)
    monkeypatch.setattr(db_core, "execute_sql_query",
                        lambda sql, params=None, **k: corridas if "FROM pipeline_metrics" in sql else completados)
    cron_tasks._review_failed_delivered_rate_alert_job()
    tick = [p for s, p in escrituras if "_review_failed_delivered_rate_alert_job_tick" in s]
    alerta = [p for s, p in escrituras if "INSERT INTO system_alerts" in s]
    return json.loads(tick[0][1]), alerta


def test_el_cron_con_el_caso_del_28_sep_ya_no_alerta(monkeypatch):
    """Tick real del 28-sep 14:53 (72 h): prod leyó 2/6 y alertó. Las 6 filas: 4 corridas del bloque 9 (1 entrega,
    fallida), el bloque 2 de 6594aae1 (entregado, aprobado) y el 3 (aprobado, pero aún sin completar al medir)."""
    b2 = {"clave": "6594aae1:chunk_worker:week_2", "plan_id": "6594aae1", "contexto": "chunk_worker:week_2", "semana": 2}
    b3 = {"clave": "6594aae1:chunk_worker:week_3", "plan_id": "6594aae1", "contexto": "chunk_worker:week_3", "semana": 3}
    corridas = [_run(_t(4, 37), True, _W9), _run(_t(4, 43), True, _W9), _run(_t(4, 51), False, _W9),
                _run(_t(16, 59), False, _W9), _run(_t(2, 0) - timedelta(days=1), True, b2),
                _run(_t(14, 51) + timedelta(days=1), True, b3)]
    completados = [{"plan_id": "3957a669", "semana": 9, "updated_at": _t(17, 2)},
                   {"plan_id": "6594aae1", "semana": 2, "updated_at": _t(2, 2) - timedelta(days=1)}]
    tick, alerta = _cron_con(monkeypatch, corridas, completados)
    assert (tick["n_delivered"], tick["n_review_failed"], tick["n_corridas"]) == (2, 1, 6), tick
    assert "insufficient_samples" in tick["skip_reason"] and not alerta


def test_el_cron_alerta_por_entregas_y_lo_dice(monkeypatch):
    corridas = [_run(_t(1, i), i < 2, {"clave": f"p{i}:chunk_worker:initial", "plan_id": f"p{i}",
                                       "contexto": "chunk_worker:initial", "semana": 1}) for i in range(5)]
    completados = [{"plan_id": f"p{i}", "semana": 1, "updated_at": _t(2, 0)} for i in range(5)]
    tick, alerta = _cron_con(monkeypatch, corridas, completados)
    assert (tick["n_delivered"], tick["n_review_failed"]) == (5, 3) and tick["alert_emitted"] is True
    meta = json.loads(alerta[0][3])
    assert meta["n_corridas"] == 5 and meta["n_delivered"] == 5 and "entregas" in alerta[0][2]


def test_el_cron_cuenta_entregas():
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    i = src.index("def _review_failed_delivered_rate_alert_job():")
    cuerpo = src[i:src.index("\ndef ", i + 10)]
    assert '__import__("entregas_revisadas").contar_entregas_revisadas(lookback_h)' in cuerpo
    assert "COUNT(*) AS delivered" not in cuerpo           # el conteo por fila se fue
    assert '"n_corridas": _n_corridas' in cuerpo


def test_la_fila_clinical_band_lleva_su_clave_de_entrega():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index('"node": "clinical_band",')
    bloque = src[i:src.index("})", i)]
    assert '"entrega": __import__("entregas_revisadas").clave_de_entrega(actual_form_data)' in bloque
    mod = (_BACKEND / "entregas_revisadas.py").read_text(encoding="utf-8")
    assert "tooltip-anchor: P1-PLAN-LOTE-746-ENTREGAS" in mod and "MEALFIT_REVFAIL_COUNT_DELIVERIES" in mod


def test_los_ficheros_con_tope_no_crecen():
    topes = {"graph_orchestrator.py": 52240, "cron_tasks.py": 36550}
    for f, tope in topes.items():
        n = (_BACKEND / f).read_text(encoding="utf-8").count("\n")
        assert n <= tope, (f, n, tope)
