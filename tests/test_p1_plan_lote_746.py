# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-746 · 2026-09-28] El revisor no rechaza con confirmaciones, y la alerta cuenta ENTREGAS.

Producción, 29-ago → 28-sep (journal de `mealfit-backend`): 140 rechazos del revisor médico con 210 razones. 172 no las
escribió el LLM: 163 las añaden los guards deterministas (nevera, piso de proteína, horario, repetición…) y son defectos
por construcción; 9 son errores de infraestructura del revisor («Error transitorio» 6, «Error en la estructura» 3), que
tampoco toca esta regla. De las 38 que escribió el revisor LLM, 7 no señalan ningún defecto; cinco de ellas llegaron
JUNTAS en una sola revisión
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


def test_los_guards_deterministas_se_anaden_despues_del_filtro():
    """[revisión 1, defecto 8] Lo que protege las razones deterministas es el ORDEN: `review_plan_node` rebaja las
    razones del LLM y SOLO DESPUÉS los guards añaden las suyas, así que el filtro nunca las ve."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("_downgrade_reviewer_non_issues(approved, issues, severity)")
    for ancla in ("ALÉRGENO DETECTADO (rechazo de seguridad clínica)", "TIRAMINA CON IMAO",
                  "ALIMENTO RECHAZADO POR EL USUARIO"):
        assert i < src.index(ancla), ancla


def test_canario_si_el_filtro_se_moviera_tras_los_guards_tampoco_los_tocaria():
    """Hoy no protege el camino vivo (ver el test de arriba): es un canario por si el filtro se moviera detrás de los
    guards. Incluye las 2 de infraestructura del revisor."""
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

# [revisión 1 · 2026-09-28] Defectos clínicos reales que la primera versión aprobaba con severidad `critical`.
# Ninguno lleva «pero/debe»: se quedan por la regla, no por un giro.
# (1) «(sin problema…)» al final no absuelve las cláusulas anteriores, y «(sin problemas)» sin «en este punto» no es
#     la conclusión del punto.
DEFECTO_1_SIN_PROBLEMA = [
    "Anemia: el desayuno del Día 2 combina leche con avena, fuente de hierro no hemo. Hidratación adecuada (sin problemas).",
    "El paciente rechazó el hígado; el Día 3 incluye hígado encebollado. El paciente rechazó la berenjena; no aparece en "
    "el plan (sin problema en este punto).",
    "Día 2: pollo en dieta vegetariana (violación). Día 3: correcto (sin problemas).",
    "Paciente renal: el Día 2 aporta 4200 mg de potasio. Sodio adecuado (sin problemas).",
    "Día 4, Almuerzo contiene 'pan de trigo', prohibido temporalmente en el perfil. Día 5 correcto (sin problema en este "
    "punto).",
    "Paciente renal: el Día 2 aporta 4200 mg de potasio. Sodio adecuado (sin problema en este punto).",
    "El Día 3 incluye hígado encebollado, que el paciente rechazó (sin problema en este punto).",
]
# (2) el texto libre (cola y paréntesis) no puede llevar un hallazgo
DEFECTO_2_TEXTO_LIBRE = [
    "No se detectan alérgenos declarados (la merienda del Día 3 contiene maní).",
    "Ningún alimento prohibido por el médico se retiró del plan.",
    "El plan respeta las restricciones del paciente y contiene maní en el Día 3.",
    "El plan respeta las restricciones declaradas (la merienda del Día 3 contiene maní).",
    "La dieta vegetariana se respeta en el Día 1 y el Día 2 incluye pollo.",
    "Dieta 'vegetariana' respetada (el Día 2 lleva pollo).",
    "No hay interacciones con la warfarina en la cena del Día 2 (espinaca diaria).",
    "Ningún alimento rechazado fue retirado del plan.",
    "El plan contiene 7 días (con maní en el Día 3) pero el plan solicitado es de 3 días; esto es una inconsistencia "
    "estructural, no un riesgo médico.",
]
# (3) «restricción declarada» con verbo de posesión o artículo definido es la restricción AUSENTE: un defecto
DEFECTO_3_RESTRICCION = [
    "El plan no incluye la restricción de sodio declarada para la hipertensión.",
    "El plan no contiene la restricción calórica declarada.",
    "No se incluye la restricción de potasio declarada.",
    "El plan no presenta la restricción de gluten declarada.",
    "El plan no tiene la restricción de purinas declarada.",
    "Sin la restricción de purinas declarada.",
    "No hay la restricción de lactosa declarada.",
]
# (1 bis) El ejemplo del hígado de la revisión salía aprobado AUNQUE este lote se apague: la regla del 227 buscaba
#     «no aparece en el plan» en CUALQUIER oración, y otra oración de la misma razón afirmaba el defecto. Ahora esa
#     regla exige que ninguna otra oración afirme contenido del plan (la de la CONCLUSIÓN, «…por lo que este punto se
#     cumple», no cambia: es la del 25-sep).
HUECO_227 = [
    "El Día 3 incluye hígado encebollado, que el paciente rechazó. La berenjena no aparece en el plan.",
    "El paciente rechazó el hígado; el Día 3 incluye hígado encebollado. El paciente rechazó la berenjena; no aparece en "
    "el plan.",
    "La cena del Día 2 contiene 150 g de pollo en una dieta vegetariana. El pescado no figura en el plan.",
    "El Día 3 incluye hígado encebollado, que el paciente rechazó, y la berenjena no aparece en el plan.",
]


def test_la_regla_del_227_sigue_rebajando_lo_que_se_niega_solo():
    """Lo que el 227 rebaja por diseño sigue igual: el veredicto sobre lo mismo que se describe, y la ausencia sin
    ninguna afirmación de contenido (con «no» delante, «no contiene» no afirma)."""
    for t in ("Día 1 | Cena: contiene queso mozzarella (lácteo) — sin alergia a lácteos declarada, no es violación.",
              # corpus de baterías: el veredicto va tras «;», en la MISMA oración
              "Día 1 | Merienda y Día 2 | Merienda: contienen mantequilla de maní; sin alergia declarada, no constituye "
              "violación.",
              "Los rechazos declarados son Pescado y Berenjena, y ninguno aparece en el plan. El plan es seguro.",
              "El plan no contiene pescado ni berenjena; ninguno aparece en el plan.",
              "El paciente rechazó la berenjena; no se incluye berenjena y no aparece en el plan."):
        ok, issues, sev, avisos = rnd._downgrade_reviewer_non_issues(False, [t], "minor")
        assert (ok, issues, sev, avisos) == (True, [], "low", [t]), t
NO_SE_DESCARTAN += DEFECTO_1_SIN_PROBLEMA + DEFECTO_2_TEXTO_LIBRE + DEFECTO_3_RESTRICCION + HUECO_227


def test_lo_que_no_es_una_confirmacion_se_queda():
    for texto in NO_SE_DESCARTAN:
        assert not rc.es_confirmacion_sin_defecto(texto), texto


def test_los_defectos_de_la_revision_siguen_rechazando_con_su_severidad():
    """Por el camino completo (lote 227 + este): con severidad `critical` el plan sigue rechazado y la razón intacta."""
    for texto in DEFECTO_1_SIN_PROBLEMA + DEFECTO_2_TEXTO_LIBRE + DEFECTO_3_RESTRICCION + HUECO_227:
        ok, issues, sev, avisos = rnd._downgrade_reviewer_non_issues(False, [texto], "critical")
        assert (ok, issues, sev, avisos) == (False, [texto], "critical", []), texto


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


def test_replay_de_los_140_rechazos_solo_cambia_la_revision_del_25_sep(monkeypatch):
    """Reproduce la cifra del informe desde lo commiteado: cadena completa de rebajas (demandas de verificación →
    227 → este lote) sobre las razones LLM de cada rechazo; un rechazo con razones deterministas o de infraestructura
    sigue rechazado. Antes: 1 (17-sep, anemia, por la rebaja de «confirmar» — riesgo abierto, fuera de este lote)."""
    import graph_orchestrator as go
    assert len(_FIX["rechazos"]) == 140
    assert sum(len(r["llm"]) for r in _FIX["rechazos"]) == 38
    assert sum(r["deterministas"] for r in _FIX["rechazos"]) == 163
    assert sum(r["infraestructura"] for r in _FIX["rechazos"]) == 9

    def aprobados(knob):
        monkeypatch.setenv("MEALFIT_REVIEWER_CONFIRMATIONS_DISCARD", knob)
        out = []
        for r in _FIX["rechazos"]:
            issues = [_LLM[n - 1]["texto"] for n in r["llm"]]
            if not issues or r["deterministas"] or r["infraestructura"]:
                continue
            ok, iss, sev, _ = go._downgrade_reviewer_verification_demands(False, issues, r["severidad"])
            ok, iss, sev, _ = rnd._downgrade_reviewer_non_issues(ok, iss, sev)
            if ok:
                out.append(r["ts"])
        return out

    antes, despues = aprobados("false"), aprobados("true")
    assert antes == ["2026-09-17 05:41:47"], antes
    assert sorted(despues) == ["2026-09-17 05:41:47", "2026-09-25 04:33:03"], despues


def test_corpus_de_baterias_solo_descarta_las_dos_sin_defecto():
    """Los 135 textos del revisor en los 919 planes de batería del VPS: solo dos se descartan, ambos sin defecto."""
    corpus = [x["t"] for x in _FIX["corpus_baterias"]]
    assert len(corpus) == 135
    assert sorted(t for t in corpus if rc.motivo(t)) == [
        "El paciente declaró que NO le gusta la berenjena; no aparece berenjena en el plan (sin problema en este punto).",
        "No se detectan alérgenos declarados (el paciente no reporta alergias).",
    ]


def test_la_regla_depende_del_knob_del_227(monkeypatch):
    """[revisión 1, defecto 7] Documentado: con `MEALFIT_REVIEWER_NON_ISSUES_ADVISORY` apagado,
    `_downgrade_reviewer_non_issues` sale antes y esta regla no corre en el revisor (aunque su knob siga en True)."""
    import graph_orchestrator as go
    monkeypatch.setattr(go, "REVIEWER_NON_ISSUES_ADVISORY", False)
    conf = _LLM[35]["texto"]
    assert rc.es_confirmacion_sin_defecto(conf)
    assert rnd._downgrade_reviewer_non_issues(False, [conf], "minor")[:2] == (False, [conf])
    for f in ("revisor_confirmaciones.py", "revisor_no_defectos.py"):
        src = (_BACKEND / f).read_text(encoding="utf-8")
        assert "MEALFIT_REVIEWER_NON_ISSUES_ADVISORY" in src and "no corre" in src, f


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


def test_la_fila_que_la_cola_de_metricas_inserta_tras_completar_cuenta():
    """[revisión 1, defecto 5] La fila `clinical_band` va por `_METRICS_EXECUTOR` (en cola): su `created_at` puede
    caer DESPUÉS de completarse el bloque. Con margen de 2 min sigue siendo la entrega; fuera del margen, no."""
    corridas = [_run(_t(4, 37), True, _W9), _run(_t(17, 2) + timedelta(seconds=40), False, _W9)]
    r = er.contar_entregas(corridas, [{"plan_id": "3957a669", "semana": 9, "updated_at": _t(17, 2)}])
    assert (r["entregas"], r["fallidas"]) == (1, 1), r
    tarde = [_run(_t(4, 37), True, _W9), _run(_t(17, 5), False, _W9)]
    r = er.contar_entregas(tarde, [{"plan_id": "3957a669", "semana": 9, "updated_at": _t(17, 2)}])
    assert (r["entregas"], r["fallidas"]) == (1, 0), r


def test_la_hora_de_entrega_es_learning_persisted_at_y_solo_cuentan_bloques_llm():
    """[revisión 1, defecto 6] `updated_at` lo mueven también la GC de snapshots y `reservation_status`; la hora de la
    compleción es `learning_persisted_at` (T2 y el chunk inicial la estampan en el mismo UPDATE). Un bloque completado
    por shuffle/edge/emergencia no entregó la corrida LLM: no se le atribuye su revisión."""
    corridas = [_run(_t(4, 37), False, _W9), _run(_t(18, 0), True, _W9)]
    q = {"plan_id": "3957a669", "semana": 9, "updated_at": _t(18, 30), "completado_en": _t(4, 45)}
    r = er.contar_entregas(corridas, [q])
    assert (r["entregas"], r["fallidas"]) == (1, 1), r          # la de las 4:37, no la de las 18:00
    for tier in ("shuffle", "edge", "emergency"):
        r = er.contar_entregas(corridas, [dict(q, quality_tier=tier)])
        assert r["entregas"] == 0, (tier, r)
    for tier in ("llm", None):                                  # el chunk inicial no estampa quality_tier
        assert er.contar_entregas(corridas, [dict(q, quality_tier=tier)])["entregas"] == 1, tier
    sql = er._SQL_COMPLETADOS
    assert "COALESCE(learning_persisted_at, updated_at) AS completado_en" in sql
    assert "COALESCE(quality_tier, 'llm') = 'llm'" in sql


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


def _cron_con(monkeypatch, corridas, completados, escrituras=None):
    import cron_tasks
    import db_core
    escrituras = [] if escrituras is None else escrituras
    monkeypatch.delenv("MEALFIT_REVFAIL_RATE_LOOKBACK_H", raising=False)
    monkeypatch.setattr(cron_tasks, "execute_sql_write",
                        lambda sql, params=None: escrituras.append((str(sql), params)), raising=False)
    monkeypatch.setattr(db_core, "execute_sql_query",
                        lambda sql, params=None, **k: corridas if "FROM pipeline_metrics" in sql else completados)
    cron_tasks._review_failed_delivered_rate_alert_job()
    tick = [p for s, p in escrituras if "_review_failed_delivered_rate_alert_job_tick" in s]
    alerta = [p for s, p in escrituras if "INSERT INTO system_alerts" in s]
    return json.loads(tick[0][1]), alerta


def _entregas(resultados):
    """Una corrida y su compleción por bloque: `resultados` = review_passed de cada entrega."""
    corridas = [_run(_t(1, i), ok, {"clave": f"p{i}:chunk_worker:initial", "plan_id": f"p{i}",
                                    "contexto": "chunk_worker:initial", "semana": 1}) for i, ok in enumerate(resultados)]
    return corridas, [{"plan_id": f"p{i}", "semana": 1, "updated_at": _t(2, 0)} for i in range(len(resultados))]


def test_en_modo_entregas_la_ventana_por_defecto_es_una_semana(monkeypatch):
    """[revisión 1, defecto 4] Prod: 15 bloques completados en 14 días (≈3,2 por 72 h) contra un mínimo de 5: con 72 h
    la alerta casi nunca evaluaba. En modo entregas la ventana por defecto es 168 h; con el knob apagado, 72 h."""
    tick, _ = _cron_con(monkeypatch, *_entregas([True] * 5))
    assert tick["lookback_h"] == 168, tick
    monkeypatch.setenv("MEALFIT_REVFAIL_COUNT_DELIVERIES", "false")
    tick, _ = _cron_con(monkeypatch, *_entregas([True] * 5))
    assert tick["lookback_h"] == 72, tick
    monkeypatch.delenv("MEALFIT_REVFAIL_COUNT_DELIVERIES")
    monkeypatch.setenv("MEALFIT_REVFAIL_RATE_LOOKBACK_H", "48")
    import cron_tasks
    import db_core
    esc = []
    monkeypatch.setattr(cron_tasks, "execute_sql_write", lambda sql, params=None: esc.append((str(sql), params)))
    monkeypatch.setattr(db_core, "execute_sql_query", lambda sql, params=None, **k: [])
    cron_tasks._review_failed_delivered_rate_alert_job()
    assert json.loads([p for s, p in esc if "_tick" in s][0][1])["lookback_h"] == 48   # el knob explícito manda


def _resuelve_heredada(escrituras):
    return [p for s, p in escrituras if "UPDATE system_alerts" in s and "n_corridas" in s]


def test_la_alerta_abierta_por_corridas_se_cierra_con_muestra_insuficiente_bajo_el_umbral(monkeypatch):
    """[revisión 1, defecto 4] La fila abierta hoy la escribió el conteo por CORRIDAS (su metadata no trae
    `n_corridas`). Con muestra insuficiente el cron nunca la tocaba: queda abierta para siempre. Si las entregas de la
    ventana están bajo el umbral, se cierra — SOLO esa fila heredada (una abierta por entregas espera la muestra)."""
    esc = []
    tick, alerta = _cron_con(monkeypatch, *_entregas([True, True, False]), escrituras=esc)   # 1/3 = 33 % > 20 %
    assert "insufficient_samples" in tick["skip_reason"] and not alerta and not _resuelve_heredada(esc)
    esc = []
    tick, alerta = _cron_con(monkeypatch, *_entregas([True, True]), escrituras=esc)          # 0/2
    assert "insufficient_samples" in tick["skip_reason"] and not alerta
    upd = _resuelve_heredada(esc)
    assert len(upd) == 1 and upd[0] == ("review_failed_delivered_rate_high",), esc
    sql = [s for s, p in esc if "UPDATE system_alerts" in s][0]
    assert "resolved_at IS NULL" in sql and "? 'n_corridas'" in sql and "NOT" in sql
    assert tick["legacy_close_attempted"] is True
    esc = []
    tick, _ = _cron_con(monkeypatch, [], [], escrituras=esc)                                   # 0 entregas: sin evidencia
    assert not _resuelve_heredada(esc) and tick["legacy_close_attempted"] is False
    monkeypatch.setenv("MEALFIT_REVFAIL_COUNT_DELIVERIES", "false")                            # modo corridas: como antes
    esc = []
    _cron_con(monkeypatch, *_entregas([True, True]), escrituras=esc)
    assert not _resuelve_heredada(esc)


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
