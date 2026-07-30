"""[P1-MICRO-PANEL-POST-FINALIZE · 2026-07-26] "Sodio 2133/2000 · SOBRE EL LÍMITE" era falso.

El owner reportó la alerta. Recalculando el sodio del plan `c162260c` **tal como está guardado**,
con el mismo estimador que usan el panel y el corrector:

    Día 1: 1587.8 mg      Día 2: 1969.3 mg      Día 3: 1965.3 mg
    promedio real = 1840.8 mg   ·   el panel decía 2133.2  ·  ningún día sobre el techo

El corrector de sodio (`_day_sodium_autofix`) no actuaba porque **tenía razón**: el plan entregado
cumple. Lo que mentía era el número mostrado — un snapshot calculado dentro del pipeline y nunca
refrescado después de los pases que cambian ingredientes (autofix de sodio, caps, cerradores,
refill de gain-muscle).

Medido sobre los 12 planes más recientes: **9 con el panel desfasado**, desvíos de −433 a +126 mg.

## Las dos direcciones importan

    c162260c   panel 2133 "alto"   real 1841   → FALSA ALARMA
    a3b9510e   panel 1648 "ok"     real 1774   → sub-reporta

La segunda es la peligrosa, y el propio `micronutrients.py` ya lo advierte: *"La dirección
PELIGROSA de un techo es la SUB-estimación (real > techo pero reportamos 'ok')"*. Refrescar cierra
las dos.

## El arreglo es el que ya existía al lado

`refresh_clinical_band_score_post_finalize` se añadió en la frontera de persistencia por ESTE
MISMO motivo, con el comentario *"la métrica persistida hasta acá era la medida DENTRO del
pipeline, pre-finalize (stale para TODO plan)"*. Y
`recompute_micronutrient_report_for_plan` ya existía para swap/regenerate-day, con el propósito
declarado de *"que el panel de micros y el PDF NO queden stale"*. Sólo faltaba cablearlo aquí.

⚠️ Cuarta vez en la sesión con la misma forma: un valor computado pronto que los pases posteriores
invalidan y nadie recalcula (ver P1-CAPS-LAST-WORD, P1-FINALIZE-TAIL-PARITY, P1-RECONCILE-LAST-WORD).
"""
from pathlib import Path

import pytest

import db_plans


_SRC = Path(db_plans.__file__).resolve().parent / "db_plans.py"
_TXT = _SRC.read_text(encoding="utf-8")


def _cuerpo_finalize() -> str:
    i = _TXT.index("def _finalize_plan_data_for_insert(")
    j = _TXT.index("\ndef ", i + 10)
    return _TXT[i:j]


# ───────────── 1. el cableado ─────────────

def test_recomputa_el_panel_en_la_frontera_de_persistencia():
    assert "recompute_micronutrient_report_for_plan" in _cuerpo_finalize(), (
        "sin esto el panel de micros que ve el usuario es un snapshot pre-finalize")


def test_va_junto_al_refresh_de_la_banda():
    """Los dos son la misma clase de dato: medida final sobre el plan ENTREGADO. Separarlos
    invita a que un futuro refactor mueva uno y olvide el otro.

    ⚠️ Escribí esto primero como `abs(i_micro - i_band) < 1800` y falló en cuanto documenté el
    fix — la quinta ventana de bytes fija que se me caduca en la misma sesión. Lo que importa no
    es la distancia sino que **nada estructural se meta en medio**: siguen siendo los dos últimos
    pases del mismo bloque, en ese orden.
    """
    cuerpo = _cuerpo_finalize()
    i_band = cuerpo.index("refresh_clinical_band_score_post_finalize")
    i_micro = cuerpo.index("recompute_micronutrient_report_for_plan")
    assert i_band < i_micro, "la banda se mide primero; los micros cierran el bloque"
    entre = cuerpo[i_band:i_micro]
    assert "\n    def " not in entre and "\ndef " not in entre, \
        "hay una definición de función entre los dos refrescos: dejaron de ser el mismo bloque"


def test_es_fail_safe():
    """El panel es advisory: un fallo suyo JAMÁS puede impedir que un plan generado se persista."""
    cuerpo = _cuerpo_finalize()
    blk = cuerpo[cuerpo.index("recompute_micronutrient_report_for_plan"):]
    blk = blk[:blk.index("\n            except") + 400] if "\n            except" in blk else blk
    assert "except Exception" in blk
    assert "logger.debug" in blk or "logger.warning" in blk


def test_no_revienta_si_falta_form_data():
    """`form_data` puede no estar en `plan_data` (planes viejos, superficies de chunk).

    [P1-PREINSERT-CLINICAL-CTX · 2026-07-30] El ancla era `"or {}" in llamada`, o sea el
    fallback escrito DENTRO de la propia llamada. Ese fallback se movió a la resolución del
    contexto clínico (`_clin_ctx = ... or _build_clinical_form(user_id) or {}`), que además
    arregló el defecto de fondo: las dos fuentes que la expresión vieja consultaba están
    SIEMPRE vacías (0 de 31 planes llevan `form_data`), así que el panel se recomputaba sin
    condiciones. El contrato que este test protege —el argumento nunca es None y la llamada
    tolera la ausencia— sigue intacto; lo que cambió es dónde vive el `or {}`."""
    cuerpo = _cuerpo_finalize()
    # La asignación ocupa dos líneas, así que se recorta hasta el primer CONSUMIDOR
    # (`_ramb(`), no hasta el primer `)` — que cae dentro de la propia expresión.
    i_ctx = cuerpo.index("_clin_ctx =")
    assert "or {}" in cuerpo[i_ctx:cuerpo.index("_ramb(", i_ctx)], (
        "la resolución del contexto clínico debe terminar en `or {}`: el argumento que llega "
        "al panel nunca puede ser None")
    # Ventana de bytes fija NO: `[i:i+260]` se llenó con el comentario del propio fix y dejó la
    # llamada fuera. Se ancla a la invocación real (`_rmr(`), que es lo que se quiere afirmar.
    i = cuerpo.index("_rmr(", cuerpo.index("recompute_micronutrient_report_for_plan"))
    llamada = cuerpo[i:cuerpo.index("\n", i)]
    assert "_clin_ctx" in llamada, f"el panel debe recibir el contexto ya resuelto: {llamada!r}"


# ───────────── 2. la función que se invoca sigue existiendo y es best-effort ─────────────

def test_la_funcion_existe_y_no_lanza():
    import graph_orchestrator as go
    assert callable(go.recompute_micronutrient_report_for_plan)
    # plan basura → devuelve False, nunca lanza
    assert go.recompute_micronutrient_report_for_plan(None, {}) is False
    assert go.recompute_micronutrient_report_for_plan({}, {}) in (True, False)


def test_respeta_el_knob_del_reporte(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setattr(go, "MICRONUTRIENT_REPORT_ENABLED", False)
    assert go.recompute_micronutrient_report_for_plan({"days": []}, {}) is False
