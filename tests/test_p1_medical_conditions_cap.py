"""[P1-MEDICAL-CONDITIONS-CAP · 2026-08-01] Cap de condiciones médicas
simultáneas por plan (decisión de producto del owner).

Contexto:
  El owner decidió acotar el alcance clínico del generador al checklist de
  condiciones que las reglas deterministas (`condition_rules.py`) SÍ saben
  defender de forma consistente:
    (A) cap de 3 condiciones médicas simultáneas para crear un plan.
    (B) eliminación de los inputs de texto libre "Otra condición médica..."
        y "Otro medicamento..." del assessment (ver frontend).

Este módulo cubre el lado servidor (A): `_validate_medical_conditions_cap`
en `routers/plans.py`, cableado en `/api/analyze` y `/api/analyze/stream`
(los dos endpoints que reciben el formulario COMPLETO para generar un plan
NUEVO). `/swap-meal` y `/{plan_id}/chunks/{chunk_id}/regenerate-simplified`
NO reciben el formulario completo (swap hidrata clinical data desde el
perfil ya persistido vía `_enrich_clinical_from_profile`; regenerate-simplified
no toma body en absoluto) — no llevan el guard, y así debe quedar.

Compatibilidad: la validación es del REQUEST de generación nueva, NO de
lectura de perfil. Perfiles YA guardados con >cap condiciones (o con
condiciones capturadas por el texto libre pre-fix) siguen leyéndose y
usándose con normalidad — solo un nuevo submit del formulario puede
disparar el 422.

Cobertura:
  (a) 3 condiciones reales → pasa.
  (b) 4 condiciones reales → rechaza (contrato `too_many_medical_conditions`).
  (c) "Ninguna" + 3 reales → pasa (negativos de `_PROFILE_RISK_NEGATIVES`
      no cuentan contra el cap, mismo SSOT que `_profile_has_medical_risk`).
  (d) knob `MEALFIT_MAX_MEDICAL_CONDITIONS` respetado (monkeypatch a 2 → la
      3ª condición rechaza) + clamp [1,7] vía validator de `_env_int`.
  (e) el knob se auto-registra en `_KNOBS_REGISTRY` al leerse.
  (f) parser: ambos call sites (`/analyze`, `/analyze/stream`) están
      cableados con el mismo `code`/`max` contract.
  (g) SAFETY: Embarazo/Lactancia (chips gender-gated de QMedical, mismo
      array `medicalConditions`) NO cuentan contra el cap — evita reabrir
      el punto ciego que P1-PREGNANCY-INTAKE-CAPTURE cerró (una usuaria
      embarazada con 3 condiciones reales NO debe recibir 422).

  [FIX-REPORT 2026-08-01, code review post-implementación — 2 CRITICAL + 1
  IMPORTANT verificados ejecutando]:
  (h) CRITICAL-1: `_close_medical_freetext_scope` — `otherConditions`/
      `otherMedications` se ignoran en requests NUEVOS de `/analyze` y
      `/analyze/stream` (mutados a "" ANTES de `_validate_form_data_min` y
      ANTES de que `pipeline_data = dict(data)` copie el payload). Cierra el
      bypass del cap vía `_merge_other_text_fields` (que corre DESPUÉS de la
      validación, a nivel router Y otra vez dentro de `arun_plan_pipeline`).
  (i) CRITICAL-2: exención de embarazo/lactancia migrada de SUBSTRING match
      (bug: "embarazo con diabetes" se eximía completo) a IGUALDAD EXACTA
      canonicalizada contra los 2 valores literales de `PREGNANCY_CHIP_LABELS`.
"""
from pathlib import Path

import pytest

from knobs import _KNOBS_REGISTRY
from routers.plans import (
    _validate_medical_conditions_cap,
    _is_pregnancy_or_lactation_condition_item,
    _close_medical_freetext_scope,
)


def _payload(conditions) -> dict:
    return {"medicalConditions": conditions}


# ---------------------------------------------------------------------------
# (a) 3 condiciones reales → pasa la validación (cap default = 3).
# ---------------------------------------------------------------------------
def test_tres_condiciones_reales_pasa_cap_default():
    ok, count, cap = _validate_medical_conditions_cap(
        _payload(["Diabetes tipo 2", "Hipertensión", "Enfermedad renal crónica"])
    )
    assert ok is True
    assert count == 3
    assert cap == 3


# ---------------------------------------------------------------------------
# (b) 4 condiciones reales → rechaza.
# ---------------------------------------------------------------------------
def test_cuatro_condiciones_reales_rechaza_cap_default():
    ok, count, cap = _validate_medical_conditions_cap(
        _payload(["Diabetes tipo 2", "Hipertensión", "Enfermedad renal crónica", "Asma"])
    )
    assert ok is False
    assert count == 4
    assert cap == 3


def test_endpoint_detail_contract_too_many_medical_conditions():
    """El detail 422 que arma el endpoint replica exactamente el contrato
    pedido por el owner: `code`, `max`, `message` es-DO con el cap dinámico."""
    ok, count, cap = _validate_medical_conditions_cap(
        _payload(["Diabetes tipo 2", "Hipertensión", "Enfermedad renal crónica", "Asma"])
    )
    assert not ok
    detail = {
        "code": "too_many_medical_conditions",
        "max": cap,
        "message": (
            "Para garantizar la calidad clínica del plan, selecciona máximo "
            f"{cap} condiciones prioritarias."
        ),
    }
    assert detail == {
        "code": "too_many_medical_conditions",
        "max": 3,
        "message": (
            "Para garantizar la calidad clínica del plan, selecciona máximo "
            "3 condiciones prioritarias."
        ),
    }


# ---------------------------------------------------------------------------
# (c) "Ninguna" (y otros sentinels) + 3 reales → pasa (negativos no cuentan).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("negative", ["Ninguna", "ninguno", "N/A", "na", "No", "Nada", "", "  "])
def test_sentinel_negativo_no_cuenta_contra_el_cap(negative):
    ok, count, cap = _validate_medical_conditions_cap(
        _payload([negative, "Diabetes tipo 2", "Hipertensión", "Enfermedad renal crónica"])
    )
    assert ok is True, f"sentinel {negative!r} no debería contar contra el cap (count={count})"
    assert count == 3


def test_solo_ninguna_cuenta_cero():
    ok, count, cap = _validate_medical_conditions_cap(_payload(["Ninguna"]))
    assert ok is True
    assert count == 0


def test_array_vacio_cuenta_cero():
    ok, count, cap = _validate_medical_conditions_cap(_payload([]))
    assert ok is True
    assert count == 0


def test_key_ausente_cuenta_cero():
    ok, count, cap = _validate_medical_conditions_cap({})
    assert ok is True
    assert count == 0


def test_string_unico_no_lista_cuenta_como_una():
    """Cliente legacy que manda `medicalConditions` como string suelto en vez
    de array — mismo comportamiento defensivo que `_profile_has_medical_risk`."""
    ok, count, cap = _validate_medical_conditions_cap({"medicalConditions": "Diabetes tipo 2"})
    assert count == 1
    assert ok is True


def test_string_unico_negativo_cuenta_cero():
    ok, count, cap = _validate_medical_conditions_cap({"medicalConditions": "Ninguna"})
    assert count == 0
    assert ok is True


# ---------------------------------------------------------------------------
# (d) knob `MEALFIT_MAX_MEDICAL_CONDITIONS` respetado + clamp [1, 7].
# ---------------------------------------------------------------------------
def test_knob_respetado_baja_el_cap_a_dos(monkeypatch):
    monkeypatch.setenv("MEALFIT_MAX_MEDICAL_CONDITIONS", "2")
    # 3ra condición real ahora rechaza.
    ok, count, cap = _validate_medical_conditions_cap(
        _payload(["Diabetes tipo 2", "Hipertensión", "Enfermedad renal crónica"])
    )
    assert cap == 2
    assert ok is False
    assert count == 3

    # 2 condiciones sigue pasando bajo el cap bajado.
    ok2, count2, cap2 = _validate_medical_conditions_cap(
        _payload(["Diabetes tipo 2", "Hipertensión"])
    )
    assert cap2 == 2
    assert ok2 is True


def test_knob_clamp_fuera_de_rango_alto_cae_a_default(monkeypatch):
    """Validator [1,7] de `_env_int`: 10 está fuera de rango → cae al
    default=3 (WARNING logueado, `parse_failed=True` en el registry)."""
    monkeypatch.setenv("MEALFIT_MAX_MEDICAL_CONDITIONS", "10")
    ok, count, cap = _validate_medical_conditions_cap(_payload([]))
    assert cap == 3


def test_knob_clamp_fuera_de_rango_bajo_cae_a_default(monkeypatch):
    monkeypatch.setenv("MEALFIT_MAX_MEDICAL_CONDITIONS", "0")
    ok, count, cap = _validate_medical_conditions_cap(_payload([]))
    assert cap == 3


def test_knob_valor_limite_superior_del_rango_se_acepta(monkeypatch):
    monkeypatch.setenv("MEALFIT_MAX_MEDICAL_CONDITIONS", "7")
    ok, count, cap = _validate_medical_conditions_cap(_payload([]))
    assert cap == 7


def test_knob_no_numerico_cae_a_default(monkeypatch):
    monkeypatch.setenv("MEALFIT_MAX_MEDICAL_CONDITIONS", "abc")
    ok, count, cap = _validate_medical_conditions_cap(_payload([]))
    assert cap == 3


# ---------------------------------------------------------------------------
# (e) el knob se auto-registra en `_KNOBS_REGISTRY` (para /health/version,
#     `_log_active_knobs()` startup, y SRE confirmando overrides sin redeploy).
# ---------------------------------------------------------------------------
def test_knob_se_autorregistra_en_knobs_registry(monkeypatch):
    monkeypatch.setenv("MEALFIT_MAX_MEDICAL_CONDITIONS", "5")
    _validate_medical_conditions_cap(_payload([]))
    entry = _KNOBS_REGISTRY.get("MEALFIT_MAX_MEDICAL_CONDITIONS")
    assert entry is not None, "el knob debe aparecer en _KNOBS_REGISTRY tras leerse"
    assert entry["type"] == "int"
    assert entry["default"] == 3
    assert entry["value"] == 5
    assert entry["is_override"] is True


def test_knob_registry_default_sin_override(monkeypatch):
    monkeypatch.delenv("MEALFIT_MAX_MEDICAL_CONDITIONS", raising=False)
    _validate_medical_conditions_cap(_payload([]))
    entry = _KNOBS_REGISTRY.get("MEALFIT_MAX_MEDICAL_CONDITIONS")
    assert entry is not None
    assert entry["value"] == 3
    assert entry["is_override"] is False


# ---------------------------------------------------------------------------
# (g) SAFETY: Embarazo/Lactancia NO cuentan contra el cap — IGUALDAD EXACTA.
#
# Bug que esto previene: los chips `PREGNANCY_CHIP_LABELS` (QMedical.jsx,
# gender-gated) escriben "Embarazo"/"Lactancia" al MISMO array
# `medicalConditions` que las 7 condiciones clínicas del checklist (ver
# P1-PREGNANCY-INTAKE-CAPTURE · 2026-06-19). Si el cap las contara igual que
# una condición clínica, una usuaria embarazada con 3 condiciones reales
# (ej. Diabetes T2 + Hipertensión + Hipotiroidismo, combinación plausible)
# recibiría 422 al marcar también "Embarazo" — bloqueada de generar CUALQUIER
# plan, reabriendo el "punto ciego de alto riesgo/prevalencia" que
# P1-PREGNANCY-INTAKE-CAPTURE cerró.
#
# [CRITICAL-2-FIX · 2026-08-01, encontrado en code review, verificado
# ejecutando] La primera versión de esta exención usaba SUBSTRING match
# contra `PREGNANCY_CONDITION_TERMS` (vocabulario amplio pensado para el
# gate de déficit calórico, intencionalmente sobre-inclusivo AHÍ). Aplicado
# al cap, el sobre-inclusivo era el bug: `"embarazo con diabetes"` contiene
# `"embaraz"` → se eximía (count no subía) PERO el string completo seguía
# disparando `condition_rules.detect_active_rules` aguas abajo (mismo tipo
# de substring match) → cap=0 pero complejidad clínica ilimitada. 5 strings
# `"embarazo con diabetes-N"` pasaban las 5 sin sumar nada al cap. El fix es
# igualdad EXACTA canonicalizada contra los 2 valores LITERALES que el chip
# puede emitir ("embarazo"/"lactancia") — cualquier frase que solo MENCIONE
# embarazo/lactancia ahora CUENTA como condición real.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("term", ["Embarazo", "embarazo", "Lactancia", "lactancia", "EMBARAZO", "LacTanCia"])
def test_is_pregnancy_or_lactation_condition_item_detecta_valores_exactos(term):
    assert _is_pregnancy_or_lactation_condition_item(term.lower()) is True


def test_is_pregnancy_or_lactation_condition_item_no_falso_positivo_en_condicion_real():
    assert _is_pregnancy_or_lactation_condition_item("diabetes t2") is False
    assert _is_pregnancy_or_lactation_condition_item("hipertension") is False


@pytest.mark.parametrize("phrase", [
    "embarazo con diabetes",
    "posible embarazo",
    "lactancia materna exclusiva",
    "gestante",  # vocabulario del gate de déficit (sobre-inclusivo AHÍ),
                 # pero NO es el valor literal del chip → cuenta para el cap.
    "postparto",
])
def test_is_pregnancy_or_lactation_condition_item_rechaza_frases_que_solo_mencionan(phrase):
    """[CRITICAL-2-FIX] Ancla del bug: una frase que CONTIENE el término pero
    no es el chip exacto ya NO se exime — debe contar como condición real."""
    assert _is_pregnancy_or_lactation_condition_item(phrase.lower()) is False


def test_embarazo_no_cuenta_contra_el_cap():
    """3 condiciones reales + Embarazo (valor exacto del chip) → pasa."""
    ok, count, cap = _validate_medical_conditions_cap(
        _payload(["Diabetes T2", "Hipertensión", "Hipotiroidismo", "Embarazo"])
    )
    assert ok is True, f"Embarazo no debería contar contra el cap (count={count})"
    assert count == 3


def test_lactancia_no_cuenta_contra_el_cap():
    ok, count, cap = _validate_medical_conditions_cap(
        _payload(["Diabetes T2", "Hipertensión", "Hipotiroidismo", "Lactancia"])
    )
    assert ok is True
    assert count == 3


def test_embarazo_y_lactancia_juntos_no_cuentan():
    """Caso límite: ambos chips de embarazo/lactancia marcados a la vez
    (UI no lo previene explícitamente) + 3 reales → sigue pasando."""
    ok, count, cap = _validate_medical_conditions_cap(
        _payload(["Diabetes T2", "Hipertensión", "Hipotiroidismo", "Embarazo", "Lactancia"])
    )
    assert ok is True
    assert count == 3


def test_cuatro_condiciones_reales_mas_embarazo_sigue_rechazando():
    """El exemption es solo para embarazo/lactancia — 4 condiciones REALES
    siguen rechazando aunque además se marque Embarazo."""
    ok, count, cap = _validate_medical_conditions_cap(
        _payload(["Diabetes T2", "Hipertensión", "Hipotiroidismo", "Colesterol Alto", "Embarazo"])
    )
    assert ok is False
    assert count == 4


def test_critical_2_embarazo_con_diabetes_cuenta_contra_el_cap():
    """[CRITICAL-2-FIX] Ancla exacta del incidente reportado: una frase que
    MENCIONA embarazo pero no es el chip exacto cuenta como condición real
    — 3 reales + esa frase = 4, excede el cap."""
    ok, count, cap = _validate_medical_conditions_cap(
        _payload(["Diabetes T2", "Hipertensión", "Hipotiroidismo", "embarazo con diabetes"])
    )
    assert ok is False
    assert count == 4


def test_critical_2_cinco_variantes_embarazo_x_cuentan_las_cinco():
    """[CRITICAL-2-FIX] Ancla exacta del incidente reportado: 5×
    "embarazo-N" (ninguna es el valor literal del chip) deben contar las 5
    contra el cap — la versión con substring las eximía todas (count=0)."""
    ok, count, cap = _validate_medical_conditions_cap(
        _payload([f"embarazo-{i}" for i in range(1, 6)])
    )
    assert count == 5
    assert ok is False


# ---------------------------------------------------------------------------
# (f) parser: ambos endpoints de generación de plan nuevo llevan el guard con
# el mismo contrato de `code`/`max`. Tooltip-anchor: si renombras
# `_validate_medical_conditions_cap` o el `code` del detail, este test falla
# ANTES de que el drift llegue a producción.
# ---------------------------------------------------------------------------
_PLANS_PY = Path(__file__).resolve().parent.parent / "routers" / "plans.py"


def _read_plans_source() -> str:
    return _PLANS_PY.read_text(encoding="utf-8")


def test_ambos_endpoints_de_generacion_cablean_el_guard():
    src = _read_plans_source()
    call_count = src.count("_validate_medical_conditions_cap(data)")
    assert call_count == 2, (
        f"Se esperaban exactamente 2 call sites de `_validate_medical_conditions_cap(data)` "
        f"(`/analyze` y `/analyze/stream`); se encontraron {call_count}. Si se añadió/quitó un "
        f"endpoint de generación de plan NUEVO, actualiza este test y verifica que el guard "
        f"esté (o deliberadamente no esté, documentando por qué) en el nuevo endpoint."
    )


def test_detail_code_too_many_medical_conditions_presente():
    src = _read_plans_source()
    assert src.count('"code": "too_many_medical_conditions"') == 2, (
        "El `code` del detail 422 debe aparecer exactamente en los 2 call sites del guard."
    )
    assert src.count('"max": _mc_cap') == 2


def test_swap_meal_no_lleva_el_guard_por_diseno():
    """`/swap-meal` hidrata clinical data DESDE el perfil ya persistido
    (`_enrich_clinical_from_profile`), no re-recibe el formulario completo de
    generación. Aplicarle el cap rompería swaps legítimos sobre perfiles
    legacy con >cap condiciones (grandfathered). Ancla la decisión: si
    alguien añade el guard aquí, este test falla para forzar review."""
    src = _read_plans_source()
    swap_start = src.index('@router.post("/swap-meal")')
    swap_persist_start = src.index('@router.post("/{plan_id}/swap-meal/persist")')
    swap_body = src[swap_start:swap_persist_start]
    assert "_validate_medical_conditions_cap" not in swap_body


def test_regenerate_simplified_no_toma_body_ni_lleva_el_guard():
    """`/{plan_id}/chunks/{chunk_id}/regenerate-simplified` no declara
    `data: dict = Body(...)` — no re-recibe el formulario, así que no puede
    (ni debe) llevar el cap de condiciones médicas."""
    src = _read_plans_source()
    start = src.index('@router.post("/{plan_id}/chunks/{chunk_id}/regenerate-simplified")')
    # siguiente @router.post tras este endpoint acota el cuerpo de la función.
    next_marker = src.index("@router.post(", start + 1)
    body = src[start:next_marker]
    assert "Body(...)" not in body
    assert "_validate_medical_conditions_cap" not in body


# ===========================================================================
# [CRITICAL-1-FIX · 2026-08-01] `_close_medical_freetext_scope` — cierre del
# bypass del cap vía `otherConditions`/`otherMedications`.
#
# Bug (encontrado en code review, verificado ejecutando): `_validate_medical_
# conditions_cap` solo contaba `medicalConditions`, pero `_merge_other_text_
# fields` corría DESPUÉS (a nivel router y otra vez dentro de
# `arun_plan_pipeline`) fusionando `otherConditions` DENTRO de
# `medicalConditions` — 3 chips + un `otherConditions` con 7 condiciones
# separadas por coma = 10 condiciones reales llegando al pipeline sin que el
# cap las hubiera visto. Cualquier browser con JS cacheado pre-deploy (el
# input viejo aún sabe rellenar `otherConditions`) lo dispara sin intención
# maliciosa.
# ===========================================================================
def test_close_medical_freetext_scope_vacia_otherConditions_poblado():
    data = {"medicalConditions": ["Diabetes T2"], "otherConditions": "Gastritis, Asma, Anemia"}
    _close_medical_freetext_scope(data)
    assert data["otherConditions"] == ""


def test_close_medical_freetext_scope_vacia_otherMedications_poblado():
    data = {"medications": ["Metformina"], "otherMedications": "Ibuprofeno, Losartán"}
    _close_medical_freetext_scope(data)
    assert data["otherMedications"] == ""


def test_close_medical_freetext_scope_no_toca_otros_campos_freetext():
    """`otherAllergies`/`otherDislikes`/`otherStruggles` NO son parte de esta
    decisión de producto (solo condiciones médicas + medicamentos) — deben
    sobrevivir intactos."""
    data = {
        "otherConditions": "Gastritis",
        "otherMedications": "Ibuprofeno",
        "otherAllergies": "Mariscos",
        "otherDislikes": "Cilantro",
        "otherStruggles": "Antojos nocturnos",
    }
    _close_medical_freetext_scope(data)
    assert data["otherConditions"] == ""
    assert data["otherMedications"] == ""
    assert data["otherAllergies"] == "Mariscos"
    assert data["otherDislikes"] == "Cilantro"
    assert data["otherStruggles"] == "Antojos nocturnos"


def test_close_medical_freetext_scope_no_op_sobre_campos_ausentes_o_vacios():
    data = {"medicalConditions": ["Ninguna"]}
    _close_medical_freetext_scope(data)
    assert data == {"medicalConditions": ["Ninguna"]}  # no añade keys nuevas


def test_close_medical_freetext_scope_defensivo_ante_no_dict():
    # No debe lanzar — mismo contrato defensivo que `_validate_form_data_min`.
    _close_medical_freetext_scope(None)
    _close_medical_freetext_scope([])
    _close_medical_freetext_scope("no soy un dict")


def test_critical_1_ancla_el_incidente_reportado_10_condiciones_via_freetext():
    """[CRITICAL-1-FIX] Reproduce el incidente EXACTO del review: 3 chips +
    otherConditions con 7 condiciones extra = 10 condiciones reales que
    ANTES del fix llegaban al pipeline sin que el cap las contara (porque
    el cap corre sobre `medicalConditions` PRE-merge). Tras el fix, el
    payload procesado por el endpoint (que llama `_close_medical_freetext_
    scope` ANTES de `_merge_other_text_fields`) nunca ve esas 7 condiciones
    — `otherConditions` llega vacío al merge, que se vuelve no-op para ese
    campo. Este test simula el orden real de operaciones del endpoint."""
    data = {
        "medicalConditions": ["Diabetes T2", "Hipertensión", "Colesterol Alto"],
        "otherConditions": "Gastritis, Asma, Anemia, Migraña, Gota, Lupus, Fibromialgia",
    }
    # Orden real del endpoint: close-scope ANTES del cap check y ANTES de
    # copiar a pipeline_data / correr el merge.
    _close_medical_freetext_scope(data)
    ok, count, cap = _validate_medical_conditions_cap(data)
    assert ok is True
    assert count == 3, "el cap debe seguir viendo solo los 3 chips, nunca las 7 de otherConditions"

    # Y el merge downstream (import perezoso para no acoplar el import-time
    # de graph_orchestrator a este test file) confirma el no-op: con
    # otherConditions="" no hay nada que fusionar.
    from graph_orchestrator import _merge_other_text_fields
    pipeline_data = dict(data)
    added = _merge_other_text_fields(pipeline_data)
    assert added == 0, f"el merge no debería añadir nada con otherConditions vaciado; añadió {added}"
    assert pipeline_data["medicalConditions"] == ["Diabetes T2", "Hipertensión", "Colesterol Alto"], (
        "las 7 condiciones de otherConditions NUNCA deben aparecer en medicalConditions post-merge"
    )


# ---------------------------------------------------------------------------
# Parser: `_close_medical_freetext_scope` cableada en AMBOS endpoints, y
# ANTES de `_validate_form_data_min` (para que tampoco pueda satisfacer el
# companion-presence-check vía texto libre) — mismo patrón de anchor que el
# resto de la sección (f).
# ---------------------------------------------------------------------------
def test_close_medical_freetext_scope_cableada_en_ambos_endpoints():
    src = _read_plans_source()
    call_count = src.count("_close_medical_freetext_scope(data)")
    assert call_count == 2, (
        f"Se esperaban exactamente 2 call sites de `_close_medical_freetext_scope(data)` "
        f"(`/analyze` y `/analyze/stream`); se encontraron {call_count}."
    )


def test_close_medical_freetext_scope_corre_antes_de_validate_form_data_min():
    src = _read_plans_source()
    analyze_start = src.index('@router.post("/analyze")')
    stream_start = src.index('@router.post("/analyze/stream")')
    for label, start, end in (
        ("/analyze", analyze_start, stream_start),
        ("/analyze/stream", stream_start, stream_start + 6000),
    ):
        body = src[start:end]
        close_pos = body.index("_close_medical_freetext_scope(data)")
        min_pos = body.index("_validate_form_data_min(data)")
        assert close_pos < min_pos, (
            f"{label}: `_close_medical_freetext_scope` debe correr ANTES de "
            f"`_validate_form_data_min` (para que el companion-presence-check "
            f"de medicalConditions no pueda satisfacerse con texto libre)."
        )
