"""[P1-PLAN-LOTE-164 · 2026-09-22] El interruptor del generador y el formulario (frontend, por OTA).

1. En el iPhone del dueño «Generación de planes» no hacía NADA. El registro de nginx lo acota: Configuración se montó
   cuatro veces en tres minutos y ni un paso del formulario llegó a montarse (cero telemetría del asistente). El
   diálogo de confirmación no llegaba a verse sobre la ventana de Configuración y su siguiente toque lo cerraba sin
   verlo. Encender sin plan abre ahora el formulario directamente — abrirlo no gasta nada; el crédito se gasta al final.
2. Y el formulario pregunta SOLO lo que falta (`utils/completarFormulario.js`): la rama del plan entera eran 26 pasos
   con 9 ya contestados, sin auto-avance y sin «Saltar».
3. Un 422 con código propio (rango biométrico, campos que faltan, alcance clínico) caía a «Conexión interrumpida · tu
   plan se sigue generando» SIN limpiar la bandera de generación, y ProtectedRoute lo devolvía a /plan en bucle.
4. «Otra condición» / «Otro medicamento» bloqueaban también la rama del CONTADOR, que no genera nada con reglas
   clínicas: quien lo declaraba con honestidad no podía usar la app.
5. «Tus Medidas» apagaba «Siguiente» en silencio; el selector de idioma del formulario no guardaba en la cuenta.

Los tests funcionales viven en `frontend/src/__tests__/lote164.test.jsx`; aquí, las anclas y el marcador.

Tooltip-anchor: P1-PLAN-LOTE-164
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def _f(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip(f"frontend ausente: {rel}")
    return p.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_encender_sin_plan_abre_el_formulario_sin_dialogo():
    st = _f("src/pages/Settings.jsx")
    i = st.index("if (!pausing && !planData) {")
    bloque = st[i:st.index("        if (pausing) {", i)]
    assert "pedirCompletarFormulario();" in bloque
    assert "confirmToast(" not in bloque, "el diálogo no se veía en el iPhone: esta rama no lo necesita"
    assert bloque.index("pedirCompletarFormulario();") < bloque.index("navigate('/assessment')")
    dt = _f("src/components/dashboard/DashboardTracking.jsx")
    j = dt.index("const irAlPlan = () => {")
    assert "pedirCompletarFormulario();" in dt[j:dt.index("};", j)]


def test_el_formulario_pregunta_solo_lo_que_falta():
    flow = _f("src/components/assessment/InteractiveAssessmentFlow.jsx")
    assert "const _completando = Boolean(completar) && !isGuest && !_isTracking && !planData;" in flow
    assert "const steps = _isTracking ? _trackingSteps : (_pasosCompletar || [_appModeStep, ...planOnlySteps]);" in flow
    # la lista se fija UNA vez y no se calcula con las alergias cifradas aún sin leer
    assert "} else if (!loadingSensitive) {" in flow
    assert "fijarPasosCompletar(_idsFijadosRef.current)" in flow
    # el envío suelta el modo (un 422 vuelve al formulario ENTERO, en el campo rechazado)
    envio = flow[flow.index("const submitAndGenerate = async () => {"):flow.index("navigate('/plan');")]
    assert "terminarCompletarFormulario();" in envio
    util = _f("src/utils/completarFormulario.js")
    assert "export const CLAVE_COMPLETAR = 'mealfit_wizard_completar';" in util
    ctx = _f("src/context/AssessmentContext.jsx")
    assert ctx.count("safeLocalStorageRemove(CLAVE_COMPLETAR);") == 3, "la marca muere con la posición del asistente"


def test_un_rechazo_del_servidor_no_es_una_conexion_cortada():
    plan = _f("src/pages/Plan.jsx")
    i = plan.index("                    if (error.terminal) {")
    assert i < plan.index("let _hasInProgressFlag = false;")
    rama = plan[i:i + 700]
    assert "safeLocalStorageRemove('mealfit_plan_in_progress');" in rama
    assert "state: error.field ? { irACampo: error.field } : undefined" in rama
    assert len(re.findall(r"\.field = campoDelRechazo\(_detail, ", plan)) == 2


def test_el_contador_no_se_cierra_por_una_condicion_fuera_de_alcance():
    q = _f("src/components/assessment/questions/QMedical.jsx")
    assert "const enContador = formData.appMode === 'tracking';" in q
    assert "(outOfScopeSelected && !enContador)" in q
    flow = _f("src/components/assessment/InteractiveAssessmentFlow.jsx")
    assert "if (!_isTracking && hasOutOfScopeMedical(formData)) {" in flow
    assert "&& (_isTracking || !hasOutOfScopeMedical(formData));" in flow


def test_medidas_avisan_y_el_idioma_del_formulario_se_guarda():
    m = _f("src/components/assessment/questions/QMeasurements.jsx")
    for clave in ("¿Son kilos? Toca KG: en libras el mínimo es {min}.", "≈ {valor} {unidad}",
                  "Escribe una edad entre {min} y {max} años."):
        assert f"t('{clave}'" in m
    ls = _f("src/components/common/LocaleSwitcher.jsx")
    assert "if (!aplicado || aplicado === SUPERSEDED || !guardarEnCuenta) return;" in ls
    assert "await import('../../config/api')" in ls, "el login no carga el cliente de API por el selector"
    lay = _f("src/components/assessment/InteractiveAssessmentLayout.jsx")
    assert 'guardarEnCuenta={!isGuest && Boolean(session)}' in lay
    for loc in ("en-US", "pt-BR", "fr-FR", "it-IT"):
        cat = json.loads(_f(f"src/i18n/locales/{loc}.json"))
        assert isinstance(cat.get("Te faltan {n} preguntas para tu plan."), dict), loc
        assert "Revisalo antes de continuar." not in cat and "Revísalo antes de continuar." in cat, loc


def test_el_marcador_esta_al_dia():
    src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'^_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · ', src, re.M)
    assert m and int(m.group(1)) >= 164
