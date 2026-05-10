"""
Tests P0-FORM-1: defensa en backend contra contradicción
sentinel "Ninguna"/"Ninguno" + texto libre `other*`.

Escenario reportado en la auditoría:
  El usuario escribe "Maní" en `otherAllergies`, luego marca el chip "Ninguna".
  El frontend (pre-fix) dejaba `allergies=["Ninguna"]` + `otherAllergies="Maní"`.
  `_merge_other_text_fields` (pre-fix) producía `allergies=["Ninguna","Maní"]` —
  contradicción de seguridad médica: el LLM podía descartar la alergia real
  al ver el sentinel exclusivo.

Fix (graph_orchestrator._merge_other_text_fields):
  Si el array canónico contiene un sentinel "Ninguna"/"Ninguno"
  (case-insensitive), descartar el `other*` y limpiar el campo.

Cobertura por field:
  - allergies / otherAllergies   (safety-critical)
  - medicalConditions / otherConditions (safety-critical)
  - dislikes / otherDislikes     (UX/calidad)
  - struggles / otherStruggles   (UX/calidad)
"""
import pytest

from graph_orchestrator import _merge_other_text_fields


# ---------------------------------------------------------------------------
# Caso primario: contradicción que causaba el incidente médico.
# ---------------------------------------------------------------------------
def test_sentinel_ninguna_descarta_other_allergies():
    """allergies=['Ninguna'] + otherAllergies='Maní' → array queda ['Ninguna']."""
    form = {
        "allergies": ["Ninguna"],
        "otherAllergies": "Maní",
    }
    added = _merge_other_text_fields(form)
    assert form["allergies"] == ["Ninguna"]
    assert form["otherAllergies"] == ""
    assert added == 0


def test_sentinel_ninguna_descarta_other_conditions():
    """medicalConditions=['Ninguna'] + otherConditions='Lupus' → ['Ninguna']."""
    form = {
        "medicalConditions": ["Ninguna"],
        "otherConditions": "Lupus, Asma",
    }
    added = _merge_other_text_fields(form)
    assert form["medicalConditions"] == ["Ninguna"]
    assert form["otherConditions"] == ""
    assert added == 0


def test_sentinel_ninguno_descarta_other_struggles():
    """struggles=['Ninguno'] + otherStruggles='Viajes' → ['Ninguno']."""
    form = {
        "struggles": ["Ninguno"],
        "otherStruggles": "Viajes frecuentes",
    }
    added = _merge_other_text_fields(form)
    assert form["struggles"] == ["Ninguno"]
    assert form["otherStruggles"] == ""
    assert added == 0


def test_sentinel_ninguno_descarta_other_dislikes():
    """dislikes=['Ninguno'] + otherDislikes='Apio' → ['Ninguno']."""
    form = {
        "dislikes": ["Ninguno"],
        "otherDislikes": "Apio, Curry",
    }
    added = _merge_other_text_fields(form)
    assert form["dislikes"] == ["Ninguno"]
    assert form["otherDislikes"] == ""
    assert added == 0


# ---------------------------------------------------------------------------
# Variantes de capitalización del sentinel — la detección es case-insensitive.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("variant", ["ninguna", "NINGUNA", "Ninguna", " Ninguna "])
def test_sentinel_es_case_insensitive_y_strip(variant):
    """El detector matchea 'ninguna'/'NINGUNA'/' Ninguna ' indistintamente."""
    form = {"allergies": [variant], "otherAllergies": "Maní"}
    _merge_other_text_fields(form)
    # El array original se preserva tal cual (no normalizamos capitalización).
    assert form["allergies"] == [variant]
    # Pero el otherAllergies sí se descarta.
    assert form["otherAllergies"] == ""


# ---------------------------------------------------------------------------
# Sin sentinel: comportamiento original del merge se preserva intacto.
# ---------------------------------------------------------------------------
def test_sin_sentinel_merge_funciona_normal():
    """Sin sentinel, el merge fusiona 'Maní' al array — comportamiento legacy."""
    form = {
        "allergies": ["Lacteos"],
        "otherAllergies": "Maní, Fresa",
    }
    added = _merge_other_text_fields(form)
    # Orden preservado: existing primero, luego nuevos en orden de aparición.
    assert form["allergies"] == ["Lacteos", "Maní", "Fresa"]
    assert added == 2


def test_sin_other_text_no_op():
    """Sin texto libre, el array no cambia y otherAllergies se queda como vino."""
    form = {"allergies": ["Lacteos"], "otherAllergies": ""}
    added = _merge_other_text_fields(form)
    assert form["allergies"] == ["Lacteos"]
    assert form["otherAllergies"] == ""
    assert added == 0


def test_array_vacio_y_other_con_texto_no_aplica_sentinel():
    """allergies=[] + otherAllergies='Maní' → mergeo normal, sentinel no aplica."""
    form = {"allergies": [], "otherAllergies": "Maní"}
    added = _merge_other_text_fields(form)
    assert form["allergies"] == ["Maní"]
    # otherAllergies preservado (no es path de sentinel).
    assert form["otherAllergies"] == "Maní"
    assert added == 1


# ---------------------------------------------------------------------------
# Idempotencia: invocar dos veces no duplica ni reintroduce el descarte.
# ---------------------------------------------------------------------------
def test_idempotencia_con_sentinel():
    """Segunda invocación es no-op porque otherAllergies ya quedó vacío."""
    form = {"allergies": ["Ninguna"], "otherAllergies": "Maní"}
    _merge_other_text_fields(form)
    snapshot = (list(form["allergies"]), form["otherAllergies"])
    _merge_other_text_fields(form)
    assert (list(form["allergies"]), form["otherAllergies"]) == snapshot


# ---------------------------------------------------------------------------
# Edge cases defensivos.
# ---------------------------------------------------------------------------
def test_array_con_sentinel_y_alergia_real_aun_descarta_other():
    """Si por alguna razón el array ya tiene `['Ninguna','Lácteos']` (estado
    inconsistente legacy), el sentinel sigue descartando el `other*`. La
    contradicción interna del array es responsabilidad de otro fix
    (frontend toggle helper); aquí solo evitamos AMPLIFICARLA con texto libre.
    """
    form = {"allergies": ["Ninguna", "Lacteos"], "otherAllergies": "Maní"}
    _merge_other_text_fields(form)
    # No agregamos Maní; el array queda como vino (no rebatimos contradicciones
    # internas existentes, solo evitamos crearlas).
    assert form["allergies"] == ["Ninguna", "Lacteos"]
    assert form["otherAllergies"] == ""


def test_form_no_dict_devuelve_cero():
    """Robustez: tipo inesperado no crashea."""
    assert _merge_other_text_fields(None) == 0
    assert _merge_other_text_fields("not a dict") == 0
    assert _merge_other_text_fields([]) == 0


def test_multiples_campos_simultaneamente():
    """Mezcla: allergies con sentinel + medicalConditions sin sentinel + dislikes con sentinel."""
    form = {
        "allergies": ["Ninguna"],
        "otherAllergies": "Maní",
        "medicalConditions": ["Hipertensión"],
        "otherConditions": "Asma",
        "dislikes": ["Ninguno"],
        "otherDislikes": "Apio",
        "struggles": [],
        "otherStruggles": "Viajes",
    }
    added = _merge_other_text_fields(form)
    # allergies: sentinel → descarta Maní.
    assert form["allergies"] == ["Ninguna"]
    assert form["otherAllergies"] == ""
    # medicalConditions: sin sentinel → mergea Asma.
    assert form["medicalConditions"] == ["Hipertensión", "Asma"]
    assert form["otherConditions"] == "Asma"
    # dislikes: sentinel → descarta Apio.
    assert form["dislikes"] == ["Ninguno"]
    assert form["otherDislikes"] == ""
    # struggles: sin sentinel y array vacío → mergea Viajes.
    assert form["struggles"] == ["Viajes"]
    # added cuenta solo Asma + Viajes.
    assert added == 2
