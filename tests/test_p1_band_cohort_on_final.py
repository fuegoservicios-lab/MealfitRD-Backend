"""[P1-BAND-COHORT-ON-FINAL · 2026-07-26] El A/B comparaba la banda que el usuario NO recibió.

Los dos primeros planes con Luna, medidos en `pipeline_metrics`:

    07:13:14  clinical_band_final   band=1.00   cohorte=None   ← lo ENTREGADO
    07:13:16  clinical_band         band=0.833  cohorte=on     ← lo que leía el A/B

Dos segundos de diferencia y 0.167 de banda. La cohorte del canario vivía SOLO en `clinical_band`,
que es la lectura PRE-finalize; la banda entregada vive en `clinical_band_final`, que no llevaba
cohorte. El lector cruzaba lo único que podía cruzar y comparaba cohortes con un número que nadie
recibió.

Es exactamente la trampa que `P1-BAND-METRIC-FINAL` cerró para el tablero (auditoría 2026-07-24,
donde yo mismo reporté 0.75 sobre un plan entregado en 1.00) y que reabrí al etiquetar sólo la fila
vieja. Cuarta instancia de "toda métrica derivada necesita refresco en la cola o miente".

La cohorte se recalcula desde `form_data`, no desde `user_id` a secas: para invitados la identidad
del bucket es `session_id`, y pasar `user_id=None` los mandaría todos al bucket 'anon'.
"""
import ast
from pathlib import Path

import graph_orchestrator as go


def _fuente() -> str:
    return (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")


def _cuerpo(nombre: str) -> str:
    src = _fuente()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == nombre:
            return ast.get_source_segment(src, n) or ""
    raise AssertionError(f"no existe {nombre}")


# ───────────── 1. la fila ENTREGADA lleva cohorte ─────────────

def test_el_emit_final_etiqueta_la_cohorte():
    cuerpo = _cuerpo("_emit_clinical_band_final_metric")
    assert '"daygen_model_cohort"' in cuerpo, \
        "sin cohorte aquí, el A/B sólo puede leer la banda pre-finalize"
    assert "_daygen_model_canary_cohort" in cuerpo


def test_la_cohorte_sale_de_form_data_no_de_user_id():
    """Los invitados se reparten por `session_id`; con `user_id=None` caerían todos en 'anon' y
    la mitad de la muestra quedaría en un solo bucket."""
    cuerpo = _cuerpo("_emit_clinical_band_final_metric")
    i = cuerpo.index('"daygen_model_cohort"')
    assert 'plan_data.get("form_data")' in cuerpo[i:i + 400]


def test_tambien_registra_que_modelo_era_el_canario():
    """Sin el modelo, una fila 'on' de la semana pasada es indistinguible de una de hoy si el
    knob cambió de valor entre medias."""
    assert '"daygen_canary_model"' in _cuerpo("_emit_clinical_band_final_metric")


# ───────────── 2. el lector usa cada nodo para lo que sirve ─────────────

def _lector() -> str:
    return (Path(go.__file__).resolve().parent / "scripts" / "daygen_canary_ab.py"
            ).read_text(encoding="utf-8")


def _sql(nombre: str) -> str:
    """Valor REAL de la constante SQL, importando el módulo.

    Nada de `s[i:i+N]`: la ventana de bytes se desbordaba al siguiente SQL y el test leía la
    consulta equivocada. Tercera vez en esta sesión — la clase se cierra leyendo el objeto, no
    el texto alrededor."""
    import importlib.util
    ruta = Path(go.__file__).resolve().parent / "scripts" / "daygen_canary_ab.py"
    spec = importlib.util.spec_from_file_location("_ab_lector", ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, nombre)


def test_la_banda_se_lee_del_nodo_final():
    sql = _sql("_SQL_BANDA_ENTREGADA")
    assert "clinical_band_final" in sql
    assert "AVG(confidence)" in sql


def test_los_reintentos_se_leen_del_nodo_del_grafo():
    """`clinical_band_final` se emite fuera del grafo con `retries=0` fijo: leer reintentos de ahí
    daría 0% siempre y el A/B diría que ningún modelo reintenta nunca."""
    sql = _sql("_SQL_COSTO_REINTENTOS")
    assert "node = 'clinical_band'" in sql and "clinical_band_final" not in sql
    assert "retries" in sql
    assert "AVG(confidence)" not in sql, "la banda de este nodo es pre-finalize"


def test_el_lector_avisa_de_la_diferencia():
    s = _lector()
    assert "pre-finalize" in s and "RECIBIÓ" in s


# ───────────── 3. no se inventa un valor cuando falta ─────────────

def test_sin_lectura_entregada_muestra_guion_no_cero():
    """Las filas anteriores a este fix no llevan cohorte. Rellenar con 0 las haría ver como
    planes horribles; con 1.0, como perfectos. Se muestra '—'."""
    s = _lector()
    assert 'c.get(\'band_entregada\')' in s or 'c.get("band_entregada")' in s
    i = s.index("def _fmt(")
    assert 'return "—"' in s[i:i + 260]


# ───────────── 4. [P1-BAND-METRIC-NO-SILENT-DROP · 2026-07-26] ninguna corrida se pierde ─────────

def test_la_fila_de_clinical_band_se_emite_siempre():
    """El guard era `if _band_val is not None:` y descartaba la fila ENTERA cuando la banda no se
    podía calcular — con ella se iba el `retries` de esa corrida.

    Medido: el plan b2464ffe (2026-07-26 14:55) **reintentó una vez** y no dejó fila; su log tampoco
    imprimió «CLINICAL BAND SCORE», o sea `_band_val` era None. El sesgo NO es aleatorio: las
    corridas que se perdían eran justo las problemáticas, las que el A/B más necesita contar."""
    src = _fuente()
    i = src.index('"node": "clinical_band",')
    ventana = src[max(0, i - 1600):i]
    assert "P1-BAND-METRIC-NO-SILENT-DROP" in ventana
    assert "if _band_val is not None:" not in ventana, \
        "volvió el guard que descarta la corrida entera"


def test_sin_score_va_cero_pero_MARCADO():
    """`confidence` es NOT NULL, así que sin score va 0.0. Un 0 sin marca sería PEOR que la fila
    ausente: bajaría la banda media de la cohorte con un plan que nunca se midió."""
    src = _fuente()
    i = src.index('"node": "clinical_band",')
    bloque = src[i:i + 1400]
    assert '"confidence": _band_val if _band_val is not None else 0.0' in bloque
    assert '"band_unavailable": _band_val is None' in bloque


def test_el_lector_cuenta_las_corridas_sin_banda_pero_no_las_promedia():
    sql_ret = _sql("_SQL_COSTO_REINTENTOS")
    sql_band = _sql("_SQL_BANDA_ENTREGADA")
    assert "band_unavailable" in sql_ret, "debe contarlas (columna sin_banda)"
    assert "FILTER (WHERE metadata->>'band_unavailable' = 'true')" in sql_ret
    assert "<> 'true'" in sql_band, "debe EXCLUIRLAS del promedio de banda"


def test_el_lector_explica_que_cuenta_corridas_y_no_planes():
    """Comparar filas de `clinical_band` contra `meal_plans` da un '362% de cobertura' que no
    significa nada — me llevó a una falsa alarma. Queda escrito en la salida."""
    s = _lector()
    assert "corridas del PIPELINE" in s
    assert "362%" in s
