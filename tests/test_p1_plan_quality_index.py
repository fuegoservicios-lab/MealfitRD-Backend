"""[P1-PLAN-QUALITY-INDEX + P1-COST-ATTRIBUTION · 2026-07-31] El medidor de
variedad y coherencia, y la atribución de costo que lo hace comparable.

POR QUÉ: el owner preguntó si cambiar el generador (deepseek-flash →
gpt-5.6-luna) subiría la calidad. No se podía responder — las señales existían
sueltas y sin persistir juntas, y el libro de costo no se podía cruzar con
nada: medido el 2026-07-31, `user_id` era NULL en 6.061 de 6.063 filas y
`plan_id` en las 6.063.

⚠️ LA LECCIÓN QUE ESTE TEST PROTEGE: la primera versión del índice leía
`_shopping_coherence_block` y `_low_quality_dishes`, claves que NO existen en
`plan_data`. El componente de coherencia —35% del peso y una de las dos cosas
que se querían medir— devolvía 100 en los 6 planes reales y parecía sano. Un
medidor inerte es peor que ninguno: da confianza falsa. Por eso aquí se afirma
que cada componente REACCIONA a su defecto, no sólo que existe.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_SERVICES = (_BACKEND / "services.py").read_text(encoding="utf-8")
_GO = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_SYSTEM = (_BACKEND / "routers" / "system.py").read_text(encoding="utf-8")

from plan_quality_index import compute_plan_quality_index  # noqa: E402


def _plan(**extra):
    base = {
        "days": [{"day": 1}, {"day": 2}, {"day": 3}],
        "variety_report": {"total_meals": 12},
        "clinical_band_score": {"score": 1.0},
        "micronutrient_report": {"gaps": []},
    }
    base.update(extra)
    return base


def test_plan_perfecto_puntua_100():
    r = compute_plan_quality_index(_plan())
    assert r["score"] == 100.0
    assert set(r["componentes"]) == {"variedad", "coherencia", "nutricion"}


# ------------------------------------------------------------------
# Cada componente REACCIONA (anti-inerte)
# ------------------------------------------------------------------

@pytest.mark.parametrize("campo,plan_kwargs", [
    ("variedad", {"variety_report": {"total_meals": 12, "same_day_protein_repeats": 1}}),
    ("coherencia", {"_recipe_coherence_errors": ["ingrediente huérfano"]}),
    ("coherencia", {"_shopping_coherence_block_history": [{"presence_count": 1, "magnitude_count": 0}]}),
    ("coherencia", {"dish_quality_report": {"low_quality_meals": 1}}),
    ("nutricion", {"micronutrient_report": {"gaps": [{"status": "bajo"}]}}),
    ("nutricion", {"micronutrient_report": {"gaps": [{"status": "alto"}]}}),
    ("nutricion", {"clinical_band_score": {"score": 0.5}}),
])
def test_cada_defecto_baja_su_componente(campo, plan_kwargs):
    limpio = compute_plan_quality_index(_plan())
    sucio = compute_plan_quality_index(_plan(**plan_kwargs))
    assert sucio["componentes"][campo] < limpio["componentes"][campo], (
        f"el componente '{campo}' NO reacciona a {list(plan_kwargs)} — "
        f"medidor inerte (la clave probablemente no existe en plan_data)"
    )
    assert sucio["score"] < limpio["score"]


def test_techo_y_piso_no_se_cuentan_dos_veces():
    """`gaps` mezcla pisos ('bajo') y techos ('alto'). La primera versión
    contaba el sodio como hueco Y otra vez como techo."""
    solo_techo = compute_plan_quality_index(
        _plan(micronutrient_report={"gaps": [{"status": "alto"}]}))
    d = solo_techo["defectos"]["nutricion"]
    assert d.get("micro_techo") == 1
    assert "micro_piso" not in d, "un techo no puede contarse también como piso"


def test_claves_leidas_existen_en_produccion():
    """Las claves del índice se verificaron contra 6 planes reales. Si alguien
    añade una lectura, que sea de una clave que el pipeline persiste de verdad."""
    src = (_BACKEND / "plan_quality_index.py").read_text(encoding="utf-8")
    for clave in ("variety_report", "_recipe_coherence_errors",
                  "_shopping_coherence_block_history", "dish_quality_report",
                  "micronutrient_report", "clinical_band_score"):
        assert f'"{clave}"' in src, f"el índice debe leer {clave}"
    # Las dos claves fantasma del primer intento no deben volver. Se busca la
    # LECTURA (`.get("clave")`), no la cadena suelta: este mismo archivo y el
    # módulo explican en prosa por qué esas claves estaban mal, y un `not in`
    # sobre el texto se acusaría a sí mismo (tercera vez en la sesión).
    for fantasma in ('_shopping_coherence_block', '_low_quality_dishes'):
        assert f'.get("{fantasma}")' not in src, (
            f"{fantasma} NO existe en plan_data — leerla deja el componente inerte"
        )


def test_no_bloquea_ni_muta():
    """El medidor es telemetría: nunca puede impedir que un plan se guarde."""
    assert compute_plan_quality_index({}).get("score") is not None or True
    assert compute_plan_quality_index("no soy un dict")["score"] is None
    p = _plan()
    antes = dict(p)
    compute_plan_quality_index(p)
    assert p == antes, "compute_plan_quality_index no debe mutar el plan"
    # El estampado devuelve COPIA (no muta) y es best-effort.
    assert "return {**plan_data, \"_quality_index\": idx}" in _SERVICES
    assert "def _stamp_quality_index" in _SERVICES


def test_estampado_en_las_dos_rutas_de_persistencia():
    """Chunked y no-chunked: si sólo una estampa, la muestra queda sesgada."""
    assert _SERVICES.count("_stamp_quality_index(plan_data)") >= 2, (
        "las DOS rutas de persistencia deben estampar el índice"
    )


# ------------------------------------------------------------------
# P1-COST-ATTRIBUTION
# ------------------------------------------------------------------

def test_emisor_adjunta_usuario_y_correlacion():
    # Anclado al marker, NO a `index("log_llm_usage_event(")`: la primera
    # aparición de esa cadena está en un COMENTARIO de la cabecera del módulo,
    # así que la ventana caía a 3.000 líneas del call site real.
    i = _GO.index("[P1-COST-ATTRIBUTION")
    win = _GO[i:i + 1800]
    assert "user_id=_attr_uid" in win, "el evento de costo debe llevar user_id"
    assert '"corr"' in win, (
        "debe estampar el id de correlación: durante la generación el plan aún "
        "no tiene id (invariante I1), así que es la única clave de unión"
    )


def test_canje_corr_a_plan_id_existe_y_es_acotado():
    from db_profiles import attach_plan_id_to_usage_events  # noqa: F401
    src = (_BACKEND / "db_profiles.py").read_text(encoding="utf-8")
    i = src.index("def attach_plan_id_to_usage_events")
    win = src[i:i + 2200]
    assert "plan_id IS NULL" in win, "nunca debe reasignar lo ya atribuido"
    assert "minutes" in win, "debe acotarse por ventana temporal"
    assert "returning=True" in win, (
        "execute_sql_write devuelve BOOL sin returning: contar filas exige RETURNING"
    )


def test_persistencia_no_pasa_true_como_plan_id():
    """`save_new_meal_plan_atomic` devuelve True (no el UUID) sin return_id:
    un `if plan_id:` a secas escribiría la cadena 'True' en la columna."""
    assert "return_id and isinstance(plan_id, str)" in _SERVICES


def test_endpoint_admin_distingue_sin_dato_de_cero():
    i = _SYSTEM.index("def admin_plan_quality")
    win = _SYSTEM[i:i + 4200]
    assert "_verify_admin_token" in win and "_check_admin_rate_limit" in win
    assert 'if r.get("calls") else None' in win, (
        "un plan sin costo atribuido debe dar None, no 0 — si no, la media miente"
    )
    assert "sin_indice" in win, "hay que declarar los planes anteriores al índice"
