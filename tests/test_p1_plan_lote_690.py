# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-690 · 2026-09-28] Las dudas de la foto se contestan antes del coach y el turno las lleva.

Caso vivo (28-sep 00:42 UTC, cuenta del dueño): «Mi cena» + foto (plátano con huevos revueltos y salami, 550 kcal, dos
dudas). El coach contestó sin las respuestas —otra cena del plan— y después «2 huevos · Maduro» sin la foto: «¿Ya te los
comiste?». Nada en el contador.
"""
from __future__ import annotations

import io
import os

import respuestas_de_la_foto as rf
from prompts.chat_agent import build_vision_context

_DESC = (
    "Plátano verde hervido en trozos acompañado de huevos revueltos con trocitos de salami. DUDAS (pregúntale solo "
    "esto): ¿Cuántos huevos usaste en el revuelto? (opciones: 1 huevo · 2 huevos (supuesto) · 3 huevos) ¿El plátano "
    "era verde o maduro? (opciones: Verde (supuesto) · Maduro) (Estimación: Calorías: 550, Proteína: 20g, "
    "Carbohidratos: 57g, Grasas Saludables: 28g)"
)


def _vision(**extra):
    v = {"kind": "multi", "items": [{"kind": "plato", "description": _DESC, "attachment_id": "x"}], "has_text": True}
    v.update(extra)
    return v


def test_sin_respuestas_conducta_de_siempre():
    v = _vision()
    assert rf.preparar_vision(v) is v
    ctx = build_vision_context(v)
    assert "DUDAS DE LA FOTO" in ctx          # la regla del 305 sigue: pregunta la primera duda
    assert "RESPUESTAS A LAS DUDAS" not in ctx


def test_con_respuestas_fuera_las_dudas_y_la_regla_de_registrar():
    ctx = build_vision_context(_vision(respuestas="2 huevos · Maduro"))
    assert "pregúntale solo esto" not in ctx     # ya contestadas: la regla del 305 no se dispara
    assert "DUDAS DE LA FOTO: si el análisis" not in ctx   # la regla del 305
    assert "RESPUESTAS A LAS DUDAS DE LA FOTO" in ctx and "«2 huevos · Maduro»" in ctx
    assert "log_consumed_meal" in ctx and "EN ESTE TURNO" in ctx
    assert "salami" in ctx                        # la foto sigue en el turno


def test_el_ajuste_de_las_opciones_llega_a_la_estimacion():
    ctx = build_vision_context(_vision(respuestas="3 huevos · Maduro",
                                       ajuste={"calories": 90, "protein": 6, "carbs": 3, "healthy_fats": 5}))
    assert "Calorías: 640, Proteína: 26g, Carbohidratos: 60g, Grasas Saludables: 33g" in ctx
    assert "Calorías: 550" not in ctx


def test_ajuste_nunca_deja_negativos_y_se_acota():
    d = rf.con_ajuste("x (Estimación: Calorías: 100, Proteína: 5g, Carbohidratos: 10g, Grasas Saludables: 2g)",
                      {"calories": -500, "protein": -9, "carbs": 0, "healthy_fats": 0})
    assert "Calorías: 0, Proteína: 0g, Carbohidratos: 10g" in d
    v = rf.preparar_vision(_vision(respuestas="x", ajuste={"calories": 99999}))
    assert "Calorías: 2550" in v["items"][0]["description"]


def test_respuesta_escrita_sin_ajuste_no_toca_las_cifras():
    ctx = build_vision_context(_vision(respuestas="4 huevos con 3 yemas"))
    assert "Calorías: 550" in ctx and "«4 huevos con 3 yemas»" in ctx


def test_varias_fotos_de_plato_no_se_ajusta_ninguna():
    v = _vision(respuestas="2 huevos", ajuste={"calories": 90})
    v["items"].append({"kind": "plato", "description": _DESC})
    out = rf.preparar_vision(v)
    assert all("Calorías: 550" in it["description"] for it in out["items"])
    assert all("pregúntale solo esto" not in it["description"] for it in out["items"])


def test_respuestas_basura_no_cuentan():
    assert rf.respuestas_de({"respuestas": 5}) == ""
    assert rf.respuestas_de({"respuestas": "   "}) == ""
    assert len(rf.respuestas_de({"respuestas": "a" * 900})) == 200


def test_sin_estimacion_se_quita_el_bloque_hasta_el_final():
    assert rf.sin_dudas("Arroz. DUDAS (pregúntale solo esto): ¿cuánto?") == "Arroz."


def test_no_muta_el_vision_del_cliente():
    v = _vision(respuestas="2 huevos · Maduro", ajuste={"calories": 10})
    rf.preparar_vision(v)
    assert "pregúntale solo esto" in v["items"][0]["description"]


def test_rama_de_una_foto_tambien():
    v = {"kind": "plato", "description": _DESC, "has_text": False, "respuestas": "1 huevo · Verde",
         "ajuste": {"calories": -72, "protein": -6, "carbs": 0, "healthy_fats": -5}}
    ctx = build_vision_context(v)
    assert "Calorías: 478" in ctx and "RESPUESTAS A LAS DUDAS" in ctx and "pregúntale solo esto" not in ctx


def test_frontend_espera_las_respuestas_antes_del_coach():
    """El chat no abre el turno del coach mientras la foto tenga dudas sin contestar."""
    ruta = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "src", "pages", "AgentPage.jsx")
    if not os.path.exists(ruta):
        return   # el backend se prueba también fuera del monorepo
    s = io.open(ruta, encoding="utf-8").read()
    assert "P1-PLAN-LOTE-690" in s and "fotoPendienteRef" in s and "respuestas:" in s
