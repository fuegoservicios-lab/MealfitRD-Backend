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


def test_prompt_que_contesta_duda_se_extrae_como_respuesta():
    v = _vision()   # DUDAS: ¿Cuántos huevos usaste en el revuelto? (opciones: 1 huevo · 2 huevos (supuesto) · 3 huevos)
    assert rf.extraer_respuesta_de_prompt(v, "2 huevos") == "2 huevos"
    v_marked = rf.marcar_rotulo(v, "2 huevos")
    assert v_marked.get("respuestas") == "2 huevos"
    assert rf.foto_para_anotar(v, "2 huevos") is True
    ctx = build_vision_context(rf.preparar_vision(v_marked))
    assert "RESPUESTAS A LAS DUDAS DE LA FOTO" in ctx
    assert "pregúntale solo esto" not in ctx
    assert "Plátano verde hervido" in ctx


def test_prompt_con_comida_y_supuesto_registra_plato_completo_con_proactividad():
    """[P1-PHOTO-PROACTIVE-PLATE · 2026-10-21] Caso vivo: 'Desayuné\\n2 huevos' con foto de plátano, queso, aguacate y 2 huevos.
    El texto aclara la duda del supuesto y la comida, sin descartar el resto del plato."""
    v = {
        "kind": "multi",
        "items": [{
            "kind": "plato",
            "description": (
                "Plátano verde hervido cortado en trozos, acompañado de 2 huevos fritos, queso blanco derretido y 2 lascas finas de aguacate. "
                "DUDAS (pregúntale solo esto): ¿Eran 1 o 2 huevos fritos? (opciones: 2 huevos (supuesto) · 1 huevo) "
                "(Estimación: Calorías: 535, Proteína: 23g, Carbohidratos: 45g, Grasas Saludables: 30g)"
            )
        }],
        "has_text": True
    }
    prompt = "Desayuné\n2 huevos"
    ans = rf.extraer_respuesta_de_prompt(v, prompt)
    assert ans == "2 huevos"
    assert rf.es_respuesta_supuesto(v, ans) is True
    assert rf.tipo_de_comida_en_texto(prompt) == "desayuno"

    v_marked = rf.marcar_rotulo(v, prompt)
    assert v_marked.get("respuestas") == "2 huevos"
    assert v_marked.get("_ajuste_aplicado") is True
    assert v_marked.get("rotulo") == "desayuno"

    ctx = build_vision_context(v_marked)
    assert "YA incluye sus respuestas: registra esas cifras del PLATO COMPLETO" in ctx
    assert "Plátano verde hervido con 2 huevos fritos, queso blanco y aguacate" in ctx
    assert "proactivamente" in ctx


def test_prompt_con_reemplazo_de_proteina_actualiza_descripcion_y_rotulo():
    """[P1-PHOTO-MEAT-REPLACE · 2026-10-22] Caso vivo: 'Almuerzo\\nSalmón' ante foto con arroz, pechuga (supuesto) y ensalada.
    Reemplaza pechuga por salmón en la descripción, fija rótulo almuerzo y no duplica comidas."""
    v = {
        "kind": "multi",
        "items": [{
            "kind": "plato",
            "description": (
                "Plato servido con una porción de arroz blanco, un filete pequeño de pechuga de pollo dorada a la plancha, rodajas de tomate y rodajas de pepino fresco. "
                "DUDAS (pregúntale solo esto): ¿Qué tipo de carne es la que está en el plato? (opciones: Pechuga de pollo (supuesto) · Salmón · Masa de cerdo) "
                "(Estimación: Calorías: 430, Proteína: 32g, Carbohidratos: 62g, Grasas Saludables: 5g)"
            )
        }],
        "has_text": True
    }
    prompt = "Almuerzo\nSalmón"
    ans = rf.extraer_respuesta_de_prompt(v, prompt)
    assert ans == "Salmón"
    assert rf.es_respuesta_supuesto(v, ans) is False
    assert rf.tipo_de_comida_en_texto(prompt) == "almuerzo"

    v_prep = rf.preparar_vision(rf.marcar_rotulo(v, prompt))
    assert v_prep.get("respuestas") == "Salmón"
    assert v_prep.get("rotulo") == "almuerzo"
    desc = v_prep["items"][0]["description"]
    assert "pechuga de pollo" not in desc.lower()
    assert "salmón" in desc.lower()

    ctx = build_vision_context(v_prep)
    assert "salmón" in ctx.lower()
    assert "ACLARACIÓN DE LA PROTEÍNA/CARNE" in ctx
    assert "almuerzo" in ctx.lower()


def test_frontend_espera_las_respuestas_antes_del_coach():
    """El chat no abre el turno del coach mientras la foto tenga dudas sin contestar."""
    ruta = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "src", "pages", "AgentPage.jsx")
    if not os.path.exists(ruta):
        return   # el backend se prueba también fuera del monorepo
    s = io.open(ruta, encoding="utf-8").read()
    assert "P1-PLAN-LOTE-690" in s and "fotoPendienteRef" in s and "respuestas:" in s
