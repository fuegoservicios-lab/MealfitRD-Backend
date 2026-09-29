"""[P1-PLAN-LOTE-765 · 2026-09-28] La foto de un pote de suplemento ES decir «esto es mío»: se guarda en la Alacena.

El dueño mandó, sin texto, la foto de su ganador de peso entero (Patriot Nutrition Atlas Gainer, 56 porciones). El
escáner lo leyó bien («…; 56 porciones por envase; no se lee la tabla nutricional») y el coach le pidió la tabla: la
instrucción de la foto de un envase solo sabía de ANOTAR una toma (la prueba de la regla S) y prohibía ofrecer la
Nevera. Y cuatro textos repetían que `guardar_suplemento` era solo «si dice que es suyo». El dueño: «si le enseñé esa
proteína completa es obvio que la quiero guardar en mi nevera, corrige eso, por si pasa con otro suplemento».
"""
import json
from pathlib import Path

from prompts import chat_agent as ca

# La foto del caso, tal como la mandó el cliente (siempre `multi`, también con una sola).
_POTE = {"kind": "multi", "has_text": False, "items": [{
    "attachment_id": "a1", "kind": "etiqueta",
    "description": "Patriot Nutrition; Atlas Gainer Advanced Mass Vanilla; 56 porciones por envase; no se lee la tabla "
                   "nutricional."}]}
_COMPRA = {"kind": "items", "has_text": False, "description": "1 pote de proteína whey, 2 manzanas"}


def test_la_foto_del_pote_se_guarda_en_la_alacena_sin_preguntar():
    for activa in (True, False):
        ctx = ca.build_vision_context(_POTE, nevera_activa=activa)
        assert "POTE DE SUPLEMENTO" in ctx and "`guardar_suplemento`" in ctx, activa
        assert "sin preguntar si lo quiere guardar" in ctx
        # sin tabla legible: se guarda SIN etiqueta, sin inventar cifras, y la tabla se pide para completarlo
        assert "guárdalo SIN etiqueta" in ctx and "un ganador de peso no es whey" in ctx
        assert "pídele una foto de la tabla para completarlo" in ctx
        # guardado el pote, no se interroga sobre una toma que nadie contó
        assert "NO le preguntes si se lo tomó ni cuántos scoops" in ctx
        # la pregunta de «¿cuántos scoops?» queda para lo de UNA porción (barra, batida lista)
        i = ctx.index("pregunta SOLO lo que falta")
        assert "Si NO es un pote de suplemento" in ctx[:i]


def test_con_la_nevera_apagada_la_regla_vale_igual_y_no_la_nombra():
    apagada = ca.build_vision_context(_POTE, nevera_activa=False)
    assert "nevera" not in apagada.lower() and "modify_pantry_inventory" not in apagada
    assert "su Alacena" in apagada   # `guardar_suplemento` decide sola si la enciende o pregunta


def test_encendida_la_prohibicion_de_ofrecer_la_nevera_es_solo_para_alimentos():
    encendida = ca.build_vision_context(_POTE)
    assert (" Si es un alimento envasado y no un suplemento, NO lo ofrezcas para la Nevera salvo que él lo pida."
            in encendida)
    assert ca._ETIQUETA_INSTRUCCION == (ca._ETIQUETA_INSTRUCCION_SIN_NEVERA
                                        + " Si es un alimento envasado y no un suplemento, NO lo ofrezcas para la "
                                          "Nevera salvo que él lo pida.")


def test_un_pote_en_la_foto_de_compra_va_a_la_alacena_no_a_los_alimentos():
    varias = {"kind": "multi", "has_text": False, "items": [{"kind": "items", "description": "1 pote de creatina"}]}
    for foto in (_COMPRA, dict(_COMPRA, has_text=True), varias):
        for activa in (True, False):
            ctx = ca.build_vision_context(foto, nevera_activa=activa)
            assert ca._POTE_EN_LA_COMPRA in ctx, (foto, activa)
            if not activa:
                assert "nevera" not in ctx.lower() and "modify_pantry_inventory" not in ctx
    assert "no con la herramienta de los alimentos" in ca._POTE_EN_LA_COMPRA


def test_los_cuatro_textos_ya_no_piden_que_diga_que_es_suyo():
    import suplementos
    import tools
    r = ca._CHAT_RESOLVE_RULES
    assert "La foto de un POTE de suplemento (o de su tabla) ES decir que lo tiene" in r
    assert "aunque no diga nada, sin preguntar si lo quiere guardar" in r
    doc = tools.guardar_suplemento.__doc__ if hasattr(tools.guardar_suplemento, "__doc__") else ""
    doc = getattr(tools.guardar_suplemento, "description", None) or doc or ""
    for texto in (r, suplementos.BLOQUE_CONOCIMIENTO, doc):
        assert "dice que es suyo" not in texto and "diga\n    que es suyo" not in texto
    assert "aunque no diga nada" in suplementos.BLOQUE_CONOCIMIENTO
    src = Path(tools.__file__).read_text(encoding="utf-8")
    i = src.index("def guardar_suplemento(")
    cuerpo = src[i:i + 2200]
    assert "no diga nada: la foto de un pote de varias porciones ES decir que lo tiene" in cuerpo.replace("\n    ", " ")
    assert "Un ganador de peso no está en la lista: sin `clave`" in cuerpo.replace("\n      ", " ")


def test_la_bateria_del_coach_tiene_los_casos_del_pote():
    ruta = Path(__file__).resolve().parents[1] / "scripts" / "coach_battery" / "battery.json"
    casos = {c["id"]: c for c in json.loads(ruta.read_text(encoding="utf-8"))["casos"]}
    p6, p7, p8 = casos["P6"], casos["P7"], casos["P8"]
    assert p6["turns"] == [""] and p6["vision"]["items"][0]["kind"] == "etiqueta"
    # [P1-PLAN-LOTE-767] antes de guardarlo sin etiqueta, busca su tabla en internet
    assert p6["expect"]["tools"][-1] == "guardar_suplemento" and "log_consumed_meal" in p6["expect"]["no_tools"]
    assert set(p7["expect"]["tools"]) == {"guardar_suplemento", "log_consumed_meal"}
    assert "guardar_suplemento" in p8["expect"]["no_tools"]   # una barra (UNA porción) no es un pote
