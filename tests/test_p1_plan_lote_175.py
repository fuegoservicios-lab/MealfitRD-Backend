# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-175 · 2026-09-23] Tercera vuelta de la batería REAL del generador (RD), con el 173 y el 174 dentro.

Lo que dejó a la vista (4 perfiles clínicos; los 4 con las macros ya en meta):

1. **Embarazo cayó OTRA VEZ al plan de emergencia** y lactancia perdió un intento entero (~4 min): el revisor leyó
   «queso blanco fresco» y «ricotta» sin «pasteurizado». La etiqueta corría en la sustitución clínica, pero después
   los cerradores y el re-renderizado de líneas vuelven a escribir el alimento sin la palabra. Ahora también corre al
   ENTRAR en el nodo revisor, que es lo que el revisor lee. Y el segundo rechazo de embarazo pedía «atún claro» y
   «edamame cocido», y el primero, el tope semanal de pescado (545 g en tres días).
2. **La nota de sodio de HTA era invisible para el revisor**: el resumen solo lleva las notas que dicen «Seguridad
   alimentaria» o «Nota clínica», y la de sodio empieza «⚠️ Sodio (hipertensión/riñón)». El mismo fallo que agosto
   cerró para embarazo (P1-REVIEWER-SEES-SAFETY-NOTES). Además: palmito en conserva, y el revisor no veía los
   MEDICAMENTOS (rechazó por IECA/ARA II a quien toma amlodipino).
3. **DM2 se quemaba dos intentos contra una puerta que el propio sistema disparaba**: el abaratador de presupuesto
   cambiaba quinoa → «Arroz integral» en la CENA y la puerta de horario lo rechazaba como «arroz de noche».
4. **La costura de los swaps en el nombre**: «Tostadas con huevo y Aguacate fresca» con la descripción diciendo
   «guayaba fresca»; «Yogurt con Guineo», «ricotta, Espinacas al limón». Y el patrón del autofix fruta-salado no
   tenía frontera de palabra al principio: «piña» casaba dentro de «es-PIÑA-cas»."""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


_EMB = {"medicalConditions": ["Embarazo"]}


# ─────────────────────────── 1. Lo que el revisor LEE: etiquetas al entrar en su nodo ───────────────────────────

def test_el_revisor_lee_el_plan_ya_etiquetado():
    src = _src("graph_orchestrator.py")
    i = src.index("async def review_plan_node(state: PlanState) -> dict:")
    cabeza = src[i:i + 400]
    # [P1-PLAN-LOTE-179] la misma línea re-encuadra además la proteína como el guardado
    assert '__import__("etiquetas_clinicas").plan_etiquetado(' in cabeza and 'state["form_data"])' in cabeza, \
        "la etiqueta clínica corre en la PRIMERA línea del nodo revisor, antes de armar lo que el revisor lee"


def test_plan_etiquetado_devuelve_el_mismo_plan_y_nunca_lanza():
    import etiquetas_clinicas as ec
    plan = {"days": [{"meals": [{"name": "Casabe con ricotta", "ingredients": ["60 g de ricotta", "1 casabe"]}]}]}
    assert ec.plan_etiquetado(plan, _EMB) is plan
    assert plan["days"][0]["meals"][0]["ingredients"][0] == "60 g de ricotta pasteurizada"
    assert ec.plan_etiquetado(None, _EMB) is None
    assert ec.plan_etiquetado({"days": "roto"}, _EMB) == {"days": "roto"}


def test_embarazo_atun_claro_edamame_cocido_y_yuca():
    import embarazo_seguro as e
    plan = {"days": [{"meals": [{"name": "Yuca con atún", "ingredients": ["80 g de atún en agua", "100 g de edamame",
                                                                          "150 g de yuca", "80 g de atún rojo"],
                                 "recipe": ["Hierve la yuca 20 minutos."]}]}]}
    assert e.etiquetar(plan, _EMB) == 1
    m = plan["days"][0]["meals"][0]
    assert m["ingredients"][0] == "80 g de atún claro en agua"
    assert m["ingredients"][1] == "100 g de edamame cocido"
    assert m["ingredients"][3] == "80 g de atún rojo", "el rojo lo cambia la sustitución de mercurio, no la etiqueta"
    assert any("desecha el agua" in p for p in m["recipe"]), "la yuca: hervida del todo y sin su agua"
    assert e.etiquetar(plan, _EMB) == 0, "idempotente"


def test_la_receta_que_ya_bota_el_agua_no_se_repite():
    import embarazo_seguro as e
    plan = {"days": [{"meals": [{"name": "Yuca", "ingredients": ["150 g de yuca"],
                                 "recipe": ["Hierve la yuca y bota el agua."]}]}]}
    e.etiquetar(plan, _EMB)
    assert len(plan["days"][0]["meals"][0]["recipe"]) == 1


def test_el_prompt_de_embarazo_pone_tope_semanal_al_pescado():
    import condition_rules as cr
    bloque = next(r for r in cr.CONDITION_RULES if r.id == "pregnancy").prompt_block
    assert "227-340 g" in bloque and "atún CLARO" in bloque
    assert "pasteurizado" in bloque and "cocidos" in bloque


# ─────────────────────────── 2. HTA: la nota de sodio, el palmito y los medicamentos ───────────────────────────

def test_la_nota_de_sodio_viaja_al_resumen_del_revisor():
    import graph_orchestrator as go
    meal = {"recipe": ["Mezcla todo.", go._LOWSODIUM_NOTE]}
    assert "bajas en sodio y enjuaga" in go._meal_safety_notes_for_summary(meal), \
        "una nota que el revisor no ve no previene ningún rechazo"


def test_palmito_en_conserva():
    import graph_orchestrator as go
    import etiquetas_clinicas as ec
    assert "palmito" in go._LOWSODIUM_NOTE_TOKENS
    assert ec._linea_hta("250 g de palmito") == "250 g de palmito bajo en sodio"
    assert ec._linea_hta("100 g de palmitos") == "100 g de palmitos bajos en sodio"


def test_el_revisor_ve_los_medicamentos_y_calibra():
    src = _src("graph_orchestrator.py")
    assert "Medicamentos declarados:" in src
    from prompts.medical_reviewer import REVIEWER_SYSTEM_PROMPT as p
    assert "6. CALIBRACIÓN" in p
    assert "medicamentos DECLARADOS" in p and "DASH" in p


# ─────────────────────────── 3. El abaratador no crea «arroz de noche» ───────────────────────────

def test_el_abaratador_no_pone_arroz_en_la_cena():
    from graph_orchestrator import _budget_candidate_collides_same_day as choca
    cena = {"meal": "Cena", "name": "Quinoa guisada criolla", "ingredients": ["60 g de quinoa"]}
    almuerzo = {"meal": "Almuerzo", "name": "Quinoa con pollo", "ingredients": ["60 g de quinoa"]}
    dia = {"meals": [almuerzo, cena]}
    assert choca(dia, cena, "Arroz integral") is True, "quinoa→arroz en la cena = rechazo seguro de la puerta de horario"
    assert choca(dia, almuerzo, "Arroz integral") is False
    assert choca(dia, cena, "Maní") is False


def test_el_prompt_de_dm2_prefiere_platano_verde():
    import condition_rules as cr
    bloque = next(r for r in cr.CONDITION_RULES if r.id == "dm2").prompt_block
    assert "VERDE antes que maduro" in bloque and "casabe" in bloque
    assert "Si el plan lleva plátano" in bloque, "restricción, no imposición: un país beta no recibe plátano dictado"


# ─────────────────────────── 4. El nombre y la descripción tras un swap ───────────────────────────

def test_concordancia_masculina_y_caja():
    import graph_orchestrator as go
    fix = go._fix_name_gender_agreement
    assert fix("Tostadas con huevo y Aguacate fresca") == "Tostadas con huevo y aguacate fresco"
    assert fix("Yogurt con Guineo y semillas de girasol") == "Yogurt con guineo y semillas de girasol"
    assert fix("Yautía majada con ricotta, Espinacas al limón y edamame") == \
        "Yautía majada con ricotta, espinacas al limón y edamame"
    # Title Case: la mayúscula es el estilo, no la costura; los nombres propios del catálogo se quedan.
    assert fix("Pollo Asado con Vegetales") is None
    assert fix("Arepitas de Negrito con pollo al limón y cebolla") is None
    assert fix("Ensalada de aguacate fresca") is None, "tras «de» el núcleo puede ser la ensalada"


def test_el_autofix_fruta_salado_no_parte_espinacas_y_cambia_la_descripcion():
    import graph_orchestrator as go
    meal = {"meal": "Desayuno", "name": "Revoltillo de huevo con espinacas y piña fresca",
            "desc": "Revoltillo cremoso con espinacas y la piña fresca del día.",
            "ingredients": ["2 huevos", "30 g de espinacas", "80 g de piña"],
            "ingredients_raw": ["2 huevos", "30 g de espinacas", "80 g de piña"],
            "recipe": ["Bate los huevos con las espinacas.", "Sirve con la piña."]}
    assert go._meal_has_sweet_savory_clash(meal), "precondición: huevo + piña es el choque que el autofix repara"
    assert go._fruit_savory_autofix([{"meals": [meal]}], {}, db=object()) == 1
    assert meal["name"] == "Revoltillo de huevo con espinacas y aguacate fresco"
    assert "30 g de espinacas" in meal["ingredients"], "«piña» no puede casar dentro de «espinacas»"
    assert meal["desc"] == "Revoltillo cremoso con espinacas y el aguacate fresco del día."


def test_sustituir_alimento_respeta_numero_y_nombre_propio():
    import dish_naming as dn
    pat = re.compile(r"\bg[uúü][aá]y[aá]b[aá]\w*", re.IGNORECASE)
    assert dn.sustituir_alimento({"name": "Guayabas asadas con queso"}, pat, "Batata") == "Batatas asadas con queso"
    m = {"name": "Guayaba fresca con huevo", "desc": "La guayaba madura, con huevo."}
    assert dn.sustituir_alimento(m, pat, "Aguacate") == "Aguacate fresco con huevo"
    assert m["desc"] == "El aguacate maduro, con huevo."


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', _src("app.py"))
    assert m and int(m.group(1)) >= 175 and m.group(2) >= "2026-09-23", "el marker nunca baja"
