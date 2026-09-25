# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-224 · 2026-09-24] El sistema de idiomas, al 100 % para producción.

Un tester con la app en inglés mandó dos capturas: el escáner decía «What is it?: Un tazón de avena cocida con leche»
y el paso de alergias sugería «Other (e.g. Maní, Fresa...)». El dueño: «el sistema de idiomas no está al 100 %, vamos
a dejarlo al 100 % listo para producción». La auditoría encontró la misma forma en todo el producto —el título
traducido y, debajo, lo que escribe el servidor o el modelo en español— y dos agujeros de fondo. Lo que ata esta
prueba en el backend:

  1. **Los alimentos en los 5 idiomas** (`food_names_i18n.py` + `data/food_names_i18n.json`). El catálogo sólo
     tenía un gloss inglés. Medido con `clinical_backstop_for_meal`: 32 de 60 alergias escritas en inglés, francés,
     italiano o portugués NO bloqueaban un plato que sí llevaba el alimento. Ahora el texto libre se traduce AL
     CANÓNICO antes de llegar al motor (la frontera de P1-I18N-DASHBOARD sigue intacta: el motor sólo ve español).
  2. **El escáner y la estimación** nombran el plato en el idioma del usuario (sin tocar el nombre que descuenta la
     Nevera), y `/api/catalog` trae los nombres en los 5 idiomas.
  3. **Los textos del servidor** que el cliente pintaba en español: el aviso de calidad del día (ahora también en
     datos), las sugerencias de ahorro, las frases fijas de «Acción requerida» y de los bloqueos, el panel de
     micronutrientes. El frontend las traduce por coincidencia exacta; aquí se exige que el espejo esté completo.
  4. **Cambiar plato / regenerar día**: el nombre del plato nuevo ya traducido para el aviso y la tarjeta, y el
     `_display` provisional nunca se guarda.
  5. **El invitado** (el embudo del plan gratis) no persiste su plan y nunca recibía traducción:
     `traducir_plan_en_memoria` + `POST /api/plans/guest-display`.
  6. **Texto libre fuera del plan** (lo que el coach recuerda, los suplementos del día): `POST /api/i18n/textos`.

Las pruebas de pantalla viven en `frontend/src/__tests__/lote224.test.jsx`; las anclas que miran el frontend se
saltan mientras el frontend de este checkout no traiga el lote (el CI del backend clona su `main`).

Tooltip-anchor: P1-PLAN-LOTE-224
"""
from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"
_LOTE_EN_EL_FRONTEND = "src/__tests__/lote224.test.jsx"


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8").replace("\r\n", "\n")


def _front(rel: str) -> str:
    if not (_FRONT / _LOTE_EN_EL_FRONTEND).exists():
        pytest.skip("el frontend de este checkout aún no trae el lote 224 (el CI del backend clona su `main`)")
    return (_FRONT / rel).read_text(encoding="utf-8").replace("\r\n", "\n")


def _literal_js(s: str) -> str:
    """Cómo aparece `s` en un fuente JS generado con JSON.stringify (el espejo de este lote)."""
    return json.dumps(s, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Alergias escritas en otro idioma
# ─────────────────────────────────────────────────────────────────────────────

_CASOS_ALERGIA = {
    "1 taza de Fresas": ["Fresa", "Strawberry", "Strawberries", "Fraise", "Fraises", "Fragola", "Fragole", "Morango", "Morangos"],
    "1 tallo de Apio": ["Apio", "Celery", "Céleri", "Sedano", "Aipo"],
    "1 taza de Mango": ["Mango", "Mangue", "Manga"],
    "1 unidad de Tomate": ["Tomate", "Tomato", "Tomates", "Pomodoro", "Pomodori"],
    "1 taza de Piña": ["Piña", "Pineapple", "Ananas", "Abacaxi"],
    "0.5 unidad de Aguacate": ["Aguacate", "Avocado", "Avocat", "Abacate"],
    "1 taza de Maíz dulce en granos": ["Maíz", "Corn", "Maïs", "Mais", "Milho"],
    "1 taza de Leche de coco": ["Coco", "Coconut", "Noix de coco", "Cocco"],
    "2 dientes de Ajo": ["Ajo", "Garlic", "Ail", "Aglio", "Alho"],
    "1 cucharadita de Mostaza": ["Mostaza", "Mustard", "Moutarde", "Senape", "Mostarda"],
    "1 cucharada de Semillas de sésamo": ["Sésamo", "Sesame", "Sésame", "Sesamo", "Gergelim"],
    "2 cucharadas de Mantequilla de maní": ["Maní", "Peanut", "Arachide", "Arachidi", "Amendoim"],
}


@pytest.fixture(scope="module")
def go():
    logging.disable(logging.CRITICAL)
    import graph_orchestrator
    yield graph_orchestrator
    logging.disable(logging.NOTSET)


@pytest.mark.parametrize("linea,declaracion", [
    (linea, d) for linea, decls in _CASOS_ALERGIA.items() for d in decls
])
def test_una_alergia_escrita_en_otro_idioma_bloquea_el_plato_que_lleva_el_alimento(go, linea, declaracion):
    plato = {"name": "Plato de prueba", "ingredients": [linea], "recipe": []}
    assert go.clinical_backstop_for_meal(plato, allergies=[declaracion]), (
        f"«{declaracion}» no bloqueó un plato con «{linea}»: el texto libre no llegó al canónico español")


_DECLARACIONES_AJENAS = [
    "Fresa", "Strawberry", "Fraise", "Fragola", "Morango", "Tomato", "Pomodoro", "Pineapple", "Ananas", "Avocado",
    "Corn", "Maïs", "Milho", "Coconut", "Garlic", "Ail", "Aglio", "Mango", "Mangue", "Manga", "Kiwi", "Celery",
    "Lait", "Milk", "Latte", "Leite", "Egg", "Œuf", "Uovo", "Ovo", "Fish", "Poisson", "Pesce", "Peixe", "Shrimp",
    "Crevettes", "Gamberi", "Peanut", "Arachide", "Soy", "Soja", "Wheat", "Blé", "Trigo", "Banana", "Apple", "Pomme",
    "Orange", "Onion", "Oignon", "Cipolla", "Mushroom", "Champignons", "Funghi",
]


def test_ninguna_declaracion_ajena_bloquea_un_plato_sin_ese_alimento(go):
    """El sesgo del backstop es «ante la duda, de más», pero no a ciegas: «lait» (leche) no puede arrastrar la
    lechuga (raíz común si se quitaran todas las vocales finales), ni «manga» un plato sin mango."""
    limpio = {"name": "Pollo con arroz y ensalada", "recipe": [], "ingredients": [
        "150 g de Pechuga de pollo", "1 taza de Arroz blanco", "80 g de Lechuga", "1 cucharada de Aceite de oliva",
        "1 unidad de Zanahoria"]}
    malos = [(d, v) for d in _DECLARACIONES_AJENAS if (v := go.clinical_backstop_for_meal(limpio, allergies=[d]))]
    assert not malos, f"falsos positivos: {malos}"


def test_la_expansion_anade_el_canonico_sin_perder_lo_escrito(go):
    out = go._expand_allergy_declarations(["Strawberry"])
    assert "strawberry" in out and any("fresa" in x for x in out), out


# ─────────────────────────────────────────────────────────────────────────────
# 2. El léxico de alimentos
# ─────────────────────────────────────────────────────────────────────────────

def test_el_lexico_cubre_el_catalogo_entero_en_los_cuatro_idiomas():
    import food_names_i18n as fn
    nombres = fn.nombres()
    snap = json.loads(_src("scripts/data/catalogo_nutricion_2026_09_12.json"))
    filas = snap.get("filas") or snap.get("rows") or snap.get("items") or snap
    canonicos = {str(f.get("name")) for f in filas if isinstance(f, dict) and f.get("name")}
    assert len(canonicos) >= 300, "el snapshot del catálogo no se leyó"
    faltan = sorted(c for c in canonicos if c not in nombres)
    assert not faltan, f"alimentos del catálogo sin fila en el léxico: {faltan[:10]}"
    huecos = [(c, loc) for c, fila in nombres.items() for loc in fn.LOCALES if not str(fila.get(loc) or "").strip()]
    assert not huecos, f"traducciones vacías: {huecos[:10]}"


@pytest.mark.parametrize("texto,canonico", [
    ("strawberry", "Fresas"), ("allergie aux fraises", "Fresas"), ("fragole", "Fresas"), ("morangos", "Fresas"),
    ("tomatoes", "Tomate"), ("pomodoro", "Tomate"), ("garlic", "Ajo"), ("aglio", "Ajo"),
])
def test_canonicos_para_texto_entiende_los_cinco_idiomas(texto, canonico):
    import food_names_i18n as fn
    assert canonico in fn.canonicos_para_texto(texto), (texto, fn.canonicos_para_texto(texto))


def test_leche_no_arrastra_la_lechuga():
    import food_names_i18n as fn
    assert not any("Lechuga" in c for c in fn.canonicos_para_texto("lait")), fn.canonicos_para_texto("lait")


def test_nombre_para_devuelve_none_en_el_idioma_base_y_el_nombre_en_los_demas():
    import food_names_i18n as fn
    assert fn.nombre_para("Fresas", "es-DO") is None
    assert fn.nombre_para("Fresas", "fr-FR")


def test_el_catalogo_trae_los_nombres_tambien_al_invitado():
    src = _src("routers/user_data.py")
    assert '_it["names"]' in src
    m = re.search(r"_CATALOG_CAMPOS_INVITADO\s*=\s*\(([^)]*)\)", src)
    assert m and '"names"' in m.group(1), "la proyección del invitado debe incluir `names`"


def test_el_filtro_rapido_del_catalogo_tambien_entiende_otros_idiomas():
    src = _src("constants.py")
    i = src.index("def _get_fast_filtered_catalogs")
    assert "_canonicos_i18n" in src[i:i + 12000], "el filtro de catálogo debe expandir alergias/gustos al canónico"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Escáner y estimación en el idioma del usuario
# ─────────────────────────────────────────────────────────────────────────────

def test_el_escaner_recibe_el_idioma_y_devuelve_el_nombre_para_leer_y_el_espanol():
    src = _src("routers/diary.py")
    assert "locale: Optional[str] = Form(None)" in src
    assert '"meal_name_es": meal_name_es' in src, "el español original viaja al lado del nombre para leer"
    assert "_nombres_para_mostrar" in src and "display_name" in src


def test_la_estimacion_escribe_nombre_y_porcion_en_el_idioma_del_usuario():
    src = _src("routers/diary.py")
    assert re.search(r"class EstimateMacrosRequest\(BaseModel\):[\s\S]{0,1500}locale: Optional\[str\]", src)
    assert "Escribe 'name' y 'portion_note' en {_idioma}." in src


def test_traducir_para_mostrar_no_llama_al_modelo_en_espanol_ni_sin_idioma(monkeypatch):
    import traduccion_para_mostrar as tpm

    def _boom(*a, **k):
        raise AssertionError("no debe llamar al modelo")
    monkeypatch.setattr(tpm, "_modelo", _boom)
    assert tpm.traducir_para_mostrar_sync(["Mangú"], "es-DO") is None
    assert tpm.traducir_para_mostrar_sync(["Mangú"], None) is None
    assert tpm.nombre_de_plato_para_mostrar("Mangú", "es-DO") is None
    assert tpm.nombre_de_plato_para_mostrar("", "en-US") is None


class _RespuestaT:
    def __init__(self, t):
        self.t = t


class _ModeloFalso:
    def __init__(self, fn):
        self._fn = fn

    def invoke(self, mensajes):
        textos = json.loads(mensajes[-1].content)
        return _RespuestaT(self._fn(textos))


def test_traducir_para_mostrar_mantiene_el_orden_y_devuelve_el_original_si_la_salida_no_cuadra(monkeypatch):
    import traduccion_para_mostrar as tpm
    monkeypatch.setattr(tpm, "_contexto", lambda node, uid: (lambda: None))
    monkeypatch.setattr(tpm, "_modelo", lambda timeout: _ModeloFalso(lambda ts: [f"EN {t}" for t in ts[:-1]] + [""]))
    out = tpm.traducir_para_mostrar_sync(["Pollo guisado", "Arroz blanco"], "en-US")
    assert out == ["EN Pollo guisado", "Arroz blanco"], "vacío ⇒ se queda el español de ESA posición"
    monkeypatch.setattr(tpm, "_modelo", lambda timeout: _ModeloFalso(lambda ts: ["solo uno"]))
    assert tpm.traducir_para_mostrar_sync(["a", "b"], "en-US") is None, "otra cantidad de textos ⇒ se descarta"


def test_el_tipo_textos_usa_su_propio_prompt_y_permite_frases_largas(monkeypatch):
    import traduccion_para_mostrar as tpm
    vistos = {}

    class _Grabador:
        def invoke(self, mensajes):
            vistos["sistema"] = mensajes[0].content
            vistos["textos"] = json.loads(mensajes[-1].content)
            return _RespuestaT(vistos["textos"])
    monkeypatch.setattr(tpm, "_contexto", lambda node, uid: (lambda: None))
    monkeypatch.setattr(tpm, "_modelo", lambda timeout: _Grabador())
    largo = " ".join(["Tomar con el desayuno para mejorar la absorción."] * 5)
    tpm.traducir_para_mostrar_sync([largo], "en-US", tipo="textos", max_chars=400)
    assert "frases cortas" in vistos["sistema"] and "nombres de platos" not in vistos["sistema"]
    assert len(vistos["textos"][0]) > 160, "una frase de suplemento no se recorta a 160 como un nombre"
    tpm.traducir_para_mostrar_sync(["Pollo guisado"], "en-US")
    assert "nombres de platos" in vistos["sistema"], "sin `tipo`, el prompt de siempre"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Avisos del servidor: en datos, y con espejo completo en el frontend
# ─────────────────────────────────────────────────────────────────────────────

def test_regenerar_dia_manda_el_aviso_tambien_en_datos():
    src = _src("routers/plans.py")
    i = src.index("def api_regenerate_day(")
    cuerpo = src[i:i + 140000]
    assert '"day_quality_warning_detail": _day_warning_detail if _day_warning else None' in cuerpo
    assert '{"kind": "deficit", "deficits": _deficit_detail,' in cuerpo
    assert '{"kind": "band", "precision_pct": round(_bmo * 100),' in cuerpo
    for eje in ('"axis": "protein"', '"axis": "kcal"', '"axis": "carbs"'):
        assert eje in cuerpo


_AVISO_PLATO = ("Este plato quedó algo alejado de tu objetivo de proteína para esta comida. "
                "Puedes volver a cambiarlo si prefieres más precisión.")


def test_el_aviso_del_plato_es_byte_identico_en_las_dos_puntas():
    assert _AVISO_PLATO in _src("routers/plans.py").replace('"\n                "', "")
    front = _front("src/utils/avisosDeCalidad.js")
    assert f"i18nKey('{_AVISO_PLATO}')" in front


def test_las_sugerencias_de_ahorro_traen_sus_piezas(monkeypatch):
    import shopping_calculator as sc
    monkeypatch.setattr(sc, "cheapest_supermarket_variant",
                        lambda name: {"brand": "Wala", "presentation": "Funda 650 gr", "price_rd": 47.4})
    sugs = sc.build_budget_suggestions([{"name": "Avena", "estimated_cost_rd": 198}])
    assert sugs and sugs[0]["brand"] == "Wala" and sugs[0]["presentation"] == "Funda 650 gr"
    assert sugs[0]["price_rd"] == 47 and "Wala Funda 650 gr" in sugs[0]["text"], "el texto de siempre, intacto"


def _textos_fijos_del_backend() -> list:
    """Las frases FIJAS de «Acción requerida» (cron_tasks: action_title/body/cta) y de `/blocked-reasons`."""
    out = []

    def _consts(path, nombres):
        for nodo in ast.walk(ast.parse(_src(path))):
            if isinstance(nodo, ast.Assign) and isinstance(nodo.value, ast.Constant) and isinstance(nodo.value.value, str):
                if any(isinstance(t, ast.Name) and t.id in nombres for t in nodo.targets):
                    out.append(nodo.value.value)

    def _dicts(path, variables, claves=("title", "body", "cta", "label", "hint")):
        for nodo in ast.walk(ast.parse(_src(path))):
            if isinstance(nodo, ast.Assign) and any(isinstance(t, ast.Name) and t.id in variables for t in nodo.targets):
                for sub in ast.walk(nodo.value):
                    if isinstance(sub, ast.Dict):
                        for k, v in zip(sub.keys, sub.values):
                            if (isinstance(k, ast.Constant) and k.value in claves and isinstance(v, ast.Constant)
                                    and isinstance(v.value, str)):
                                out.append(v.value)
    _consts("cron_tasks.py", {"action_title", "action_body", "action_cta"})
    _dicts("routers/plans.py", {"reason_to_text", "_UNKNOWN_REASON_TEMPLATE"})
    return sorted(set(out))


def test_se_extraen_las_frases_fijas_del_backend():
    fijas = _textos_fijos_del_backend()
    assert len(fijas) >= 45, len(fijas)
    assert "Tu plan necesita atención" in fijas and "Bloqueo sin clasificar" in fijas


def test_cada_frase_fija_del_backend_tiene_su_traduccion_en_el_frontend():
    front = _front("src/utils/textosDelServidor.js")
    # El catálogo no hornea la marca (P3-I18N-MARCA-HORNEADA del frontend): «Abre Bioboros» es «Abre {app}» allí.
    faltan = [s for s in _textos_fijos_del_backend()
              if f"i18nKey({_literal_js(s.replace('Bioboros', '{app}'))})" not in front]
    assert not faltan, f"frases del servidor que el frontend pintaría en español: {faltan}"
    # Los dos nombres fijos que el servidor escribe en datos.
    assert '"name": "Plan en preparación"' in _src("generation_lifecycle.py")
    assert 'i18nKey("Plan en preparación")' in front
    import food_search
    assert food_search.derive_meal_name([]) == "Comida registrada"
    assert 'i18nKey("Comida registrada")' in front


def _micros_textos() -> dict:
    import micronutrients as mn
    import micros_seguros as ms
    textos = {"labels": list(mn._LABELS.values()), "notas": list(mn._SUPPLEMENT_NOTE.values())}
    textos["especiales"] = [mn._POTASSIUM_RESTRICTED_NOTE, mn._FIBER_RENAL_NOTE, mn._CEILING_ESTIMADO_NOTE,
                            ms.SIN_FUENTE_SEGURA, mn._FLOOR_ESTIMADO_SUFIJO]
    sup, alimentos = [], []
    for plantilla in mn._SUPPLEMENT_TEMPLATES.values():
        for k, v in plantilla.items():
            if k == "alimentos":
                alimentos.extend(ms._partes(v))
            elif isinstance(v, str):
                sup.append(v)
    textos["suplementos"], textos["alimentos"] = sup, alimentos
    compuestas = []
    for prefijo, opciones, sufijo in ms._NOTAS.values():
        compuestas.extend([prefijo, sufijo, *opciones])
    for extras in ms._EXTRAS.values():
        compuestas.extend(extras)
    textos["compuestas"] = compuestas
    return textos


def test_el_panel_de_micronutrientes_tiene_espejo_completo_en_el_frontend():
    front = _front("src/utils/microsCopy.js")
    faltan = {grupo: [s for s in lista if _literal_js(s) not in front]
              for grupo, lista in _micros_textos().items()}
    faltan = {g: l for g, l in faltan.items() if l}
    assert not faltan, f"textos del panel sin espejo (se pintarían en español): {faltan}"


def test_la_coletilla_del_piso_estimado_es_la_constante():
    src = _src("micronutrients.py")
    assert 'entry["nota"] = str(entry["nota"]).rstrip() + " " + _FLOOR_ESTIMADO_SUFIJO' in src


# ─────────────────────────────────────────────────────────────────────────────
# 5. Cambiar plato / regenerar día / arreglar sodio
# ─────────────────────────────────────────────────────────────────────────────

def test_el_swap_devuelve_el_nombre_traducido_despues_de_persistir():
    src = _src("routers/plans.py")
    i = src.index("def api_swap_meal(")
    cuerpo = src[i:i + 30000]
    i_persist = cuerpo.index('result["persisted"] = _persist_swap_server_side(')
    i_nombre = cuerpo.index('result["display_name"] = _dn')
    assert i_nombre > i_persist, "el nombre para leer se añade DESPUÉS del persist: no se guarda"
    assert 'data.get("locale")' in cuerpo[i_persist:i_nombre]


def test_arreglar_sodio_devuelve_el_plato_nuevo_para_leer():
    src = _src("routers/plans.py")
    assert '"new_meal_display": _new_meal_display,' in src


def test_regenerar_dia_devuelve_los_nombres_para_leer_alineados(monkeypatch):
    from routers import plans as rp
    import traduccion_para_mostrar as tpm
    llamadas = []

    def _fake(textos, locale, **kw):
        llamadas.append((list(textos), locale, kw.get("node")))
        return [f"EN {t}" for t in textos]
    monkeypatch.setattr(tpm, "traducir_para_mostrar_sync", _fake)
    meals = [{"name": "Mangú con huevo"}, {"name": "Pollo guisado"}]
    assert rp._nombres_visibles_del_dia(meals, "en-US", "u1") == ["EN Mangú con huevo", "EN Pollo guisado"]
    assert llamadas[-1][2] == "display_i18n_regen_day"
    assert rp._nombres_visibles_del_dia(meals, "es-DO", "u1") is None
    assert rp._nombres_visibles_del_dia(meals, None, "u1") is None
    assert '"meals_display_names": _nombres_visibles_del_dia(new_meals, _locale_nombres_rd, verified_user_id),' in _src(
        "routers/plans.py")


def test_lo_provisional_nunca_se_guarda():
    from routers import plans as rp
    plan = {"days": [{"meals": [
        {"name": "A", "_display": {"en-US": {"name": "A!", "_provisional": True}}},
        {"name": "B", "_display": {"en-US": {"name": "B!", "_provisional": True}, "fr-FR": {"name": "B?"}}},
        {"name": "C"},
    ]}]}
    rp._quitar_display_provisional(plan)
    comidas = plan["days"][0]["meals"]
    assert "_display" not in comidas[0]
    assert comidas[1]["_display"] == {"fr-FR": {"name": "B?"}}
    src = _src("routers/plans.py")
    for fn in ("def api_restore_plan_local(", "def api_adopt_guest_plan("):
        i = src.index(fn)
        assert "_quitar_display_provisional(" in src[i:i + 9000], f"{fn} debe quitar lo provisional"


# ─────────────────────────────────────────────────────────────────────────────
# 6. El plan del invitado
# ─────────────────────────────────────────────────────────────────────────────

class _RespuestaLLM:
    def __init__(self, content):
        self.content = content
        self.usage_metadata = {}


class _LLMDisplay:
    def invoke(self, mensajes):
        return _RespuestaLLM(json.dumps({"meals": [{
            "i": 0, "name": "Chicken with rice", "description": "Grilled breast with white rice.",
            "recipe": ["Cook the chicken for 10 minutes."],
            "ingredients": ["150 g Chicken breast (Pechuga de pollo)"],
        }], "plan_name": "Strong flavor", "insights": ["Eat more fiber."]}))


def test_traducir_plan_en_memoria_devuelve_las_traducciones_sin_tocar_la_bd(monkeypatch):
    import plan_display_i18n as pdi
    monkeypatch.setattr(pdi, "_circuit_breaker_can_proceed", lambda model: True)
    monkeypatch.setattr(pdi, "build_chat_llm", lambda *a, **k: _LLMDisplay())
    monkeypatch.setattr(pdi, "_emit_usage_telemetry", lambda *a, **k: None)

    def _no_bd(*a, **k):
        raise AssertionError("el plan del invitado no se lee ni se escribe en la BD")
    monkeypatch.setattr(pdi, "_fetch_plan_data", _no_bd)
    monkeypatch.setattr(pdi, "_persist_batch", _no_bd)
    plan = {"name": "Sazón fuerte", "insights": ["Come más fibra."], "days": [{"meals": [{
        "name": "Pollo con arroz", "desc": "Pechuga a la plancha con arroz blanco.",
        "recipe": ["Cocina el pollo 10 minutos."], "ingredients": ["150 g de Pechuga de pollo"],
    }]}]}
    res = pdi.traducir_plan_en_memoria(plan, "en-US")
    assert res["skipped"] is None, res
    assert res["meals"][0]["day"] == 0 and res["meals"][0]["meal"] == 0
    assert res["meals"][0]["name"] == "Pollo con arroz", "el cliente fusiona sólo si el plato sigue siendo ése"
    assert res["meals"][0]["display"]["name"] == "Chicken with rice"
    assert res["plan_name"] == "Strong flavor" and res["insights"] == ["Eat more fiber."]


def test_una_entrada_provisional_no_cuenta_como_traducida():
    """Lo provisional (sólo el nombre, puesto por el cliente) se vuelve a traducir, también en un plato sin receta ni
    ingredientes: sin la guarda, ese plato se quedaba con el nombre traducido y la descripción en español."""
    import plan_display_i18n as pdi
    sin_arrays = {"name": "Té", "_display": {"en-US": {"name": "Tea", "_provisional": True}}}
    assert pdi._display_ya_usable(sin_arrays, "en-US") is False
    assert pdi._display_ya_usable({"name": "Té", "_display": {"en-US": {"name": "Tea"}}}, "en-US") is True
    objetivos = pdi._collect_targets([{"meals": [sin_arrays]}], [0], locale="en-US")
    assert [t["name"] for t in objetivos] == ["Té"]


def test_traducir_plan_en_memoria_no_hace_nada_en_espanol_ni_con_basura(monkeypatch):
    import plan_display_i18n as pdi
    monkeypatch.setattr(pdi, "build_chat_llm", lambda *a, **k: (_ for _ in ()).throw(AssertionError("sin LLM")))
    assert pdi.traducir_plan_en_memoria({"days": [{"meals": [{"name": "X"}]}]}, "es-DO")["skipped"] == "locale"
    assert pdi.traducir_plan_en_memoria({"days": []}, "en-US")["skipped"] == "no_days"
    assert pdi.traducir_plan_en_memoria("no es un plan", "en-US")["skipped"] == "no_days"


def test_el_endpoint_del_invitado_tiene_cupo_por_ip_y_topes():
    src = _src("routers/plans.py")
    assert "_GUEST_DISPLAY_LIMITER = RateLimiter(max_calls=6, period_seconds=600)" in src
    i = src.index('@router.post("/guest-display")')
    cuerpo = src[i:i + 2500]
    assert "Depends(_GUEST_DISPLAY_LIMITER)" in cuerpo
    assert "_GUEST_DISPLAY_MAX_BYTES" in cuerpo and "status_code=413" in cuerpo
    assert "traducir_plan_en_memoria(" in cuerpo
    assert "verify_api_quota" not in cuerpo, "no cuesta créditos: el invitado no tiene"


def test_el_par_del_limitador_del_invitado_es_unico():
    """La ventana de Redis es `rl:<max>:<periodo>:<uid>`: dos limitadores con el mismo par compartirían cupo."""
    fuentes = [p for p in (*_BACKEND.glob("routers/*.py"), *_BACKEND.glob("*.py"))]
    pares = re.findall(r"RateLimiter\(max_calls=(\d+), period_seconds=(\d+)\)",
                       "".join(p.read_text(encoding="utf-8") for p in fuentes))
    assert pares.count(("6", "600")) == 1
    assert pares.count(("8", "120")) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 7. Texto libre fuera del plan: /api/i18n/textos
# ─────────────────────────────────────────────────────────────────────────────

def test_el_endpoint_de_textos_valida_y_no_traduce_en_espanol(monkeypatch):
    from fastapi import HTTPException
    from routers import user_data as ud
    import traduccion_para_mostrar as tpm
    for malo in ({}, {"textos": []}, {"textos": ["x"] * 41}, {"textos": [1]}, {"textos": ["y" * 401]}):
        with pytest.raises(HTTPException):
            ud.api_traducir_textos(malo, verified_user_id="u1")

    def _boom(*a, **k):
        raise AssertionError("en español no se llama al modelo")
    monkeypatch.setattr(tpm, "traducir_para_mostrar_sync", _boom)
    assert ud.api_traducir_textos({"textos": ["Creatina"], "locale": "es-DO"}, verified_user_id="u1") == {"textos": None}
    assert ud.api_traducir_textos({"textos": ["Creatina"], "locale": "xx-XX"}, verified_user_id="u1") == {"textos": None}
    vistos = {}

    def _fake(textos, locale, **kw):
        vistos.update(kw)
        return ["Creatine"]
    monkeypatch.setattr(tpm, "traducir_para_mostrar_sync", _fake)
    assert ud.api_traducir_textos({"textos": ["Creatina"], "locale": "en-US"}, verified_user_id="u1") == {
        "textos": ["Creatine"]}
    assert vistos["tipo"] == "textos" and vistos["node"] == "display_i18n_textos"


def test_el_endpoint_de_textos_no_cuesta_creditos():
    src = _src("routers/user_data.py")
    i = src.index('@router.post("/i18n/textos")')
    cuerpo = src[i:i + 1800]
    assert "Depends(_TEXTOS_TRADUCIDOS_LIMITER)" in cuerpo and "verify_api_quota" not in cuerpo


# ─────────────────────────────────────────────────────────────────────────────
# 8. El marcador
# ─────────────────────────────────────────────────────────────────────────────

def test_marker_bumpeado():
    # `>=` como sus hermanos: un lote posterior sube el marcador sin romper este test.
    m = re.search(r'^_LAST_KNOWN_PFIX\s*=\s*"P1-PLAN-LOTE-(\d+) · \d{4}-\d{2}-\d{2}"', _src("app.py"), re.M)
    assert m and int(m.group(1)) >= 224
