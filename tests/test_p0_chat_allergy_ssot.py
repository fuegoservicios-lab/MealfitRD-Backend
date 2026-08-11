"""[P0-CHAT-ALLERGY-SSOT · 2026-08-11] El chat recomendaba alimentos a los que el
usuario es alérgico, y le decía al modelo que la lista estaba filtrada.

EL DEFECTO. `suggest_foods_for_nutrient` filtraba así (tools.py, pre-fix):

    allergies = [strip_accents(str(a).lower()) for a in _as_list(hp.get("allergies"))]
    exclude_tokens = [t for t in (allergies + dislikes) if t]
    ...
    if any(tok in name_norm for tok in exclude_tokens): continue

Compara **la etiqueta del chip** contra el **nombre del alimento**, como subcadena. El
formulario guarda `"Lacteos"` (QAllergies.jsx:48, `val` es lo que se persiste; `label`
«Lácteos» es solo lo que se pinta). Ningún alimento del catálogo se llama «lácteo», así
que el filtro no bloquea NADA.

MEDIDO contra las 206 filas reales de `master_ingredients`:

    chip            bloquea   debería
    Lácteos               0        17
    Gluten                0        15
    Mariscos              0         4
    Frutos Secos          0         6
    Soya                  3         5
    Huevo                 3         3   <-- acierta
    Maní                  2         2   <-- acierta

CORRECCIÓN DE MI PROPIA CIFRA: primero conté 24 lácteos. Son 17. Las otras 7 filas que
contienen «leche», «mantequilla» o «yogur» son leche de almendras/avena/coco/soya,
mantequilla de almendras/maní y yogur de coco — NO son lácteos, y el matcher SSOT las
deja pasar a propósito (la «excusa plant-adjacent» que documenta `_scan_allergen_
violations`). O sea que el SSOT es más fino que mi conteo: bloquea 17 de 17 y no se
lleva por delante ninguna de las 7 vegetales. Post-fix, verificado contra las 206 filas
reales.

El patrón lo explica todo: **el filtro solo acierta cuando la etiqueta del chip resulta
ser parte del nombre del alimento.** Por eso «Huevo» funciona y «Lácteos» no. Y por eso
el test que ya existía (`test_p3_micro_food_suggest.py`) pasaba: su fixture usa
`allergies=["leche"]` — la palabra que sí aparece en el nombre. *La muestra con la que
uno prueba es justo la que esconde el fallo.*

LO QUE LO HACÍA PEOR. La tool cerraba su respuesta diciéndole al modelo que los
alimentos venían «ya filtrados por las restricciones del usuario». Una afirmación falsa
que el modelo no tiene forma de contradecir es peor que no filtrar: le quita el motivo
para dudar.

EL ARREGLO no inventa un matcher nuevo: reusa el que el pipeline de planes lleva usando
desde siempre (`_allergen_pool_item_banned` y `_diet_pool_item_banned`, respaldados por
`_ALLERGEN_SYNONYMS` y por los `_DIET_*_TERMS`), y borra `_VEGETARIAN_EXCLUDE` /
`_VEGAN_EXCLUDE` de tools.py, que eran la 4ª tabla de dieta a mano que
`P1-DIET-CANON-SSOT` prohíbe por escrito.

Estos casos afirman el RESULTADO —qué alimentos salen y cuáles no— y no el mecanismo,
para que cambiar el matcher por uno mejor no los rompa.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools.py"
_PROMPT = Path(__file__).resolve().parents[1] / "prompts" / "chat_agent.py"


# ─────────────────────────── Anclas de código (sin importar) ───────────────────────────

def test_no_queda_la_afirmacion_falsa():
    """La frase «ya filtrados por las restricciones» no puede volver mientras la tool
    no sea capaz de sostenerla — y aunque lo sea, decírselo al modelo le quita el
    escepticismo que es su última defensa."""
    src = _TOOLS.read_text(encoding="utf-8")
    assert "ya filtrados por las" not in src, (
        "volvió la afirmación de que la lista está filtrada en tools.py"
    )
    prompt = _PROMPT.read_text(encoding="utf-8")
    assert "ya filtrados por" not in prompt and "ya filtrada por" not in prompt, (
        "el system prompt del chat vuelve a afirmar que la lista viene filtrada"
    )


def test_no_hay_cuarta_tabla_de_dieta():
    """[P1-DIET-CANON-SSOT] «No escribas una 4ª». `_VEGETARIAN_EXCLUDE`/`_VEGAN_EXCLUDE`
    eran exactamente eso, y su lista se quedó corta: no tenía 'gelatina', ni 'manteca',
    ni los plurales que el SSOT sí cubre."""
    # Se busca la DEFINICIÓN (`NOMBRE = [`), no la mención: el comentario que explica
    # por qué se borraron las nombra, y un guard que se encuentra a sí mismo en su
    # propia explicación no puede pasar nunca. Sexta vez esta semana con esta forma.
    src = _TOOLS.read_text(encoding="utf-8")
    for nombre in ("_VEGETARIAN_EXCLUDE", "_VEGAN_EXCLUDE"):
        assert not re.search(rf"^{nombre}\s*=", src, re.M), (
            f"reapareció la definición de {nombre} en tools.py: es una 4ª tabla de dieta "
            "a mano. Usa `_diet_pool_item_banned` / `constants.canonicalize_diet_type`."
        )


def test_la_tool_usa_el_ssot_de_alergenos():
    """Ancla el REUSO. Si alguien vuelve a escribir un matcher propio aquí, este caso
    cae antes de que llegue a producción."""
    src = _TOOLS.read_text(encoding="utf-8")
    assert "_allergen_pool_item_banned" in src, (
        "la tool dejó de usar el matcher SSOT de alérgenos del pipeline de planes"
    )


# ─────────────────────────── Funcional ───────────────────────────

def _load():
    try:
        import tools as tools_mod  # noqa
        import shopping_calculator  # noqa
        return tools_mod, shopping_calculator
    except Exception as e:  # pragma: no cover
        pytest.skip(f"tools/shopping_calculator no importable: {e}")


# El catálogo real, en miniatura: nombres tal cual están en `master_ingredients`.
CATALOGO = [
    {"name": "Queso cheddar", "calcium_mg_per_100g": 720.0},
    {"name": "Leche", "calcium_mg_per_100g": 125.0},
    {"name": "Yogurt griego entero", "calcium_mg_per_100g": 110.0},
    {"name": "Almendras fileteadas", "calcium_mg_per_100g": 264.0},
    {"name": "Sardinas en lata", "calcium_mg_per_100g": 382.0},
    {"name": "Brocoli", "calcium_mg_per_100g": 47.0},
    {"name": "Espinaca", "calcium_mg_per_100g": 99.0},
]


def _sugerir(monkeypatch, perfil, nutriente="calcio", top_n=7):
    tools_mod, shopping_calculator = _load()
    monkeypatch.setattr(shopping_calculator, "get_master_ingredients", lambda: CATALOGO)
    monkeypatch.setattr(tools_mod, "get_user_profile", lambda uid: {"health_profile": perfil})
    return tools_mod.suggest_foods_for_nutrient.func(user_id="u1", nutrient=nutriente, top_n=top_n)


def test_lacteos_el_chip_real_bloquea_los_lacteos(monkeypatch):
    """EL CASO. Con el valor que el formulario guarda de verdad —«Lacteos», no
    «leche»— antes salían los tres lácteos del catálogo, y encabezando la lista porque
    son los más ricos en calcio."""
    out = _sugerir(monkeypatch, {"allergies": ["Lacteos"], "dietType": "balanced"})
    for prohibido in ("Queso cheddar", "Leche", "Yogurt griego entero"):
        assert prohibido not in out, (
            f"la tool sigue recomendando «{prohibido}» a un alérgico a lácteos"
        )
    assert "Brocoli" in out, "se llevó por delante también los alimentos seguros"


def test_las_vegetales_NO_se_bloquean_por_error(monkeypatch):
    """El otro lado del filtro, y el que se olvida: «leche de coco» no es un lácteo.
    Un matcher por subcadena que buscara «leche» se las llevaría todas y dejaría a un
    alérgico sin las alternativas que justamente puede comer. El SSOT las respeta
    (excusa plant-adjacent) — verificado contra las 7 filas reales del catálogo."""
    catalogo = [
        {"name": "Leche de coco", "calcium_mg_per_100g": 16.0},
        {"name": "Leche de almendras", "calcium_mg_per_100g": 184.0},
        {"name": "Yogur de coco", "calcium_mg_per_100g": 12.0},
        {"name": "Queso cheddar", "calcium_mg_per_100g": 720.0},
    ]
    tools_mod, shopping_calculator = _load()
    monkeypatch.setattr(shopping_calculator, "get_master_ingredients", lambda: catalogo)
    monkeypatch.setattr(tools_mod, "get_user_profile",
                        lambda uid: {"health_profile": {"allergies": ["Lacteos"]}})
    out = tools_mod.suggest_foods_for_nutrient.func(user_id="u1", nutrient="calcio", top_n=5)
    assert "Queso cheddar" not in out, "el lácteo de verdad tiene que caer"
    for vegetal in ("Leche de coco", "Leche de almendras", "Yogur de coco"):
        assert vegetal in out, (
            f"«{vegetal}» se bloqueó como si fuera lácteo: al alérgico se le quitan "
            "justo las alternativas que sí puede comer"
        )


def test_frutos_secos_y_mariscos_con_sus_chips_reales(monkeypatch):
    out = _sugerir(monkeypatch, {"allergies": ["Frutos Secos"], "dietType": "balanced"})
    assert "Almendras fileteadas" not in out, "«Frutos Secos» no bloqueó las almendras"

    out2 = _sugerir(monkeypatch, {"allergies": ["Mariscos", "Pescado"], "dietType": "balanced"})
    assert "Sardinas en lata" not in out2, "«Pescado» no bloqueó las sardinas"


def test_huevo_sigue_funcionando(monkeypatch):
    """Control. «Huevo» ya acertaba antes del arreglo porque la etiqueta coincide con el
    nombre; si este caso se rompe, el matcher nuevo perdió lo que el viejo sí hacía."""
    catalogo = [
        {"name": "Huevo entero", "protein_g_per_100g": 13.0},
        {"name": "Lentejas", "protein_g_per_100g": 9.0},
    ]
    tools_mod, shopping_calculator = _load()
    monkeypatch.setattr(shopping_calculator, "get_master_ingredients", lambda: catalogo)
    monkeypatch.setattr(tools_mod, "get_user_profile",
                        lambda uid: {"health_profile": {"allergies": ["Huevo"]}})
    out = tools_mod.suggest_foods_for_nutrient.func(user_id="u1", nutrient="proteina", top_n=5)
    assert "Huevo entero" not in out
    assert "Lentejas" in out


def test_sin_alergias_no_bloquea_nada(monkeypatch):
    """Un filtro que se pasa de celoso es otro defecto: dejaría al coach sin nada que
    recomendar y el usuario no sabría por qué."""
    out = _sugerir(monkeypatch, {"allergies": [], "dietType": "balanced"})
    for esperado in ("Queso cheddar", "Sardinas en lata", "Brocoli"):
        assert esperado in out, f"«{esperado}» desapareció sin que nadie lo prohibiera"


def test_dieta_vegana_por_el_ssot(monkeypatch):
    """La dieta se canoniza con `constants.canonicalize_diet_type`, así que un perfil
    legacy en español («vegana») tiene que funcionar igual que «vegan». Antes la
    comparación era `diet_type == "vegan"` y con el valor en español el filtro de dieta
    desaparecía entero (routers/plans.py:510 documenta que esos perfiles existen)."""
    for valor in ("vegan", "vegana"):
        out = _sugerir(monkeypatch, {"allergies": [], "dietType": valor})
        assert "Queso cheddar" not in out, f"dietType={valor!r} no excluyó el queso"
        assert "Sardinas en lata" not in out, f"dietType={valor!r} no excluyó el pescado"
        assert "Brocoli" in out, f"dietType={valor!r} se llevó también los vegetales"


# ─────────────────────────── update_form_field no borra alergias ───────────────────────────

def test_actualizar_una_alergia_no_borra_las_otras():
    """[P0-CHAT-ALLERGY-MERGE] `update_form_field` hacía `_hp[field] = nuevo_valor` —
    REEMPLAZO— y el system prompt ordena al modelo llamarla «OBLIGATORIO y SIN
    EXCEPCIÓN» cuando el usuario mencione un dato nuevo, con «soy intolerante a la
    lactosa» de ejemplo literal.

    Secuencia real: perfil con ["Mariscos","Frutos Secos"] → el usuario menciona la
    lactosa → el modelo obedece → queda ["Lacteos"]. Los otros dos desaparecen, y acto
    seguido se borran los `user_facts` de categoría alergia, que eran el respaldo.
    `health_profile` no tiene historial: no hay de dónde recuperarlos.

    Se ancla en el fuente porque la función escribe en DB y montar eso aquí sería
    probar el mock. Lo que se afirma es que el mutator FUNDE."""
    src = _TOOLS.read_text(encoding="utf-8")
    i = src.index("def _field_mutator")
    cuerpo = src[i:i + 900]
    assert "_hp[field] = _new_field_value" not in cuerpo, (
        "el mutator volvió a REEMPLAZAR la lista: mencionar una alergia borra las demás"
    )
    assert "P0-CHAT-ALLERGY-MERGE" in cuerpo, (
        "el mutator perdió su marcador: sin él nadie sabe por qué funde en vez de asignar"
    )


def test_la_fusion_funciona_de_verdad(monkeypatch):
    """El caso de arriba lee el fuente; este EJECUTA el mutator.

    Se captura la función que `update_form_field` le pasa a
    `update_user_health_profile_atomic` y se corre contra un perfil de mentira. Así se
    afirma el RESULTADO —qué queda en el perfil— y no la forma de escribirlo.

    La secuencia es la real: el usuario ya declaró mariscos y frutos secos en el
    formulario, y en el chat menciona la lactosa."""
    tools_mod, _ = _load()
    capturado = {}

    def _fake_atomic(user_id, mutator):
        perfil = {"allergies": ["Mariscos", "Frutos Secos"], "medicalConditions": ["Hipertension"]}
        mutator(perfil)
        capturado["perfil"] = perfil
        return perfil

    monkeypatch.setattr(tools_mod, "update_user_health_profile_atomic", _fake_atomic)
    monkeypatch.setattr(tools_mod, "delete_user_facts_by_metadata", lambda *a, **k: 0, raising=False)

    tools_mod.update_form_field.func(user_id="u1", field="allergies", new_value="Lacteos")

    quedaron = capturado["perfil"]["allergies"]
    for previo in ("Mariscos", "Frutos Secos"):
        assert previo in quedaron, (
            f"«{previo}» desapareció al mencionar otra alergia en el chat. "
            "El perfil no tiene historial: no hay de dónde recuperarlo."
        )
    assert "Lacteos" in quedaron, "no se añadió la alergia nueva"


def test_la_fusion_no_duplica_ni_por_acentos(monkeypatch):
    """«Lacteos» y «Lácteos» son la misma alergia. Sin normalizar, la lista crece cada
    vez que el modelo la escribe distinto y acaba siendo ilegible en Configuración."""
    tools_mod, _ = _load()
    capturado = {}

    def _fake_atomic(user_id, mutator):
        perfil = {"allergies": ["Lácteos"]}
        mutator(perfil)
        capturado["perfil"] = perfil
        return perfil

    monkeypatch.setattr(tools_mod, "update_user_health_profile_atomic", _fake_atomic)
    monkeypatch.setattr(tools_mod, "delete_user_facts_by_metadata", lambda *a, **k: 0, raising=False)

    tools_mod.update_form_field.func(user_id="u1", field="allergies", new_value="Lacteos")
    assert len(capturado["perfil"]["allergies"]) == 1, (
        f"se duplicó por acentos: {capturado['perfil']['allergies']}"
    )


def test_los_gustos_SI_se_reemplazan(monkeypatch):
    """La otra mitad de la decisión. Acumular preferencias para siempre convertiría un
    «no me gusta el tomate» de un martes en una restricción que nadie puede quitar."""
    tools_mod, _ = _load()
    capturado = {}

    def _fake_atomic(user_id, mutator):
        perfil = {"dislikes": ["Tomate", "Cebolla"]}
        mutator(perfil)
        capturado["perfil"] = perfil
        return perfil

    monkeypatch.setattr(tools_mod, "update_user_health_profile_atomic", _fake_atomic)
    monkeypatch.setattr(tools_mod, "delete_user_facts_by_metadata", lambda *a, **k: 0, raising=False)

    tools_mod.update_form_field.func(user_id="u1", field="dislikes", new_value="Cilantro")
    assert capturado["perfil"]["dislikes"] == ["Cilantro"], (
        "los gustos se acumularon: quitar uno se vuelve imposible desde el chat"
    )


def test_los_DOS_prompts_del_chat_dicen_lo_mismo():
    """PARIDAD, y este caso nace de un fallo mío.

    Arreglé la promesa falsa en `build_tools_instructions` y dejé intacta
    `build_tools_instructions_stream`, que decía «filtrados por sus alergias/dieta» —
    la misma mentira con otras palabras. El guard pasó, porque solo buscaba la frase
    literal del primero.

    Es exactamente el modo de fallo del que la cabecera de este archivo avisa por
    escrito para `_CHAT_BREVITY_RULES`: «una futura edición no puede arreglar 3 de los 4
    prompts y dejar el cuarto con el wording viejo». Lo hice igual.

    Se afirma la PROPIEDAD en ambos: ninguno promete la lista depurada de antemano, y
    los dos mandan LEER lo que la herramienta responda."""
    prompt = _PROMPT.read_text(encoding="utf-8")
    for fn in ("def build_tools_instructions(", "def build_tools_instructions_stream("):
        i = prompt.index(fn)
        # Hasta el siguiente `def ` de nivel 0, contando desde el cuerpo.
        j = prompt.find("\ndef ", i + 1)
        cuerpo = prompt[i:j if j > 0 else len(prompt)]
        assert "suggest_foods_for_nutrient" in cuerpo, f"{fn} dejó de documentar la tool"
        for promesa in ("filtrados por sus alergias", "ya filtrad", "filtrada por las"):
            assert promesa not in cuerpo, (
                f"{fn} vuelve a prometer la lista depurada de antemano («{promesa}»). "
                "Si arreglas uno, arregla los DOS: la cabecera de este archivo avisa "
                "justo de esto."
            )


# ─────────────────── El coach recibe el perfil clínico (P0-CHAT-CLINICAL-BLOCK) ───────────────────

def test_el_bloque_clinico_existe_y_dice_lo_que_debe():
    from prompts.chat_agent import build_clinical_guard_context as bloque
    out = bloque({
        "allergies": ["Lacteos", "Mariscos"],
        "medicalConditions": ["Hipertension"],
        "medications": ["Losartan"],
    })
    assert "Lacteos" in out and "Mariscos" in out, "las alergias no llegan al coach"
    assert "Hipertension" in out, "las condiciones no llegan al coach"
    assert "Losartan" in out, "los medicamentos no llegan al coach"
    assert "NUNCA" in out, "el bloque no es imperativo: es contexto, tiene que mandar"


def test_el_bloque_calla_cuando_no_hay_nada_declarado():
    """Un bloque que dice «ninguna alergia» gasta tokens en cada llamada y, peor, le da
    al modelo una certeza que no tiene: un perfil incompleto no es un perfil sin
    alergias."""
    from prompts.chat_agent import build_clinical_guard_context as bloque
    assert bloque({}) == ""
    assert bloque(None) == ""
    # Los centinelas del formulario no son datos clínicos.
    assert bloque({"allergies": ["Ninguna"], "medicalConditions": ["ninguno"]}) == ""


def test_el_bloque_se_inyecta_en_LOS_DOS_call_sites():
    """La paridad otra vez, y por el mismo motivo: `agent.py` monta el contexto en dos
    sitios (chat normal y stream). Cablear uno y olvidar el otro dejaría al coach ciego
    en la mitad de las conversaciones, y sería indetectable mirando el prompt de una."""
    src = (Path(__file__).resolve().parents[1] / "agent.py").read_text(encoding="utf-8")
    n_identidad = src.count("build_user_identity_context(form_data or {}")
    n_clinico = src.count("build_clinical_guard_context(form_data or {})")
    assert n_clinico == n_identidad, (
        f"el bloque clínico se inyecta en {n_clinico} sitios y la identidad en "
        f"{n_identidad}: hay un camino del chat que no recibe las alergias"
    )
    assert n_clinico >= 2, "esperaba al menos los dos call sites (chat y stream)"
