"""[P0-ALLERGEN-VOCAB-I18N · 2026-08-21] El backstop determinista de alergias entendía UNA
grafía por alérgeno — la dominicana/mexicana — y era CIEGO al resto.

Medido contra el código de producción antes del fix (plato con maní + ajonjolí + almejas + queso
+ huevo + atún + leche):

    declara 'cacahuete'  ->  0 violaciones     declara 'cacahuate' ->  1
    declara 'sesamo'     ->  0                 declara 'ajonjoli'  ->  1
    declara 'shellfish'  ->  0                 declara 'mariscos'  ->  1
    declara 'dairy'      ->  0                 declara 'lacteos'   ->  2
    declara 'egg'        ->  0                 declara 'huevo'     ->  1
    declara 'fish'       ->  0                 declara 'pescado'   ->  1

POR QUÉ ES P0 Y NO P1. `clinical_backstop_for_meal` es la ÚNICA defensa determinista de las tres
superficies de UPDATE (swap individual, chat-modify, camino degradado sin LLM) por diseño de
P0-UPDATE-CLINICAL-GUARD — ahí no corre el reviewer LLM ni
`_apply_deterministic_clinical_layer`. Y el wizard ofrece SEIS chips (Lácteos/Gluten/Huevo/
Mariscos/Frutos Secos/Soya): **no hay chip de maní**, así que el alérgeno más anafiláctico del
espectro es el único de los grandes que obliga a texto libre — y la palabra que un español teclea
ahí es «cacahuete» («cacahuate» es mexicanismo). Con el sistema de países VIVO en producción
(`MEALFIT_COUNTRY_SYSTEM=true`, selector de 6 países visible) y el dashboard en 5 idiomas desde
P1-I18N-DASHBOARD, las dos poblaciones existen hoy.

LAS DOS MITADES DEL VOCABULARIO. El fix separa lo que estaba mezclado:

  * `_ALLERGEN_SYNONYMS`  — cómo se LLAMA el alimento. Viaja al set `forbidden` y se escanea
    contra los ingredientes. Sólo entran aquí términos que pueden aparecer en una línea de
    receta en español ('cacahuete' sí; 'dairy' no).
  * `_ALLERGEN_DECLARATION_ALIASES` (nuevo) — cómo lo DECLARA el usuario. Sólo se consulta para
    decidir a qué clase pertenece la declaración; JAMÁS entra en `forbidden`. Aquí viven el
    inglés ('dairy', 'shellfish', 'egg'…) y las palabras de CATEGORÍA del Reglamento UE
    ('moluscos', 'crustáceos') que ningún ingrediente lleva por nombre.

    Esa separación es la que evita romper `test_paridad_dieta_alergeno_bidireccional`
    (test_p1_country_system_f2.py): meter 'dairy' en `_ALLERGEN_SYNONYMS['lacteos']` obligaría a
    espejarlo en `_DIET_DAIRY_TERMS`, que es una lista de NOMBRES DE ALIMENTO — y ninguna receta
    en español dice «dairy». El guard seguiría verde y significaría menos.

CLASE NUEVA `sesamo`. No existía ninguna (el comentario M5 de la ola final de F2 lo dejó por
escrito: «candidato de clase futura si aparece evidencia real, medido, no especulativo»). Esta
auditoría es esa evidencia: el catálogo tiene Ajonjolí, Tahini, Hummus y Aceite de sésamo, y el
sésamo es el nº 11 de los 14 alérgenos de declaración obligatoria del Reglamento UE 1169/2011 —
relevante para España, que está viva desde el flip. Antes del fix la única protección era que el
usuario tecleara exactamente «ajonjolí» y cayera al fallback de match literal.

SSOT. La expansión declaración→términos-prohibidos estaba DUPLICADA palabra por palabra en
`_scan_allergen_violations` y `_verified_catalog_excluded_tokens` (el propio docstring del segundo
decía «MISMA expansión que...»). Dos copias de la misma regla es exactamente lo que
P1-DIET-CANON-SSOT pagó una vez. El fix extrae `_expand_allergy_declarations` y las dos la llaman.

Cubre:
  A. Las grafías ciegas medidas arriba, una por una, contra `clinical_backstop_for_meal` real.
  B. La clase `sesamo` nueva (los 4 alimentos del catálogo).
  C. No-regresión: cada grafía que YA funcionaba sigue funcionando (el riesgo de esta clase de
     cambio es la sobre-detección, no la sub-detección).
  D. No-regresión: un plato inocuo no dispara con ninguno de los términos nuevos.
  E. SSOT: las dos superficies expanden IGUAL (control del refactor).
  F. Los alias de declaración NO contaminan el set de términos escaneados.
  G. Parser-based: el marker y las claves nuevas viven en el fuente.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_GO_PATH = _BACKEND_ROOT / "graph_orchestrator.py"


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


# Plato-sonda: un alimento real del catálogo por cada clase que la auditoría midió ciega.
_PLATO = {
    "meal": "Almuerzo",
    "name": "Bowl mixto de sonda",
    "ingredients": [
        "30 g de Mantequilla de maní",
        "10 g de Ajonjolí",
        "120 g de Almejas",
        "40 g de Queso blanco",
        "1 Huevos",
        "80 g de Atún",
        "50 g de Leche entera",
    ],
}

_PLATO_INOCUO = {
    "meal": "Cena",
    "name": "Ensalada de sonda",
    "ingredients": ["150 g de Lechuga", "80 g de Tomate", "10 g de Aceite de oliva"],
}


def _viola(go, declaracion):
    return go.clinical_backstop_for_meal(dict(_PLATO), allergies=[declaracion])


# ── A. Las grafías que la auditoría midió CIEGAS ────────────────────────────────────────────────

@pytest.mark.parametrize("declaracion", ["cacahuete", "cacahuetes", "alergia al cacahuete"])
def test_cacahuete_la_palabra_peninsular_del_mani_dispara_el_backstop(go, declaracion):
    """RED pre-fix: 0 violaciones. `_ALLERGEN_SYNONYMS['mani']` traía 'cacahuate' (grafía MX) y
    'peanut', y el match es por substring bidireccional — 'cacahuete' no es subcadena de
    'cacahuate' ni al revés (divergen en la 6ª letra). Es el caso más grave del conjunto: no hay
    chip de maní, así que el texto libre es la ÚNICA vía, y anafilaxia es el desenlace."""
    assert _viola(go, declaracion), (
        f"'{declaracion}' no protegió contra 'Mantequilla de maní' — el backstop es la única "
        f"defensa determinista de swap/chat-modify/degradado"
    )


@pytest.mark.parametrize("declaracion", ["sesamo", "sésamo", "sesame", "tahini", "ajonjoli"])
def test_sesamo_en_cualquiera_de_sus_grafias_dispara_el_backstop(go, declaracion):
    """RED pre-fix: sólo 'ajonjoli' pasaba, y no por conocer el sésamo sino por el fallback de
    match literal (`forbidden.add(a_low)`), que atrapa la palabra EXACTA tecleada y ningún
    derivado. Sin clase, «sésamo» no alcanzaba «Ajonjolí» y «ajonjolí» no alcanzaba «Aceite de
    sésamo»."""
    assert _viola(go, declaracion), f"'{declaracion}' no protegió contra 'Ajonjolí'"


def test_la_clase_sesamo_cubre_los_cuatro_alimentos_del_catalogo(go):
    """Los 4 nombres reales de `master_ingredients` que llevan sésamo. El Hummus (alta del top-up
    RD de F2-T8) lleva tahini y ninguna declaración posible lo cubría."""
    plato = {
        "meal": "Merienda",
        "name": "Sonda de sésamo",
        "ingredients": ["10 g de Ajonjolí", "20 g de Tahini", "60 g de Hummus",
                        "5 g de Aceite de sésamo"],
    }
    viol = go.clinical_backstop_for_meal(plato, allergies=["sésamo"])
    assert len(viol) >= 4, f"esperaba las 4 líneas marcadas, salieron {len(viol)}: {viol}"


@pytest.mark.parametrize("declaracion,ingrediente", [
    ("dairy", "Queso blanco"),
    ("milk", "Leche entera"),
    ("egg", "Huevos"),
    ("eggs", "Huevos"),
    ("fish", "Atún"),
    ("seafood", "Atún"),
    ("shellfish", "Almejas"),
    ("sesame", "Ajonjolí"),
])
def test_alergia_declarada_en_ingles_dispara_el_backstop(go, declaracion, ingrediente):
    """RED pre-fix: las 10 daban 0. El dashboard está en 5 idiomas desde P1-I18N-DASHBOARD
    (2026-08-15) y hay un usuario real con `locale='en-US'`: un formulario en inglés que sólo
    entiende alergias en español es una trampa silenciosa. Sólo 'peanut', 'wheat' y 'soy'
    sobrevivían, y por accidente — alguien los metió en su día como nombres de alimento."""
    assert _viola(go, declaracion), f"'{declaracion}' no protegió contra '{ingrediente}'"


@pytest.mark.parametrize("declaracion", ["nuts", "tree nuts", "tree-nuts"])
def test_frutos_secos_declarado_en_ingles_dispara_sobre_un_fruto_seco(go, declaracion):
    """Plato aparte a propósito: el maní NO es un fruto seco, así que emparejar 'tree nuts' con
    «Mantequilla de maní» habría sido pedir la sobre-detección que el test de al lado prohíbe.
    La sonda correcta es una almendra."""
    plato = {"meal": "Merienda", "name": "Sonda de frutos secos",
             "ingredients": ["30 g de Almendras", "10 g de Pistachos"]}
    assert go.clinical_backstop_for_meal(plato, allergies=[declaracion]), (
        f"'{declaracion}' no protegió contra 'Almendras'"
    )


@pytest.mark.parametrize("declaracion", ["moluscos", "molusco", "crustaceos", "crustáceos"])
def test_las_categorias_del_reglamento_ue_disparan_el_backstop(go, declaracion):
    """Moluscos y crustáceos son 2 de los 14 alérgenos de declaración obligatoria del Reglamento
    UE 1169/2011 y en España la distinción es estándar de etiquetado — mucha gente es alérgica a
    uno y no al otro. RED pre-fix: 0 para ambas; sólo el chip 'Mariscos' funcionaba, y un español
    no tiene por qué usar esa palabra."""
    assert _viola(go, declaracion), f"'{declaracion}' no protegió contra 'Almejas'"


# ── B. No-regresión: lo que YA funcionaba sigue funcionando ─────────────────────────────────────

@pytest.mark.parametrize("declaracion,esperado_min", [
    ("cacahuate", 1), ("mani", 1), ("peanut", 1),
    ("mariscos", 1), ("marisco", 1),
    ("lacteos", 2), ("lactosa", 2),
    ("huevo", 1), ("huevos", 1),
    ("pescado", 1),
    ("frutos secos", 0),   # el maní NO es fruto seco: categorías separadas, a propósito
])
def test_las_grafias_que_ya_funcionaban_siguen_funcionando(go, declaracion, esperado_min):
    """El riesgo de ampliar un vocabulario de match por substring no es sub-detectar: es
    sobre-detectar y mover el conteo de una clase vecina. Este parámetro fija el conteo medido
    ANTES del fix para cada declaración que ya pasaba."""
    viol = _viola(go, declaracion)
    assert len(viol) >= esperado_min, f"'{declaracion}' regresó: {len(viol)} < {esperado_min}"


def test_frutos_secos_sigue_sin_cubrir_el_mani(go):
    """Control de la asimetría intencional: el maní es una legumbre, no un fruto seco, y el chip
    «Frutos Secos» del wizard NO lo cubre. Si el fix hiciera que 'nuts' arrastrara al maní por
    una expansión perezosa, este test lo caza — la sobre-detección se acepta en la dirección
    declaración→clase (quien declara 'peanuts' recibe también los frutos secos, consejo clínico
    habitual), nunca al revés."""
    assert not go.clinical_backstop_for_meal(
        {"meal": "Merienda", "name": "x", "ingredients": ["30 g de Mantequilla de maní"]},
        allergies=["frutos secos"],
    )


# ── C. No-regresión: el plato inocuo no dispara con NINGÚN término nuevo ────────────────────────

@pytest.mark.parametrize("declaracion", [
    "cacahuete", "sesamo", "sesame", "dairy", "milk", "egg", "eggs", "fish", "seafood",
    "shellfish", "nuts", "tree nuts", "moluscos", "crustaceos",
])
def test_ningun_termino_nuevo_dispara_sobre_un_plato_inocuo(go, declaracion):
    """Lechuga + tomate + aceite de oliva. Este repo ya pagó la sobre-detección por subcadena
    tres veces ('sal'⊂'salsa', 'res'⊂'fresco', 'pollo'⊂'repollo'): cada término añadido necesita
    su caso de no-regresión."""
    assert not go.clinical_backstop_for_meal(dict(_PLATO_INOCUO), allergies=[declaracion]), (
        f"'{declaracion}' sobre-detecta en un plato sin alérgenos"
    )


def test_mantequilla_de_cacahuete_no_dispara_una_alergia_a_lacteos(go):
    """La excusa plant-adjacent (`_PLANT_ADJ_EXCUSE_RX`) listaba 'cacahuate' y no 'cacahuete', así
    que en un plan ESPAÑOL la línea «mantequilla de cacahuete» marcaba una violación de LÁCTEOS
    por la palabra «mantequilla». Es el mismo defecto de grafía, en el guard que existe justo
    para evitar el falso positivo."""
    assert not go.clinical_backstop_for_meal(
        {"meal": "Desayuno", "name": "x", "ingredients": ["20 g de mantequilla de cacahuete"]},
        allergies=["lacteos"],
    )


# ── D. SSOT: las dos superficies expanden IGUAL ─────────────────────────────────────────────────

@pytest.mark.parametrize("declaracion", [
    "cacahuete", "sesamo", "dairy", "shellfish", "moluscos", "egg", "mariscos", "gluten",
])
def test_el_catalogo_verificado_y_el_escaner_expanden_la_misma_declaracion(go, declaracion):
    """La expansión declaración→clase estaba DUPLICADA palabra por palabra en
    `_scan_allergen_violations` y `_verified_catalog_excluded_tokens` — el docstring del segundo
    decía «MISMA expansión que...», que es exactamente la promesa que dos copias no pueden
    sostener. Tras el fix ambas llaman a `_expand_allergy_declarations`, así que arreglar una
    arregla la otra: si alguien vuelve a bifurcarlas, este test se pone rojo."""
    del_escaner = go._expand_allergy_declarations([declaracion])
    del_catalogo = go._verified_catalog_excluded_tokens({"allergies": [declaracion]})
    assert del_escaner, f"'{declaracion}' no expandió a ningún término"
    assert del_escaner <= set(del_catalogo), (
        f"'{declaracion}': el catálogo verificado excluye MENOS que el escáner — "
        f"faltan {sorted(del_escaner - set(del_catalogo))}"
    )


def test_los_alias_de_declaracion_no_contaminan_los_terminos_escaneados(go):
    """`_ALLERGEN_DECLARATION_ALIASES` responde «¿de qué clase habla el usuario?», nunca «¿qué
    busco en el plato?». Si sus términos entraran en `forbidden`, el escáner buscaría la palabra
    'dairy' dentro de ingredientes escritos en español — ruido inútil hoy y una trampa el día que
    alguien los espeje a `_DIET_*_TERMS` creyendo que son nombres de alimento."""
    expandido = go._expand_allergy_declarations(["dairy"])
    assert "leche" in expandido, "la clase correcta no se resolvió"
    for alias in ("dairy", "shellfish", "tree nuts", "moluscos"):
        assert alias not in expandido, (
            f"el alias de declaración '{alias}' se coló en el set de términos escaneados"
        )


def test_una_alergia_desconocida_sigue_cayendo_al_match_literal(go):
    """Contrato preservado: una alergia free-text que no casa con ninguna clase se busca tal
    cual. Es lo que protege hoy a quien declara algo que el sistema no modela (fresa, kiwi)."""
    assert go._expand_allergy_declarations(["fresa"]) == {"fresa"}


# ── E. Parser-based: el marker y las claves viven en el fuente ──────────────────────────────────

def test_el_fuente_declara_el_marker_y_las_dos_mitades_del_vocabulario():
    """Si alguien renombra el dict de alias o la clase nueva, este test cae antes que producción
    (convención del repo: tooltip-anchor para todo test que parsea fuente)."""
    src = _GO_PATH.read_text(encoding="utf-8", errors="replace")
    assert "P0-ALLERGEN-VOCAB-I18N" in src
    assert "_ALLERGEN_DECLARATION_ALIASES" in src
    assert "_expand_allergy_declarations" in src
    assert '"sesamo":' in src, "la clase de sésamo desapareció de _ALLERGEN_SYNONYMS"


def test_la_expansion_esta_en_un_solo_sitio():
    """Control del refactor SSOT: el bucle `for cat, syns in _ALLERGEN_SYNONYMS.items():` debe
    existir UNA sola vez en producción. Dos ocurrencias significan que alguien volvió a copiar la
    expansión en vez de llamar al helper — la forma exacta del defecto que este P-fix cerró."""
    src = _GO_PATH.read_text(encoding="utf-8", errors="replace")
    assert src.count("for cat, syns in _ALLERGEN_SYNONYMS.items():") == 1, (
        "la expansión declaración→clase volvió a estar duplicada"
    )
