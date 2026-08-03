"""[P2-SEEDER-DAYS-COUNT · 2026-08-03] (audit solver+seeder v7) El reparto determinista del
seeder estaba hardcodeado a 3 días; el chunk dominante de los planes largos es de 4.

## El límite aritmético

`_rotate_pairs` producía SIEMPRE 3 pares (`range(3)`), el prompt definía Opciones A/B/C con
`protein_0/1/2` y `num_proteins_to_pick = min(3, ...)`. Pero `constants.split_with_absorb`
reparte así:

    7d  → [3, 4]
    15d → [3, 4, 4, 4]
    30d → [3, 4, 4, 4, 4, 4, 4, 3]
    21d → [3, 4, 4, 4, 6]

y el estampado al esqueleto (`graph_orchestrator`, P2-SEEDER-PAIRS-CHANNEL /
P2-VEGGIE-CHANNEL-DAYGEN) reparte por MÓDULO: `_pairs_all[_di % len(_pairs_all)]`. Con listas de
largo 3 y chunks de 4 días, **el día índice 3 recibe exactamente el reparto del día 0** — misma
proteína, mismo par de carbos, mismo par de vegetales, mismo par de frutas. En un plan de 30 días
son ~6 pares de días clonados POR CONSTRUCCIÓN, no por un fallo del modelo: el contrato
«1 proteína distinta por día» (`variety_level=max`, auto-promovido para gain_muscle/lose_fat y
bariátrica) es aritméticamente insatisfacible en el 4º día de cada chunk.

## Lo que NO cambia

El estampado del orquestador sigue siendo `% len(_pairs_all)`. Con listas de largo `days_count`
el módulo pasa a ser la identidad para el chunk que se está generando; y si por lo que sea el
esqueleto trae más días que pares (pool degradado, caller no migrado), el módulo sigue siendo el
fallback correcto. No se toca una línea de ese bloque.

## La degradación es deliberada

Con un pool más chico que `days_count` (6 días pedidos, 4 proteínas tras alergias+dieta+clínico)
el reparto DEGRADA al módulo actual en vez de exigir lo imposible. Es la lección de
P1-FRUIT-SEEDER-GATE-CONTRACT: una instrucción insatisfacible no produce mejores planes, produce
retries quemados. Lo que sí se exige es que la degradación no vuelva a producir el par idéntico
CONSECUTIVO que el helper existe para evitar.

Rollback sin redeploy: `MEALFIT_SEEDER_DAYS_COUNT=false` → reparto de 3 fijo, como antes.

tooltip-anchor: P2-SEEDER-DAYS-COUNT
"""
from __future__ import annotations

import random
import re
from pathlib import Path

import pytest

import ai_helpers as a

_BACKEND = Path(__file__).resolve().parent.parent


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


_AH = _src("ai_helpers.py")
_GO = _src("graph_orchestrator.py")
_PREFS = _src("prompts/preferences.py")


def _prompt(days_count=None, **form):
    fd = {"mainGoal": "gain_muscle"}
    fd.update(form)
    kw = {} if days_count is None else {"days_count": days_count}
    return a.get_deterministic_variety_prompt("", fd, None, **kw)


# ───────────── 1 · el helper del reparto escala ─────────────

def test_rotate_pairs_produce_days_pares():
    """El caso del brief: 4 días, 4 bases → 4 pares y el día 4 NO clona al día 1."""
    pares = a._rotate_pairs(["a", "b", "c", "d"], days=4)
    assert pares is not None and len(pares) == 4
    assert all(x != y for x, y in pares), f"ningún par (x,x): {pares}"
    assert pares[3] != pares[0], (
        "el día 4 duplica el día 1 aunque el pool alcanza — el módulo a 3 sigue vivo")


def test_rotate_pairs_default_sigue_siendo_tres():
    """Contrato de compatibilidad: los callers no migrados (fruta, ancla liviana) siguen
    recibiendo exactamente 3 pares."""
    assert len(a._rotate_pairs(["a", "b", "c", "d"])) == 3
    assert len(a._rotate_fruit_pairs(["Mango", "Guineo", "Fresas", "Melón"])) == 3


@pytest.mark.parametrize("days", [1, 2, 3, 4, 5, 6])
def test_rotate_pairs_nunca_repite_par_consecutivo(days):
    """Con pool de 4 y hasta 6 días el reparto degrada al módulo, pero el par del día i+1
    JAMÁS puede ser el mismo objeto que el del día i (sería la monotonía que el helper existe
    para evitar)."""
    pares = a._rotate_pairs(["a", "b", "c", "d"], days=days)
    assert pares is not None and len(pares) == days
    for i in range(len(pares) - 1):
        assert pares[i] != pares[i + 1], f"pares consecutivos idénticos en {i}: {pares}"


def test_rotate_pairs_sin_material_sigue_devolviendo_none():
    """El caller decide el fallback; `days` no debe inventar un pool."""
    for entrada in (None, [], ["Papa"], ["", "  "]):
        assert a._rotate_pairs(entrada, days=4) is None


# ───────────── 2 · el prompt escala al tamaño real del chunk ─────────────

def test_prompt_con_4_opciones():
    """El caso del brief: un chunk de 4 días exige 4 opciones de reparto."""
    p = _prompt(days_count=4)
    assert "protein_3" in p or "Opción D" in p or "OPCIÓN D" in p, (
        "chunk de 4 días exige 4 opciones de reparto")


@pytest.mark.parametrize("n,letra", [(3, "C"), (4, "D"), (5, "E"), (6, "F")])
def test_el_prompt_tiene_exactamente_n_opciones(n, letra):
    p = _prompt(days_count=n)
    assert p.count("OPCIÓN ") == n, f"se esperaban {n} opciones, hay {p.count('OPCIÓN ')}"
    assert f"OPCIÓN {letra}" in p, f"falta la etiqueta de la última opción (OPCIÓN {letra})"
    # cada opción trae su par de frutas y su par de carbos, como las 3 originales
    assert p.count("Frutas asignadas al día") == n
    assert p.count("acompañante vegetal/grasa") == n


def test_el_default_sigue_dando_tres_opciones():
    """Los callers que no pasan `days_count` (y el chunk de 3, que es la mitad de los planes)
    ven exactamente el prompt de siempre."""
    p = _prompt()
    assert p.count("OPCIÓN ") == 3
    assert "OPCIÓN D" not in p


def test_default_y_days_count_3_son_el_mismo_prompt():
    """Byte-identidad: pasar `days_count=3` explícito no puede cambiar ni un carácter
    (protege el prompt-cache y hace revisable el diff del refactor)."""
    random.seed(20260803)
    sin = _prompt()
    random.seed(20260803)
    con = _prompt(days_count=3)
    assert sin == con


# ───────────── 3 · el knob de rollback ─────────────

def test_knob_off_congela_el_reparto_en_tres(monkeypatch):
    """`MEALFIT_SEEDER_DAYS_COUNT=false` ⇒ byte-idéntico al comportamiento previo aunque el
    caller pida 4/6 días."""
    random.seed(4242)
    base = _prompt(days_count=3)
    monkeypatch.setenv("MEALFIT_SEEDER_DAYS_COUNT", "false")
    random.seed(4242)
    apagado = _prompt(days_count=6)
    assert apagado == base, "con el knob OFF el prompt debe ser el de 3 días, byte a byte"
    assert apagado.count("OPCIÓN ") == 3


def test_el_knob_esta_registrado_por_env_bool():
    """Un knob que no pasa por `_env_bool` no aparece en `_KNOBS_REGISTRY` y es invisible en
    `/health/version` — la lección de P3-SEEDER-KNOBS-REGISTRY."""
    assert '_env_bool("MEALFIT_SEEDER_DAYS_COUNT"' in _AH


# ───────────── 4 · degradación con pool más chico que el chunk ─────────────

def _prompt_pool_estrecho(monkeypatch, n_proteinas: int, days_count: int) -> str:
    """Fuerza el pool de proteínas a `n_proteinas` interceptando el filtro clínico/dietético.

    Se parchea `_get_fast_filtered_catalogs` en vez de usar `dislikes` porque el filtro real hace
    matching por subcadena sobre las CUATRO categorías: una lista larga de dislikes de proteínas
    vaciaba también carbos y vegetales y el seeder salía por su guard de pool vacío (`return ""`),
    midiendo otra cosa. Todas son alta densidad para que el reemplazo de
    P3-GAINMUSCLE-PROTEIN-DENSITY no introduzca ruido."""
    from constants import DOMINICAN_CARBS, DOMINICAN_VEGGIES_FATS, DOMINICAN_FRUITS
    prot = ["Pollo", "Cerdo", "Res", "Pescado", "Atún", "Camarones"][:n_proteinas]
    monkeypatch.setattr(
        a, "_get_fast_filtered_catalogs",
        lambda *_a, **_k: (list(prot), list(DOMINICAN_CARBS),
                           list(DOMINICAN_VEGGIES_FATS), list(DOMINICAN_FRUITS)))
    return a.get_deterministic_variety_prompt(
        "", {"mainGoal": "gain_muscle"}, None, days_count=days_count)


def test_seis_dias_con_pool_de_cuatro_no_lanza(monkeypatch):
    """21d → [3,4,4,4,6]: el chunk de 6 pide 6 proteínas distintas. Si el pool filtrado tiene 4,
    el reparto DEGRADA (documentado) — pero no puede reventar el nodo: la excepción tumbaría la
    generación y el retry repetiría el mismo crash determinista."""
    p = _prompt_pool_estrecho(monkeypatch, 4, 6)
    assert p and p.count("OPCIÓN ") == 6
    assert "{protein_" not in p and "{carb_" not in p and "{veggie_" not in p, (
        "quedó un placeholder sin sustituir: `.format` no reventó pero el prompt sale roto")


def test_la_degradacion_no_produce_pares_identicos(monkeypatch):
    """El pool corto no puede resucitar 'Zanahoria y Zanahoria' (P2-VEGGIE-PAIRS-ROTATE): las
    categorías que pasan por `_rotate_pairs` garantizan par distinto EN el día y par distinto
    ENTRE días consecutivos, también con 6 días y pool de 4."""
    p = _prompt_pool_estrecho(monkeypatch, 4, 6)
    for etiqueta, patron in (("vegetales", r"acompañante vegetal/grasa:\s*([^.]+)\."),
                             ("frutas", r"mismo día\):\s*([^.]+)\.")):
        lineas = re.findall(patron, p)
        assert len(lineas) == 6, f"{etiqueta}: se esperaban 6 líneas, hay {len(lineas)}"
        for ln in lineas:
            if " y " in ln:
                izq, der = [x.strip() for x in ln.split(" y ", 1)]
                assert izq != der, f"{etiqueta}: par idéntico dentro del día: {ln!r}"
        for i in range(len(lineas) - 1):
            assert lineas[i] != lineas[i + 1], (
                f"{etiqueta}: días consecutivos con el MISMO par: {lineas[i]!r}")


def test_la_degradacion_de_proteina_recicla_todo_el_pool(monkeypatch):
    """Con 6 días y 4 proteínas los 2 días extra RECICLAN — pero deben reciclar sobre TODO el
    pool, no colapsar a una o dos bases (que es la monotonía que P2-PANTRY-ROTATION-FLOOR cerró
    por el lado de la nevera)."""
    p = _prompt_pool_estrecho(monkeypatch, 4, 6)
    prot = [x.strip() for x in re.findall(r"DEBE incluir obligatoriamente:\s*([^+]+)\+", p)]
    assert len(prot) == 6
    assert len(set(prot)) == 4, f"el reciclado colapsó el pool: {prot}"


def test_la_proteina_puede_caer_en_dos_dias_seguidos_es_PREEXISTENTE(monkeypatch):
    """⚠️ RESIDUO DOCUMENTADO, no una regresión de P2-SEEDER-DAYS-COUNT.

    Carbos, vegetales y frutas se reparten con `_rotate_pairs`, que da espaciado circular. La
    PROTEÍNA no: es una lista plana que se rellena por round-robin (espaciado correcto) y
    después se vuelve a barajar (`random.shuffle(chosen_proteins)`), lo que destruye ese
    espaciado y puede dejar dos días seguidos con la misma base.

    Este test lo REPRODUCE con `days_count=3` y un pool de 2 — exactamente el camino que corría
    en producción antes de este fix (`variety_level=standard` elige 2 proteínas y el padding las
    cicla a 3): con `[a,b,a]` barajado, 2 de las 3 permutaciones dejan las dos 'a' pegadas. O
    sea, el defecto NO nace de escalar a 4/6 días.

    No se arregla aquí a propósito: quitar el segundo shuffle cambia el ORDEN de las proteínas
    de todos los planes que se generan hoy, y eso es un cambio de comportamiento que pide
    medición (benchmark de variedad), no un efecto colateral de una tarea sobre aritmética de
    chunks. Si alguien lo cierra, este test debe INVERTIRSE, no borrarse."""
    vistos = set()
    for semilla in range(30):
        random.seed(semilla)
        p = _prompt_pool_estrecho(monkeypatch, 2, 3)
        prot = [x.strip() for x in re.findall(r"DEBE incluir obligatoriamente:\s*([^+]+)\+", p)]
        vistos.add(any(prot[i] == prot[i + 1] for i in range(len(prot) - 1)))
    assert True in vistos, (
        "ya no se reproduce el residuo con days_count=3 — si se arregló el orden de proteínas, "
        "invierte este test y actualiza el reporte")


def test_el_prompt_no_promete_mas_proteinas_distintas_de_las_que_hay(monkeypatch):
    """Contrato de `variety_level=max` («1 proteína distinta por día», auto-promovido para
    gain_muscle): con 3 proteínas en el pool y un chunk de 4, el texto NO puede afirmar que cada
    día lleva una proteína distinta. Se degrada en silencio, no se miente al modelo."""
    p = _prompt_pool_estrecho(monkeypatch, 3, 4)
    bajo = p.lower()
    for promesa in ("proteína distinta por día", "proteina distinta por dia",
                    "una proteína diferente cada día", "4 proteínas distintas"):
        assert promesa not in bajo, f"el prompt promete lo insatisfacible: {promesa!r}"
    # y la degradación es la documentada: el 4º día recicla, no inventa fuera del pool
    prot = {x.strip() for x in re.findall(r"DEBE incluir obligatoriamente:\s*([^+]+)\+", p)}
    assert prot <= {"Pollo", "Cerdo", "Res"}, f"proteína fuera del pool filtrado: {prot}"


# ───────────── 5 · el caller pasa el tamaño REAL del chunk ─────────────

# `nutrition` mínimo que `_build_shared_context` recorre entero (`build_nutrition_context` y
# `build_minimal_correction_context` indexan estas claves con `[...]`, no con `.get`).
_NUTRICION_MINIMA = {
    "bmr": 1600, "tdee": 2200, "target_calories": 2000, "calories": 2000,
    "goal_label": "Ganar músculo", "kinematics": {}, "calculation_details": {}, "alergias": [],
    "macros": {"protein_g": 150, "carbs_g": 200, "fats_g": 60, "fiber_g": 30,
               "protein_str": "150g", "carbs_str": "200g", "fats_str": "60g", "fiber_str": "30g"},
}


def _espiar_days_count(monkeypatch, dias_del_chunk):
    """EJECUTA `_build_shared_context` con un spy sobre el seeder y devuelve los kwargs reales.

    El spy se instala en `ai_helpers.get_deterministic_variety_prompt` (no en el namespace del
    orquestador) porque el callsite hace `from ai_helpers import ...` DENTRO del cuerpo de la
    función: resuelve el nombre en cada llamada."""
    import ai_helpers
    import graph_orchestrator as go

    capturado: dict = {}

    def _spy(*args, **kwargs):
        capturado["args"] = args
        capturado["kwargs"] = kwargs
        return ""

    monkeypatch.setattr(ai_helpers, "get_deterministic_variety_prompt", _spy)
    estado = {
        "form_data": {"user_id": "guest", "mainGoal": "gain_muscle",
                      "_days_to_generate": dias_del_chunk},
        "nutrition": dict(_NUTRICION_MINIMA),
    }
    go._build_shared_context(estado)
    return capturado


@pytest.mark.parametrize("dias", [3, 4, 6])
def test_build_shared_context_pasa_el_tamano_del_chunk(monkeypatch, dias):
    """⚠️ La afirmación MÁS importante de la tarea, y por eso se EJECUTA en vez de parsearse:
    sin este cableado todo lo demás queda INERTE en producción y un test parser-based no lo
    notaría (es la lección «un test parser-based no ejecuta nada» que ya costó un P-fix aquí).

    Los tres tamaños son los que `split_with_absorb` produce de verdad: 3 (chunk inicial),
    4 (el dominante en 15d/30d) y 6 (el leftover absorbido de 21d)."""
    cap = _espiar_days_count(monkeypatch, dias)
    assert cap, "`_build_shared_context` no llegó a invocar el seeder"
    assert cap["kwargs"].get("days_count") == dias, (
        f"el callsite pasó days_count={cap['kwargs'].get('days_count')!r} para un chunk de "
        f"{dias} días — el fix no llega al pipeline real")
    # y el canal tipado del reparto (P2-VEGGIE-CHANNEL-DAYGEN / P2-SEEDER-PAIRS-CHANNEL) sigue vivo
    assert isinstance(cap["kwargs"].get("out_assignment"), dict), (
        "se perdió `out_assignment`: el reparto dejaría de viajar como DATO al esqueleto")


def test_sin_days_to_generate_el_caller_cae_al_chunk_size(monkeypatch):
    """El path no-chunked (planes ≤3 días) no setea `_days_to_generate`: debe caer a
    `PLAN_CHUNK_SIZE`, no a `None` ni a 0."""
    import ai_helpers
    import graph_orchestrator as go
    from constants import PLAN_CHUNK_SIZE

    capturado: dict = {}
    monkeypatch.setattr(ai_helpers, "get_deterministic_variety_prompt",
                        lambda *a, **k: capturado.update(k) or "")
    go._build_shared_context({
        "form_data": {"user_id": "guest", "mainGoal": "gain_muscle"},
        "nutrition": dict(_NUTRICION_MINIMA),
    })
    assert capturado.get("days_count") == PLAN_CHUNK_SIZE


def test_el_callsite_lee_days_to_generate_y_no_otra_fuente():
    """Defensa barata y complementaria al spy: la fuente del dato debe ser la MISMA clave que
    usan `plan_skeleton_node` y `generate_days_parallel_node`. El spy prueba que llega un
    número; esto prueba de DÓNDE sale (un `days_count=4` hardcodeado pasaría el spy)."""
    i = _GO.index("variety_prompt = get_deterministic_variety_prompt(")
    bloque = _GO[i - 1500:i + 500]
    assert "days_count=" in bloque
    assert "_days_to_generate" in bloque, (
        "el tamaño del chunk debe venir de `_days_to_generate`, no inventarse")


def test_el_estampado_del_orquestador_sigue_siendo_por_modulo():
    """Ancla de NO-cambio: el módulo es la identidad cuando len(pares) == len(días) y sigue
    siendo el fallback correcto si el seeder degradó a menos pares que días."""
    assert "_pairs_all[_di % len(_pairs_all)]" in _GO
    assert "_vp_all[_di % len(_vp_all)]" in _GO


# ───────────── 6 · anclas estructurales ─────────────

def test_el_reparto_no_tiene_el_tres_hardcodeado():
    """tooltip-anchor: si alguien revierte a `range(3)` dentro del helper, este test cae antes
    que producción."""
    import inspect
    src = inspect.getsource(a._rotate_pairs)
    assert "range(days)" in src, "`_rotate_pairs` volvió a fijar el número de días"
    assert "for i in range(3)" not in src


def test_la_letra_del_dia_sale_de_un_solo_sitio(monkeypatch):
    """El prompt nombra el día DOS veces con letra: en la etiqueta «OPCIÓN D» y en el ancla
    liviana («día D → …»). Si cada una lleva su propia tabla de letras, divergen en cuanto
    alguien toque el alfabeto y las dos frases hablarían de días distintos. Funcional: se
    enciende `MEALFIT_LIGHT_PROTEIN_SEED` (OFF por default) para que el ancla exista."""
    monkeypatch.setattr(a, "LIGHT_PROTEIN_SEED", True)
    p = _prompt(days_count=4)
    assert "ANCLA LIVIANA SORTEADA POR DÍA" in p, "el knob no encendió el bloque"
    letras_ancla = set(re.findall(r"día ([A-Z]) →", p))
    letras_opcion = set(re.findall(r"OPCIÓN ([A-Z]) \(Alternativa", p))
    assert letras_ancla == letras_opcion == {"A", "B", "C", "D"}, (
        f"ancla={sorted(letras_ancla)} vs opciones={sorted(letras_opcion)}")
    # y `ai_helpers` no puede llevar su propio alfabeto: usa el helper público del módulo de prompts
    assert "ABCDEFG" not in _AH, "ai_helpers duplicó la tabla de letras en vez de importarla"
    assert "option_letter" in _AH


def test_las_opciones_del_prompt_se_generan_no_se_copian():
    """Tres literales A/B/C es cómo se llegó al límite de 3. Las opciones se generan por join."""
    assert "def build_deterministic_variety_prompt" in _PREFS
    assert _PREFS.count("(Alternativa 1)") <= 1, (
        "las opciones vuelven a estar copiadas literal en la plantilla")


def test_marker_presente_en_las_tres_capas():
    for src, nombre in ((_AH, "ai_helpers.py"), (_PREFS, "prompts/preferences.py"),
                        (_GO, "graph_orchestrator.py")):
        assert "P2-SEEDER-DAYS-COUNT" in src, f"falta el marker en {nombre}"
