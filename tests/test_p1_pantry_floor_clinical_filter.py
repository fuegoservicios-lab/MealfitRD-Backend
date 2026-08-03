"""[P1-PANTRY-FLOOR-CLINICAL-FILTER · 2026-08-02] (audit solver+seeder v7 · Task 3)

Lo que la nevera aporta al esqueleto debe pasar por los MISMOS filtros de "puede ser base
principal" que el sorteo, y no lo hacía.

El seeder aplica sus penalties clínicos sobre los PESOS del sorteo (embutidos ×0.1 por goal,
curados en sal ×0.1 universal, carnes grasas ×0.3, reemplazo de mains de baja densidad para
gain_muscle/bariátrica, frutas alto-IG ×0.15 para bariátrica). Cientos de líneas después, el
modo rotación REEMPLAZA el pool entero por lo extraído de la nevera y —si aporta
`PANTRY_ROTATION_MIN_PROTEINS` proteínas— activa `cycle_locked`, que mete en el prompt
"REGLA DE AHORRO EXTREMA… NO SUGIERAS ALIMENTOS BASE NUEVOS".

Resultado: una nevera con «Salami Dominicano» + «Longaniza» convierte dos embutidos en la base
OBLIGATORIA de los 3 días, con el LLM impedido de proponer otra cosa. El revisor clínico
rechaza de forma determinista, el retry vuelve a correr el seeder con la MISMA nevera, y se
queman todos los reintentos. Los penalties existían: el reemplazo por la nevera los saltaba.

Contrato que ancla este archivo:

  1. Con goal que penaliza procesados (o perfil bariátrico), los embutidos/quesos de relleno de
     la nevera dejan de ser candidatos a MAIN. Siguen en la nevera (acompañantes) — solo salen
     del pool de bases.
  2. Si el 100% de lo extraído es curado/embutido, NO se activa el lock para NINGÚN goal.
  3. Espejo en frutas alto-IG SOLO para bariátrica (que es donde existe el penalty).
  4. El filtro corre ANTES del `len(extracted_p) >= PANTRY_ROTATION_MIN_PROTEINS`, así que un
     pool que queda corto cae por la rama existente (nevera primero + sorteo completa, sin lock).
  5. El matching es por LÍMITE DE PALABRA. Este repo lleva 13+ incidentes de subcadena
     (`"sal"`⊂`"Salami"`, `"res"`⊂`"fresas"`, `"pollo"`⊂`"repollo"`, `"molida"`⊂`"linaza
     molida"`); aquí el caso REAL del catálogo es `"pina"` ⊂ `"Espinacas"`.
  6. Alergia y dieta mandan: el filtro solo QUITA de lo ya extraído, jamás reintroduce.
"""
from __future__ import annotations

import logging
import os
import random
import re

import pytest

import ai_helpers as ah
from constants import (
    DOMINICAN_CARBS,
    DOMINICAN_FRUITS,
    DOMINICAN_PROTEINS,
    DOMINICAN_VEGGIES_FATS,
    strip_accents,
)

_LOCK = "REGLA DE AHORRO EXTREMA"


def _prompt(pantry, **extra) -> str:
    """Prompt del seeder con la nevera dada. `user_id=None` ⇒ cero I/O (test offline)."""
    fd = {"dietType": "ninguna", "current_pantry_ingredients": list(pantry)}
    fd.update(extra)
    return ah.get_deterministic_variety_prompt("", fd, user_id=None)


def _mains(prompt: str) -> list[str]:
    """Las 3 proteínas que el prompt ASIGNA como base de cada día."""
    return re.findall(r"DEBE incluir obligatoriamente: (.+?) \+ ", prompt)


def _fruits(prompt: str) -> str:
    """Bloque de frutas asignadas (para comprobar el espejo alto-IG)."""
    return " | ".join(re.findall(r"NUNCA la misma dos veces el mismo día\): (.+?)\.", prompt))


def _pantry_proteins(caplog, pantry, **extra):
    """Nº de proteínas que la nevera aportó tras el filtro. `None` = se activó el lock.

    Mismo vehículo que `test_p1_pantry_extract_filtered_wb`: el log del FLOOR distingue
    "no aportó nada" de "aportó una y el floor la completó", cosa que mirar solo el lock no
    permite. Afirmar sobre el NOMBRE en el prompt daría falsos positivos: la prosa genérica
    del prompt ya menciona pollo/res/pescado."""
    caplog.clear()
    caplog.set_level(logging.INFO, logger=ah.logger.name)
    out = _prompt(pantry, **extra)
    if _LOCK in out.upper():
        return None
    for r in caplog.records:
        m = re.search(r"la nevera aportó (\d+) proteína", r.getMessage())
        if m:
            return int(m.group(1))
    return 0


# ═══════════════ 1 · caso del audit: nevera 100% embutidos ═══════════════

def test_nevera_100pct_embutidos_no_activa_lock():
    """Salami + Longaniza ≥ MIN_PROTEINS ⇒ pre-fix el pool se reemplazaba y el lock los volvía
    la base OBLIGATORIA de los 3 días → rechazo clínico determinista + retries quemados."""
    out = _prompt(["Salami Dominicano", "Longaniza", "Repollo"], mainGoal="gain_muscle")
    assert _LOCK not in out.upper(), (
        "una nevera de puros embutidos no puede convertirse en las bases obligatorias del plan")


def test_nevera_100pct_embutidos_no_es_main_con_goal_que_penaliza():
    """El lock es la mitad del daño; la otra es que ocupen los slots de main.

    ⚠️ [re-review audit-v7-p1 · 2026-08-03] Este test era FLAKY y su redacción prometía un
    "nunca" que el sistema no da. Se cazó en la corrida completa de la suite (con la junction
    puesta, o sea con todos los ficheros colectándose): falló con
    `['Longaniza', 'Queso de hoja', 'Salmón']` mientras el log confirmaba que el filtro SÍ había
    corrido (`2 proteína(s) de la nevera fuera de las BASES clínicas`). No era una regresión: es
    que la defensa de esta capa es un PESO, no una exclusión. El filtro de la nevera saca a los
    embutidos de las bases derivadas de la nevera —eso sí es determinista— pero en el sorteo
    general siguen en el pool con `×0.1` (goal penalty) y `×0.1` (P1-SODIUM-BOMB-POOL), así que
    con mala suerte salen igual. El seeder usa el `random` global SIN semilla y el test no
    sembraba: cualquier cambio en el orden o el número de tests de la suite mueve el stream del
    RNG y voltea el resultado.

    Medido barriendo 400 semillas: **2/400 = 0,50%**  (semillas 134 y 280).

    Reescrito para afirmar lo que es cierto y de forma REPRODUCIBLE: un barrido determinista de
    100 semillas fijas con techo de tasa. Si alguien borra el `×0.1` de embutidos o el de
    sodio-bomba, la tasa se dispara muy por encima del techo y el test cae. Lo que NO se hace es
    sembrar una sola semilla afortunada y seguir escribiendo "nunca": eso sería un verde comprado.
    """
    ocurrencias = []
    for semilla in range(100):
        random.seed(semilla)
        out = _prompt(["Salami Dominicano", "Longaniza", "Repollo"], mainGoal="gain_muscle")
        mains = _mains(out)
        assert len(mains) == 3, f"no se pudieron leer las 3 bases del prompt: {mains!r}"
        malos = [b for b in ("Salami Dominicano", "Longaniza") if b in mains]
        if malos:
            ocurrencias.append((semilla, malos, mains))
    tasa = len(ocurrencias) / 100.0
    assert tasa <= 0.03, (
        f"los embutidos de la nevera llegan a base principal en {tasa:.0%} de los sorteos "
        f"(techo 3%, medido 0,5% sobre 400 semillas). Probable: se debilitó el ×0.1 de embutidos "
        f"por goal o el ×0.1 de P1-SODIUM-BOMB-POOL. Casos: {ocurrencias[:3]!r}")


def test_el_filtro_de_nevera_es_deterministico_aunque_el_sorteo_no():
    """[re-review · 2026-08-03] La mitad DETERMINISTA de la garantía, que el test de arriba ya no
    puede afirmar: pase lo que pase con el sorteo general, el filtro tiene que sacar a los
    embutidos de las bases derivadas de la NEVERA en las 100 semillas. Ésa es la promesa literal
    de P1-PANTRY-FLOOR-CLINICAL-FILTER; el resto es probabilidad."""
    for semilla in range(100):
        random.seed(semilla)
        proteinas, _frutas = ah._pantry_clinical_main_filter(
            ["Salami Dominicano", "Longaniza", "Pechuga de pollo"], [],
            penaliza_procesados=True, exige_densidad=True,
        )
        assert "Salami Dominicano" not in proteinas and "Longaniza" not in proteinas, (
            f"semilla {semilla}: el filtro de nevera dejó pasar un embutido a las bases: "
            f"{proteinas!r}")
        assert "Pechuga de pollo" in proteinas, (
            "el filtro no puede ser un borrado indiscriminado: la proteína sana se queda")


@pytest.mark.parametrize("goal", ["gain_muscle", "lose_fat", "maintenance", "performance", ""])
def test_nevera_100pct_curada_nunca_activa_lock_para_ningun_goal(goal):
    """Decisión explícita: la regla del 100% curado es UNIVERSAL, no solo para los goals que
    penalizan procesados. El presupuesto de sodio de la OMS no depende del objetivo."""
    out = _prompt(["Salami Dominicano", "Longaniza", "Repollo"], mainGoal=goal)
    assert _LOCK not in out.upper(), f"lock activado con nevera 100% curada y goal={goal!r}"


def test_nevera_mixta_conserva_la_proteina_sana(caplog):
    """No es un borrado indiscriminado: lo sano de la nevera sigue yendo PRIMERO (ahorro)."""
    n = _pantry_proteins(caplog, ["Salami Dominicano", "Pechuga de pollo", "Arroz Blanco"],
                         mainGoal="gain_muscle")
    assert n == 1, (
        f"esperado 1 (el pollo sobrevive, el salami sale de las bases), obtenido {n!r} "
        f"(None = lock activado con el embutido dentro de las bases obligatorias)")


# ═══════════════ 2 · bariátrica: baja densidad fuera de main ═══════════════

def test_bariatrica_excluye_low_density_y_embutido_como_main(caplog):
    """El pouch bariátrico necesita proteína animal densa; queso de freír y salami como base
    son rechazo clínico (corr=5ffd78cf / 3b318e57). El sorteo ya los reemplaza — la nevera no."""
    n = _pantry_proteins(caplog, ["Queso de Freír", "Salami", "Huevos"],
                         medicalConditions=["Cirugía bariátrica"])
    assert n == 1, (
        f"esperado 1 (solo Huevos sobrevive como base), obtenido {n!r}")


def test_bariatrica_no_pone_queso_ni_salami_de_base():
    out = _prompt(["Queso de Freír", "Salami", "Huevos"],
                  medicalConditions=["Cirugía bariátrica"])
    mains = _mains(out)
    assert len(mains) == 3
    for banned in ("Queso de Freír", "Salami Dominicano"):
        assert banned not in mains, f"'{banned}' como base principal en bariátrica: {mains!r}"


def test_bariatrica_filtra_frutas_alto_ig_de_la_nevera():
    """Espejo de frutas: SOLO bariátrica (es donde existe el penalty ×0.15)."""
    out = _prompt(["Piña", "Fresa", "Manzana"], medicalConditions=["Cirugía bariátrica"])
    asignadas = _fruits(out)
    assert asignadas, "no se pudieron leer las frutas asignadas del prompt"
    assert "Piña" not in asignadas, (
        f"fruta de alto índice glucémico como asignada de bariátrica: {asignadas!r}")
    assert "Fresa" in asignadas or "Manzana" in asignadas, (
        f"las frutas sanas de la nevera deben seguir siendo el piso: {asignadas!r}")


def test_no_bariatrica_conserva_frutas_alto_ig():
    """La regla NO se inventa para otros perfiles: sin bariátrica, la piña de la nevera sigue."""
    out = _prompt(["Piña", "Fresa", "Manzana"], mainGoal="gain_muscle")
    assert "Piña" in _fruits(out), "el espejo de frutas se aplicó fuera de bariátrica"


# ═══════════════ 3 · la nevera sana no cambia de comportamiento ═══════════════

def test_nevera_sana_conserva_floor_y_lock():
    """Regresión inversa: el ahorro es la razón de existir del modo rotación."""
    out = _prompt(["Pechuga de pollo", "Filete de pescado", "Arroz Blanco"],
                  mainGoal="gain_muscle")
    assert _LOCK in out.upper(), (
        "con 2 proteínas sanas en la nevera el cycle-lock debe seguir activándose")
    mains = _mains(out)
    assert "Pollo" in mains and "Pescado" in mains, f"bases de la nevera perdidas: {mains!r}"


def test_una_sola_proteina_sana_sigue_sin_lock(caplog):
    """P2-PANTRY-ROTATION-FLOOR intacto."""
    assert _pantry_proteins(caplog, ["Pechuga de pollo", "Arroz Blanco"],
                            mainGoal="gain_muscle") == 1


# ═══════════════ 4 · el modo de fallo recurrente: SUBCADENA ═══════════════

def test_alimento_sano_con_token_prohibido_dentro_no_se_filtra():
    """`"pina"` (token alto-IG de «Piña») es SUBCADENA de «Es-PINA-cas», un alimento real del
    catálogo. Con matching por subcadena —el operador que ya mordió 13 veces en este repo—
    Espinacas quedaría marcada como fruta de alto índice glucémico."""
    trampa = strip_accents("Espinacas".lower())
    assert "pina" in trampa, "la trampa dejó de ser real; revisa el caso antes de borrar el test"
    assert ah._token_matches_wb("Espinacas", ah._HIGH_GI_FRUITS) is False, (
        "'pina' ⊂ 'espinacas' — el matcher debe ser por límite de palabra, no subcadena")


def test_ningun_alimento_del_catalogo_cae_por_subcadena():
    """Barrido sobre los 4 catálogos REALES: todo alimento donde subcadena y límite-de-palabra
    discrepan debe quedar FUERA del filtro. Se exige que el barrido no sea vacío (si no, el
    test no probaría nada) y que contenga el caso documentado.

    [review final audit-v7-p1 · 2026-08-03 · T6] Aquí vivía `assert wb is False` DENTRO de
    `if naive and not wb:` — verdadera por álgebra (`_token_matches_wb` está anotada `-> bool` y
    solo retorna literales), o sea inalcanzable como fallo. Daba la impresión de que el barrido
    comprobaba algo por alimento cuando lo único que probaba eran las dos líneas finales.
    Sustituida por la comprobación que el docstring ya prometía y el cuerpo no hacía: se recoge
    también el complemento (`naive and wb` — subcadena Y límite de palabra coinciden) y se afirma
    que TODO alimento de ese complemento sí es filtrado por el matcher real. Así el barrido
    contrasta las dos ramas y no solo una."""
    grupos = (
        (DOMINICAN_PROTEINS, ah._CURED_OR_PROCESSED_TOKENS),
        (DOMINICAN_PROTEINS, ah._PROCESSED_MEAT_KEYWORDS),
        (DOMINICAN_FRUITS, ah._HIGH_GI_FRUITS),
        (DOMINICAN_CARBS, ah._CURED_OR_PROCESSED_TOKENS),
        (DOMINICAN_VEGGIES_FATS, ah._HIGH_GI_FRUITS),
    )
    divergencias = []      # subcadena dice sí, límite-de-palabra dice no → NO se filtra
    coincidencias = []     # ambos dicen sí → SÍ se filtra
    for catalogo, tokens in grupos:
        for food in catalogo:
            n = strip_accents(str(food).lower())
            naive = any(strip_accents(str(t).lower()) in n for t in tokens)
            wb = ah._token_matches_wb(food, tokens)
            if naive and wb:
                coincidencias.append((food, tokens))
            elif naive:
                divergencias.append(food)
    assert divergencias, (
        "el barrido no encontró ninguna colisión de subcadena en el catálogo — el test sería "
        "vacío; añade el caso a mano antes de aceptarlo")
    assert "Espinacas" in divergencias, f"colisiones detectadas: {divergencias!r}"
    # El complemento: donde los dos operadores coinciden, el filtro DEBE morder. Si el
    # word-boundary se volviera un no-op (o se relajara a "nunca matchea"), `divergencias`
    # crecería y esta lista se vaciaría — y sin esta comprobación el test seguiría verde.
    #
    # [ronda 2 · 2026-08-03] Mi primera versión cerraba con
    # `for food, tokens in coincidencias: assert _token_matches_wb(food, tokens) is True`, que es
    # tautológico: `coincidencias` se construyó con `if naive and wb`. Sustituido por el conjunto
    # ESPERADO, escrito a mano desde el catálogo real: así el test dice qué debe filtrarse y falla
    # si el matcher deja de morder en un alimento concreto, no solo si se apaga entero.
    nombres = {f for f, _ in coincidencias}
    esperados = {
        "Salami Dominicano", "Longaniza", "Jamón de pavo",   # embutidos/curados
        "Bacalao", "Arenque",                                # salazones
        "Piña", "Mango", "Melón", "Sandía", "Uva", "Guineo",  # frutas alto-IG
    }
    assert esperados <= nombres, (
        f"el filtro dejó de morder en alimentos que SÍ debe filtrar: "
        f"{sorted(esperados - nombres)!r}. Si un alimento salió del catálogo, quítalo de "
        f"`esperados`; si el matcher cambió, ese es el bug.")


def test_los_matches_legitimos_siguen_vivos():
    """Regresión inversa del word-boundary: no puede volver el filtro un no-op."""
    assert ah._token_matches_wb("Salami Dominicano", ah._PROCESSED_MEAT_KEYWORDS) is True
    assert ah._token_matches_wb("Longaniza", ah._PROCESSED_MEAT_KEYWORDS) is True
    assert ah._token_matches_wb("Jamón de pavo", ah._PROCESSED_MEAT_KEYWORDS) is True
    assert ah._token_matches_wb("Bacalao", ah._SALT_CURED_PROTEIN_TOKENS) is True
    assert ah._token_matches_wb("Arenque", ah._CURED_OR_PROCESSED_TOKENS) is True
    assert ah._token_matches_wb("Piña", ah._HIGH_GI_FRUITS) is True
    assert ah._token_matches_wb("Pollo", ah._CURED_OR_PROCESSED_TOKENS) is False
    assert ah._token_matches_wb("Fresa", ah._HIGH_GI_FRUITS) is False


# ═══════════════ 5 · el filtro como unidad (determinista) ═══════════════
#
# El filtro recibe DECISIONES ya tomadas (`penaliza_procesados` / `exige_densidad`), no el goal:
# las condiciones viven junto a los penalties del sorteo que replican. La cobertura por goal la
# dan los tests end-to-end de arriba, que sí entran por `mainGoal`.

def test_filtro_unidad_goal_penaliza_procesados():
    p, f = ah._pantry_clinical_main_filter(
        ["Salami Dominicano", "Longaniza", "Huevos"], [],
        penaliza_procesados=True, exige_densidad=True, is_bariatric=False)
    assert p == ["Huevos"] and f == []


def test_filtro_unidad_100pct_curado_vacia_sin_penalty_de_goal():
    """Un goal neutro no penaliza procesados, pero el 100% curado sí vacía (regla universal)."""
    p, _ = ah._pantry_clinical_main_filter(
        ["Salami Dominicano", "Longaniza"], [],
        penaliza_procesados=False, exige_densidad=False, is_bariatric=False)
    assert p == []


def test_filtro_unidad_goal_neutro_con_mezcla_no_toca_nada():
    """Fuera de la regla universal, un goal neutro conserva el comportamiento previo."""
    p, _ = ah._pantry_clinical_main_filter(
        ["Salami Dominicano", "Pollo"], [],
        penaliza_procesados=False, exige_densidad=False, is_bariatric=False)
    assert p == ["Salami Dominicano", "Pollo"]


def test_filtro_unidad_bariatrica_quesos_y_frutas():
    p, f = ah._pantry_clinical_main_filter(
        ["Queso de Freír", "Salami Dominicano", "Huevos"], ["Piña", "Fresa"],
        penaliza_procesados=True, exige_densidad=True, is_bariatric=True)
    assert p == ["Huevos"]
    assert f == ["Fresa"]


def test_filtro_unidad_frutas_solo_con_perfil_bariatrico():
    """El espejo de frutas NO se dispara por goal: es exclusivo del perfil bariátrico."""
    _, f = ah._pantry_clinical_main_filter(
        [], ["Piña", "Fresa"],
        penaliza_procesados=True, exige_densidad=True, is_bariatric=False)
    assert f == ["Piña", "Fresa"]


def test_filtro_unidad_nunca_reintroduce_ni_reordena():
    """Alergia/dieta mandan: el filtro es un SUBCONJUNTO ordenado de lo que recibe."""
    entrada = ["Pollo", "Salami Dominicano", "Pescado"]
    p, _ = ah._pantry_clinical_main_filter(
        entrada, [], penaliza_procesados=True, exige_densidad=False, is_bariatric=False)
    assert p == [x for x in entrada if x in p], "el filtro reordenó el pool"
    assert set(p).issubset(set(entrada)), "el filtro reintrodujo un alimento que no recibió"


def test_filtro_unidad_nunca_vacia_una_nevera_sana():
    p, f = ah._pantry_clinical_main_filter(
        ["Pollo", "Pescado"], ["Fresa"],
        penaliza_procesados=True, exige_densidad=True, is_bariatric=True)
    assert p == ["Pollo", "Pescado"] and f == ["Fresa"]


def test_filtro_unidad_leguminosa_sobrevive_sin_exigencia_de_densidad():
    """`lose_fat` penaliza embutidos pero el sorteo NO le reemplaza mains de baja densidad:
    el filtro no puede ser más estricto que la capa que replica."""
    p, _ = ah._pantry_clinical_main_filter(
        ["Lentejas", "Salami Dominicano"], [],
        penaliza_procesados=True, exige_densidad=False, is_bariatric=False)
    assert p == ["Lentejas"]


# ═══════════════ 6 · knob ═══════════════

def test_knob_registrado_con_default_true():
    from knobs import _KNOBS_REGISTRY
    row = _KNOBS_REGISTRY.get("MEALFIT_PANTRY_FLOOR_CLINICAL_FILTER")
    assert row is not None, "el knob debe auto-registrarse vía _env_bool (no os.environ crudo)"
    assert row["default"] is True


def test_knob_off_restaura_el_comportamiento_previo(monkeypatch):
    """Rollback sin redeploy: con el knob apagado, la nevera de embutidos vuelve a bloquear."""
    monkeypatch.setattr(ah, "PANTRY_FLOOR_CLINICAL_FILTER", False)
    out = _prompt(["Salami Dominicano", "Longaniza", "Repollo"], mainGoal="gain_muscle")
    assert _LOCK in out.upper(), (
        "con el knob OFF debe reaparecer el comportamiento previo (si no, el knob es cosmético)")


# ═══════════════ 7 · estructural ═══════════════

def test_el_filtro_corre_antes_del_minimo_de_rotacion():
    """Decisión de diseño: un pool que queda corto tras filtrar debe caer por la rama EXISTENTE
    (`< _min_p`: nevera primero + sorteo completa, sin lock), no por una rama nueva."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "ai_helpers.py"), encoding="utf-8") as f:
        src = f.read()
    i = src.index("ROTATION MODE")
    blk = src[i:src.index("FORCED INGREDIENT INJECTION", i)]
    assert "P1-PANTRY-FLOOR-CLINICAL-FILTER" in blk, "marker inline ausente del bloque de nevera"
    assert blk.index("_pantry_clinical_main_filter") < blk.index("_min_p = PANTRY_ROTATION_MIN"), (
        "el filtro debe correr ANTES del chequeo del mínimo de rotación")


def test_el_matcher_es_word_boundary_y_los_tokens_son_derivados():
    """SSOT: el conjunto curado/embutido se DERIVA de las dos listas que ya existen; una cuarta
    lista escrita a mano drifearía (patrón documentado del repo)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "ai_helpers.py"), encoding="utf-8") as f:
        src = f.read()
    i = src.index("def _token_matches_wb")
    cuerpo = src[i:src.index("\ndef ", i + 10)]
    assert r"re.search(r'\b' + re.escape(" in cuerpo, (
        "el matcher debe ser word-boundary (patrón canónico del repo), no subcadena")
    j = src.index("_CURED_OR_PROCESSED_TOKENS = ")
    win = src[j:j + 300]
    assert "_SALT_CURED_PROTEIN_TOKENS" in win and "_PROCESSED_MEAT_KEYWORDS" in win, (
        "el set curado/embutido debe derivarse de los dos SSOT existentes, no escribirse a mano")


def test_el_filtro_no_re_deriva_las_condiciones_del_sorteo():
    """El filtro recibe las decisiones; las condiciones viven UNA sola vez, junto al penalty del
    sorteo que replican. Si el filtro volviera a derivarlas tendríamos la misma regla clínica
    escrita en dos capas — el drift que este P-fix cierra."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "ai_helpers.py"), encoding="utf-8") as f:
        src = f.read()
    i = src.index("def _pantry_clinical_main_filter")
    cuerpo = src[i:src.index("\n# [P1-FRUIT-SEEDER-GATE-CONTRACT", i)]
    assert "_GOALS_PENALIZE_PROCESSED" not in cuerpo, (
        "el filtro no debe re-derivar qué goals penalizan procesados")
    assert 'main_goal == "gain_muscle"' not in cuerpo, (
        "el filtro no debe re-derivar la condición de densidad del sorteo")
