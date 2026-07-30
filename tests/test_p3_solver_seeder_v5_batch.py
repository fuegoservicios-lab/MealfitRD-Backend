"""[P3-SOLVER-SEEDER-V5 · 2026-07-30] Los P3 del audit solver+seeder v5.

Misma convención de batch por oleada que `test_p2_solver_seeder_v5_batch.py`.

  21 P3-FEASIBILITY-NO-CARRIER     el guard `_mx > 0` ocultaba la infactibilidad MÁS fuerte
  22 P3-REFINE-OUTOFBOUNDS-INWARD  una línea fuera de [lo,hi] quedaba congelada en AMBAS direcciones
  23 P3-EXEMPT-KNOBS-INDEPENDENT   apagar un knob desactivaba en silencio al otro
  25 P3-REFINE-EXCEPT-LOGGED       el `except` exterior del refinador era mudo
  26 P3-SWAP-NUMMEALS-SAMEDAY      el fallback ignoraba la estructura REAL del día que edita
  27 P3-FORCED-ALLOWED-CLEANUP     rama muerta (0 productores) con 3 defectos dentro
  28 P3-GROCERY-LOCK-REFILTER      el lock restauraba bases sin re-filtrar alergias actuales
  29 P3-MICRO-SEED-MARKER-LIST     marker ESCALAR: la 2ª siembra pisaba a la 1ª
  30 P3-REINFORCE-CEILING-GUARD    el refuerzo era el 4º path sin el guard clínico
  35 P3-CHUNK1-LESSON-HONEST       lessons que certifican 0 violaciones sin medirlas
  36 P3-DISHLIB-MERIENDA-SLOTS     la biblioteca de platos saltaba las 3 meriendas

**Tarea 24 (P3-LSQ-FIXED-CONTRIBUTION) — CERRADA COMO REFUTADA POR MEDICIÓN.**
El hallazgo era: el LSQ apunta al target COMPLETO sin restar la contribución FIJA de los entries
resolubles-SIN-grupo (kcal>0 con P/C/F todos 0, que el solver nunca escala) ⇒ sobre-dispararía
kcal por construcción. El SELECT decisivo sobre `master_ingredients` (2026-07-30) devolvió
**0 filas** con `kcal_per_100g > 5` y los tres macros en cero, sobre 204 filas con kcal. O sea:
un entry sin grupo solo puede ser tipo-agua (kcal≈0) y su contribución fija es despreciable.
Sin filas que lo produzcan, no hay sesgo que restar. Si algún día el catálogo gana una fila así
(un edulcorante o un alcohol con kcal y macros nulos), el gap se reabre — por eso queda escrito
aquí y no solo en el plan.

La deuda de tests (31/32/33/34) se salda EN los archivos afectados, no aquí.
"""
from __future__ import annotations

import logging
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)


def _read(rel: str) -> str:
    with open(os.path.join(_BACKEND, *rel.split("/")), encoding="utf-8") as f:
        return f.read()


_PS = _read("portion_solver.py")
_GO = _read("graph_orchestrator.py")
_AH = _read("ai_helpers.py")
_AG = _read("agent.py")
_PL = _read("routers/plans.py")
_CT = _read("cron_tasks.py")
_DL = _read("dish_library.py")


# ═════════════ 21 · P3-FEASIBILITY-NO-CARRIER ═════════════

def _entry(macros: dict, group: str | None = None) -> dict:
    return {"s": "x", "macros": macros, "group": group}


def test_21_macro_sin_portador_se_marca():
    """El guard `if _mx > 0 and t > _mx` ocultaba justo el caso más grave.

    Con CERO entries aportando el macro, `_mx` queda en 0.0 → la rama 'high' no entra (por el
    guard) y la rama 'low' tampoco (`t < _mn` es falso con `_mn=0`). O sea: el macro para el que
    NO existe ningún portador en el plato pasaba como factible, que es exactamente la distinción
    que esta señal existe para hacer ("falta escalado" vs "falta un PORTADOR")."""
    import portion_solver as ps
    # merienda tipo gelatina + claras: todas las líneas con fats=0.00 tras el round del catálogo
    entries = [_entry({"kcal": 60, "protein": 12, "carbs": 2, "fats": 0.0}, "protein"),
               _entry({"kcal": 40, "protein": 0, "carbs": 10, "fats": 0.0}, "carbs")]
    rep = ps._feasibility_report(entries, {"fats": 6.0}, 0.3, [3.5, 3.5])
    assert rep and rep.get("fats"), (
        "un macro con target>0 y CERO portadores debe marcarse infactible, no pasar como OK")


def test_21_distingue_no_carrier_de_falta_de_escalado():
    """La señal tiene que separar los dos diagnósticos: sin portador NO se arregla escalando."""
    import portion_solver as ps
    sin_portador = ps._feasibility_report(
        [_entry({"kcal": 60, "protein": 12, "carbs": 2, "fats": 0.0}, "protein")],
        {"fats": 6.0}, 0.3, [3.5])
    con_portador_corto = ps._feasibility_report(
        [_entry({"kcal": 60, "protein": 0, "carbs": 0, "fats": 1.0}, "fats")],
        {"fats": 99.0}, 0.3, [3.5])
    assert sin_portador.get("fats") == "no_carrier"
    assert con_portador_corto.get("fats") == "high"


def test_21_caso_factible_sigue_limpio():
    """Regresión inversa: un target alcanzable no puede empezar a marcarse infactible."""
    import portion_solver as ps
    rep = ps._feasibility_report(
        [_entry({"kcal": 60, "protein": 0, "carbs": 0, "fats": 10.0}, "fats")],
        {"fats": 12.0}, 0.3, [3.5])
    assert not rep.get("fats")


# ═════════════ 22 · P3-REFINE-OUTOFBOUNDS-INWARD ═════════════

class _RefDB:
    def macros_from_ingredient_string(self, s: str):
        import re
        m = re.match(r"\s*([\d.]+)\s*g\b", str(s))
        g = float(m.group(1)) if m else 0.0
        return {"kcal": 1.30 * g, "protein": .027 * g, "carbs": .28 * g, "fats": .003 * g}


def test_22_linea_sobre_el_cap_puede_bajar():
    """`lo`/`hi` se computan desde `orig` y el test evaluaba SOLO el punto DESTINO, así que una
    línea que ARRANCA fuera del intervalo quedaba congelada en las DOS direcciones: con
    orig=400 y cap_g=300, `hi=300`, y tanto 395 como 405 caen fuera ⇒ `continue`.

    El refinador perdía su palanca justo en la línea más grande del día — y esa línea seguía
    sobre el cap también para el usuario."""
    import portion_solver as ps
    meal = {"ingredients": ["400 g de arroz"], "protein": 11, "carbs": 112, "fats": 1, "cals": 520}
    moves = ps.refine_day_portions_integer(
        [meal], {"kcal": 300, "protein": 6, "carbs": 65, "fats": 1},
        _RefDB(), step_g=5.0, floor_g=15.0, cap_g=300.0, max_iters=60)
    gramos = float(meal["ingredients"][0].split()[0])
    assert moves > 0, "la línea sobre el cap quedó congelada: el refinador no tuvo palanca"
    assert gramos < 400.0, f"debió poder BAJAR hacia el intervalo, quedó en {gramos}"
    # El intervalo válido de esta línea es [max(floor, 0.5·orig), min(cap, 2·orig)] = [200, 300].
    # Una vez DENTRO, el refinador sigue optimizando el error hacia el target — que aquí es bajo —
    # y eso es lo correcto: el bound no es un imán, es una frontera. Lo que se prueba es que la
    # línea deja de estar fuera y nunca cruza el piso.
    assert 200.0 - 1e-6 <= gramos <= 300.0 + 1e-6, f"quedó fuera del intervalo válido: {gramos}"


def test_22_linea_dentro_del_rango_se_comporta_igual():
    """Regresión inversa: desde dentro del intervalo el comportamiento es byte-idéntico."""
    import portion_solver as ps
    meal = {"ingredients": ["200 g de arroz"], "protein": 5, "carbs": 56, "fats": 1, "cals": 260}
    ps.refine_day_portions_integer(
        [meal], {"kcal": 200, "protein": 4, "carbs": 43, "fats": 1},
        _RefDB(), step_g=5.0, floor_g=15.0, cap_g=300.0, max_iters=60)
    g = float(meal["ingredients"][0].split()[0])
    assert 100.0 <= g <= 300.0, g


# ═════════════ 23 · P3-EXEMPT-KNOBS-INDEPENDENT ═════════════

def test_23_los_dos_knobs_son_independientes(monkeypatch):
    """El comentario del fix promete "rollback independiente por knob" y el código lo contradecía:
    la rama `if not REFINE_EXEMPT_WORD_BOUNDARY` construía el matcher con SOLO los tokens del
    caller, y el append de `_MICRO_CARRIER_TOKENS` vivía únicamente en la rama `else`.

    Consecuencia: un operador que apagara el word-boundary (rollback documentado) desactivaba
    TAMBIÉN, en silencio, la protección de portadores de micros — el refinador volvía a poder
    recortar la linaza sembrada por el closer, deshaciendo cierres de micros sin un solo log,
    mientras /health/version seguía mostrando PROTECT_MICRO_CARRIERS en true."""
    import portion_solver as ps
    monkeypatch.setattr(ps, "REFINE_EXEMPT_WORD_BOUNDARY", False)
    monkeypatch.setattr(ps, "REFINE_PROTECT_MICRO_CARRIERS", True)
    ps._EXEMPT_RE_CACHE.clear()
    fn = ps._exempt_matcher(("aceite",))
    assert fn("2 cdas de linaza molida"), (
        "con PROTECT_MICRO_CARRIERS=true la linaza debe seguir exenta aunque el "
        "word-boundary esté apagado")
    ps._EXEMPT_RE_CACHE.clear()


def test_23_protect_off_deja_de_proteger(monkeypatch):
    """La otra dirección: apagar la protección SÍ debe dejar de eximir portadores."""
    import portion_solver as ps
    monkeypatch.setattr(ps, "REFINE_PROTECT_MICRO_CARRIERS", False)
    ps._EXEMPT_RE_CACHE.clear()
    fn = ps._exempt_matcher(("aceite",))
    assert not fn("2 cdas de linaza molida")
    assert fn("1 cda de aceite de oliva"), "los tokens del caller siguen vigentes"
    ps._EXEMPT_RE_CACHE.clear()


# ═════════════ 25 · P3-REFINE-EXCEPT-LOGGED ═════════════

def test_25_el_failsafe_del_refinador_no_es_mudo(monkeypatch, caplog):
    """`except Exception: return 0` sin log. Un fallo estructural (p.ej. el NameError-en-helper-
    recién-extraído que ya mordió dos veces en este repo) apagaría el ÚLTIMO optimizador de la
    banda all-4 en generación Y updates, y cada llamada retornaría 0 en silencio: la banda de la
    flota degradaría sin una sola línea de diagnóstico."""
    import portion_solver as ps

    def _boom(*a, **k):
        raise RuntimeError("refinador roto")

    monkeypatch.setattr(ps, "_refine_error", _boom)
    caplog.set_level(logging.WARNING, logger=ps._log.name)
    meal = {"ingredients": ["200 g de arroz"], "protein": 5, "carbs": 56, "fats": 1, "cals": 260}
    assert ps.refine_day_portions_integer(
        [meal], {"kcal": 200, "protein": 4, "carbs": 43, "fats": 1}, _RefDB()) == 0
    hits = [r for r in caplog.records
            if r.levelno == logging.WARNING and "REFINE" in r.getMessage()]
    assert hits, "el fail-safe debe dejar traza: un motor apagado es indistinguible de uno inerte"
    assert "RuntimeError" in hits[0].getMessage(), hits[0].getMessage()


# ═════════════ 26 · P3-SWAP-NUMMEALS-SAMEDAY ═════════════

def test_26_el_fallback_usa_la_estructura_real_del_dia():
    """El fallback deriva `num_meals` del PERFIL (`decide_meals_per_day`) ignorando que el mismo
    `form_data` ya trae la estructura REAL del día 40 líneas más abajo (`same_day_other_meals`,
    poblado por el caller para la variedad intra-día).

    Tres vectores en los que el perfil miente sobre el plan vivo:
      (a) el usuario eligió num_meals explícito en el formulario de generación y el plan se
          enforzó a ese conteo, pero el form del swap no lo trae;
      (b) la regla de alto gasto depende del kcal, que el callsite no pasa;
      (c) el perfil cambió DESPUÉS de generar el plan (registra bypass gástrico → 6).
    En los tres, `allocate_macros_per_slot` reparte con N distinto del real y el solver
    re-escala el plato a la cuota equivocada."""
    # Anclar al bloque INLINE, no al comentario del knob: `[P2-SWAP-NUM-MEALS` aparece también
    # en la definición del knob (muy arriba del archivo) y un `.index()` sobre el marker pelado
    # aterriza ahí, produciendo un recorte de 26k que abarca medio módulo. Es la colisión por
    # prefijo compartido que ya costó un rojo en este repo: se ancla a la línea que asigna.
    i = _AG.index("_num8 = int(form_data.get(\"num_meals\")")
    blk = _AG[i:_AG.index("_slots8 = ", i)]
    assert "_num_meals_from_same_day" in blk, (
        "el fallback debe preferir el ground-truth del día que se edita sobre la derivación "
        "del perfil")
    assert blk.index("_num_meals_from_same_day") < blk.index("decide_meals_per_day"), (
        "la estructura real del día va ANTES de la derivación por perfil")


def test_26_conteo_derivado_del_dia(monkeypatch):
    import agent
    assert agent._num_meals_from_same_day({"same_day_other_meals": ["a", "b", "c"]}) == 4
    assert agent._num_meals_from_same_day({"same_day_other_meals": []}) is None
    assert agent._num_meals_from_same_day({}) is None
    assert agent._num_meals_from_same_day(None) is None


# ═════════════ 27 · P3-FORCED-ALLOWED-CLEANUP ═════════════

def test_27_la_rama_forced_base_proteins_ya_no_existe():
    """Rama con CERO productores en todo el repo (ni backend ni frontend emiten
    `_force_base_proteins`) y tres defectos dentro:
      · `if item_n in banned or banned in item_n` — doble substring: 'res' ⊂ 'ensalada fresca'
        y 'pollo' ⊂ 'repollo guisado' banean antes de llegar a las keywords;
      · los fallbacks restauran la lista forzada COMPLETA, bans legítimos incluidos, cuando el
        filtro deja menos del mínimo — filtro simultáneamente sobre-inclusivo y no-op;
      · con la lista vacía, el padding hace `% len([])` → ZeroDivisionError.
    Código muerto con bugs es deuda que solo espera a un caller nuevo para convertirse en
    incidente. Se elimina; si vuelve a hacer falta, se escribe con word-boundary desde el día 1."""
    # Se afirma sobre el CÓDIGO, no sobre el vocabulario: el comentario que documenta la
    # eliminación nombra `_force_base_proteins` a propósito (si no, el próximo lector no sabría
    # qué se quitó ni por qué). Un `not in` sobre el archivo entero fallaría por la propia doc.
    assert 'if form_data and "_force_base_proteins" in form_data:' not in _AH, (
        "la rama muerta sigue viva")
    assert "def _is_forced_allowed(" not in _AH, "el filtro con doble substring sigue definido"
    assert "P3-FORCED-ALLOWED-CLEANUP" in _AH, (
        "la eliminación debe quedar documentada donde estaba el bloque")


def test_27_el_seeder_sigue_funcionando_sin_la_rama():
    import ai_helpers as ah
    p = ah.get_deterministic_variety_prompt("", {"allergies": [], "dislikes": []}, user_id=None)
    assert isinstance(p, str) and p


# ═════════════ 28 · P3-GROCERY-LOCK-REFILTER ═════════════

def test_28_el_lock_reintersecta_con_los_filtros_actuales():
    """`MEALFIT_GROCERY_CYCLE_LOCK` (default OFF, rollback documentado del owner) restauraba
    `base_proteins/carbs/veggies` persistidos al inicio del ciclo SIN re-filtrarlos contra las
    alergias/dislikes/dieta de HOY — y encima el prompt añade "REGLA DE AHORRO EXTREMA… usa
    EXACTAMENTE las proteínas asignadas".

    Escenario: ciclo mensual que arrancó con ['Pollo','Salmón']; al día 10 el usuario añade
    alergia a pescado. El lock re-impone Salmón como base OBLIGATORIA durante el resto del
    ciclo → el allergen-guard lo convierte en rechazo→retry en cada regeneración."""
    i = _AH.index("GROCERY CYCLE LOCK] Reutilizando")
    blk = _AH[_AH.rindex("cycle_locked = True", 0, i):i + 400]
    assert "filtered_proteins" in blk, (
        "el restore del lock debe intersectar con el pool filtrado de HOY, no confiar en el "
        "que se persistió al inicio del ciclo")


def test_28_interseccion_conserva_lo_permitido():
    import ai_helpers as ah
    assert ah._intersect_cycle_base(["Pollo", "Salmón"], ["Pollo", "Res"]) == ["Pollo"]
    # sin interseccion utilizable → se conserva el sorteo ya computado (fail-open, nunca vacío)
    assert ah._intersect_cycle_base(["Salmón"], ["Pollo", "Res"]) is None
    assert ah._intersect_cycle_base([], ["Pollo"]) is None
    assert ah._intersect_cycle_base(None, ["Pollo"]) is None


# ═════════════ 29 · P3-MICRO-SEED-MARKER-LIST ═════════════

def test_29_el_marker_de_siembra_es_una_coleccion():
    """`meal["_micro_seed_applied"] = k` es un slot ESCALAR: si el mismo meal se siembra para un
    segundo micro (los pools tienen fallback explícito a meals YA sembradas cuando no quedan
    candidatas sin sembrar), la 2ª asignación PISA a la 1ª.

    Consecuencias: en una re-entrada del closer (swap-persist / chat-modify, que persisten el
    marker en plan_data) `_already_seeded` da False para el micro pisado ⇒ se siembra una
    SEGUNDA línea del mismo portador en vez de entrar al refuerzo; y el refuerzo post-erosión,
    que localiza su meal por `== k`, nunca vuelve a encontrarla."""
    import graph_orchestrator as go
    m = {}
    go._mark_micro_seed(m, "vit_a_mcg")
    go._mark_micro_seed(m, "omega3_g")
    assert go._has_micro_seed(m, "vit_a_mcg"), "la 2ª siembra pisó el marker de la 1ª"
    assert go._has_micro_seed(m, "omega3_g")
    assert not go._has_micro_seed(m, "vit_e_mg")


def test_29_lee_el_marker_escalar_legacy():
    """Planes ya persistidos traen el escalar: el shim debe reconocerlo o el anti-doble-siembra
    se apagaría para todo el parque existente."""
    import graph_orchestrator as go
    assert go._has_micro_seed({"_micro_seed_applied": "vit_a_mcg"}, "vit_a_mcg")
    assert not go._has_micro_seed({"_micro_seed_applied": "vit_a_mcg"}, "omega3_g")
    assert not go._has_micro_seed({}, "vit_a_mcg")
    assert not go._has_micro_seed(None, "vit_a_mcg")


def test_29_el_closer_usa_los_helpers():
    i = _GO.index("_already_seeded = any(")
    blk = _GO[i:_GO.index("_seed_trigger", i)]
    assert "_has_micro_seed" in blk, "el anti-doble-siembra sigue comparando el escalar por ="


# ═════════════ 30 · P3-REINFORCE-CEILING-GUARD ═════════════

def test_30_el_refuerzo_honra_el_guard_clinico():
    """`_ceiling_risky_contributor` aterrizó en 3 de los 4 paths que ESCALAN una línea (loop
    principal, fatswap y candidato de siembra) — el refuerzo post-erosión se quedó fuera.

    Población concreta: planes generados entre 2026-07-06 (cuando la espinaca entró como
    candidato de vit A) y 2026-07-29 (cuando nació el guard) para usuarios con warfarina que
    rechazan zanahoria y auyama. `_micro_seed_applied` persiste en plan_data, así que un swap
    re-entra al closer, encuentra el día aún corto y escala la espinaca ×1.8 — carga asimétrica
    de vitamina K en un solo día bajo anticoagulante, que es la definición operativa del riesgo
    que el guard existe para evitar. Los otros 3 paths sí la habrían saltado."""
    # Recorte por ORDEN RELATIVO entre dos anclas del PROPIO bloque de refuerzo: desde el
    # chequeo de idempotencia que lo abre (`_micro_seed_reinforced") == k`) hasta la asignación
    # que lo cierra. Un `rindex` sobre un prefijo suelto ('_reinforce') aterriza en la variable
    # local más cercana y produce un recorte demasiado corto — el test daría rojo con el guard
    # correctamente puesto 4.6k caracteres antes.
    ini = _GO.index('_micro_seed_reinforced") == k')
    fin = _GO.index('_micro_seed_reinforced"] = k', ini)
    blk = _GO[ini:fin]
    assert "_ceiling_risky_contributor" in blk, (
        "el refuerzo escala sin consultar el guard clínico de contribuyente")
    assert blk.index("_ceiling_risky_contributor") < blk.index("_g_new"), (
        "el guard debe evaluarse ANTES de calcular el nuevo gramaje, no después")


# ═════════════ 35 · P3-CHUNK1-LESSON-HONEST ═════════════

def test_35_la_lesson_no_certifica_cero_sin_medir():
    """`_calculate_learning_metrics(..., allergy_keywords=[], fatigued_ingredients=[])` con las
    listas HARDCODEADAS vacías, y la lesson resultante persiste `allergy_violations: 0` con
    `metrics_unavailable: False`.

    O sea: afirma haber medido y no midió. El chunk 2 lee esa lesson y concluye que el chunk
    anterior salió limpio, suprimiendo el refuerzo correctivo que la cadena de lessons existe
    para dar. Los 3 callsites del chunk worker sí pasan listas reales — la asimetría era solo
    de estos dos."""
    # Se afirma sobre el VALOR PERSISTIDO, no sobre el literal del argumento: la fatiga sigue
    # sin tener quién la alimente en este seed, así que la llamada conserva la lista vacía — lo
    # que NO puede seguir es que el campo persistido diga 0 (que significa "lo miré y estaba
    # limpio") cuando nadie lo miró. Un test anclado al literal del argumento habría exigido un
    # cambio cosmético y dejado el contrato igual de mentiroso.
    for src, nombre in ((_PL, "routers/plans.py"), (_CT, "cron_tasks.py")):
        assert "allergy_keywords=[]" not in src, (
            f"{nombre} sigue midiendo alergias contra una lista vacía hardcodeada")
        assert "violations_measured" in src, (
            f"{nombre} debe declarar si las violaciones se midieron de verdad")
    i = _PL.index("'allergy_violations':")
    blk = _PL[i:_PL.index("'chunk': 1", i)]
    assert "if _c1_measured else None" in blk, (
        "sin lista contra la que comparar, el campo va a None (no-medido), nunca a 0")
    assert "'fatigued_violations': None" in blk, (
        "nadie alimenta `fatigued_ingredients` en este seed: certificar 0 es afirmar sin medir")


# ═════════════ 36 · P3-DISHLIB-MERIENDA-SLOTS ═════════════

def test_36_la_biblioteca_cubre_las_meriendas():
    """`_SLOT_LABELS` tiene 'merienda' a secas y el slot real es 'Merienda AM'/'PM'/'Nocturna'
    (los que `MEAL_TYPES_BY_COUNT[5|6]` fuerza en TODO plan clínico) ⇒ `continue` y esas comidas
    se generan sin biblioteca de inspiración.

    La función hermana del MISMO módulo (`build_swap_inspiration_context`) sí tiene el fallback
    por substring: un swap de esa merienda recibía inspiración y la generación no. En un plan
    bariátrico son 3 de 6 slots — justo el perfil que más depende de meriendas bien diseñadas."""
    import dish_library as dl
    sk = {"meal_types": ["Desayuno", "Almuerzo", "Merienda AM", "Merienda Nocturna", "Cena"]}
    out = dl.build_dish_library_context(sk, 1)
    assert out, "sanity: la biblioteca debe emitir algo para este esqueleto"
    assert "Merienda" in out, (
        "las meriendas con calificativo se saltan: el day-gen las compone sin inspiración")


def test_36_slots_sin_equivalente_siguen_ignorados():
    """Regresión inversa: el fallback es por substring de un slot conocido, no un cajón abierto."""
    import dish_library as dl
    out = dl.build_dish_library_context({"meal_types": ["Pre-entreno"]}, 1)
    assert not out or "Pre-entreno" not in out


# ═════════════ 33 · P3-AUDIT-V5-TESTDEBT-FATIGUE ═════════════

def test_33_la_fatiga_por_recencia_tiene_cobertura(monkeypatch):
    """`_apply_recency_fatigue` — la mitad "recencia" del anti mode-collapse — tenía CERO tests
    con aserciones, ni directos ni transitivos: todos los callers del seeder en la suite pasan
    `user_id=None` o `"guest"`, así que la rama nunca se ejecutaba.

    Y su `except` degrada a `freq_map` sin fatiga con un solo `logger.warning`: si un refactor
    de las claves de frecuencia (como el que P2-FREQ-LOOKUP-CANONICAL acaba de hacer) rompiera
    el solape, el boost de recencia quedaría en 0 permanente para TODOS los usuarios logueados
    con la suite entera en verde. Un motor que no recibe datos es indistinguible de uno apagado."""
    import ai_helpers as ah
    monkeypatch.setattr(ah, "get_user_ingredient_frequencies",
                        lambda uid, days_limit=None: ({"pollo": 2} if days_limit == 3
                                                      else {"pollo": 3, "arroz": 1}))
    out = ah._apply_recency_fatigue({"pollo": 10.0, "arroz": 4.0, "mero": 1.0}, "u1")
    assert out["pollo"] == pytest.approx(10.0 + 2 * 3.0 + 3 * 1.5)   # 3d ×3 + 7d ×1.5
    assert out["arroz"] == pytest.approx(4.0 + 0 + 1 * 1.5)
    assert out["mero"] == pytest.approx(1.0), "sin uso reciente, la frecuencia no se toca"


def test_33_la_fatiga_degrada_con_traza(monkeypatch, caplog):
    import ai_helpers as ah

    def _boom(*a, **k):
        raise RuntimeError("DB down")

    monkeypatch.setattr(ah, "get_user_ingredient_frequencies", _boom)
    caplog.set_level(logging.WARNING, logger=ah.logger.name)
    base = {"pollo": 10.0}
    assert ah._apply_recency_fatigue(base, "u1") == base, "degrada al mapa sin fatiga"
    assert [r for r in caplog.records if "FATIGUE" in r.getMessage()], (
        "la degradación debe dejar traza o el motor apagado es invisible")


@pytest.mark.parametrize("uid", [None, "", "guest"])
def test_33_early_returns_no_consultan_la_db(monkeypatch, uid):
    """Guest / sin user: passthrough SIN tocar la query (es el camino más caliente del seeder)."""
    import ai_helpers as ah
    llamadas = {"n": 0}
    monkeypatch.setattr(ah, "get_user_ingredient_frequencies",
                        lambda *a, **k: llamadas.__setitem__("n", llamadas["n"] + 1) or {})
    base = {"pollo": 1.0}
    assert ah._apply_recency_fatigue(base, uid) == base
    assert llamadas["n"] == 0


def test_33_freq_map_vacio_es_passthrough(monkeypatch):
    import ai_helpers as ah
    llamadas = {"n": 0}
    monkeypatch.setattr(ah, "get_user_ingredient_frequencies",
                        lambda *a, **k: llamadas.__setitem__("n", llamadas["n"] + 1) or {})
    assert ah._apply_recency_fatigue({}, "u1") == {}
    assert llamadas["n"] == 0
