"""[P3-SEEDER-TEMPLATE-COVERAGE · 2026-08-04] (audit solver+seeder v7 · Task 19)

El seeder repartía bases sin saber cuántas variantes de plato REALES soporta cada una.

`dish_library` ofrece al day-generator ~87 plantillas dominicanas curadas, pero
`_protein_matches_pool` sólo deja pasar las que NOMBRAN la base asignada (más las de clase
genérica, que aplican a todo pool). Una base como Chivo / Conejo / Pulpo no aparece en NINGUNA
plantilla de almuerzo: si el sorteo se la impone —y bajo nevera/lock puede ocupar 2-3 días del
chunk— el day-gen compone esos platos SIN recombinación y tiende a la fórmula clonada que el
detector de P1-CONTAINER-SERVABLE mide *después*.

El audit lo ajustó a la baja a propósito: las plantillas de clase genérica ('none'/'mixta')
aplican siempre, así que esto es **degradación suave, no bloqueo**. Por eso el cierre es un
multiplicador ×0.5 sobre el peso del sorteo — jamás una exclusión— más telemetría.

Contrato que ancla este archivo:

  1. **El mapa se DERIVA del JSON vivo** (`data/dish_templates.json` vía `dish_library`), no de
     una lista a mano: si mañana entra una plantilla de chivo, la cobertura sube sola y este
     test lo nota (patrón live-anchor de P3-MICRO-CARRIER-LIVE-ANCHOR).
  2. **Knob** `MEALFIT_LOW_TEMPLATE_COVERAGE_PENALTY`: float, **default 1.0 (OFF)**, clamp
     [0.1, 1.0]. Registrado en `_KNOBS_REGISTRY` (visible en `/health/version`).

     [ronda 1 · 2026-08-04] El default pasó de 0.5 a 1.0: encenderlo compra −1,25 pp sobre la
     base descubierta y paga +15,5 pp de concentración en 15/49 proteínas, contra la preferencia
     declarada del dueño por variedad; y el metro tiene dos defectos abiertos (mide sólo
     `almuerzo` cuando el reparto dice «Almuerzo o Cena» ⇒ 12 falsos positivos medidos; la mitad
     carbo no tiene consumidor aguas abajo). Los tests de EFECTO de más abajo siguen midiendo el
     mecanismo con el knob forzado — apagado por default no significa sin probar.
  3. **El multiplicador es exacto y quirúrgico**: sólo toca las bases de cobertura 0, deja el
     resto byte-idéntico. Es un sesgo, no una redistribución.
  4. **Efecto MEDIDO end-to-end** sobre semillas fijas, no afirmado.
  5. **Con el knob en 1.0 el sorteo es idéntico al de la biblioteca ausente** — o sea, el
     bloque queda demostrablemente inerte (equivalencia no-circular).
  6. **Jamás excluye**: ni con el knob en su mínimo (0.1) desaparece la base del sorteo.
  7. **WARNING** cuando una base extraída de la NEVERA con cobertura 0 va a ocupar ≥2 días.
  8. **Matching por LÍMITE DE PALABRA** (14+ incidentes de subcadena en este repo), con la
     trampa MEDIDA contra los 4 catálogos vivos, no elegida a ojo.
  9. **Fail-open**: sin biblioteca (archivo ausente/corrupto) no hay penalty y el seeder
     entrega su prompt igual.

Todo el archivo es OFFLINE: los lectores de DB del seeder van stubeados.
"""
from __future__ import annotations

import logging
import random
import re

import pytest

import ai_helpers as ah

MARKER = "P3-SEEDER-TEMPLATE-COVERAGE"
KNOB = "MEALFIT_LOW_TEMPLATE_COVERAGE_PENALTY"

# Nombres REALES del catálogo (`constants.DOMINICAN_*`).
CHIVO = "Chivo"          # sin ninguna plantilla de almuerzo (se verifica contra el JSON vivo)
CONEJO = "Conejo"        # idem
POLLO = "Pollo"          # la base mejor cubierta de la biblioteca
CAMARONES = "Camarones"  # control CUBIERTO y sin contaminación de alias en el lookup de freqs


# --------------------------------------------------------------------------------------
# Andamiaje offline (mismo patrón que test_p1_cycle_base_affinity)
# --------------------------------------------------------------------------------------
@pytest.fixture
def offline(monkeypatch):
    """Neutraliza los lectores/escritores de DB del seeder. Devuelve un setter de frecuencias."""
    monkeypatch.setattr(ah, "get_user_ingredient_frequencies",
                        lambda uid, days_limit=None: {}, raising=False)
    monkeypatch.setattr(ah, "_apply_recency_fatigue", lambda m, uid: m, raising=False)
    monkeypatch.setattr(ah, "get_user_profile", lambda uid: {"health_profile": {}}, raising=False)
    monkeypatch.setattr(ah, "update_user_health_profile_atomic",
                        lambda uid, mut: None, raising=False)
    monkeypatch.setattr(ah, "get_latest_meal_plan_with_id", lambda uid: None, raising=False)

    def _install(freqs=None):
        monkeypatch.setattr(ah, "get_user_ingredient_frequencies",
                            lambda uid, days_limit=None: dict(freqs or {}), raising=False)

    _install()
    return _install


def _seed_call(seed, **extra):
    fd = {
        "dietType": "ninguna",
        "mainGoal": "gain_muscle",   # salta la GARANTÍA de leguminosa (ruido ajeno a este fix)
        "allergies": [],
        "dislikes": [],
    }
    fd.update(extra)
    random.seed(seed)
    return ah.get_deterministic_variety_prompt("", fd, user_id="u-tpl")


_RE_PROT = re.compile(r"DEBE incluir obligatoriamente: (.+?) \+ ")


def _prot_picks(prompt):
    return _RE_PROT.findall(prompt)


def _rate(name, *, seeds, **kw):
    """Fracción de semillas en las que `name` sale como proteína base. La MEDICIÓN."""
    hits = 0
    for s in seeds:
        hits += name in _prot_picks(_seed_call(s, **kw))
    return hits / len(seeds)


# Fatiga fuerte todo el catálogo MENOS la pareja medida (objetivo + control), para que el sorteo
# se juegue entre esos dos y las tasas caigan en un rango medible (~40%) en vez de en la cola.
def _freqs_pareja_fresca(pareja=(CHIVO, CAMARONES), fresca=6, fatigada=40):
    from constants import DOMINICAN_PROTEINS, strip_accents
    return {strip_accents(p.lower()): (fresca if p in pareja else fatigada)
            for p in DOMINICAN_PROTEINS}


def _biblioteca_sintetica_sin(base):
    """Biblioteca donde TODA base del catálogo tiene plantilla propia MENOS `base`.

    Aísla el mecanismo: con la biblioteca VIVA la cobertura 0 la comparten 34 de las 49
    proteínas, así que penalizarlas a todas se auto-normaliza y ninguna cae ×0.5 (ver el test
    del efecto real, más abajo). Aquí el multiplicador toca UNA sola base y el efecto end-to-end
    se puede leer sin la dilución del denominador."""
    from constants import DOMINICAN_PROTEINS, DOMINICAN_CARBS, strip_accents
    return ([{"name": f"Plato de {p}", "slots": ["almuerzo"], "base": "none",
              "protein": strip_accents(p.lower())}
             for p in DOMINICAN_PROTEINS if p != base]
            + [{"name": f"Base {c}", "slots": ["almuerzo"], "protein": "none",
                "base": strip_accents(c.lower())} for c in DOMINICAN_CARBS])


SEEDS = range(200)


# --------------------------------------------------------------------------------------
# Contrato 1 — el mapa se deriva del JSON VIVO
# --------------------------------------------------------------------------------------
def _proteinas_de_almuerzo_vivas():
    """Set de proteínas del catálogo con ≥1 plantilla de ALMUERZO, derivado del JSON vivo."""
    from constants import DOMINICAN_PROTEINS
    return {p for p in DOMINICAN_PROTEINS if ah._template_coverage(p, "protein") > 0}


def test_marker_presente():
    import inspect
    assert MARKER in inspect.getsource(ah), "el marker inline debe vivir en el código"


def test_el_mapa_sale_del_json_vivo_y_no_de_una_lista_a_mano():
    """Sanity + live-anchor: la biblioteca existe, tiene plantillas de almuerzo y el conteo
    coincide EXACTAMENTE con recorrer el JSON a mano aquí."""
    from dish_library import load_dish_templates
    tpls = load_dish_templates()
    assert len(tpls) >= 50, "sanity: la biblioteca viva debe tener plantillas"

    esperado_pollo = sum(
        1 for t in tpls
        if "almuerzo" in (t.get("slots") or []) and str(t.get("protein") or "").lower() == "pollo")
    assert esperado_pollo > 0, "sanity: la biblioteca viva tiene plantillas de pollo al almuerzo"
    assert ah._template_coverage(POLLO, "protein") == esperado_pollo, (
        "la cobertura de Pollo debe ser exactamente las plantillas de pollo del slot principal")


def test_chivo_y_conejo_no_tienen_plantilla_de_almuerzo_hoy():
    """Si mañana entra una plantilla de chivo esto falla y hay que elegir otro caso — que es
    justo lo que se quiere de un anclaje vivo (no un literal que envejece en silencio)."""
    vivas = _proteinas_de_almuerzo_vivas()
    assert POLLO in vivas, "sanity: el caso positivo debe estar dentro"
    assert CHIVO not in vivas and CONEJO not in vivas, (
        f"'{CHIVO}'/'{CONEJO}' ya tienen plantilla de almuerzo: el caso dejó de ser representativo")


def test_la_biblioteca_no_deja_sin_cobertura_a_las_leguminosas_de_la_GARANTIA():
    """Anti-oscilación: la 'GARANTÍA NUTRICIONAL' IMPONE una leguminosa como proteína principal.
    Si este penalty las castigara a la vez, serían dos guardas sobre el mismo campo tirando en
    direcciones opuestas (el modo de fallo que este repo ya pagó varias veces).

    Las 5 leguminosas de esa garantía salen del SSOT `NUTRITIONAL_CATEGORIES['legumbres']`, que es
    exactamente el mismo conjunto que `LEGUME_NAMES` del seeder — 'Habichuelas Blancas' está fuera
    de AMBOS (omisión documentada en `_LOW_DENSITY_AS_MAIN`), así que no hay contradicción."""
    from constants import NUTRITIONAL_CATEGORIES
    sin_cobertura = [x for x in NUTRITIONAL_CATEGORIES["legumbres"]
                     if ah._template_coverage(x, "protein") == 0]
    assert sin_cobertura == [], (
        f"leguminosas que la GARANTÍA impone y este penalty castiga: {sin_cobertura}")


# --------------------------------------------------------------------------------------
# Contrato 2 — knob
# --------------------------------------------------------------------------------------
def test_knob_default(monkeypatch):
    """[ronda 1 · 2026-08-04] NACE APAGADO. Ver el contrato 2 del docstring del módulo."""
    monkeypatch.delenv(KNOB, raising=False)
    assert ah._low_template_coverage_penalty() == 1.0


@pytest.mark.parametrize("raw,esperado", [
    ("0.0", 1.0),      # bajo el clamp → default (= OFF)
    ("0.05", 1.0),     # bajo el clamp → default (= OFF)
    ("1.5", 1.0),      # sobre el clamp → default (= OFF)
    ("basura", 1.0),   # no-parseable → default (= OFF)
    ("1.0", 1.0),      # el OFF explícito
    ("0.1", 0.1),      # el mínimo
    ("0.8", 0.8),
])
def test_knob_clamp(monkeypatch, raw, esperado):
    monkeypatch.setenv(KNOB, raw)
    assert ah._low_template_coverage_penalty() == esperado


def test_knob_registrado_en_el_registry(offline):
    """Un knob invisible en /health/version no existe para el operador durante un incidente."""
    from knobs import get_knobs_registry_snapshot
    _seed_call(1)
    assert KNOB in get_knobs_registry_snapshot()


# --------------------------------------------------------------------------------------
# Contrato 3 — el multiplicador es exacto y quirúrgico (determinista, sin sorteo)
# --------------------------------------------------------------------------------------
def test_el_multiplicador_solo_toca_las_bases_sin_plantilla():
    nombres = [POLLO, CHIVO, "Res"]
    pesos = [1.0, 1.0, 0.25]

    intacto, n = ah._apply_low_template_coverage_penalty(nombres, pesos, "protein", 1.0)
    assert intacto == pesos and n == 0, "factor 1.0 = no-op"

    out, n = ah._apply_low_template_coverage_penalty(nombres, pesos, "protein", 0.5)
    assert n == 1, f"sólo Chivo debía penalizarse, se penalizaron {n}"
    assert out[0] == pesos[0] and out[2] == pesos[2], "las cubiertas quedan INTACTAS"
    assert out[1] == pytest.approx(0.5), "la base sin plantillas baja ×0.5"


def test_el_multiplicador_tambien_pesa_carbos():
    """La decisión cubre proteínas Y carbos (el campo `base` de la plantilla)."""
    assert ah._template_coverage("Arroz Blanco", "base") > 0
    assert ah._template_coverage("Quinoa", "base") == 0
    out, n = ah._apply_low_template_coverage_penalty(
        ["Arroz Blanco", "Quinoa"], [1.0, 1.0], "base", 0.5)
    assert n == 1 and out == [1.0, 0.5]


# --------------------------------------------------------------------------------------
# Contrato 4 — efecto MEDIDO end-to-end
# --------------------------------------------------------------------------------------
def test_la_base_sin_plantillas_pierde_la_mitad_de_probabilidad(offline, monkeypatch):
    """Medición, no afirmación: con el multiplicador aislado a UNA base, la caída end-to-end es
    ≈×0.5 (medido 0.46 sobre 200 semillas, 0.50 sobre 400; ver el reporte del lote).

    Banda TOLERANTE a propósito: el sorteo es sin reemplazo (`random.choices` + descarte del
    elegido), así que halvar el peso no halva EXACTAMENTE la probabilidad — comprime hacia 1."""
    import dish_library as dl
    monkeypatch.setattr(dl, "load_dish_templates",
                        lambda _s=_biblioteca_sintetica_sin(CHIVO): _s, raising=False)
    offline(_freqs_pareja_fresca())
    assert ah._template_coverage(CHIVO, "protein") == 0
    assert ah._template_coverage(CAMARONES, "protein") > 0

    monkeypatch.setenv(KNOB, "1.0")
    off = _rate(CHIVO, seeds=SEEDS)
    monkeypatch.setenv(KNOB, "0.5")
    on = _rate(CHIVO, seeds=SEEDS)

    assert off > 0.15, f"sanity: sin penalty la base debe salir a menudo ({off:.1%})"
    assert 0.32 <= on / off <= 0.70, (
        f"la caída debe ser ≈×0.5: off={off:.1%} on={on:.1%} razón={on / off:.3f}")


def test_con_la_biblioteca_VIVA_el_efecto_es_relativo_no_una_caida_a_la_mitad(offline,
                                                                             monkeypatch):
    """Lo que el fix hace HOY, medido y escrito para no prometer de más.

    Con el catálogo real, 34 de las 49 proteínas y 15 de los 26 carbos tienen cobertura 0: como
    el sorteo NORMALIZA los pesos, penalizarlas a todas se auto-compensa y la base sin plantillas
    apenas baja en términos absolutos (38,5% → 37,3% sobre 400 semillas). Lo que se mueve de
    verdad es la base CUBIERTA, que sube fuerte (41,5% → 57,0%). O sea: el efecto es un
    DESPLAZAMIENTO hacia las bases con plantilla, no un castigo a las que no la tienen.

    Si alguien reduce el conjunto de cobertura 0 (añadiendo plantillas), este test seguirá
    valiendo y el efecto absoluto se parecerá cada vez más al ×0.5 del test anterior."""
    offline(_freqs_pareja_fresca())

    monkeypatch.setenv(KNOB, "1.0")
    off_t, off_c = _rate(CHIVO, seeds=SEEDS), _rate(CAMARONES, seeds=SEEDS)
    monkeypatch.setenv(KNOB, "0.5")
    on_t, on_c = _rate(CHIVO, seeds=SEEDS), _rate(CAMARONES, seeds=SEEDS)

    assert off_c > 0 and on_c > 0, "sanity: el control debe salir sorteado"
    assert on_c > off_c * 1.10, (
        f"la base CUBIERTA debe ganar terreno: {off_c:.1%} → {on_c:.1%}")
    razon = (on_t / on_c) / (off_t / off_c)
    assert 0.45 <= razon <= 0.90, (
        f"la razón sin-plantilla/con-plantilla debe caer: off={off_t / off_c:.3f} "
        f"on={on_t / on_c:.3f} (×{razon:.3f})")


def test_nunca_excluye_ni_en_el_minimo_del_knob(offline, monkeypatch):
    """Contrato 6: es un sesgo, no un filtro. Ni con ×0.1 desaparece del sorteo."""
    offline(_freqs_pareja_fresca())
    monkeypatch.setenv(KNOB, "0.1")
    assert _rate(CHIVO, seeds=SEEDS) > 0.0, "el penalty NO puede convertirse en exclusión"


# --------------------------------------------------------------------------------------
# Contrato 5 — con el knob en 1.0 el bloque es demostrablemente inerte
# --------------------------------------------------------------------------------------
def test_knob_en_uno_equivale_a_no_tener_biblioteca(offline, monkeypatch):
    """Equivalencia NO circular: se compara el DEFAULT (sin env var) contra la biblioteca AUSENTE
    (el otro camino por el que el bloque no aplica). Si coinciden semilla a semilla, el fix entra
    a producción byte-idéntico al comportamiento previo.

    [ronda 1 · 2026-08-04] La rama de referencia BORRA la variable en vez de setearla a "1.0":
    así el test ancla «nace apagado» y no sólo «1.0 apaga». Si alguien devuelve el default a 0.5,
    esto falla — que es exactamente lo que debe pasar."""
    import dish_library as dl

    offline(_freqs_pareja_fresca())
    monkeypatch.delenv(KNOB, raising=False)
    con_knob_off = [_prot_picks(_seed_call(s)) for s in range(80)]

    monkeypatch.setenv(KNOB, "0.5")
    monkeypatch.setattr(dl, "load_dish_templates", lambda: [], raising=False)
    sin_biblioteca = [_prot_picks(_seed_call(s)) for s in range(80)]

    assert con_knob_off == sin_biblioteca, (
        "knob=1.0 debe ser exactamente 'como si el bloque no existiera'")


def test_con_el_knob_activo_el_sorteo_SI_se_mueve(offline, monkeypatch):
    """El gemelo del anterior: sin esto, `knob=1.0 == sin biblioteca` podría pasar porque el
    bloque entero fuera no-op (la clase de fallo P1-G: el modo `block` que no bloqueaba)."""
    offline(_freqs_pareja_fresca())
    monkeypatch.setenv(KNOB, "1.0")
    apagado = [_prot_picks(_seed_call(s)) for s in range(80)]
    monkeypatch.setenv(KNOB, "0.5")
    encendido = [_prot_picks(_seed_call(s)) for s in range(80)]
    assert apagado != encendido, "el penalty encendido tiene que mover el sorteo"


# --------------------------------------------------------------------------------------
# Contrato 7 — WARNING con base de NEVERA sin cobertura ocupando ≥2 días
# --------------------------------------------------------------------------------------
def test_warning_cuando_una_base_de_nevera_sin_plantillas_ocupa_dos_dias(offline, monkeypatch,
                                                                         caplog):
    """Dos proteínas de nevera (≥ `PANTRY_ROTATION_MIN_PROTEINS`) reemplazan el pool; el padding
    las cicla hasta cubrir los 3 días, así que una de las dos ocupa 2. Ninguna tiene plantilla de
    almuerzo ⇒ telemetría antes de decidir un guard más duro."""
    offline()
    monkeypatch.setenv(KNOB, "0.5")
    with caplog.at_level(logging.WARNING, logger="ai_helpers"):
        random.seed(7)
        ah.get_deterministic_variety_prompt(
            "", {"dietType": "ninguna", "mainGoal": "maintenance", "allergies": [], "dislikes": [],
                 "current_pantry_ingredients": ["2 lb de chivo", "1 conejo entero"]},
            user_id="u-tpl")
    msgs = [r.getMessage() for r in caplog.records if MARKER in r.getMessage()]
    assert msgs, f"no se emitió el WARNING de cobertura 0 desde la nevera: {caplog.text[-800:]}"
    assert any((CHIVO in m or CONEJO in m) for m in msgs), msgs


def test_sin_nevera_no_hay_warning(offline, monkeypatch, caplog):
    """La telemetría es sobre lo EXTRAÍDO DE LA NEVERA (donde la base se impone), no sobre
    cualquier base del sorteo — o el log se volvería ruido en cada plan."""
    offline()
    monkeypatch.setenv(KNOB, "0.5")
    with caplog.at_level(logging.WARNING, logger="ai_helpers"):
        for s in range(15):
            _seed_call(s)
    assert not [r for r in caplog.records if MARKER in r.getMessage()], (
        "el WARNING no puede dispararse sin nevera")


def test_una_base_de_nevera_CON_plantillas_no_dispara_el_warning(offline, monkeypatch, caplog):
    offline()
    monkeypatch.setenv(KNOB, "0.5")
    with caplog.at_level(logging.WARNING, logger="ai_helpers"):
        random.seed(7)
        ah.get_deterministic_variety_prompt(
            "", {"dietType": "ninguna", "mainGoal": "maintenance", "allergies": [], "dislikes": [],
                 "current_pantry_ingredients": ["2 lb de pollo", "1 lb de res"]},
            user_id="u-tpl")
    assert not [r for r in caplog.records if MARKER in r.getMessage()]


# --------------------------------------------------------------------------------------
# Contrato 8 — subcadena (14+ incidentes en este repo)
# --------------------------------------------------------------------------------------
def _cobertura_ingenua(base, field="protein", slot="almuerzo"):
    """El matcher INGENUO (`in` crudo) que este repo lleva 14+ incidentes pagando. Existe para
    PROBAR que el caso elegido es una trampa de verdad, no para afirmarlo."""
    from constants import strip_accents
    from dish_library import load_dish_templates
    _n = strip_accents(str(base).lower())
    return sum(1 for t in load_dish_templates()
               if slot in (t.get("slots") or [])
               and str(t.get(field) or "none").lower() not in ("none", "mixta")
               and str(t.get(field) or "").lower() in _n)


def _trampas_medidas_contra_los_catalogos():
    """Barrido de los 4 catálogos vivos × los tokens de clase vivos: devuelve los nombres donde
    el matcher ingenuo se traga una plantilla que el de límite de palabra rechaza."""
    from constants import (DOMINICAN_PROTEINS, DOMINICAN_CARBS, DOMINICAN_VEGGIES_FATS,
                           DOMINICAN_FRUITS)
    out = []
    for cat, field in ((DOMINICAN_PROTEINS, "protein"), (DOMINICAN_CARBS, "base"),
                       (DOMINICAN_VEGGIES_FATS, "protein"), (DOMINICAN_FRUITS, "protein")):
        for n in cat:
            if _cobertura_ingenua(n, field) > 0 and ah._template_coverage(n, field) == 0:
                out.append((n, field, _cobertura_ingenua(n, field)))
    return out


@pytest.mark.parametrize("nombre,field", [("Repollo", "protein"), ("Fresa", "protein")])
def test_no_hay_colision_por_subcadena(nombre, field):
    """`'pollo'` ⊂ `'re-POLLO'` y `'res'` ⊂ `'f-RES-a'`: dos nombres REALES del catálogo que
    heredarían decenas de plantillas ajenas bajo un matcher de subcadena."""
    ingenua = _cobertura_ingenua(nombre, field)
    assert ingenua > 0, (
        f"'{nombre}' ya no es trampa para el matcher ingenuo: el caso dejó de ser representativo")
    assert ah._template_coverage(nombre, field) == 0, (
        f"'{nombre}' heredó {ingenua} plantillas ajenas por subcadena")


def test_las_unicas_trampas_vivas_estan_fuera_de_proteinas_y_carbos():
    """Medición, no afirmación: el barrido completo dice exactamente DÓNDE muerde hoy la
    subcadena. Si aparece una en proteínas o carbos —las dos categorías que este penalty pesa—,
    hay que moverla al test parametrizado de arriba para probarla end-to-end."""
    alcanzables = [(n, f) for n, f, _ in _trampas_medidas_contra_los_catalogos()
                   if f in ("protein", "base") and n in _nombres_de_pools_pesados()]
    assert alcanzables == [], (
        f"colisión alcanzable en el pool que este fix pesa: {alcanzables}")


def _nombres_de_pools_pesados():
    from constants import DOMINICAN_PROTEINS, DOMINICAN_CARBS
    return set(DOMINICAN_PROTEINS) | set(DOMINICAN_CARBS)


# --------------------------------------------------------------------------------------
# Contrato 9 — fail-open
# --------------------------------------------------------------------------------------
def test_fail_open_sin_biblioteca(offline, monkeypatch):
    import dish_library as dl
    monkeypatch.setattr(dl, "load_dish_templates", lambda: [], raising=False)
    monkeypatch.setenv(KNOB, "0.5")
    out = _seed_call(1)
    assert isinstance(out, str) and len(out) > 100
    assert ah._template_coverage(CHIVO, "protein") == -1, (
        "sin biblioteca la cobertura es 'no sé' (-1), NUNCA 0 — un 0 penalizaría el pool entero")


def test_fail_open_si_la_biblioteca_revienta(offline, monkeypatch):
    import dish_library as dl

    def _boom():
        raise RuntimeError("JSON corrupto a propósito")

    monkeypatch.setattr(dl, "load_dish_templates", _boom, raising=False)
    monkeypatch.setenv(KNOB, "0.5")
    out = _seed_call(1)
    assert isinstance(out, str) and len(out) > 100


# --------------------------------------------------------------------------------------
# Paridad con el matcher de la biblioteca (SSOT único)
# --------------------------------------------------------------------------------------
def test_paridad_con_protein_matches_pool_en_los_tokens_especificos():
    """El matcher de este fix debe coincidir con el de `dish_library` en TODOS los tokens que
    nombran un alimento — si divergen, la cobertura mediría una biblioteca que el day-gen no ve."""
    from constants import DOMINICAN_PROTEINS, DOMINICAN_CARBS, strip_accents
    from dish_library import load_dish_templates, _protein_matches_pool

    genericos = set(ah._TEMPLATE_CLASS_NAMELESS) | set(ah._TEMPLATE_CLASS_CATEGORY)
    tokens = sorted({str(t.get("protein") or "").lower()
                     for t in load_dish_templates()} - genericos - {""})
    assert tokens, "sanity: la biblioteca viva tiene tokens de proteína específicos"

    divergencias = []
    for tok in tokens:
        for n in list(DOMINICAN_PROTEINS) + list(DOMINICAN_CARBS):
            a = strip_accents(str(n).lower())
            if ah._template_token_names_base(tok, a, str(n).lower()) != _protein_matches_pool(tok, a):
                divergencias.append((tok, n))
    assert divergencias == [], f"el matcher divergió de dish_library: {divergencias[:5]}"
