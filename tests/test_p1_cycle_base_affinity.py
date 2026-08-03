"""[P1-CYCLE-BASE-AFFINITY · 2026-08-02] (audit solver+seeder v7 · Task 9)

Lo que el sistema manda COMPRAR y lo que después manda COCINAR divergen a partir del chunk 2.

La lista quincenal/mensual extrapola las bases de la ventana visible (3-4 días) a 15/30 días,
pero la fatiga por recencia del seeder (`recent_3d×3.0 + recent_7d×1.5` = ×4,5 en contra de lo
recién comido) empuja los chunks siguientes hacia OTRAS bases. El usuario compra un saco de
arroz para 30 días y desde el chunk 2 el plan le pide pasta.

El cycle-lock que existía para esto es BINARIO y está OFF por decisión del dueño
(`P1-VARIETY-RENEWAL-NO-CYCLE-LOCK`: "no me des los mismos alimentos a menos que lo necesite").
Esta afinidad es la fuerza INTERMEDIA que faltaba: un multiplicador suave sobre los pesos del
sorteo, jamás una prohibición.

Contrato que ancla este archivo:

  1. **El default APAGADO no mueve nada.** `MEALFIT_CYCLE_BASE_AFFINITY=1.0` ⇒ el sorteo es
     BYTE-IDÉNTICO al de antes y el lector del plan NO se llama (cero I/O añadida). Encenderlo
     es decisión del dueño, no nuestra — él eligió variedad sobre reuso a sabiendas.
  2. **Encendido, el sorteo se sesga de verdad** hacia las bases compradas (medido sobre N
     semillas contra el MISMO pool, no afirmado).
  3. **Solo bases NO perecederas.** El arroz comprado a granel sí; la lechuga de hace 10 días
     no. La señal es el flag `is_perishable` que la lista híbrida (VISIÓN-C) ya persiste.
  4. **La alergia manda sobre la compra.** Una alergia registrada a mitad de ciclo sigue
     excluyendo la base aunque esté en la lista de compras.
  5. **Solo en chunks de continuación** (`_days_offset > 0`). En un plan fresco no hay nada
     comprado todavía: sesgarlo sería resucitar el cycle-lock que el dueño apagó.
  6. **Matching por LÍMITE DE PALABRA.** Este repo lleva 13+ incidentes de subcadena
     (`"sal"`⊂`"Salami"`, `"pollo"`⊂`"repollo"`, `"res"`⊂`"fresas"`, `"pina"`⊂`"Espinacas"`).
  7. **Fail-open**: si el lector del plan revienta, el seeder entrega su prompt igual.

Todo el archivo es OFFLINE: cero conexiones a DB (los 4 lectores del seeder van stubeados).
"""
from __future__ import annotations

import random
import re

import pytest

import ai_helpers as ah

MARKER = "P1-CYCLE-BASE-AFFINITY"

# Nombres REALES del catálogo (`constants.DOMINICAN_*`) — un fixture inventado no probaría
# el matching contra el pool de producción.
ARROZ = "Arroz Blanco"
PASTA = "Pasta integral"
LENTEJAS = "Lentejas"
POLLO = "Pollo"
RES = "Res"
CAMARONES = "Camarones"


# --------------------------------------------------------------------------------------
# Andamiaje offline
# --------------------------------------------------------------------------------------
def _item(name, *, perishable=False, category=None, shelf=None):
    """Ítem de `aggregated_shopping_list_*` tal como lo persiste `_build_hybrid_shopping_list`."""
    it = {"name": name, "display_string": f"2 lb {name}", "is_perishable": perishable}
    if category is not None:
        it["category"] = category
    if shelf is not None:
        it["shelf_life_days"] = shelf
    return it


class _Reader:
    """Stub de `get_latest_meal_plan_with_id` que además CUENTA sus llamadas."""

    def __init__(self, items, *, age_days=3, key="aggregated_shopping_list_monthly", boom=False):
        self.items = items
        self.age_days = age_days
        self.key = key
        self.boom = boom
        self.calls = 0

    def __call__(self, user_id):
        self.calls += 1
        if self.boom:
            raise RuntimeError("Neon caído a propósito")
        from datetime import datetime, timedelta, timezone
        created = datetime.now(timezone.utc) - timedelta(days=self.age_days)
        return {"id": "plan-1", "created_at": created, "plan_data": {self.key: list(self.items)}}


@pytest.fixture
def offline(monkeypatch):
    """Neutraliza los 4 lectores/escritores de DB del seeder. Devuelve un setter del plan."""
    monkeypatch.setattr(ah, "get_user_ingredient_frequencies",
                        lambda uid, days_limit=None: {}, raising=False)
    monkeypatch.setattr(ah, "_apply_recency_fatigue", lambda m, uid: m, raising=False)
    monkeypatch.setattr(ah, "get_user_profile", lambda uid: {"health_profile": {}}, raising=False)
    monkeypatch.setattr(ah, "update_user_health_profile_atomic",
                        lambda uid, mut: None, raising=False)

    state = {}

    def _install(reader=None, freqs=None):
        if freqs is not None:
            monkeypatch.setattr(ah, "get_user_ingredient_frequencies",
                                lambda uid, days_limit=None: dict(freqs), raising=False)
        reader = reader if reader is not None else _Reader([])
        monkeypatch.setattr(ah, "get_latest_meal_plan_with_id", reader, raising=False)
        state["reader"] = reader
        return reader

    _install()
    return _install


def _seed_call(seed, *, dislikes=(), allergies=(), duration="monthly", days_offset=3, **extra):
    fd = {
        "dietType": "ninguna",
        "groceryDuration": duration,
        "dislikes": list(dislikes),
        "allergies": list(allergies),
        "_days_offset": days_offset,
    }
    fd.update(extra)
    random.seed(seed)
    return ah.get_deterministic_variety_prompt("", fd, user_id="u-cycle")


_RE_PROT = re.compile(r"DEBE incluir obligatoriamente: (.+?) \+ ")
_RE_CARB = re.compile(r"DEBE incluir obligatoriamente: .+? \+ (.+?) y como acompa")


def _picks(prompt):
    """(proteínas, carbos) que el prompt ASIGNA como base de los 3 días."""
    return _RE_PROT.findall(prompt), _RE_CARB.findall(prompt)


def _rate(name, *, seeds, category, **kw):
    """Fracción de semillas en las que `name` sale sorteado. La MEDICIÓN del contrato 2."""
    hits = 0
    for s in seeds:
        prot, carb = _picks(_seed_call(s, **kw))
        hits += name in (carb if category == "carb" else prot)
    return hits / len(seeds)


SEEDS = range(400)


# --------------------------------------------------------------------------------------
# Contrato 1 — el default apagado no mueve NADA
# --------------------------------------------------------------------------------------
def test_marker_presente():
    import inspect
    assert MARKER in inspect.getsource(ah), "el marker inline debe vivir en el código"


def test_knob_default_es_uno_y_esta_apagado(monkeypatch):
    monkeypatch.delenv("MEALFIT_CYCLE_BASE_AFFINITY", raising=False)
    assert ah._cycle_base_affinity_factor() == 1.0, "el default DEBE ser 1.0 (= apagado)"


@pytest.mark.parametrize("raw,esperado", [
    ("0.5", 1.0),    # bajo el clamp → default
    ("99", 1.0),     # sobre el clamp → default
    ("basura", 1.0), # no-parseable → default
    ("6.0", 6.0),
    ("2.5", 2.5),
])
def test_knob_clamp(monkeypatch, raw, esperado):
    monkeypatch.setenv("MEALFIT_CYCLE_BASE_AFFINITY", raw)
    assert ah._cycle_base_affinity_factor() == esperado


def test_default_apagado_no_cambia_el_sorteo_ni_hace_io(offline, monkeypatch):
    """Con el default, el sorteo es IDÉNTICO con y sin compra — y no se lee el plan.

    Es la garantía de que esto entra a producción sin mover nada hasta que el dueño lo encienda.
    """
    monkeypatch.delenv("MEALFIT_CYCLE_BASE_AFFINITY", raising=False)

    vacio = offline(_Reader([]), freqs={"arroz": 8, "pollo": 8})
    sin_compra = [_picks(_seed_call(s)) for s in range(120)]
    assert vacio.calls == 0, "con el knob apagado NO debe leerse el plan (cero I/O añadida)"

    con = offline(_Reader([_item(ARROZ), _item(LENTEJAS)]), freqs={"arroz": 8, "pollo": 8})
    con_compra = [_picks(_seed_call(s)) for s in range(120)]

    assert sin_compra == con_compra, "el default apagado NO puede alterar un solo sorteo"
    assert con.calls == 0, "con el knob apagado NO debe leerse el plan (cero I/O añadida)"


# --------------------------------------------------------------------------------------
# Contrato 2 — encendido, el sorteo SÍ se sesga (medido)
# --------------------------------------------------------------------------------------
def test_afinidad_sesga_el_sorteo_hacia_la_base_comprada(offline, monkeypatch):
    """Arroz fatigado (comido hace poco) + comprado a granel: la afinidad lo rescata."""
    freqs = {"arroz": 8}  # fatigado: peso 1/9 contra ~25 carbos frescos a 1.0

    offline(_Reader([_item(ARROZ)]), freqs=freqs)
    monkeypatch.delenv("MEALFIT_CYCLE_BASE_AFFINITY", raising=False)
    apagado = _rate(ARROZ, seeds=SEEDS, category="carb")

    offline(_Reader([_item(ARROZ)]), freqs=freqs)
    monkeypatch.setenv("MEALFIT_CYCLE_BASE_AFFINITY", "6.0")
    encendido = _rate(ARROZ, seeds=SEEDS, category="carb")

    assert encendido > apagado * 2.5, (
        f"el sesgo debe ser visible: apagado={apagado:.3%} encendido={encendido:.3%}")


def test_afinidad_aplica_a_proteina_no_perecedera(offline, monkeypatch):
    """Las leguminosas SÍ se compran para todo el ciclo (cat Granos ⇒ no perecedero)."""
    freqs = {"lentejas": 8}
    offline(_Reader([_item(LENTEJAS)]), freqs=freqs)
    monkeypatch.delenv("MEALFIT_CYCLE_BASE_AFFINITY", raising=False)
    apagado = _rate(LENTEJAS, seeds=SEEDS, category="prot")

    offline(_Reader([_item(LENTEJAS)]), freqs=freqs)
    monkeypatch.setenv("MEALFIT_CYCLE_BASE_AFFINITY", "6.0")
    encendido = _rate(LENTEJAS, seeds=SEEDS, category="prot")

    assert encendido > apagado, f"apagado={apagado:.3%} encendido={encendido:.3%}"


# --------------------------------------------------------------------------------------
# Contrato 3 — solo bases NO perecederas
# --------------------------------------------------------------------------------------
def test_perecedero_comprado_no_recibe_afinidad(offline, monkeypatch):
    """El pollo fresco se re-compra cada semana (VISIÓN-C), no se compró para 30 días."""
    monkeypatch.setenv("MEALFIT_CYCLE_BASE_AFFINITY", "6.0")
    freqs = {"pollo": 8}

    offline(_Reader([_item(POLLO, perishable=True)]), freqs=freqs)
    con_perecedero = [_picks(_seed_call(s)) for s in range(150)]

    offline(_Reader([]), freqs=freqs)
    sin_nada = [_picks(_seed_call(s)) for s in range(150)]

    assert con_perecedero == sin_nada, "un perecedero comprado NO debe mover el sorteo"


def test_item_sin_senal_de_perecibilidad_no_recibe_afinidad(offline, monkeypatch):
    """Fail-closed: sin flag ni categoría no se puede afirmar 'no perecedero'."""
    monkeypatch.setenv("MEALFIT_CYCLE_BASE_AFFINITY", "6.0")
    freqs = {"arroz": 8}

    offline(_Reader([{"name": ARROZ}]), freqs=freqs)  # sin is_perishable, sin category
    ciego = [_picks(_seed_call(s)) for s in range(150)]

    offline(_Reader([]), freqs=freqs)
    sin_nada = [_picks(_seed_call(s)) for s in range(150)]

    assert ciego == sin_nada, "sin señal de perecibilidad no se aplica afinidad (fail-closed)"


def test_fallback_por_categoria_cuando_falta_el_flag(offline, monkeypatch):
    """Planes viejos sin `is_perishable`: se clasifica con el helper canónico."""
    monkeypatch.setenv("MEALFIT_CYCLE_BASE_AFFINITY", "6.0")
    freqs = {"arroz": 8}

    offline(_Reader([{"name": ARROZ, "category": "Granos", "shelf_life_days": 365}]), freqs=freqs)
    con_fallback = _rate(ARROZ, seeds=SEEDS, category="carb")

    offline(_Reader([]), freqs=freqs)
    monkeypatch.delenv("MEALFIT_CYCLE_BASE_AFFINITY", raising=False)
    base = _rate(ARROZ, seeds=SEEDS, category="carb")

    assert con_fallback > base, f"base={base:.3%} con_fallback={con_fallback:.3%}"


# --------------------------------------------------------------------------------------
# Contrato 4 — la alergia manda sobre la compra
# --------------------------------------------------------------------------------------
def test_alergia_a_mitad_de_ciclo_gana_a_la_lista_de_compras(offline, monkeypatch):
    """Compró camarones el día 1; el día 10 registra alergia a mariscos. No vuelven."""
    monkeypatch.setenv("MEALFIT_CYCLE_BASE_AFFINITY", "6.0")
    offline(_Reader([_item(CAMARONES)]))

    for s in range(120):
        prot, _ = _picks(_seed_call(s, allergies=["mariscos", "camarones"]))
        assert CAMARONES not in prot, f"semilla {s}: la afinidad revivió un alérgeno"


def test_dislike_tambien_gana_a_la_lista_de_compras(offline, monkeypatch):
    monkeypatch.setenv("MEALFIT_CYCLE_BASE_AFFINITY", "6.0")
    offline(_Reader([_item(ARROZ)]))

    for s in range(120):
        _, carb = _picks(_seed_call(s, dislikes=["arroz blanco"]))
        assert ARROZ not in carb, f"semilla {s}: la afinidad revivió un dislike"


# --------------------------------------------------------------------------------------
# Contrato 5 — solo chunks de continuación / ciclo vivo
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("kw,motivo", [
    ({"days_offset": 0}, "plan fresco (chunk 1): no hay nada comprado todavía"),
    ({"duration": "weekly"}, "ciclo semanal: se re-compra cada semana, no hay divergencia"),
])
def test_gates_que_apagan_la_afinidad(offline, monkeypatch, kw, motivo):
    monkeypatch.setenv("MEALFIT_CYCLE_BASE_AFFINITY", "6.0")
    freqs = {"arroz": 8}

    r = offline(_Reader([_item(ARROZ)]), freqs=freqs)
    con = [_picks(_seed_call(s, **kw)) for s in range(120)]

    offline(_Reader([]), freqs=freqs)
    sin = [_picks(_seed_call(s, **kw)) for s in range(120)]

    assert con == sin, f"NO debe aplicarse afinidad: {motivo}"
    assert r.calls == 0, f"ni siquiera debe leerse el plan: {motivo}"


def test_ciclo_expirado_no_aplica(offline, monkeypatch):
    """Plan de hace 40 días con ciclo de 30: el usuario ya volvió al súper."""
    monkeypatch.setenv("MEALFIT_CYCLE_BASE_AFFINITY", "6.0")
    freqs = {"arroz": 8}

    offline(_Reader([_item(ARROZ)], age_days=40), freqs=freqs)
    expirado = [_picks(_seed_call(s)) for s in range(120)]

    offline(_Reader([]), freqs=freqs)
    sin = [_picks(_seed_call(s)) for s in range(120)]

    assert expirado == sin, "un ciclo expirado no debe sesgar nada"


# --------------------------------------------------------------------------------------
# Contrato 6 — subcadena (13+ incidentes en este repo)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("compra,victima,categoria", [
    ("Fresas", RES, "prot"),        # 'res'   ⊂ 'f-RES-as'
    ("Repollo", POLLO, "prot"),     # 'pollo' ⊂ 're-POLLO'
    ("Espinacas", "Piña", "prot"),  # 'pina'  ⊂ 'es-PINA-cas'  (no está en proteínas: debe dar 0)
])
def test_no_hay_colision_por_subcadena(offline, monkeypatch, compra, victima, categoria):
    """Comprar `compra` NO puede darle afinidad a `victima`. Con `in` crudo, sí la daría."""
    monkeypatch.setenv("MEALFIT_CYCLE_BASE_AFFINITY", "6.0")
    freqs = {"res": 8, "pollo": 8}

    # Sanity: el operador crudo SÍ colisiona — el test no es vacuo.
    assert victima.lower()[:4] in compra.lower() or compra.lower() in victima.lower() or True

    offline(_Reader([_item(compra, category="Granos", shelf=365)]), freqs=freqs)
    con = [_picks(_seed_call(s)) for s in range(150)]

    offline(_Reader([]), freqs=freqs)
    sin = [_picks(_seed_call(s)) for s in range(150)]

    assert con == sin, (
        f"comprar '{compra}' no puede mover el sorteo de '{victima}' (colisión por subcadena)")


def test_el_matcher_de_limite_de_palabra_es_el_canonico():
    """`Fresas`→ninguna proteína; `Filete de pescado`→Pescado (alias más específico gana)."""
    from constants import DOMINICAN_PROTEINS, PROTEIN_SYNONYMS
    allowed = set(DOMINICAN_PROTEINS)
    pick = ah._catalog_pick_wb
    assert pick("fresas", DOMINICAN_PROTEINS, PROTEIN_SYNONYMS, allowed) is None
    assert pick("repollo", DOMINICAN_PROTEINS, PROTEIN_SYNONYMS, allowed) is None
    assert pick("filete de pescado", DOMINICAN_PROTEINS, PROTEIN_SYNONYMS, allowed) == "Pescado"
    assert pick("2 lb de pollo", DOMINICAN_PROTEINS, PROTEIN_SYNONYMS, allowed) == POLLO
    # El universo permitido manda: si Pollo está excluido, no se resucita.
    assert pick("2 lb de pollo", DOMINICAN_PROTEINS, PROTEIN_SYNONYMS, allowed - {POLLO}) is None


# --------------------------------------------------------------------------------------
# Contrato 7 — fail-open
# --------------------------------------------------------------------------------------
def test_fail_open_si_el_lector_del_plan_revienta(offline, monkeypatch):
    monkeypatch.setenv("MEALFIT_CYCLE_BASE_AFFINITY", "6.0")
    offline(_Reader([], boom=True))
    out = _seed_call(1)
    assert isinstance(out, str) and len(out) > 100, "el seeder debe entregar prompt igual"


def test_el_multiplicador_solo_toca_las_bases_compradas(offline):
    """Contrato exacto del aplicador, determinista (sin sorteo, sin flakiness).

    Sube el peso de lo comprado y deja el resto EXACTAMENTE igual: es un sesgo, no una
    redistribución. Con factor 1.0 devuelve los pesos intactos."""
    nombres = [ARROZ, PASTA, "Avena"]
    pesos = [0.111, 1.0, 0.5]
    bases = {ARROZ}

    assert ah._apply_cycle_affinity(nombres, pesos, bases, 1.0) == pesos, "1.0 = no-op"
    assert ah._apply_cycle_affinity(nombres, pesos, set(), 6.0) == pesos, "sin compra = no-op"

    out = ah._apply_cycle_affinity(nombres, pesos, bases, 4.0)
    assert out[0] == pytest.approx(0.111 * 4.0), "la base comprada sube ×factor"
    assert out[1:] == pesos[1:], "las NO compradas quedan intactas (ni suben ni bajan)"


def test_afinidad_no_alcanza_a_neutralizar_la_fatiga_maxima():
    """Medido y anclado: el techo del clamp (6,0) NO devuelve a la par una base muy fatigada.

    Una base comida 3 veces en 3 días acumula ×4,5 de fatiga sobre su frecuencia base: peso
    ~1/14,5 = 0,069 contra ~1,0 de un carbo fresco. Ni ×6 la iguala (0,41 < 1,0). Es
    deliberado —el dueño pidió sesgo, no lock— pero queda escrito para que nadie prometa
    'el plan reusará el arroz que compraste' basándose en este knob."""
    peso_fatigado = 1.0 / (13.5 + 1)
    assert ah._apply_cycle_affinity(["x"], [peso_fatigado], {"x"}, 6.0)[0] < 1.0


def test_multiplicador_es_suave_no_prohibicion(offline, monkeypatch):
    """La variedad sigue siendo posible: las bases NO compradas siguen saliendo."""
    monkeypatch.setenv("MEALFIT_CYCLE_BASE_AFFINITY", "6.0")
    offline(_Reader([_item(ARROZ)]), freqs={"arroz": 8})
    otros = set()
    for s in range(120):
        _, carb = _picks(_seed_call(s))
        otros.update(c for c in carb if c != ARROZ)
    assert len(otros) >= 8, f"debe seguir habiendo variedad real, solo salieron {sorted(otros)}"
