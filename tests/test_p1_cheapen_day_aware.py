"""[P1-CHEAPEN-DAY-AWARE · 2026-07-10] El lever económico NO puede crear la repetición
same-day que el reviewer rechaza — y el autofix debe poder repararla en contexto
dulce/ligero aunque el goal sea gain_muscle.

Cadena causal medida en vivo (renovación cb150867, 2026-07-10 19:21-19:33, 660s,
2 rechazos, entrega degradada band 0.83 con banner amarillo):
  1. marker-regen (PRO) devolvía Día 2 VÁLIDO: camarones (almuerzo) + pescado (cena).
  2. `_apply_budget_cheapen_pass` convertía camarones→'Filete de pescado blanco'
     → pescado×2 same-day (el pase no miraba las otras comidas del día).
  3. `_protein_repeat_autofix` lo detectaba pero quedaba IMPOTENTE
     (`no_safe_target_sweet_or_light`): contexto dulce exige queso/legumbres y la
     escalera gain_muscle es carnes-only → intersección VACÍA.
  4. Rechazo del reviewer → OTRA reparación PRO → el cheapen volvía a romperla.
  El lever deshacía la reparación en CADA ciclo. Costo: retries PRO + banner.

tooltip-anchor: P1-CHEAPEN-DAY-AWARE
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))
_GO_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


def _mk_day(meals):
    return {"day": 2, "meals": meals}


def _meal(name, ingredients, slot="Almuerzo"):
    return {"name": name, "meal": slot, "ingredients": list(ingredients),
            "ingredients_raw": list(ingredients), "recipe": []}


# ---------------------------------------------------------------------------
# 1. Guard helper: el candidato que repite proteína del día colisiona
# ---------------------------------------------------------------------------

def test_candidate_collides_when_sibling_has_same_label():
    from graph_orchestrator import _budget_candidate_collides_same_day
    cena = _meal("Filete de pescado blanco al Horno", ["120 g de filete de pescado blanco"], "Cena")
    almuerzo = _meal("Camarones Guisados", ["150 g de camarones"], "Almuerzo")
    day = _mk_day([almuerzo, cena])
    # sustituir camarones→pescado en el ALMUERZO colisiona con la CENA (pescado ya presente)
    assert _budget_candidate_collides_same_day(day, almuerzo, "Filete de pescado blanco") is True
    # un candidato sin label del gate (maní) jamás colisiona
    assert _budget_candidate_collides_same_day(day, almuerzo, "Maní") is False
    # sin hermano con pescado → no colisiona
    solo = _mk_day([almuerzo])
    assert _budget_candidate_collides_same_day(solo, almuerzo, "Filete de pescado blanco") is False


# ---------------------------------------------------------------------------
# 2. Cheapen estático: skip de la sustitución que crearía el repeat
# ---------------------------------------------------------------------------

def _fake_price_map():
    return {"camarones": 299.0, "camaron": 299.0, "filete de pescado blanco": 128.0}


def test_cheapen_pass_skips_colliding_substitution(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setattr(go, "_budget_build_master_price_map", _fake_price_map)
    almuerzo = _meal("Camarones Guisados con Casabe", ["150 g de camarones", "50 g de casabe"])
    cena = _meal("Pescado al Horno", ["120 g de filete de pescado blanco"], "Cena")
    days = [_mk_day([almuerzo, cena])]
    subs = go._apply_budget_cheapen_pass(days, {}, force=True)
    assert subs == 0, "camarones→pescado con pescado ya en la cena debe SKIPPEARSE"
    assert "camarones" in " ".join(almuerzo["ingredients"]).lower(), "el premium queda intacto"


def test_cheapen_pass_still_substitutes_without_collision(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setattr(go, "_budget_build_master_price_map", _fake_price_map)
    almuerzo = _meal("Camarones Guisados con Casabe", ["150 g de camarones", "50 g de casabe"])
    cena = _meal("Pechuga a la Plancha", ["120 g de pechuga de pollo"], "Cena")
    days = [_mk_day([almuerzo, cena])]
    subs = go._apply_budget_cheapen_pass(days, {}, force=True)
    assert subs == 1, "sin colisión, la sustitución económica sigue operando"
    assert "pescado" in " ".join(almuerzo["ingredients"]).lower()


# ---------------------------------------------------------------------------
# 3. Driver-aware: mismo guard (parser — la maquinaria es compartida)
# ---------------------------------------------------------------------------

def test_driver_aware_pass_has_same_guard():
    i = _GO_SRC.find("def _apply_budget_driver_aware_pass")
    assert i > 0
    blk = _GO_SRC[i: i + 9000]
    assert "_budget_candidate_collides_same_day(_d, meal, candidate)" in blk, (
        "P1-CHEAPEN-DAY-AWARE: la familia mariscos→pescado del driver-aware colapsa labels "
        "distintos en uno — sin el guard recrea el repeat que el reviewer rechaza."
    )
    # el estático también lo lleva (dos callsites en total)
    assert _GO_SRC.count("_budget_candidate_collides_same_day(_d, meal, candidate)") >= 2


# ---------------------------------------------------------------------------
# 4. Deadlock dulce/ligero × gain_muscle roto: el autofix repara con queso
# ---------------------------------------------------------------------------

def test_autofix_fixes_light_slot_repeat_for_gainmuscle():
    from graph_orchestrator import _protein_repeat_autofix
    cena = _meal("Filete de pescado blanco al Horno con Batata", ["120 g de filete de pescado blanco"], "Cena")
    merienda = _meal("Casabe con Pescado Desmenuzado", ["60 g de pescado", "30 g de casabe"], "Merienda")
    days = [_mk_day([cena, merienda])]
    fixed = _protein_repeat_autofix(days, {"mainGoal": "gain_muscle"})
    assert fixed >= 1, (
        "P1-CHEAPEN-DAY-AWARE: gain_muscle + slot ligero volvió al deadlock (escalera "
        "carnes-only ∩ permitidos-ligero = ∅ → impotencia → rechazo → PRO)."
    )
    _blob = " ".join(str(i) for i in merienda["ingredients"]).lower() + " " + str(merienda["name"]).lower()
    assert "queso" in _blob, "el target del slot ligero para gain_muscle es queso (proteína animal densa)"


def test_autofix_gainmuscle_main_slots_still_meat_only():
    # En slots PRINCIPALES la doctrina P2-FALLBACK-NONGATED-GAINMUSCLE sigue: carnes, no queso/legumbre.
    from graph_orchestrator import _protein_repeat_autofix
    almuerzo = _meal("Pescado Guisado con Arroz", ["150 g de filete de pescado blanco", "arroz"], "Almuerzo")
    cena = _meal("Pescado al Horno con Yuca", ["140 g de filete de pescado blanco", "yuca hervida"], "Cena")
    days = [_mk_day([almuerzo, cena])]
    fixed = _protein_repeat_autofix(days, {"mainGoal": "gain_muscle"})
    if fixed:  # si reparó, el target debe ser CARNE (no queso ni legumbre)
        _blob = (" ".join(str(i) for i in cena["ingredients"]) + " " + str(cena["name"])).lower()
        assert not any(t in _blob for t in ("queso", "habichuela", "lenteja")), (
            "slot principal gain_muscle debe recibir proteína animal densa (escalera de carnes)"
        )


# ---------------------------------------------------------------------------
# 5. Rename de huevo sin residuo: "Claras de Huevo" completo
# ---------------------------------------------------------------------------

def test_egg_phrase_rename_consumes_de_huevo_tail():
    from graph_orchestrator import _replace_egg_phrase_in_name
    out = _replace_egg_phrase_in_name(
        "Cazuela de Avena y Arándanos Horneados con Claras de Huevo", "Pechuga de pollo")
    assert out == "Cazuela de Avena y Arándanos Horneados con Pechuga de pollo", (
        f"residuo 'de Huevo' en el rename (nombre mutante vivo cb150867): {out!r}"
    )
    # formas previas intactas
    assert _replace_egg_phrase_in_name("Arepitas con Huevo Revuelto", "Queso blanco") == \
        "Arepitas con Queso blanco"
    assert _replace_egg_phrase_in_name("Batido de Guineo", "Queso blanco") is None


# ---------------------------------------------------------------------------
# 6. Marker vivo
# ---------------------------------------------------------------------------

def test_marker_anchored_in_source():
    # [P1-CHUNK-AUTONOMY · 2026-07-10] durable: anclar el marker en el CÓDIGO (comments
    # tooltip-anchor), no en _LAST_KNOWN_PFIX vigente — pinnear el marker actual rota
    # con cada bump posterior (clase de fallo de test_p1_account_delete_1 y compañía).
    assert _GO_SRC.count("P1-CHEAPEN-DAY-AWARE") >= 3, (
        "los anchors del guard day-aware desaparecieron de graph_orchestrator"
    )
