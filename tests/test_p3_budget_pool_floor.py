"""[P3-BUDGET-POOL-FLOOR · 2026-08-04] (audit solver+seeder v7 · Task 20)

El boost económico ×2 del sorteo perdía contra la fatiga en ciclos largos.

`P1-BUDGET-TIER-LEVERS` multiplica ×2 el peso de las proteínas/carbos del TERCIO MÁS BARATO
cuando el presupuesto pide economía. Pero el peso base es `1/(freq+1)`: un staple barato comido
33 veces pesa `1/34 = 0,029`, y ×2 sigue siendo `0,059` contra el `1,0` de un premium fresco —
**17:1 en contra del barato**. El caso general está sano (un barato NO reciente pesa 2,0 y gana);
el residuo real es el ciclo de 15/30 días con pool barato chico, donde el tercio barato ENTERO se
fatiga y el sorteo se va a los premium, que el cheapen-pass de `assemble` tiene que corregir
después con churn y colapso de variedad.

El cierre es un PISO RELATIVO: los ítems del tercio barato no pueden pesar menos que
`mediana(pesos_del_pool) × MEALFIT_BUDGET_POOL_FLOOR`. Relativo y no absoluto a propósito — un
número fijo no significa nada contra una escala de pesos que depende del historial del usuario.

Contrato que ancla este archivo:

  1. **Knob** `MEALFIT_BUDGET_POOL_FLOOR`: float, default 0.8, clamp [0.0, 1.5], `0.0` = OFF.
     Registrado en `_KNOBS_REGISTRY`.
  2. **Con el knob en 0 el sorteo es BYTE-IDÉNTICO** al de antes del fix (mismo `w × boost`).
  3. **El piso es un PISO**: nunca baja un peso, sólo lo sube; y sólo toca al tercio barato.
  4. **Sólo cuando el boost económico ya aplica** (budget low / custom ajustado). Con presupuesto
     no económico el sorteo no se mueve ni un ápice.
  5. **Efecto MEDIDO end-to-end**, no afirmado: el staple barato fatigado pasa de casi nunca a
     salir con regularidad.
  6. **Techo honesto**: con el default 0,8 el piso queda POR DEBAJO de la mediana por
     construcción, así que un barato fatigado NO adelanta a un premium fresco. Queda escrito
     para que nadie prometa "el plan reusará el arroz que compraste" citando este knob. Sólo
     con el knob en su tope (1,5) lo adelanta — medido.
  7. **El piso se aplica sobre el pool YA FILTRADO**: jamás resucita un alérgeno.

Todo el archivo es OFFLINE: los lectores de DB del seeder y el catálogo maestro van stubeados.
"""
from __future__ import annotations

import random
import re

import pytest

import ai_helpers as ah

MARKER = "P3-BUDGET-POOL-FLOOR"
KNOB = "MEALFIT_BUDGET_POOL_FLOOR"

POLLO = "Pollo"          # barato del catálogo, FATIGADO en el escenario del audit
SALMON = "Salmón"        # premium fresco (freq 0)


# --------------------------------------------------------------------------------------
# Andamiaje offline
# --------------------------------------------------------------------------------------
def _pool_de_diez():
    """Pool reducido y MEDIDO: `dislikes` arrastra a los parientes de sinónimo (un dislike de
    'Mero' se lleva a 'Pescado'), así que el conjunto no se elige a ojo — se comprueba contra
    `_get_fast_filtered_catalogs` en el test de sanity."""
    return ["Pollo", "Muslo de pollo", "Pescado", "Atún", "Mero", "Tilapia", "Salmón",
            "Bacalao", "Arenque", "Sardinas en lata"]


def _dislikes_para_el_pool_de_diez():
    from constants import DOMINICAN_PROTEINS
    keep = set(_pool_de_diez())
    return [p.lower() for p in DOMINICAN_PROTEINS if p not in keep]


# Tercio barato = 3 de los 10 (el corte del código es `_valid[int(0.33 * (n - 1))]`), los TRES
# fatigados: es el escenario que el audit describe (pool barato chico + ciclo largo).
BARATOS = ("Pollo", "Muslo de pollo", "Sardinas en lata")


def _master_rows():
    from constants import DOMINICAN_PROTEINS, DOMINICAN_CARBS
    filas = [{"name": p, "price_per_lb": (30.0 if p in BARATOS else 200.0)}
             for p in DOMINICAN_PROTEINS]
    filas += [{"name": c, "price_per_lb": 40.0 if i % 3 == 0 else 180.0}
              for i, c in enumerate(DOMINICAN_CARBS)]
    return filas


@pytest.fixture
def offline(monkeypatch):
    import shopping_calculator as sc
    from constants import strip_accents

    monkeypatch.setattr(ah, "_apply_recency_fatigue", lambda m, uid: m, raising=False)
    monkeypatch.setattr(ah, "get_user_profile", lambda uid: {"health_profile": {}}, raising=False)
    monkeypatch.setattr(ah, "update_user_health_profile_atomic",
                        lambda uid, mut: None, raising=False)
    monkeypatch.setattr(ah, "get_latest_meal_plan_with_id", lambda uid: None, raising=False)
    monkeypatch.setattr(sc, "get_master_ingredients", _master_rows, raising=False)
    # El penalty de cobertura de plantillas (P3-SEEDER-TEMPLATE-COVERAGE) también pesa el sorteo
    # y castigaría a Salmón: neutralizado para que lo medido aquí sea SÓLO el piso económico.
    # [ronda 1 · 2026-08-04] Hoy es REDUNDANTE (ese knob nació apagado) y se queda a propósito: es
    # lo que impide que encenderlo algún día cambie esta medición en silencio.
    monkeypatch.setenv("MEALFIT_LOW_TEMPLATE_COVERAGE_PENALTY", "1.0")

    # El tercio barato ENTERO fatigado — el residuo que el audit describe.
    freqs = {strip_accents(p.lower()): 33 for p in BARATOS}
    monkeypatch.setattr(ah, "get_user_ingredient_frequencies",
                        lambda uid, days_limit=None: dict(freqs), raising=False)
    return freqs


def _seed_call(seed, *, budget="low", **extra):
    fd = {
        "dietType": "ninguna",
        "mainGoal": "maintenance",
        "budget": budget,
        "allergies": [],
        "dislikes": _dislikes_para_el_pool_de_diez(),
    }
    fd.update(extra)
    random.seed(seed)
    return ah.get_deterministic_variety_prompt("", fd, user_id="u-bud")


_RE_PROT = re.compile(r"DEBE incluir obligatoriamente: (.+?) \+ ")


def _prot_picks(prompt):
    return _RE_PROT.findall(prompt)


def _rate(name, *, seeds, **kw):
    return sum(name in _prot_picks(_seed_call(s, **kw)) for s in seeds) / len(seeds)


SEEDS = range(500)


# --------------------------------------------------------------------------------------
# Sanity del escenario (sin esto, todo lo demás mediría otra cosa)
# --------------------------------------------------------------------------------------
def test_marker_presente():
    import inspect
    assert MARKER in inspect.getsource(ah), "el marker inline debe vivir en el código"


def test_el_pool_reducido_es_exactamente_el_esperado():
    """MEDIDO, no supuesto: `dislikes` arrastra parientes de sinónimo."""
    from constants import _get_fast_filtered_catalogs
    fp = _get_fast_filtered_catalogs((), tuple(_dislikes_para_el_pool_de_diez()), "ninguna")[0]
    assert set(fp) == set(_pool_de_diez()), f"el pool filtrado no es el del escenario: {sorted(fp)}"


def test_el_presupuesto_low_activa_el_boost_economico():
    from nutrition_calculator import budget_prefers_economy
    assert budget_prefers_economy({"budget": "low"}) is True
    assert budget_prefers_economy({"budget": "medium"}) is False


# --------------------------------------------------------------------------------------
# Contrato 1 — knob
# --------------------------------------------------------------------------------------
def test_knob_default(monkeypatch):
    monkeypatch.delenv(KNOB, raising=False)
    assert ah._budget_pool_floor() == 0.8


@pytest.mark.parametrize("raw,esperado", [
    ("-1", 0.8),       # bajo el clamp → default
    ("2.0", 0.8),      # sobre el clamp → default
    ("basura", 0.8),   # no-parseable → default
    ("0", 0.0),        # el OFF explícito
    ("1.5", 1.5),      # el tope
    ("0.4", 0.4),
])
def test_knob_clamp(monkeypatch, raw, esperado):
    monkeypatch.setenv(KNOB, raw)
    assert ah._budget_pool_floor() == esperado


def test_knob_registrado_en_el_registry(offline):
    from knobs import get_knobs_registry_snapshot
    _seed_call(1)
    assert KNOB in get_knobs_registry_snapshot()


# --------------------------------------------------------------------------------------
# Contratos 2 y 3 — el aplicador puro (determinista, sin sorteo)
# --------------------------------------------------------------------------------------
def test_con_el_piso_en_cero_es_el_comportamiento_previo():
    """Byte-idéntico a `w × boost` para el tercio barato y `w` para el resto."""
    pesos = [0.03, 1.0, 1.0, 1.0]
    precios = [30.0, 200.0, 200.0, 200.0]
    out = ah._budget_boost_with_floor(pesos, precios, 2.0, 0.0)
    assert out == [0.06, 1.0, 1.0, 1.0]


def test_el_piso_solo_SUBE_y_solo_al_tercio_barato():
    pesos = [0.03, 1.0, 1.0, 1.0]      # mediana = 1.0 → piso = 0.8
    precios = [30.0, 200.0, 200.0, 200.0]
    out = ah._budget_boost_with_floor(pesos, precios, 2.0, 0.8)
    assert out[0] == pytest.approx(0.8), "el barato fatigado sube al piso"
    assert out[1:] == pesos[1:], "los caros quedan INTACTOS (ni suben ni bajan)"


def test_el_piso_nunca_baja_un_peso_que_ya_estaba_arriba():
    """`max(w × boost, piso)`: un barato FRESCO conserva su ×2, no lo pierde contra el piso."""
    pesos = [1.0, 1.0, 1.0, 1.0]       # mediana = 1.0 → piso = 0.8, boost lo lleva a 2.0
    precios = [30.0, 200.0, 200.0, 200.0]
    out = ah._budget_boost_with_floor(pesos, precios, 2.0, 0.8)
    assert out[0] == pytest.approx(2.0), f"el piso no puede degradar un peso ya alto: {out}"


def test_sin_precios_suficientes_no_toca_nada():
    """Contrato heredado de P1-BUDGET-TIER-LEVERS: con <4 precios resolubles no hay tercil."""
    pesos = [0.03, 1.0, 1.0]
    assert ah._budget_boost_with_floor(pesos, [30.0, None, None], 2.0, 0.8) == pesos


def test_el_item_sin_precio_resoluble_queda_intacto():
    pesos = [0.03, 1.0, 1.0, 1.0, 0.02]
    precios = [30.0, 200.0, 200.0, 200.0, None]
    out = ah._budget_boost_with_floor(pesos, precios, 2.0, 0.8)
    assert out[4] == 0.02, "sin precio no se puede afirmar que sea barato → peso intacto"


# --------------------------------------------------------------------------------------
# Contrato 5 — efecto MEDIDO end-to-end
# --------------------------------------------------------------------------------------
def test_el_staple_barato_fatigado_deja_de_ser_invisible(offline, monkeypatch):
    """El residuo del audit, medido: con el piso apagado el pollo fatigado casi no sale frente al
    salmón fresco; con el piso encendido sube un orden de magnitud."""
    monkeypatch.setenv(KNOB, "0")
    off_pollo, off_salmon = _rate(POLLO, seeds=SEEDS), _rate(SALMON, seeds=SEEDS)

    monkeypatch.setenv(KNOB, "0.8")
    on_pollo, on_salmon = _rate(POLLO, seeds=SEEDS), _rate(SALMON, seeds=SEEDS)

    assert off_salmon > 0.10, f"sanity: el premium fresco debe salir a menudo ({off_salmon:.1%})"
    assert on_pollo > off_pollo * 3.0, (
        f"el piso debe rescatar al barato fatigado: {off_pollo:.1%} → {on_pollo:.1%}")
    razon_off = off_pollo / off_salmon
    razon_on = on_pollo / on_salmon
    assert razon_on > razon_off * 3.0, (
        f"la razón barato/premium debe mejorar mucho: {razon_off:.3f} → {razon_on:.3f}")


def test_techo_honesto_el_piso_por_default_NO_adelanta_al_premium_fresco(offline, monkeypatch):
    """Contrato 6. Por construcción `piso = mediana × 0,8` queda BAJO la mediana, así que un
    barato pisado no puede superar a un premium de peso mediano. Es deliberado —sesgo, no lock—
    pero queda MEDIDO para que nadie prometa reuso garantizado citando el default."""
    monkeypatch.setenv(KNOB, "0.8")
    assert _rate(POLLO, seeds=SEEDS) < _rate(SALMON, seeds=SEEDS), (
        "con el default el barato fatigado NO adelanta al premium fresco (y está bien)")


def test_en_el_tope_del_knob_el_barato_SI_adelanta_al_premium(offline, monkeypatch):
    """El clamp llega a 1,5 justamente para poder pasar la mediana cuando el dueño lo quiera."""
    monkeypatch.setenv(KNOB, "1.5")
    assert _rate(POLLO, seeds=SEEDS) > _rate(SALMON, seeds=SEEDS), (
        "con el piso por encima de la mediana el barato debe ganar")


# --------------------------------------------------------------------------------------
# Contratos 2 y 4 — no-op medible
# --------------------------------------------------------------------------------------
def test_con_el_knob_en_cero_el_sorteo_es_identico_semilla_a_semilla(offline, monkeypatch):
    monkeypatch.setenv(KNOB, "0")
    a = [_prot_picks(_seed_call(s)) for s in range(150)]
    monkeypatch.setenv(KNOB, "0.0")
    b = [_prot_picks(_seed_call(s)) for s in range(150)]
    assert a == b

    monkeypatch.setenv(KNOB, "0.8")
    c = [_prot_picks(_seed_call(s)) for s in range(150)]
    assert a != c, "sanity: encendido el piso TIENE que mover el sorteo (si no, es no-op)"


def test_con_presupuesto_no_economico_el_piso_no_existe(offline, monkeypatch):
    """Contrato 4: el gating de P1-BUDGET-TIER-LEVERS manda — sin economía no hay boost, y sin
    boost tampoco piso."""
    monkeypatch.setenv(KNOB, "0")
    apagado = [_prot_picks(_seed_call(s, budget="medium")) for s in range(150)]
    monkeypatch.setenv(KNOB, "1.5")
    encendido = [_prot_picks(_seed_call(s, budget="medium")) for s in range(150)]
    assert apagado == encendido, "con budget no económico el piso no puede mover nada"


# --------------------------------------------------------------------------------------
# Contrato 7 — el piso se aplica sobre el pool YA FILTRADO
# --------------------------------------------------------------------------------------
def test_el_piso_no_resucita_un_alergeno(offline, monkeypatch):
    """`available_proteins` ya pasó por alergias/dieta/dislikes cuando el piso corre: por barato
    y fatigado que esté, un alérgeno no puede volver."""
    monkeypatch.setenv(KNOB, "1.5")
    for s in range(150):
        picks = _prot_picks(_seed_call(s, allergies=["pollo"]))
        assert POLLO not in picks and "Muslo de pollo" not in picks, (
            f"semilla {s}: el piso económico resucitó un alérgeno → {picks}")


def test_el_piso_no_resucita_un_dislike(offline, monkeypatch):
    monkeypatch.setenv(KNOB, "1.5")
    dis = _dislikes_para_el_pool_de_diez() + ["pollo"]
    for s in range(150):
        picks = _prot_picks(_seed_call(s, dislikes=dis))
        assert POLLO not in picks, f"semilla {s}: el piso resucitó un dislike → {picks}"
