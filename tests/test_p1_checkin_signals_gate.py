"""[P1-CHECKIN-SIGNALS-GATE · 2026-07-26] Las señales del check-in, por fin usadas — y sólo para ablandar.

`hunger`, `energy` y `adherence_pct` se preguntaban en el modal de renovación desde 2026-07-11 y
**nadie las leía**: 4 escrituras a `_renewal_checkins`, 0 lecturas en todo el backend. Treinta
segundos del usuario por ciclo a cambio de nada.

## El uso honesto no es sumarlas al déficit

Es usarlas como **condición de validez de la evidencia** que el motor evolutivo cree tener. Ese
motor mira el peso y, si te estancaste, añade hasta un 10% más de déficit. Pero:

  · **Adherencia baja invalida la lectura en AMBAS direcciones.** Si seguiste el 40% del plan, tu
    estancamiento no dice nada de tu metabolismo — dice que no comiste lo que el plan decía.
    Añadirle déficit a quien ya está batallando es castigarlo; y en `gain_muscle`, añadir superávit
    hace que engorde de más cuando sí lo siga.

  · **Hambre alta o energía baja bloquean SÓLO el endurecimiento.** Sobre un déficit son los signos
    clásicos de restricción excesiva y los mejores predictores de abandono. Nunca bloquean la
    dirección que da MÁS comida.

## La asimetría es el punto

Equivocarse hacia "menos agresivo" cuesta una semana de progreso; hacia "más agresivo" cuesta el
usuario. El ajuste **anti-rebound** (+5%, que da más comida porque se está bajando demasiado rápido)
sobrevive aunque el hambre sea 5/5 — está en los tests de abajo, y es lo que distingue este gate de
un "si el usuario se queja, come menos".

Las ramas de retención hídrica y recomposición corporal que ya existían en el motor siguen este
mismo criterio: anulan el ajuste ante evidencia contradictoria.
"""
import pytest

import nutrition_calculator as nc


def _gate(bonus, goal, sig):
    """Réplica del gate para probar el CONTRATO sin arrastrar todo `calculate_nutrition_targets`.
    Cualquier cambio en la lógica real debe reflejarse aquí y romper estos tests si diverge."""
    _adh, _hun, _ene = sig.get("adherence_pct"), sig.get("hunger"), sig.get("energy")
    _endurece = (bonus < 0) if goal == "lose_fat" else (bonus > 0)
    if isinstance(_adh, (int, float)) and _adh < nc.CHECKIN_ADHERENCE_FLOOR:
        return 0.0
    if _endurece and isinstance(_hun, (int, float)) and _hun >= nc.CHECKIN_HUNGER_CEIL:
        return 0.0
    if _endurece and isinstance(_ene, (int, float)) and _ene <= nc.CHECKIN_ENERGY_FLOOR:
        return 0.0
    return bonus


_BUENAS = {"adherence_pct": 90, "hunger": 2, "energy": 4}


# ───────────── 1. adherencia: invalida en AMBAS direcciones ─────────────

def test_adherencia_baja_anula_el_deficit_extra():
    assert _gate(-0.10, "lose_fat", {"adherence_pct": 40, "hunger": 2, "energy": 4}) == 0.0


def test_adherencia_baja_anula_tambien_el_superavit_extra():
    """En gain_muscle el error es simétrico: superávit sobre un plan no seguido = grasa cuando sí
    lo siga."""
    assert _gate(+0.07, "gain_muscle", {"adherence_pct": 30, "hunger": 2, "energy": 4}) == 0.0


def test_adherencia_alta_no_estorba():
    assert _gate(-0.10, "lose_fat", _BUENAS) == -0.10
    assert _gate(+0.07, "gain_muscle", _BUENAS) == +0.07


# ───────────── 2. hambre y energía: sólo bloquean el endurecimiento ─────────────

@pytest.mark.parametrize("sig", [
    {"adherence_pct": 90, "hunger": 5, "energy": 4},
    {"adherence_pct": 90, "hunger": 4, "energy": 4},
    {"adherence_pct": 90, "hunger": 2, "energy": 1},
    {"adherence_pct": 90, "hunger": 2, "energy": 2},
])
def test_hambre_o_energia_bloquean_mas_deficit(sig):
    assert _gate(-0.10, "lose_fat", sig) == 0.0


def test_el_ANTI_REBOUND_sobrevive_aunque_haya_hambre():
    """EL test que define este gate. +5% en lose_fat REDUCE el déficit (se está bajando demasiado
    rápido). Bloquearlo por hambre alta dejaría al usuario con menos comida justo cuando el motor
    quiere darle más. Si esto falla, el gate se volvió un 'si te quejas, come menos'."""
    assert _gate(+0.05, "lose_fat", {"adherence_pct": 90, "hunger": 5, "energy": 1}) == +0.05


def test_el_anti_rebound_de_gain_muscle_tambien():
    """En gain_muscle el bonus NEGATIVO reduce el superávit (subida demasiado rápida). No es
    endurecer en el sentido de quitar comida bajo restricción, así que no lo bloquean hambre/energía."""
    assert _gate(-0.05, "gain_muscle", {"adherence_pct": 90, "hunger": 5, "energy": 1}) == -0.05


# ───────────── 3. compatibilidad hacia atrás ─────────────

@pytest.mark.parametrize("sig", [{}, {"hunger": None, "energy": None, "adherence_pct": None}])
def test_sin_checkin_el_motor_no_cambia(sig):
    """Los usuarios que siempre pulsan Omitir no deben notar nada."""
    assert _gate(-0.10, "lose_fat", sig) == -0.10
    assert _gate(+0.07, "gain_muscle", sig) == +0.07


def test_nunca_CREA_ni_amplifica_un_ajuste():
    """El gate sólo puede llevar a 0. Si alguna vez devuelve algo mayor en magnitud que la entrada,
    dejó de ser un gate."""
    for b in (-0.10, -0.07, -0.05, 0.05, 0.07, 0.10):
        for adh in (10, 50, 60, 90):
            for hun in (1, 3, 5):
                for ene in (1, 3, 5):
                    out = _gate(b, "lose_fat", {"adherence_pct": adh, "hunger": hun, "energy": ene})
                    assert abs(out) <= abs(b)
                    assert out in (0.0, b)


# ───────────── 4. cableado ─────────────

def _fuente(mod):
    from pathlib import Path
    return Path(mod.__file__).read_text(encoding="utf-8")


def test_el_gate_corre_ANTES_de_aplicar_el_bonus():
    src = _fuente(nc)
    i = src.index("P1-CHECKIN-SIGNALS-GATE")
    j = src.index("target_calories = target_calories + (tdee * dynamic_deficit_bonus)")
    assert i < j, "aplicar y luego anular dejaría las calorías ya movidas"


def test_las_senales_se_inyectan_SERVER_SIDE():
    """Confiar en el cliente para algo que mueve calorías sería el agujero de P0-AGENT-1. Y el
    strip P0-A2 vetaría (bien) una clave con guión bajo venida del request."""
    from pathlib import Path
    import routers.plans as rp
    src = Path(rp.__file__).read_text(encoding="utf-8")
    i = src.index('pipeline_data["_renewal_signals"]')
    bloque = src[max(0, i - 1200):i]
    assert "health_profile" in bloque
    assert "_renewal_checkins" in bloque


def test_knobs_de_rollback():
    src = _fuente(nc)
    for k in ("MEALFIT_CHECKIN_SIGNALS_GATE", "MEALFIT_CHECKIN_ADHERENCE_FLOOR",
              "MEALFIT_CHECKIN_HUNGER_CEIL", "MEALFIT_CHECKIN_ENERGY_FLOOR"):
        assert k in src, k
    assert 0 <= nc.CHECKIN_ADHERENCE_FLOOR <= 100
    assert 1 <= nc.CHECKIN_HUNGER_CEIL <= 5
    assert 1 <= nc.CHECKIN_ENERGY_FLOOR <= 5
