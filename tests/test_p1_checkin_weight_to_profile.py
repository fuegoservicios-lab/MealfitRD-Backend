"""[P1-CHECKIN-WEIGHT-TO-PROFILE · 2026-07-26] El peso del check-in no llegaba al cálculo.

El modal "Antes de tu nuevo ciclo" pide **Peso actual** antes de renovar. El endpoint escribía ese
peso en `weight_history` — que alimenta la TENDENCIA del motor evolutivo (±5-10%) — y **nada más**.
El BMR se calcula con `health_profile.weight`, que no se tocaba.

Consecuencia medida: perfil en 130 lb, tres planes seguidos a 2050 kcal por la fórmula estándar, y
`engine_active=False` (0 registros). Si el usuario escribía 125, el plan salía igualmente para 130.
La etiqueta dice "Peso actual" y el usuario espera que el plan sea para ese peso.

## Por qué NO se convierte entre unidades

Se guarda el par `(valor, unidad)` tal como lo declaró el usuario. Convertir exigiría confiar en la
unidad vieja del perfil, que es justo la que puede estar mal: hay perfiles con `weight=75` y
`weightUnit=None` — 75 lb no es un adulto plausible, casi seguro son kg. Escribir ambos campos SANA
esa ambigüedad en vez de propagarla.

## Lo que este fix NO toca

`hunger`, `energy` y `adherence_pct` siguen archivándose en `_renewal_checkins` sin que nadie los
lea (verificado: 4 escrituras, 0 lecturas en todo el backend). Usarlos para mover calorías es una
decisión clínica, no mecánica, y queda pendiente de decidir — no de implementar a ciegas.
"""
import os

import pytest


def _mutador(hp, peso, unidad, hoy="2026-07-26"):
    """Replica del mutator del endpoint, para probar el CONTRATO sin tocar la DB."""
    import routers.plans as rp
    _wh = [e for e in list(hp.get("weight_history") or [])
           if not (isinstance(e, dict) and e.get("date") == hoy)]
    _wh.append({"date": hoy, "weight": peso, "unit": unidad})
    hp["weight_history"] = _wh
    if os.environ.get("MEALFIT_CHECKIN_WEIGHT_TO_PROFILE", "true").lower() in ("1", "true", "yes", "on"):
        hp["weight"] = peso
        hp["weightUnit"] = unidad
    return hp


def test_el_peso_actualiza_el_perfil():
    """Lo que el BMR lee es `health_profile.weight`."""
    hp = {"weight": 130, "weightUnit": "lb"}
    _mutador(hp, 125.0, "lb")
    assert hp["weight"] == 125.0
    assert hp["weightUnit"] == "lb"


def test_tambien_sigue_alimentando_la_tendencia():
    """El fix AÑADE, no sustituye: `weight_history` es lo que usa el motor evolutivo."""
    hp = {"weight": 130, "weightUnit": "lb"}
    _mutador(hp, 125.0, "lb")
    assert hp["weight_history"][-1] == {"date": "2026-07-26", "weight": 125.0, "unit": "lb"}


def test_sana_el_perfil_con_unidad_ambigua():
    """Caso real en la base: `weight=75, weightUnit=None`. 75 lb no es un adulto plausible."""
    hp = {"weight": 75, "weightUnit": None}
    _mutador(hp, 165.0, "lb")
    assert (hp["weight"], hp["weightUnit"]) == (165.0, "lb")


def test_no_convierte_entre_unidades():
    """Convertir exigiría confiar en la unidad vieja, que es la que puede estar mal."""
    hp = {"weight": 75, "weightUnit": None}
    _mutador(hp, 56.0, "kg")
    assert hp["weight"] == 56.0 and hp["weightUnit"] == "kg"


def test_dedupe_por_dia():
    """Dos check-ins el mismo día: gana el último, no se duplica el registro."""
    hp = {}
    _mutador(hp, 130.0, "lb")
    _mutador(hp, 129.0, "lb")
    assert len(hp["weight_history"]) == 1
    assert hp["weight_history"][0]["weight"] == 129.0
    assert hp["weight"] == 129.0


def test_knob_de_rollback(monkeypatch):
    monkeypatch.setenv("MEALFIT_CHECKIN_WEIGHT_TO_PROFILE", "false")
    hp = {"weight": 130, "weightUnit": "lb"}
    _mutador(hp, 125.0, "lb")
    assert hp["weight"] == 130, "con el knob apagado el perfil no se toca"
    assert hp["weight_history"][-1]["weight"] == 125.0, "la tendencia SÍ se sigue registrando"


# ───────────── el endpoint conserva sus guardas ─────────────

def _fuente_endpoint() -> str:
    from pathlib import Path
    import routers.plans as rp
    return Path(rp.__file__).read_text(encoding="utf-8")


def test_sigue_validando_rango_y_unidad():
    src = _fuente_endpoint()
    i = src.index('@router.post("/renewal-checkin")')
    bloque = src[i:i + 4000]
    assert "0 < _w <= 2000" in bloque
    assert '_unit not in ("lb", "kg")' in bloque
    assert "_clamp_int" in bloque


def test_la_escritura_del_perfil_va_dentro_del_mutator_atomico():
    """Fuera del mutator sería un read-modify-write con ventana de lost-update contra el resto de
    escritores del health_profile."""
    src = _fuente_endpoint()
    i = src.index("def _checkin_mutator")
    j = src.index("update_user_health_profile_atomic", i)
    assert '_hp["weight"] = _w' in src[i:j]


def test_el_endpoint_reporta_si_actualizo_el_perfil():
    src = _fuente_endpoint()
    assert '"profile_weight_updated"' in src
    assert '"profile_weight_prev"' in src
