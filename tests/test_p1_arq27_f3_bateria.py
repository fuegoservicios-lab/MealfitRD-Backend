# -*- coding: utf-8 -*-
"""[P1-ARQ27-F3-BATERIA · 2026-09-06] La suite medía otro producto (ARQ27-P1-06).

`conftest.py` apaga gates a propósito, para que los tests del camino OFF no dependan del entorno. El
efecto lateral no estaba medido: **seis knobs corren en un estado que producción no usa.**

Leído del `.env` del VPS el 2026-09-06, no deducido:

| knob | producción | la suite |
|---|---|---|
| `MEALFIT_COUNTRY_SYSTEM` | `true` | `False` (default; conftest no lo toca) |
| `MEALFIT_VERIFIED_INGREDIENTS_ONLY` | `true` | `false` (conftest) |
| `MEALFIT_UPDATE_DISHES_STRICT_ALL_REASONS` | `true` | `false` (conftest) |
| `MEALFIT_SODIUM_EXCESS_GATE` | ausente ⇒ default `True` | `false` (conftest) |
| `MEALFIT_RECIPE_CONTRACT_GATE` | ausente ⇒ default `True` | `false` (conftest) |
| `MEALFIT_MICRO_CLOSER_PERDAY` | ausente ⇒ default `True` | `false` (conftest) |

El más caro es el primero: con `COUNTRY_SYSTEM` apagado los seis países colapsan a DO, así que
**ningún test de la suite ha visto nunca el catálogo de ES, US, MX, PR ni CO**.

Los otros tres enseñan una lección aparte: **un knob que no aparece en un `.env` no está apagado —
está en su default.** Los tres valen `True` en el código, así que producción los tiene encendidos y
la suite los prueba apagados. Leer un `.env` y concluir «no está configurado, luego no actúa» es el
mismo error de forma que `int(x or -1)` con `attempts=0`.

## La batería

`scripts/delivery_battery.py` aplica el perfil y recorre 16 cohortes publicando **tasas con su
denominador** por dimensión. Nunca un promedio: ahí es donde se esconde la regresión de una cohorte
pequeña, y la pequeña siempre es la vegetal.

**Dos dimensiones nacieron mal y se corrigieron antes de publicar nada.** Contaban «¿mordió la etapa
del filtro?», así que castigaban a `renal_do` y `hta_do` por no descartar ninguna plantilla — cuando
eso significa que toda la biblioteca dominicana tiene fósforo, potasio y sodio medidos, o sea el
mejor resultado posible. Ahora se miden sobre los SUPERVIVIENTES: ninguno puede llevar un alérgeno
excluido ni un nutriente exigido desconocido.

**Y la cohorte «imposible» no lo era.** Pedía cero candidatos para una vegana alérgica a gluten, soja,
frutos secos, legumbres, maní y ajonjolí con condición renal — y hay 6 desayunos que lo cumplen de
verdad. Exigir cero habría inventado un defecto. Lo que sí se le exige, y es más útil, es que sus
supervivientes sean un SUBCONJUNTO de los de la cohorte laxa: un motor que relajara en silencio
devolvería algo que la laxa no tenía, y un contador no lo vería.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import prod_profile as pp  # noqa: E402

_CONFTEST = (_BACKEND / "tests" / "conftest.py").read_text(encoding="utf-8")


# ── el perfil ─────────────────────────────────────────────────────────────────────────────────
def test_el_perfil_declara_su_procedencia_y_su_fecha():
    """Un perfil «equivalente a producción» sin fecha ni fuente es una suposición con buena letra.
    Cuando el operador cambie un knob, esto queda obsoleto y hay que releerlo."""
    assert pp.PROFILE_READ_AT == "2026-09-06"
    assert "VPS" in pp.PROFILE_SOURCE and ".env" in pp.PROFILE_SOURCE


@pytest.mark.parametrize("knob,valor", [
    ("MEALFIT_COUNTRY_SYSTEM", "true"),
    ("MEALFIT_VERIFIED_INGREDIENTS_ONLY", "true"),
    ("MEALFIT_UPDATE_DISHES_STRICT_ALL_REASONS", "true"),
    ("MEALFIT_SHOPPING_COHERENCE_GUARD", "block"),
    ("MEALFIT_PLAN_POLICY_MODE", "enforce"),
])
def test_los_knobs_de_produccion_estan_en_el_perfil(knob, valor):
    assert pp.PROD_KNOBS.get(knob) == valor


@pytest.mark.parametrize("knob", ["MEALFIT_SODIUM_EXCESS_GATE", "MEALFIT_RECIPE_CONTRACT_GATE",
                                  "MEALFIT_MICRO_CLOSER_PERDAY"])
def test_los_ausentes_del_env_corren_en_su_default(knob):
    """Un knob que no aparece en un `.env` no está apagado: está en su default. Los tres valen `True`
    en el código, así que producción los tiene ENCENDIDOS."""
    assert pp.PROD_DEFAULTS_ON.get(knob) == "true"
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert f'_env_bool("{knob}", True)' in src, (
        f"{knob} ya no tiene default True en el código: el perfil de producción miente")


@pytest.mark.parametrize("knob", ["MEALFIT_SODIUM_EXCESS_GATE", "MEALFIT_RECIPE_CONTRACT_GATE",
                                  "MEALFIT_MICRO_CLOSER_PERDAY", "MEALFIT_VERIFIED_INGREDIENTS_ONLY",
                                  "MEALFIT_UPDATE_DISHES_STRICT_ALL_REASONS"])
def test_conftest_los_apaga_y_por_eso_existe_este_gap(knob):
    """El gap no es teórico: está escrito en `conftest.py`. Si algún día deja de apagarlos, este test
    cae y habrá que quitar esa fila del perfil — que es exactamente cuándo hay que revisarlo."""
    assert f'environ.setdefault("{knob}", "false")' in _CONFTEST


def test_country_system_no_lo_toca_nadie_en_la_suite():
    """El más caro de los seis, y el único que diverge por OMISIÓN: nadie lo apaga, simplemente nadie
    lo enciende, y su default es False. Con él apagado los seis países colapsan a DO."""
    assert "MEALFIT_COUNTRY_SYSTEM" not in _CONFTEST
    src = (_BACKEND / "constants.py").read_text(encoding="utf-8")
    assert '_env_bool("MEALFIT_COUNTRY_SYSTEM", False)' in src


def test_dentro_de_la_suite_hay_divergencias_de_verdad():
    """La medida del gap. Si esto se vacía, el gap está cerrado por otra vía y la batería ya no hace
    falta — pero mientras devuelva filas, un test que no declara el flag mide otro producto."""
    div = list(pp.divergencias())
    assert div, "no hay divergencias: ¿alguien alineó conftest con producción?"
    nombres = {k for k, _, _ in div}
    assert "MEALFIT_COUNTRY_SYSTEM" in nombres


def test_lo_excluido_lleva_su_motivo():
    """Excluir sin decir por qué es indistinguible de olvidarlo. Los user_id reales no van a un
    perfil de pruebas, y esa ausencia cambia el canary: se declara, no se disimula."""
    assert "MEALFIT_PLAN_POLICY_ENFORCE_USERS" in pp.EXCLUIDOS_A_SABIENDAS
    assert "MEALFIT_INITIAL_VIA_QUEUE_USERS" in pp.EXCLUIDOS_A_SABIENDAS
    assert all(len(v) > 20 for v in pp.EXCLUIDOS_A_SABIENDAS.values()), "un motivo de dos palabras no es un motivo"


def test_el_perfil_no_lleva_secretos():
    """`CRON_SECRET` y `SUPERMARKET_ADMIN_TOKEN` no salen del VPS. Este fichero está en git."""
    todo = " ".join(list(pp.perfil_completo()) + list(pp.EXCLUIDOS_A_SABIENDAS))
    for prohibido in ("SECRET", "TOKEN", "KEY", "PASSWORD", "DATABASE_URL"):
        assert prohibido not in todo.upper(), prohibido


def test_aplicar_sobrescribe_no_hace_setdefault():
    """El sentido de `aplicar` es pisar lo que conftest dejó apagado. Con `setdefault` sería inerte —
    el modo de fallo que este repo ya conoce: una defensa cableada que nadie invoca."""
    env = {"MEALFIT_COUNTRY_SYSTEM": "false"}
    pp.aplicar(env)
    assert env["MEALFIT_COUNTRY_SYSTEM"] == "true"


# ── la batería ────────────────────────────────────────────────────────────────────────────────
def _hay_registry() -> bool:
    try:
        import dish_registry as dr
        return bool(dr.registry_hash("DO"))
    except Exception:
        return False


bateria = pytest.mark.skipif(not _hay_registry(), reason="sin snapshot compilado del registry")


@pytest.fixture(scope="module")
def resultado():
    from db_core import connection_pool
    if connection_pool is not None:
        try:
            connection_pool.open()
        except Exception:
            pass
    from scripts.delivery_battery import correr
    return correr()


@bateria
def test_cada_tasa_publica_su_denominador(resultado):
    """«Cero hallazgos en N casos no demuestra una garantía universal» — el criterio, literal. Un
    porcentaje sin su N invita justo a esa lectura."""
    for nombre, v in resultado["dimensiones"].items():
        assert "de" in v and isinstance(v["de"], int), nombre
        assert v["de"] > 0, f"{nombre} publica una tasa sobre cero casos"


@bateria
@pytest.mark.parametrize("dim", ["seguridad", "nutricion", "sin_relajacion"])
def test_cero_violaciones_duras(dim, resultado):
    """Las tres dimensiones donde cualquier valor distinto de 100 % es un fallo del producto: un
    alérgeno excluido que sobrevive, un nutriente exigido desconocido, o una restricción que devuelve
    algo que la cohorte laxa no tenía."""
    assert resultado["dimensiones"][dim]["pct"] == 100.0, resultado["dimensiones"][dim]


@bateria
def test_ninguna_cohorte_se_queda_sin_franja(resultado):
    vacias = {f["id"]: f["franjas_vacias"] for f in resultado["cohortes"] if f["franjas_vacias"]}
    assert not vacias, f"cohortes sin candidatos en alguna franja: {vacias}"


@bateria
def test_las_seis_cocinas_estan_en_la_matriz(resultado):
    """Con `COUNTRY_SYSTEM` apagado las seis colapsan a DO; la batería existe para verlas separadas."""
    assert {f["pais"] for f in resultado["cohortes"]} == {"DO", "ES", "US", "MX", "PR", "CO"}


@bateria
def test_la_bateria_declara_lo_que_no_mide(resultado):
    """El canary —latencia y coste por plan entregado— necesita generaciones reales. Fingirlo en una
    batería determinista sería peor que no tenerlo, así que queda escrito como trabajo abierto."""
    no_medido = " ".join(resultado["no_medido"]).lower()
    assert "latencia" in no_medido and "coste" in no_medido
    assert "swap" in no_medido


# ── la batería no puede contaminar a los demás ────────────────────────────────────────────────
def test_el_perfil_se_restaura_al_salir():
    """`perfil_aplicado` devuelve el entorno EXACTO, incluidas las claves que no existían: si se
    quedaran con el valor de producción, el siguiente test mediría otro producto."""
    env = {"MEALFIT_COUNTRY_SYSTEM": "false"}
    with pp.perfil_aplicado(env):
        assert env["MEALFIT_COUNTRY_SYSTEM"] == "true"
        assert env["MEALFIT_SODIUM_EXCESS_GATE"] == "true"
    assert env == {"MEALFIT_COUNTRY_SYSTEM": "false"}, f"quedaron restos: {env}"


def test_importar_la_bateria_no_toca_el_entorno():
    """**El defecto que este test existe para impedir.** La primera versión aplicaba el perfil en
    tiempo de import, y como este fichero importa la batería, 17 tests de coherencia, compras y
    hierbas empezaron a correr con los gates de producción encendidos por debajo — pasaban 8 de 8
    aislados y caían 4 de 8 detrás de la batería.

    Una herramienta que existe para denunciar que la suite corre con otras banderas no puede ser
    quien se las cambie a los demás."""
    import os
    antes = {k: os.environ.get(k) for k in pp.perfil_completo()}
    import scripts.delivery_battery  # noqa: F401
    import importlib
    importlib.reload(scripts.delivery_battery)
    despues = {k: os.environ.get(k) for k in pp.perfil_completo()}
    assert antes == despues, f"importar la batería mutó el entorno: "\
        f"{ {k: (antes[k], despues[k]) for k in antes if antes[k] != despues[k]} }"


def test_correr_deja_el_entorno_como_estaba(resultado):
    """Y tampoco al EJECUTARLA: el fixture ya corrió la batería entera antes de este test."""
    import os
    div = {k for k, _, _ in pp.divergencias()}
    assert "MEALFIT_COUNTRY_SYSTEM" in div, (
        "tras correr la batería el entorno se quedó con las banderas de producción")
    assert os.environ.get("MEALFIT_SHOPPING_COHERENCE_GUARD") != "block" or \
        "MEALFIT_SHOPPING_COHERENCE_GUARD" not in pp.PROD_KNOBS
