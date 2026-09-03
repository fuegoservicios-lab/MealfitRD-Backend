"""[P2-COND-SUBS-COUNTRY-HALF · 2026-08-23] De las DOS puertas de sustitución clínica,
`P1-CONDITION-RULES-COUNTRY` sólo cerró una.

LO MEDIDO ANTES DE TOCAR NADA (`collect_substitutions`, perfil con Embarazo):

    Embarazo DO: n=3 repl=['Filete de pescado blanco', 'Mango', 'Tayota']
    Embarazo ES: n=3 repl=['Filete de pescado blanco', 'Mango', 'Tayota']   ← IDÉNTICO
    (contraste con la puerta que SÍ se cerró: Gluten DO n=6 · beta n=4)

«Tayota» es el nombre dominicano del chayote y está en la propia lista `_DO_ONLY_SUB_TARGETS`.
Ese string viaja al plato, a la receta y a la lista de la compra de un usuario de Madrid.

EL ARREGLO OBVIO CONVERTÍA UN P2 DE RECONOCIBILIDAD EN EXPOSICIÓN CLÍNICA. La puerta hermana
(`collect_allergen_substitutions`) resuelve esto con `continue`, y ahí está bien: omitir la
sustitución de un ALÉRGENO es seguro porque el plan cae al path crítico→fallback, que RECHAZA el
plato. Para las condiciones NO existe ese path: la única sustitución es-DO de las 12 reglas es el
reemplazo del **cundeamor** (uterotónico/abortivo) en embarazo, así que omitirla no deja al
usuario sin plato — lo deja con el ingrediente contraindicado dentro.

Por eso el cierre REMAPEA en vez de omitir, con un mapa CERRADO: target sin equivalente
registrado ⇒ se conserva el dominicano (peor nombre, misma seguridad).

Este fichero ancla las cuatro propiedades que no pueden romperse:
  A. Byte-identidad DO.
  B. Beta no pierde NI UNA sustitución (el conteo es el mismo que el de DO).
  C. Ningún target es-DO sobrevive en beta.
  D. El equivalente resuelve a fila viva del catálogo (e2e) — un target fantasma sería
     P2-SUBS-RESOLVE otra vez.
"""
from __future__ import annotations

import pytest

import condition_rules as cr
from constants import COUNTRY_PROFILES

_BETA = sorted(cc for cc, p in COUNTRY_PROFILES.items() if p.get("is_beta"))
#: El perfil que dispara la única sustitución es-DO del módulo.
_EMBARAZO = {"medicalConditions": ["Embarazo"]}


@pytest.fixture(autouse=True)
def _sistema_de_paises_encendido(monkeypatch):
    """Sin el knob se mide el sistema APAGADO: `country_for_form_data` devuelve 'DO' para todos y
    el test pasaría por la razón equivocada."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")


def _subs(cc, diet=None):
    fd = dict(_EMBARAZO, country=cc)
    return cr.collect_substitutions(fd, diet_type=diet)


# ── A. Byte-identidad dominicana ────────────────────────────────────────────────────────────────

def test_dominicana_conserva_su_target_criollo():
    repl = [s["replacement"] for s in _subs("DO")]
    assert any(cr._is_do_only_target(r) for r in repl), (
        f"DO dejó de recibir su target es-DO: {repl}")


# ── B. Beta no pierde ninguna sustitución ───────────────────────────────────────────────────────

@pytest.mark.parametrize("cc", _BETA)
def test_beta_conserva_el_mismo_numero_de_sustituciones_que_do(cc):
    """La propiedad que separa este cierre del `continue` de la puerta hermana. Si alguien
    'simplifica' remapear→omitir, aquí se ve: beta se quedaría con una sustitución menos y el
    cundeamor seguiría en el plato."""
    assert len(_subs(cc)) == len(_subs("DO")), (
        f"{cc} recibe {len(_subs(cc))} sustituciones y DO {len(_subs('DO'))}: "
        "omitir una sustitución de CONDICIÓN deja el ingrediente contraindicado servido")


@pytest.mark.parametrize("cc", _BETA)
def test_beta_sigue_teniendo_reemplazo_para_el_uterotonico(cc):
    from constants import strip_accents
    labels = [s["label"] for s in _subs(cc)]
    assert any("uteroton" in strip_accents(l.lower()) for l in labels), (
        f"{cc} perdió la sustitución del uterotónico: {labels}")


# ── C. Ningún target es-DO sobrevive en beta ────────────────────────────────────────────────────

@pytest.mark.parametrize("cc", _BETA)
@pytest.mark.parametrize("diet", [None, "vegana", "vegetariana", "pescetariano"])
def test_ningun_target_solo_dominicano_llega_a_un_pais_beta(cc, diet):
    """Se cruza con la dieta a propósito: el redirect veg* corre sobre el MISMO valor, y un orden
    equivocado entre los dos dejaría el target criollo pasar para alguna combinación."""
    malos = [s["replacement"] for s in _subs(cc, diet) if cr._is_do_only_target(s["replacement"])]
    assert not malos, f"{cc}/{diet}: targets sólo-RD servidos a un país beta: {malos}"


@pytest.mark.parametrize("cc", _BETA)
def test_el_equivalente_no_es_vacio_ni_es_el_mismo_target_criollo(cc):
    for s in _subs(cc):
        assert str(s["replacement"] or "").strip(), "un reemplazo vacío borraría el ingrediente"


def test_el_mapa_de_equivalentes_es_cerrado_y_fail_safe():
    """Sin entrada NO se omite: se conserva el target dominicano. Es la diferencia entre 'peor
    nombre' y 'ingrediente contraindicado servido'."""
    assert cr._neutralize_do_only_target("Casabe") == "Casabe"
    assert cr._neutralize_do_only_target(None) is None
    for clave, destino in cr._DO_ONLY_TARGET_NEUTRAL_EQUIVALENTS.items():
        assert clave in cr._DO_ONLY_SUB_TARGETS, (
            f"{clave!r} no es un target es-DO: el mapa neutraliza lo que no hacía falta")
        assert not cr._is_do_only_target(destino), (
            f"{clave!r} → {destino!r} sigue siendo un target sólo dominicano")


# ── D. El equivalente existe de verdad ──────────────────────────────────────────────────────────

@pytest.mark.e2e
def test_todo_equivalente_neutro_resuelve_a_fila_viva_del_catalogo():
    """P2-SUBS-RESOLVE: un target que no resuelve entra en la receta y sale de la lista de la
    compra como fantasma, con el delta de macros perdiendo el ingrediente."""
    try:
        from shopping_calculator import get_master_ingredients
        filas = get_master_ingredients() or []
    except Exception as e:  # pragma: no cover
        pytest.skip(f"catálogo no disponible: {e}")
    if not filas:
        pytest.skip("catálogo vacío (¿pool de Neon sin abrir?)")
    vivos = {str(r.get("name") or "").strip() for r in filas}
    faltan = [d for d in cr._DO_ONLY_TARGET_NEUTRAL_EQUIVALENTS.values() if d not in vivos]
    assert not faltan, f"equivalentes sin fila en master_ingredients: {faltan}"
