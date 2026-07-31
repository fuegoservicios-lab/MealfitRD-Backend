"""[P3-AUDIT-V6-LOCALES · 2026-07-31] (audit solver+seeder v6 · P3 / F25, F30, F16, F26)
Cuatro P3 locales, cada uno con su propio marker de producción.

F25 · P3-SWAP-SAMEDAY-ARITY — `_same_day_other_meals_for_swap` devolvía `[]` (un valor) en 3 de sus
returns cuando su contrato son DOS listas. Hoy lo tapa el try/except del caller, que degrada en
silencio a swap sin variedad same-day ni blobs para el gate determinista. Un contrato que solo se
sostiene por el `except` de su llamador no es un contrato.

F30 · P3-SEEDER-KNOBS-REGISTRY — dos knobs leían `os.environ` en crudo, así que no se
auto-registraban y eran invisibles en `/health/version`. Durante un incidente el operador consulta
el snapshot, no los ve, y concluye que la palanca no existe.

F16 · P3-PANTRY-INTERSECT-WB — substring BIDIRECCIONAL con umbral de 3 chars en la intersección con
la nevera del Edge Recipe: `'sal'` hacía preferir `'Salami Dominicano'`. La 13ª subcadena de la
serie del repo.

F26 · P3-SWAP-INSPIRATION-DIET — la inspiración de swap ofrecía el pool ancho (pollo/res/pescado…)
sin mirar dieta ni alergias: el LLM adaptaba una plantilla de carne para un vegetariano, el backstop
la rechazaba y el swap quemaba reintentos — un fallo de producto inducido por una señal que el
propio sistema fabricó.
"""
import re

import pytest


# --------------------------------------------------------------- F25 · aridad

def test_el_helper_de_swap_devuelve_siempre_dos_listas():
    """Los 3 returns de guarda deben respetar el contrato, no depender del except del caller."""
    import inspect
    from routers import plans

    src = inspect.getsource(plans._same_day_other_meals_for_swap)
    sueltos = [l.strip() for l in src.splitlines() if l.strip() == "return []"]
    assert not sueltos, (
        f"{len(sueltos)} return(s) devuelven una sola lista donde el contrato son dos; "
        f"el caller hace `a, b = ...` y lanzaría ValueError"
    )


def test_el_helper_de_swap_desempaqueta_sin_datos():
    """Funcional: con un usuario sin planes debe desempaquetar limpio."""
    from routers.plans import _same_day_other_meals_for_swap

    nombres, blobs = _same_day_other_meals_for_swap(
        "00000000-0000-0000-0000-000000000000", "Plato Inexistente")
    assert nombres == [] and blobs == []


# --------------------------------------------------------------- F30 · knobs visibles

def test_los_knobs_del_seeder_estan_en_el_registry():
    """Un knob que no se registra es invisible en /health/version justo cuando hace falta."""
    import ai_helpers
    from knobs import get_knobs_registry_snapshot

    ai_helpers.get_deterministic_variety_prompt("", {"mainGoal": "maintenance"}, user_id=None)
    reg = get_knobs_registry_snapshot()
    faltan = [k for k in ("MEALFIT_TRANSFORM_BASE_BOOST", "MEALFIT_BUDGET_POOL_WEIGHT")
              if k not in reg]
    assert not faltan, f"knobs que eluden `_KNOBS_REGISTRY`: {faltan}"


def test_los_knobs_del_seeder_no_leen_environ_en_crudo():
    """tooltip-anchor de producción: P3-SEEDER-KNOBS-REGISTRY"""
    import inspect
    import ai_helpers

    src = inspect.getsource(ai_helpers.get_deterministic_variety_prompt)
    for knob in ("MEALFIT_TRANSFORM_BASE_BOOST", "MEALFIT_BUDGET_POOL_WEIGHT"):
        m = re.search(r"^.*" + knob + r".*$", src, re.M)
        assert m and "environ" not in m.group(0), (
            f"{knob} sigue leyéndose con os.environ crudo: {m.group(0).strip() if m else '?'}"
        )


# --------------------------------------------------------------- F16 · word-boundary en la nevera

def test_la_sal_de_la_nevera_no_hace_preferir_salami():
    """'sal' (3 chars) matcheaba dentro de 'Salami Dominicano' por substring bidireccional."""
    import cron_tasks

    dia = cron_tasks._build_filtered_edge_recipe_day(
        [], [], "", pantry_items=["sal", "arroz", "cebolla"])
    if dia is None:
        pytest.skip("el catálogo no permitió construir un edge day en este entorno")
    texto = " ".join(i for m in dia["meals"] for i in m["ingredients"]).lower()
    assert "salami" not in texto, (
        f"un salero en la nevera hizo entrar embutido al día degradado: {texto}"
    )


def test_la_nevera_sigue_intersectando_por_nombre_real():
    """Control: el mecanismo no se rompe — un alimento REAL de la nevera sí debe preferirse."""
    import cron_tasks

    dia = cron_tasks._build_filtered_edge_recipe_day(
        [], [], "", pantry_items=["arroz blanco"])
    if dia is None:
        pytest.skip("el catálogo no permitió construir un edge day en este entorno")
    texto = " ".join(i for m in dia["meals"] for i in m["ingredients"]).lower()
    assert "arroz" in texto, f"el arroz de la nevera no se prefirió: {texto}"


# --------------------------------------------------------------- F26 · inspiración diet-aware

def test_la_inspiracion_de_swap_no_propone_carne_a_un_vegano():
    import dish_library as dl

    pool = dl._diet_safe_pool_ascii(diet_type="vegana", allergies=[])
    for prohibido in ("pollo", "res", "cerdo", "pavo", "pescado", "atun", "camarones", "salmon"):
        assert prohibido not in pool.split(), f"'{prohibido}' sigue en el pool de un vegano: {pool}"


def test_la_inspiracion_de_swap_respeta_alergias():
    import dish_library as dl

    pool = dl._diet_safe_pool_ascii(diet_type=None, allergies=["Mariscos"]).split()
    assert "camarones" not in pool, f"camarones ofrecidos a un alérgico a mariscos: {pool}"


def test_la_inspiracion_sin_restricciones_no_cambia():
    """Control: el caso común conserva el pool ancho (no se degrada la creatividad)."""
    import dish_library as dl
    assert dl._diet_safe_pool_ascii(None, []) == dl._BROAD_POOL_ASCII
    assert dl._diet_safe_pool_ascii("balanced", []).split() == dl._BROAD_POOL_ASCII.split()


def test_el_callsite_del_swap_pasa_dieta_y_alergias():
    """tooltip-anchor de producción: P3-SWAP-INSPIRATION-DIET"""
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "agent.py").read_text(
        encoding="utf-8", errors="ignore")
    i = src.index("_bsi_swap(")
    bloque = src[i: i + 700]
    assert "diet_type=" in bloque and "allergies=" in bloque, (
        "el swap invoca la inspiración sin pasar dieta/alergias, que ya están en scope"
    )
