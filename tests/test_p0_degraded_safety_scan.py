"""[P0-DEGRADED-SAFETY-SCAN · 2026-07-31] (audit solver+seeder v6 · P0) El path degradado
entregaba comidas construidas del catálogo mal filtrado SIN ningún escáner de dieta/alérgenos.

Cuando el chunk worker entra en modo degraded (probe del LLM caído), `_build_filtered_edge_recipe_day`
arma días con `random.choice()` sobre el output de `_get_fast_filtered_catalogs` y el resultado se
PERSISTE en `plan_data`. Esa rama BYPASEA `assemble_plan_node` — o sea que ni `review_plan_node`, ni
el allergen guard (C2-ALLERGEN-GUARD), ni el diet hard guard (P1-DIET-HARD-GUARD) llegan a verla.
Grep sobre las 33.773 líneas de cron_tasks.py: CERO llamadas a `_scan_diet_violations` /
`_scan_allergen_violations` / `clinical_backstop_for_meal`.

La única defensa era el blocklist substring de `_is_blocked`, que falla por dos lados:
  (a) NO incluye la dieta (blocklist = allergies + dislikes);
  (b) compara el LABEL del chip contra el texto — 'frutos secos' no es substring de
      'Almendras fileteadas', así que un chip de categoría no bloquea nada.

Y el filtro de catálogo upstream tiene agujeros medidos que ese blocklist no tapa:
  · dietType femenino legacy ('vegetariana'/'vegana') no matchea NINGUNA rama de
    `_get_fast_filtered_catalogs` (constants.py:2417-2422 solo lista masculinos) ⇒ el pool sale
    con Pollo/Res para una vegetariana;
  · los catch-alls de alérgenos expanden a SINGULARES contra un catálogo en PLURAL
    ('Almendras fileteadas', 'Pistachos', 'Nueces/Almendras' sobreviven al chip 'Frutos Secos').

La cura NO es parchear el filtro (eso es F11/F13, aguas arriba y para todos los callers): es que
esta superficie tenga el MISMO backstop determinista que el path LLM. `clinical_backstop_for_meal`
ya expande sinónimos vía `_ALLERGEN_SYNONYMS` (plurales incluidos) y canonicaliza la dieta vía
`_canonicalize_diet_type` (femeninos incluidos) — reusarlo cierra el P0 con independencia de los
agujeros del filtro.

Anchor de producción: P0-DEGRADED-SAFETY-SCAN.
"""
import random
import re
from pathlib import Path

import pytest

CRON = Path(__file__).resolve().parent.parent / "cron_tasks.py"

# Suficientes vueltas para que el sorteo de 3 slots proteicos sobre un pool con ~30 animales
# de 47 no pueda esquivar la violación por suerte (pre-fix la probabilidad de 0 hallazgos es ~0).
_VUELTAS = 25


def _violaciones(dia, allergies, diet):
    """Veredicto INDEPENDIENTE del código bajo prueba: el mismo backstop que corre el path LLM."""
    from graph_orchestrator import clinical_backstop_for_meal

    out = []
    for meal in (dia or {}).get("meals", []) or []:
        out.extend(clinical_backstop_for_meal(meal, allergies=allergies, diet_type=diet))
    return out


def _construir_muchas_veces(allergies, dislikes, diet, semilla=20260730):
    """Devuelve la lista de días construidos (los None se descartan: negarse es seguro)."""
    from cron_tasks import _build_filtered_edge_recipe_day

    random.seed(semilla)
    dias = []
    for _ in range(_VUELTAS):
        d = _build_filtered_edge_recipe_day(allergies, dislikes, diet)
        if d is not None:
            dias.append(d)
    return dias


# --------------------------------------------------------------- el P0: dieta

def test_vegetariana_femenino_nunca_recibe_carne():
    """El femenino legacy es el que elude el filtro de catálogo — y el que más importa."""
    dias = _construir_muchas_veces([], [], "vegetariana")
    malos = [v for d in dias for v in _violaciones(d, [], "vegetariana")]
    assert not malos, (
        f"{len(malos)} violaciones de dieta en {len(dias)} Edge Recipes para 'vegetariana'. "
        f"Muestra: {malos[:5]}"
    )


def test_vegetariano_masculino_nunca_recibe_carne():
    """Control: la forma que el filtro SÍ matchea debe seguir limpia (el fix no la rompe)."""
    dias = _construir_muchas_veces([], [], "vegetariano")
    malos = [v for d in dias for v in _violaciones(d, [], "vegetariano")]
    assert not malos, f"violaciones con el masculino (regresión): {malos[:5]}"


def test_vegano_nunca_recibe_producto_animal():
    """Vegano prohíbe además huevo y lácteo — el catálogo de vegetales/grasas tiene quesos."""
    dias = _construir_muchas_veces([], [], "vegana")
    malos = [v for d in dias for v in _violaciones(d, [], "vegana")]
    assert not malos, f"violaciones veganas: {malos[:5]}"


@pytest.mark.parametrize("femenino,masculino", [
    ("vegetariana", "vegetariano"),
    ("vegana", "vegano"),
    ("pescetariana", "pescetariano"),
])
def test_el_femenino_acaba_con_el_mismo_pool_que_el_masculino(femenino, masculino):
    """La invariante fuerte: el femenino no queda "más seguro" ni "menos", queda IGUAL.

    Es mejor aserción que "cero violaciones" porque fija el destino, no solo la ausencia de daño:
    la forma que eludía el filtro converge exactamente a la que ya funcionaba. Pre-fix el pool
    proteico del femenino era el catálogo entero (49) y el del masculino 23.
    """
    from cron_tasks import _sieve_catalog_for_safety
    from constants import _get_fast_filtered_catalogs

    def _pool(d):
        p, _c, _v, _f = _get_fast_filtered_catalogs((), (), d)
        return set(_sieve_catalog_for_safety(p, [], d))

    fem, mas = _pool(femenino), _pool(masculino)
    assert fem == mas, (
        f"'{femenino}' y '{masculino}' deben ofrecer el mismo pool proteico. "
        f"Solo en femenino: {sorted(fem - mas)[:6]} · solo en masculino: {sorted(mas - fem)[:6]}"
    )
    assert fem, "el pool no puede quedar vacío: sería fail-secure inútil (cero Edge Recipes)"


def test_el_dia_se_sigue_construyendo_para_dietas_restrictivas():
    """Un fix que devolviera siempre None sería seguro e inútil. Debe SEGUIR entregando días."""
    for dieta in ("vegetariana", "vegana", "pescetariana"):
        dias = _construir_muchas_veces([], [], dieta)
        assert len(dias) == _VUELTAS, (
            f"con dieta {dieta!r} solo se construyeron {len(dias)}/{_VUELTAS} Edge Recipes: "
            f"el guard está degradando el producto, no solo protegiéndolo"
        )


# --------------------------------------------------------------- el P0: alérgenos

def test_chip_frutos_secos_no_deja_pasar_los_plurales():
    """'Almendras fileteadas' / 'Pistachos' / 'Nueces/Almendras' viven en el catálogo de grasas."""
    alerg = ["Frutos Secos"]
    dias = _construir_muchas_veces(alerg, [], "")
    malos = [v for d in dias for v in _violaciones(d, alerg, "")]
    assert not malos, f"alérgeno IgE servido en el path degradado: {malos[:5]}"


def test_chip_gluten_no_deja_pasar_bulgur_ni_cebada():
    """El catch-all de gluten del filtro está drifteado vs `_ALLERGEN_SYNONYMS`."""
    alerg = ["Gluten"]
    dias = _construir_muchas_veces(alerg, [], "")
    malos = [v for d in dias for v in _violaciones(d, alerg, "")]
    assert not malos, f"gluten servido a un celíaco en el path degradado: {malos[:5]}"


# --------------------------------------------------------------- el helper reusable

def test_helper_detecta_violacion_en_un_dia_del_pool():
    """Los prior days del Smart Shuffle también pasan por aquí: las restricciones CAMBIAN."""
    from cron_tasks import _degraded_safety_violations

    dia = {"meals": [{"name": "Almuerzo", "ingredients": ["200g Pollo", "150g Arroz Blanco"]}]}
    assert _degraded_safety_violations(dia, [], "vegetariana"), \
        "un día con pollo debe violar la dieta vegetariana (femenino incluido)"
    assert _degraded_safety_violations(dia, ["arroz"], ""), \
        "un día con arroz debe violar una alergia declarada a arroz"


def test_helper_es_no_op_sin_restricciones():
    """Sin alergias y con dieta balanced no puede inventar violaciones (no romper el caso común)."""
    from cron_tasks import _degraded_safety_violations

    dia = {"meals": [{"name": "Almuerzo", "ingredients": ["200g Pollo", "150g Arroz Blanco"]}]}
    assert _degraded_safety_violations(dia, [], "") == []
    assert _degraded_safety_violations(dia, [], "balanced") == []


def test_helper_tolera_basura():
    """Un día malformado no puede tumbar el worker degradado."""
    from cron_tasks import _degraded_safety_violations

    assert _degraded_safety_violations(None, [], "vegetariana") == []
    assert _degraded_safety_violations({}, [], "vegetariana") == []
    assert _degraded_safety_violations({"meals": None}, [], "vegetariana") == []


def test_knob_off_desactiva_el_scan():
    """Rollback sin redeploy: MEALFIT_DEGRADED_SAFETY_SCAN=false vuelve al comportamiento previo."""
    import cron_tasks

    dia = {"meals": [{"name": "Almuerzo", "ingredients": ["200g Pollo"]}]}
    previo = cron_tasks.DEGRADED_SAFETY_SCAN
    try:
        cron_tasks.DEGRADED_SAFETY_SCAN = False
        assert cron_tasks._degraded_safety_violations(dia, [], "vegetariana") == []
    finally:
        cron_tasks.DEGRADED_SAFETY_SCAN = previo
    assert cron_tasks._degraded_safety_violations(dia, [], "vegetariana"), \
        "con el knob restaurado el guard debe volver a detectar"


# --------------------------------------------------------------- anclaje estructural

def test_el_filtro_del_pool_usa_el_helper():
    """El blocklist de prior days (`_is_blocked`) debe consultar el escáner, no solo el substring.

    tooltip-anchor de producción: P0-DEGRADED-SAFETY-SCAN
    """
    src = CRON.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"def _is_blocked\(day\):(.{0,1600})", src, re.S)
    assert m, "no se encontró `_is_blocked` — ¿renombrado? actualizar este anclaje"
    assert "_degraded_safety_violations" in m.group(1), (
        "`_is_blocked` filtra prior days por substring del label del chip y sin dieta; "
        "debe consultar además `_degraded_safety_violations`"
    )


def test_el_builder_verifica_el_dia_antes_de_devolverlo():
    """Defensa en profundidad: aunque el pool de candidatos se tamice, el día se re-verifica."""
    src = CRON.read_text(encoding="utf-8", errors="ignore")
    m = re.search(
        r"def _build_filtered_edge_recipe_day\(.*?\n(?=def |\Z)", src, re.S
    )
    assert m, "no se encontró `_build_filtered_edge_recipe_day`"
    cuerpo = m.group(0)
    assert "P0-DEGRADED-SAFETY-SCAN" in cuerpo, "falta el tooltip-anchor en el builder"
    assert "_degraded_safety_violations" in cuerpo, (
        "el builder debe verificar el día construido antes de devolverlo"
    )
