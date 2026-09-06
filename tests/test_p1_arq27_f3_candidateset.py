# -*- coding: utf-8 -*-
"""[P1-ARQ27-F3-CANDIDATESET · 2026-09-06] El prompt consume los candidatos FIJADOS al run (ARQ27-P1-04).

El blueprint ya fijaba 3 candidatos por día y franja. Pero `slice_for_chunk` **no se los llevaba**, así
que `registry_prompt_lines` volvía a consultar `template_candidates` sobre el registro ACTIVO — y encima
recitaba solo 2 de los 3. Consecuencia: recompilar el Dish Registry entre dos chunks cambiaba los platos
que se le proponían a un plan **ya empezado**, sin que nada lo declarara.

Tres defectos distintos en el mismo camino:

1. **La rebanada tiraba el CandidateSet.** Ahora lo lleva, y por eso entra en `slice_hash` → `input_hash`:
   un cambio de catálogo se ve como revisión distinta en vez de colarse callado.
2. **El orden era el de inserción en el fichero, con corte al llegar a `k`.** Añadir una plantilla al
   principio del JSON cambiaba los candidatos de todas las consultas, y las últimas del fichero no se
   ofrecían jamás. Ahora el orden sale del hash del `template_id` con la consulta como sal, y `rotate`
   (el índice del día) desplaza la lista.
3. **Fijar el ID no bastaba.** Si el registro activo retiraba la plantilla, el ID no resolvía a nombre y
   el conjunto del run encogía en silencio — lo contrario de «fijado». Se fija también el NOMBRE. Esto lo
   encontró la verificación contra el registry real, no el diseño: el primer intento pasaba tres de los
   cuatro criterios y fallaba justo el que da nombre al gap.

**El swap no entra aquí y eso es la respuesta, no una omisión.** El criterio pide que el swap declare si
usa el snapshot del plan o migra a otro; la respuesta medida es que **no usa el registry en absoluto**.
El test de abajo ancla ese hecho para que, si alguien lo cablea mañana, tenga que declarar cuál usa.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import dish_registry as dr  # noqa: E402
import horizon as H  # noqa: E402

_EFF = {
    "diet": {"type": "omnivora", "allergies": []},
    "culture_weights": [{"profile_id": "dominican_criolla", "weight": 1.0}],
    "market_country": "DO",
    "shopping": {"main_cycle_days": 7, "freezer_mode": "limited"},
    "recurrence": {"global_mode": "balanced"},
}


def _hay_registry() -> bool:
    return bool(dr.registry_hash("DO"))


pytestmark = pytest.mark.skipif(not _hay_registry(), reason="sin snapshot compilado del registry")


@pytest.fixture
def bp():
    return H.build_blueprint(_EFF, total_days=14, meals_per_day=4)


@pytest.fixture
def registry_intacto():
    """Restaura la caché del snapshot pase lo que pase: estos tests la mutan a propósito."""
    ruta = dr.snapshot_path(dr.library_for_country("DO"))
    orig = dr.load_registry("DO")
    yield ruta, orig
    dr._CACHE[ruta] = orig
    dr._BY_ID.clear()


# ── la rebanada lleva lo suyo ─────────────────────────────────────────────────────────────────
def test_la_rebanada_lleva_el_candidateset(bp):
    sl = H.slice_for_chunk(bp, 0, 7)
    reg = sl.get("registry") or {}
    assert reg.get("candidates"), "la rebanada tiene que llevar los candidatos fijados"
    assert reg.get("candidate_names"), "…y sus nombres, o un registro retirado los borra"
    assert reg.get("snapshot_hash") == (bp.get("registry") or {}).get("snapshot_hash")


def test_cada_rebanada_lleva_solo_sus_dias(bp):
    a = H.slice_for_chunk(bp, 0, 7)["registry"]["candidates"]
    b = H.slice_for_chunk(bp, 7, 7)["registry"]["candidates"]
    assert a and b
    assert not (set(a) & set(b)), "las rebanadas no deben solaparse"
    assert all(0 <= int(k.split(":")[0]) < 7 for k in a)
    assert all(7 <= int(k.split(":")[0]) < 14 for k in b)


def test_el_candidateset_entra_en_el_hash_de_la_rebanada(bp):
    """Es lo que convierte «cambió el catálogo» en una revisión visible en vez de un cambio callado."""
    sl = H.slice_for_chunk(bp, 0, 7)
    otra = copy.deepcopy(sl)
    otra["registry"]["candidates"] = {"0:lunch": ["tpl_inventado"]}
    assert H.slice_hash(otra) != H.slice_hash(sl)
    assert H.chunk_input_hash("fp", otra) != H.chunk_input_hash("fp", sl)


# ── un run fijado no se mueve ─────────────────────────────────────────────────────────────────
def test_cambiar_el_registro_activo_no_cambia_un_run_fijado(bp, registry_intacto, monkeypatch):
    """El criterio que da nombre al gap. Se retiran plantillas del snapshot ACTIVO y se invierte su
    orden: las líneas del prompt de un run ya fijado no se mueven ni un carácter."""
    monkeypatch.setenv("MEALFIT_DISH_REGISTRY_PROMPT", "1")
    ruta, orig = registry_intacto
    sl = H.slice_for_chunk(bp, 0, 7)
    antes = H.registry_prompt_lines(_EFF, sl)
    assert antes, "el escenario no prueba nada si no hay líneas"

    mutado = copy.deepcopy(orig)
    mutado["templates"] = list(reversed(mutado["templates"]))[:-5]
    mutado["snapshot_hash"] = "otro_hash"
    dr._CACHE[ruta] = mutado
    dr._BY_ID.clear()

    assert H.registry_prompt_lines(_EFF, sl) == antes


def test_sin_candidateset_el_adaptador_reconsulta(bp, monkeypatch):
    """Rollback declarado en el gap: los runs fijados ANTES de este cambio no traen `registry` en su
    rebanada y su historial no se reescribe — siguen produciendo líneas."""
    monkeypatch.setenv("MEALFIT_DISH_REGISTRY_PROMPT", "1")
    sl = {k: v for k, v in H.slice_for_chunk(bp, 0, 7).items() if k != "registry"}
    assert H.registry_prompt_lines(_EFF, sl)


def test_el_nombre_fijado_gana_al_id(bp):
    """Capa 1 de `_pinned_candidate_names`: es la única que sobrevive a que retiren la plantilla."""
    reg = {"candidate_names": {"0:lunch": ["Plato fijado"]}, "candidates": {"0:lunch": ["tpl_x"]}}
    assert H._pinned_candidate_names(dr, reg, 0, "lunch", "DO") == ["Plato fijado"]


def test_sin_nombres_resuelve_por_id(bp):
    """Capa 2: rebanadas fijadas antes de que se guardaran los nombres."""
    sl = H.slice_for_chunk(bp, 0, 7)
    clave, ids = next(iter(sl["registry"]["candidates"].items()))
    dia, slot = clave.split(":", 1)
    reg = {"candidates": {clave: ids}}
    esperado = sl["registry"]["candidate_names"][clave]
    assert H._pinned_candidate_names(dr, reg, int(dia), slot, "DO") == esperado


# ── el ranking ────────────────────────────────────────────────────────────────────────────────
def test_mismo_snapshot_mismo_conjunto():
    a = [c["template_id"] for c in dr.template_candidates("DO", "cena", None, k=5)]
    b = [c["template_id"] for c in dr.template_candidates("DO", "cena", None, k=5)]
    assert a == b and a


def test_el_ranking_no_es_el_orden_del_fichero(registry_intacto):
    _, orig = registry_intacto
    del_fichero = [t["template_id"] for t in orig["templates"]
                   if t.get("status") == "ok" and "cena" in (t.get("slots") or [])][:5]
    del_ranking = [c["template_id"] for c in dr.template_candidates("DO", "cena", None, k=5)]
    assert del_ranking != del_fichero, (
        "si coinciden, el ranking sigue premiando a quien se insertó primero")


def test_invertir_el_fichero_no_cambia_los_candidatos(registry_intacto):
    """«Cambiar el orden de inserción no altera candidatos» — el criterio, literal."""
    ruta, orig = registry_intacto
    antes = [c["template_id"] for c in dr.template_candidates("DO", "cena", None, k=5)]
    invertido = copy.deepcopy(orig)
    invertido["templates"] = list(reversed(invertido["templates"]))
    dr._CACHE[ruta] = invertido
    dr._BY_ID.clear()
    assert [c["template_id"] for c in dr.template_candidates("DO", "cena", None, k=5)] == antes


def test_rotate_reparte_la_cabeza_entre_dias():
    """Sin `rotate`, los días consecutivos con la misma franja y familia recibían siempre los mismos
    tres. Determinista: el mismo día da siempre el mismo conjunto."""
    conjuntos = {tuple(c["template_id"] for c in dr.template_candidates("DO", "cena", None, k=3, rotate=d))
                 for d in range(6)}
    assert len(conjuntos) >= 3, f"rotate apenas reparte: {len(conjuntos)} conjuntos en 6 días"
    uno = tuple(c["template_id"] for c in dr.template_candidates("DO", "cena", None, k=3, rotate=4))
    otro = tuple(c["template_id"] for c in dr.template_candidates("DO", "cena", None, k=3, rotate=4))
    assert uno == otro, "rotate tiene que ser determinista"


def test_templates_by_id_se_cachea_por_hash_de_contenido(registry_intacto):
    """«Resolver por hash de contenido, no solo por nombre v1»: el repo recompila EN `v1`, así que una
    caché indexada por nombre de versión serviría plantillas viejas."""
    ruta, orig = registry_intacto
    antes = dr.templates_by_id("DO")
    assert antes
    mutado = copy.deepcopy(orig)
    mutado["templates"] = mutado["templates"][:3]
    mutado["snapshot_hash"] = "hash_distinto"
    dr._CACHE[ruta] = mutado
    assert len(dr.templates_by_id("DO")) == 3, "la caché sirvió el índice viejo pese al hash nuevo"


# ── el swap ───────────────────────────────────────────────────────────────────────────────────
def test_el_swap_no_usa_el_registry_y_eso_esta_declarado():
    """El criterio pide que el swap declare qué snapshot usa. La respuesta medida es: ninguno. Si
    alguien lo cablea, este test cae y tendrá que declararlo — que es justo lo que el gap quiere."""
    src = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    i = src.index("def api_swap_meal(")
    cuerpo = src[i:src.index("def api_swap_meal_persist(")]
    for prohibido in ("dish_registry", "registry_prompt_lines", "template_candidates", "_blueprint_slice"):
        assert prohibido not in cuerpo, (
            f"el swap ahora toca {prohibido}: declara si usa el snapshot del plan o migra a otro")
