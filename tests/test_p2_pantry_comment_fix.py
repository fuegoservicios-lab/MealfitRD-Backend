"""[P2-PANTRY-COMMENT-FIX · 2026-06-21] Contrato del mínimo de nevera: el plan INICIAL
está EXENTO (onboarding), el MANTENIMIENTO lo enforce.

Decisión de producto del owner (Fase 4 del build "todo terreno"): el plan inicial — el primero
tras crear cuenta + llenar el formulario — DEBE dar la lista de compras completa con todo lo que
el usuario necesita, AUNQUE su nevera esté vacía (un usuario nuevo no tiene nada en la nevera;
ese es el punto de onboarding). El mínimo de nevera aplica SOLO al MANTENIMIENTO: cuando, con el
tiempo, el usuario borra/agota los alimentos de su nevera y gestiona su inventario manualmente,
los chunks rolling (semana 2+) SÍ deben respetar lo que hay y pausar/pedir reabastecer si cae
bajo el mínimo.

Estos tests anclan el contrato para que un refactor futuro NO:
  (a) añada un gate de "mínimo de nevera" al plan inicial (rompería onboarding), ni
  (b) elimine el guard proactivo del mantenimiento.

El audit (2026-06-21) flageó que el comentario en constants.py daba a entender que
routers/plans.py VALIDA un piso de nevera inicial — cuando en realidad lo SALTA
(P1-PANTRY-GUARD-INITIAL-SKIP). Este P-fix corrige el comentario y ancla la verdad.
"""
import constants


def _plans_src():
    import routers.plans as _p
    return open(_p.__file__, encoding="utf-8").read()


def _constants_src():
    return open(constants.__file__, encoding="utf-8").read()


# ---------------------------------------------------------------------------
# 1. El plan inicial está EXENTO del mínimo (onboarding)
# ---------------------------------------------------------------------------
def test_initial_plan_exento_del_minimo_de_nevera():
    src = _plans_src()
    # La función de validación inicial existe y hace el short-circuit (skip) cuando
    # la nevera tiene menos de PANTRY_GUARD_MIN_ITEMS — NO enforce un piso.
    assert "def _run_pantry_validation_for_initial_chunk" in src
    assert "P1-PANTRY-GUARD-INITIAL-SKIP" in src
    assert "PANTRY_GUARD_MIN_ITEMS" in src
    # El skip retorna el result intacto (no rechaza el plan inicial por nevera baja).
    assert "len(pantry_ingredients) < _PANTRY_MIN" in src


# ---------------------------------------------------------------------------
# 2. El mantenimiento SÍ enforce el mínimo (guard proactivo)
# ---------------------------------------------------------------------------
def test_mantenimiento_tiene_guard_de_minimo():
    assert hasattr(constants, "CHUNK_MIN_FRESH_PANTRY_ITEMS")
    assert constants.CHUNK_MIN_FRESH_PANTRY_ITEMS >= 1
    assert hasattr(constants, "CHUNK_PANTRY_PROACTIVE_GUARD")
    # Deeplink a Mi Nevera para el push de "refresca tu nevera".
    assert hasattr(constants, "CHUNK_STALE_PANTRY_DEEPLINK")


# ---------------------------------------------------------------------------
# 3. El comentario corregido es honesto (no afirma un piso inicial inexistente)
# ---------------------------------------------------------------------------
def test_comentario_corregido_es_honesto():
    src = _constants_src()
    assert "P2-PANTRY-COMMENT-FIX" in src
    # El comentario debe dejar claro que el inicial es EXENTO / se SALTA, y que el
    # mínimo aplica SOLO al mantenimiento.
    guard_idx = src.find("CHUNK_PANTRY_PROACTIVE_GUARD = (")
    assert guard_idx > -1
    region = src[max(0, guard_idx - 1200):guard_idx]
    assert "P1-PANTRY-GUARD-INITIAL-SKIP" in region, (
        "El comentario debe referenciar el skip real del plan inicial."
    )
    assert ("SALTA" in region or "EXENTO" in region or "exento" in region), (
        "El comentario debe decir que el inicial está exento / se salta, no que se valida."
    )
    assert "MANTENIMIENTO" in region or "mantenimiento" in region
