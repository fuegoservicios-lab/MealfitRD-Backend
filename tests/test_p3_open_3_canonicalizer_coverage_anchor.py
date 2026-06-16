"""[P3-OPEN-3 · 2026-05-11] Anchor de cobertura de canonicalizers RD
para el shopping coherence guard + lista de familias candidatas deferred.

Contexto:
    El guard `run_shopping_coherence_guard` (shopping_calculator.py) usa
    12 canonicalizers para reducir falsos positivos `cap_swallowed_modifier`
    y `unknown` cuando el aggregator normaliza variantes RD del mismo
    alimento (p.ej., "Pollo entero" vs "Pechuga de pollo").

    El audit 2026-05-11 identificó 5 familias adicionales **candidatas**
    a canonicalización (sin evidencia operacional de divergencias hoy):
    cítricos, tomate, cebolla, quesos blancos, frutos secos.

    Decisión: NO accionar sin signal (YAGNI). Trigger explícito:
        Si pipeline_metrics del cron diario
        `_shopping_coherence_alert_job` muestra que >5% del bucket
        `unknown` corresponden a una de las familias candidatas
        (foods normalizados con prefijo de la familia), entonces
        promover esa familia a canonicalizer con su propio P-fix.

    Este test ancla:
      1. La lista **canónica** de canonicalizers existentes (12).
      2. La lista **candidato** documentada (5 familias).
      3. El **criterio numérico** del trigger (`>5%` del bucket).

    Si alguien añade un canonicalizer nuevo, debe añadirlo a
    `_CANONICALIZERS_EXISTING` para que el test pase Y removerlo de
    `_CANDIDATES_DEFERRED` si era candidato. Si una familia candidata
    nueva surge del análisis operacional, añadirla a la lista deferred.

Tooltip-anchor: P3-OPEN-3-START | gap P3 audit 2026-05-11
"""
from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SHOPPING = _REPO_ROOT / "backend" / "shopping_calculator.py"

# Canonicalizers existentes en `shopping_calculator.py` al momento del
# audit 2026-05-11. Si añades uno nuevo, lista aquí su nombre exacto
# (sin prefix `def`).
_CANONICALIZERS_EXISTING: set[str] = {
    "canonicalize_pavo",
    "canonicalize_protein",          # pollo/cerdo/res
    "canonicalize_fish_seafood",
    "canonicalize_huevo",
    "canonicalize_lacteo",
    "canonicalize_grano",            # arroz/avena
    "canonicalize_legumino",
    "canonicalize_viveres",          # yuca/yautía/batata/papa/auyama
    "canonicalize_musaceae",         # plátano/guineo
    "canonicalize_frutas_tropicales", # mango/piña/papaya-lechosa
    "canonicalize_verduras_hoja",    # lechuga/espinaca/rúcula/acelga/berro
    "canonicalize_aceites",          # oliva/girasol/coco/aguacate
    # [P3-NEW-12 · 2026-05-11] Las 5 familias candidato deferred fueron
    # promovidas a canonicalizers activos (wired al guard en
    # shopping_calculator.py:4639-4660 y 5493-5509). Antes en
    # `_CANDIDATES_DEFERRED`, ahora implementadas.
    "canonicalize_citricos",         # limón/lima/naranja/mandarina/toronja
    "canonicalize_tomate",           # tomate/tomate cherry
    "canonicalize_cebolla",          # cebolla/cebollín
    "canonicalize_quesos_blancos_rd", # queso blanco/de freír/mozzarella/etc.
    "canonicalize_frutos_secos",     # almendras/maní/nueces/etc.
}

# Familias candidato deferred (audit 2026-05-11). YAGNI hasta que el
# cron diario `_shopping_coherence_alert_job` muestre evidencia.
#
# [P3-NEW-12 · 2026-05-11] Las 5 familias originales (cítricos, tomate,
# cebolla, quesos blancos, frutos secos) fueron promovidas a
# canonicalizers activos y movidas a `_CANONICALIZERS_EXISTING`. No
# quedan candidatos deferred hoy; si surge una familia nueva del análisis
# operacional (>5% del bucket unknown), añadirla aquí.
_CANDIDATES_DEFERRED: dict[str, str] = {}

# Threshold del criterio (porcentaje del bucket unknown). Si SRE bumpea
# esto, actualizar también la sección "Cómo decidir promover".
_TRIGGER_PCT_OF_UNKNOWN_BUCKET = 5  # %


def _read_shopping_src() -> str:
    return _SHOPPING.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Todos los canonicalizers anclados existen en el código
# ---------------------------------------------------------------------------
def test_all_canonicalizers_defined_in_source() -> None:
    """Cada función en `_CANONICALIZERS_EXISTING` debe existir como
    `def canonicalize_<x>(...)` en `shopping_calculator.py`. Si una
    función se elimina, el test falla pidiendo decisión: ¿se removió
    intencionalmente (refactor a otro módulo) o accidentalmente?
    """
    src = _read_shopping_src()
    missing: list[str] = []
    for name in _CANONICALIZERS_EXISTING:
        if not re.search(rf"\bdef\s+{re.escape(name)}\s*\(", src):
            missing.append(name)
    assert not missing, (
        f"P3-OPEN-3: los siguientes canonicalizers están listados en "
        f"`_CANONICALIZERS_EXISTING` pero NO se encontraron en "
        f"`shopping_calculator.py`:\n"
        + "\n".join(f"  - {n}" for n in missing)
        + "\n\nSi fueron renombrados/refactorizados, actualizar la lista "
          "aquí. Si fueron eliminados por error, restaurarlos."
    )


# ---------------------------------------------------------------------------
# 2. No hay canonicalizers nuevos sin anclar
# ---------------------------------------------------------------------------
def test_no_unanchored_canonicalizers_in_source() -> None:
    """Detecta `def canonicalize_<x>(...)` que NO esté en la lista
    anclada. Fuerza decisión humana al introducir uno nuevo: ¿es
    legítimo? Si sí, añadir a `_CANONICALIZERS_EXISTING` + ¿es una
    promoción de un candidato deferred? Entonces removerlo de
    `_CANDIDATES_DEFERRED`.
    """
    src = _read_shopping_src()
    pat = re.compile(r"^def\s+(canonicalize_[a-z_]+)\s*\(", re.MULTILINE)
    found = {m.group(1) for m in pat.finditer(src)}
    # Excluir helpers internos que NO sean canonicalizers de coherencia.
    # Hoy no hay ninguno; si en el futuro algún `canonicalize_X` no es
    # del guard (e.g., un helper de display), añadirlo aquí.
    _NON_GUARD_PREFIX_ALLOWLIST: set[str] = set()
    found_guard = found - _NON_GUARD_PREFIX_ALLOWLIST

    unexpected = found_guard - _CANONICALIZERS_EXISTING
    assert not unexpected, (
        f"P3-OPEN-3: encontrados canonicalizers NUEVOS en "
        f"`shopping_calculator.py` no anclados:\n"
        + "\n".join(f"  + {n}" for n in sorted(unexpected))
        + "\n\nDecide manualmente:\n"
          "  1. ¿Es un canonicalizer legítimo del guard? Añadirlo a "
             "`_CANONICALIZERS_EXISTING`.\n"
          "  2. ¿Era un candidato deferred que se promovió? Removerlo "
             "de `_CANDIDATES_DEFERRED` y añadirlo a `_CANONICALIZERS_EXISTING`.\n"
          "  3. ¿Es un helper interno NO del guard? Añadirlo a "
             "`_NON_GUARD_PREFIX_ALLOWLIST` con razón."
    )


# ---------------------------------------------------------------------------
# 3. Candidatos deferred NO están implementados (sanity check)
# ---------------------------------------------------------------------------
def test_deferred_candidates_not_implemented() -> None:
    """Las familias candidato NO deben tener una función
    `canonicalize_<familia>` ya implementada. Si alguien la añade sin
    actualizar `_CANDIDATES_DEFERRED`, hay drift entre código y plan.

    Esto NO bloquea la promoción de un candidato — bloquea la
    promoción SILENCIOSA. La promoción legítima requiere:
      1. Documentar la evidencia (memoria + métrica).
      2. Mover el item de `_CANDIDATES_DEFERRED` a `_CANONICALIZERS_EXISTING`.
      3. CI verifica que el código y los anchors estén sincronizados.
    """
    src = _read_shopping_src()
    leaked: list[tuple[str, str]] = []
    for family, _reason in _CANDIDATES_DEFERRED.items():
        # Match canonical name = "canonicalize_<family>".
        if re.search(rf"\bdef\s+canonicalize_{re.escape(family)}\s*\(", src):
            leaked.append((family, _reason))
    assert not leaked, (
        f"P3-OPEN-3: familias listadas como `_CANDIDATES_DEFERRED` ya tienen "
        f"un canonicalizer implementado en `shopping_calculator.py`:\n"
        + "\n".join(f"  ✗ canonicalize_{f}: {r}" for f, r in leaked)
        + "\n\nEsto es drift entre código y plan. Pasos a seguir:\n"
          "  1. Verificar que la promoción fue documentada con evidencia "
             "(memoria + pipeline_metrics).\n"
          "  2. Mover el item de `_CANDIDATES_DEFERRED` a "
             "`_CANONICALIZERS_EXISTING`.\n"
          "  3. Re-ejecutar este test — debería pasar tras la actualización."
    )


# ---------------------------------------------------------------------------
# 4. Trigger numérico documentado
# ---------------------------------------------------------------------------
def test_trigger_percentage_anchored() -> None:
    """El threshold del trigger (`>5%`) está anclado como constante del test.
    Si el valor cambia, el commit que lo cambia debe modificar también
    `_TRIGGER_PCT_OF_UNKNOWN_BUCKET` arriba — y este test es la red de
    seguridad para que el cambio sea deliberado.
    """
    # Sanity: el valor es positivo y razonable (cubre 1..50%).
    assert 1 <= _TRIGGER_PCT_OF_UNKNOWN_BUCKET <= 50, (
        f"P3-OPEN-3: el trigger pct {_TRIGGER_PCT_OF_UNKNOWN_BUCKET}% está "
        "fuera de rango razonable [1, 50]. Probable typo."
    )


# ---------------------------------------------------------------------------
# 5. Slug del marker en filename
# ---------------------------------------------------------------------------
def test_marker_anchor_present() -> None:
    """Filename contiene `p3_open_3` para cross-link audit."""
    assert "p3_open_3" in __file__.replace("\\", "/").lower(), (
        "El nombre de este archivo debe contener el slug `p3_open_3`."
    )
