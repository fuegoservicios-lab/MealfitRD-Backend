"""[P1-PANTRY-EXTRACT-FILTERED-WB · 2026-07-30] (audit solver+seeder v5 · P1-5)

El 13º incidente de subcadena de este repo, y el primero que se retroalimenta con un lock.

La extracción de la nevera (ROTATION MODE) recorre los catálogos **completos** (`DOMINICAN_*`,
NO los `filtered_*` que el propio seeder ya calculó 660 líneas antes) y matchea sinónimos con
`in` crudo:

    if any(strip_accents(s) in item for s in syns) and p not in extracted_p:

`PROTEIN_SYNONYMS['res']` contiene el alias `'res'` y `['pollo']` contiene `'pollo'`. Con una
nevera de vegetales:

    'res'   ⊂ '**fres**as'     → extrae **Res**
    'pollo' ⊂ 're**pollo**'    → extrae **Pollo**

Dos proteínas fantasma ≥ `PANTRY_ROTATION_MIN_PROTEINS` ⇒ el pool se REEMPLAZA por ellas y se
activa `cycle_locked`, que mete en el prompt "REGLA DE AHORRO EXTREMA… EXACTAMENTE las proteínas
asignadas". El usuario recibe 3 días de pollo y res que NO tiene en la nevera (anti-ahorro: la
lista de compras añade carnes). Para un vegetariano —repollo y fresas son ítems típicamente
vegetarianos— el pool filtrado por dieta se descarta entero y el backstop de dieta lo convierte
en rechazo → retry-storm.

Dos correcciones, no una: **word-boundary** (el operador) y **catálogos filtrados** (el universo).
Cualquiera de las dos sola deja la mitad del agujero: con solo word-boundary, una nevera que
mencione 'pollo' de verdad seguiría pudiendo re-sembrar una proteína que la alergia excluyó.
"""
from __future__ import annotations

import pytest

import ai_helpers as ah


def _prompt(pantry, **extra):
    fd = {"current_pantry_ingredients": pantry}
    fd.update(extra)
    return ah.get_deterministic_variety_prompt("", fd, user_id=None)


# ═════════════════ 1 · las subcadenas fantasma ═════════════════

def _extracted_proteins(caplog, pantry) -> int:
    """Nº de proteínas que la nevera aportó, leído del log del FLOOR.

    Con UNA sola proteína fantasma no se activa el lock (el floor de
    P2-PANTRY-ROTATION-FLOOR lo impide), así que mirar solo el lock no distingue
    "no extrajo nada" de "extrajo un fantasma y el floor lo tapó" — y el fantasma queda
    PRIMERO en el pool. El log del floor sí lo distingue."""
    import logging
    import re
    caplog.clear()
    caplog.set_level(logging.INFO, logger=ah.logger.name)
    _prompt(pantry)
    for r in caplog.records:
        m = re.search(r"la nevera aportó (\d+) proteína", r.getMessage())
        if m:
            return int(m.group(1))
    return 0


def test_repollo_no_extrae_pollo(caplog):
    assert _extracted_proteins(caplog, ["Repollo", "Arroz Blanco"]) == 0, (
        "'pollo' ⊂ 'repollo' → proteína fantasma al frente del pool")


def test_fresas_no_extrae_res(caplog):
    assert _extracted_proteins(caplog, ["Fresas", "Avena"]) == 0, (
        "'res' ⊂ 'fresas' → proteína fantasma al frente del pool")


def test_el_pantry_real_sigue_contando(caplog):
    """Sanity del vehículo: el detector de arriba SÍ ve una extracción legítima."""
    assert _extracted_proteins(caplog, ["Pechuga de pollo", "Arroz Blanco"]) == 1


def test_repollo_y_fresas_juntos_no_activan_cycle_lock():
    """El escenario exacto del audit: 2 fantasmas ≥ MIN_PROTEINS ⇒ pool reemplazado + lock."""
    out = _prompt(["Repollo", "Fresas", "Arroz Blanco"])
    assert "REGLA DE AHORRO EXTREMA" not in out.upper(), (
        "'pollo'⊂'repollo' + 'res'⊂'fresas' produjeron 2 proteínas fantasma y activaron el lock")


# ═════════════════ 2 · los matches legítimos siguen vivos ═════════════════

def test_pantry_real_sigue_extrayendo():
    """Regresión inversa: el word-boundary NO puede romper la extracción legítima — es la
    funcionalidad de ahorro que el ROTATION MODE existe para dar."""
    out = _prompt(["Pechuga de pollo", "Carne de res molida", "Arroz Blanco"])
    up = out.upper()
    assert "REGLA DE AHORRO EXTREMA" in up, (
        "con 2 proteínas REALES en la nevera el cycle-lock debe activarse (ahorro)")


def test_una_sola_proteina_real_no_bloquea():
    """P2-PANTRY-ROTATION-FLOOR sigue vigente: 1 proteína < mínimo ⇒ se completa con el sorteo."""
    out = _prompt(["Pechuga de pollo", "Arroz Blanco"])
    assert "REGLA DE AHORRO EXTREMA" not in out.upper()


# ═════════════════ 3 · el universo: catálogos FILTRADOS ═════════════════

def _extracted_count(caplog, pantry, **extra) -> int | None:
    """Nº de proteínas extraídas de la nevera. `None` = se activó el cycle-lock (≥ mínimo).

    NO se puede afirmar sobre la presencia del NOMBRE en el prompt: el prompt trae reglas
    genéricas que mencionan 'pescado'/'res' en prosa ("no uses pollo, res ni pescado salvo…"),
    así que un `not in` daría falso positivo por una frase que no tiene nada que ver con la
    extracción. Medimos el efecto real: cuántas proteínas aportó la nevera."""
    import logging
    import re
    caplog.clear()
    caplog.set_level(logging.INFO, logger=ah.logger.name)
    out = _prompt(pantry, **extra)
    if "REGLA DE AHORRO EXTREMA" in out.upper():
        return None
    for r in caplog.records:
        m = re.search(r"la nevera aportó (\d+) proteína", r.getMessage())
        if m:
            return int(m.group(1))
    return 0


def test_extraccion_respeta_alergias(caplog):
    """La nevera no puede re-introducir un alimento que la alergia excluyó del pool.

    Este es el lado del fix que el word-boundary NO cubre: aunque el match sea exacto, iterar
    los catálogos COMPLETOS permite que la nevera resucite un alérgeno — y con `cycle_locked`
    el prompt lo vuelve OBLIGATORIO."""
    n = _extracted_count(caplog, ["Pechuga de pollo", "Filete de pescado", "Arroz Blanco"],
                         allergies=["pescado"])
    assert n == 1, (
        f"esperado 1 (solo el pollo; el pescado está excluido por alergia), obtenido {n!r} "
        f"(None = cycle-lock activado con el alérgeno dentro de las bases obligatorias)")


def test_extraccion_respeta_dieta_vegetariana(caplog):
    # NOTA: `_get_fast_filtered_catalogs` matchea la dieta contra los literales
    # ["vegetariano", "vegetarian"] (constants.py) — la forma femenina "vegetariana" NO entra
    # por ninguna de esas ramas. Ese hueco de normalización es OTRO defecto (fuera del alcance
    # de este P-fix); aquí usamos la forma que el filtro sí reconoce para probar ESTE fix y no
    # el ajeno. Un test que falla por un gap distinto al que dice cubrir no protege ninguno.
    n = _extracted_count(caplog, ["Pechuga de pollo", "Carne de res molida", "Arroz Blanco"],
                         dietType="vegetariano")
    assert n == 0, (
        f"un plan vegetariano no puede recibir pollo/res por la puerta de la nevera; "
        f"proteínas extraídas: {n!r}")


def test_alias_generico_no_gana_al_especifico(caplog):
    """`'filete'` es alias de Res y es palabra COMPLETA dentro de "Filete de pescado": el
    word-boundary no puede distinguirlo y el pool filtrado tampoco (Res está permitida).
    Gana el alias más específico sobre el catálogo completo, y si el ganador está excluido
    el ítem no aporta base."""
    n = _extracted_count(caplog, ["Filete de pescado", "Arroz Blanco"])
    assert n == 1, f"esperado 1 (Pescado), obtenido {n!r} — ¿se extrajo Res por el alias 'filete'?"
    # y con alergia a pescado ese mismo ítem no debe degradar a Res
    n2 = _extracted_count(caplog, ["Filete de pescado", "Arroz Blanco"], allergies=["pescado"])
    assert n2 == 0, (
        f"con pescado excluido, el ítem NO puede caer a un match genérico (Res): obtenido {n2!r}")


# ═════════════════ 4 · estructural: el operador y el universo ═════════════════

def test_el_loop_usa_word_boundary_y_pools_filtrados():
    """Ancla el patrón canónico. El repo ya tiene DOS implementaciones correctas del mismo
    match (cpu_tasks.py y constants.py `fast_regex`); esta era la tercera, con `in` crudo."""
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "ai_helpers.py"), encoding="utf-8") as f:
        src = f.read()
    i = src.index("ROTATION MODE")
    blk = src[i:src.index("FORCED INGREDIENT INJECTION", i)]
    assert "for p in DOMINICAN_PROTEINS" not in blk, "el loop debe iterar filtered_proteins"
    assert "for f in DOMINICAN_FRUITS" not in blk, "el loop debe iterar filtered_fruits"
    assert "filtered_proteins" in blk and "filtered_carbs" in blk
    assert "filtered_veggies" in blk and "filtered_fruits" in blk
    # [P1-CYCLE-BASE-AFFINITY · 2026-08-02] El cuerpo del matcher se subió a `_catalog_pick_wb`
    # (nivel módulo) para que la afinidad de ciclo —que corre ANTES del sorteo— use EXACTAMENTE
    # este match en vez de nacer como cuarta implementación. El `\b` ya no vive dentro de este
    # bloque, así que el ancla sigue al SSOT: el bloque DELEGA y el delegado es word-boundary.
    assert "_catalog_pick_wb" in blk, "el bloque debe delegar en el matcher SSOT"
    import inspect
    import ai_helpers as _ah
    assert r"\b" in inspect.getsource(_ah._catalog_pick_wb), \
        "el match debe ser word-boundary, no substring"
