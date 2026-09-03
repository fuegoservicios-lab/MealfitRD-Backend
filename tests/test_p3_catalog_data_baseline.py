"""[P3-CATALOG-DATA-BASELINE · 2026-08-22] Los tres huecos de datos del catálogo, medidos — y el
que resultó no serlo.

El plan de auditoría los listaba de memoria. Las cuatro cifras estaban mal, y una de las
correcciones cambia por completo qué había que hacer.

┌─ LO QUE DECÍA EL PLAN ──────────────┬─ LO MEDIDO HOY (347 filas) ─────────────────────────────┐
│ «136 de las 141 filas beta sin      │ 145 de 187 beta, **y 21 de 160 con precio**. No es un   │
│  densidad»                          │ problema de beta: el lote dominicano también tiene.     │
│ «15 filas con ready_to_eat NULL»    │ **64**.                                                 │
│ «un alias duplicado dentro de su    │ **107 filas**, y casi todas son el alias repitiendo su   │
│  propia fila»                       │ propio nombre: redundante, inofensivo al resolver.      │
│ (no lo mencionaba)                  │ **4 alias reclamados por MÁS DE UNA fila** — que sonaba  │
│                                     │ al defecto de verdad… y no lo es. Ver abajo.            │
└─────────────────────────────────────┴─────────────────────────────────────────────────────────┘

EL QUE PARECÍA GRAVE Y NO LO ERA. Cuatro alias los reclaman varias filas: `mariscos`
(Pulpo/Mejillones/Calamar), `mero` y `tilapia` (la fila específica **y** «Filete de pescado
blanco»), y `nueces` (Nueces mixtas **y** Almendras fileteadas). Leído en la tabla parece el
«alias bare es un arma» de siempre: pedir mero y que te sirvan un filete genérico; pedir nueces y
que te den almendras.

Medido en la conducta, no en el dato:

    normalize_name('mero')     → 'Mero'            ← gana la fila específica
    normalize_name('tilapia')  → 'Tilapia'         ← gana la fila específica
    normalize_name('nueces')   → 'Nueces mixtas'   ← NO almendras
    normalize_name('mariscos') → 'Pulpo'           ← arbitrario, pero es un término de categoría

    pantry_names_match('nueces', 'Nueces mixtas')        = False
    pantry_names_match('nueces', 'Almendras fileteadas') = False   ← ambiguo ⇒ no resuelve

O sea: **los dos resolutores ya tratan bien la ambigüedad.** El índice de alias de
`P1-PANTRY-NAME-RESOLUTION` cuenta los reclamantes ANTES de filtrar y descarta lo ambiguo, que es
exactamente para lo que se escribió. Estuve a punto de escribir una migración para «limpiar» estos
alias — habría sido `P2-11` otra vez: mover un nombre que es un IDENTIFICADOR, cambiando
«resuelve bien» por «no resuelve». Lo que hace falta aquí es un ancla, no un cambio.

POR QUÉ ESTE FICHERO NO RELLENA DENSIDAD NI `ready_to_eat`. Es la lección de
`P1-BEDCA-DEPROXY-ES`: «un `fdc_id` es una AFIRMACIÓN, no una nota al pie», donde 47 filas
compartían id y una daba 404. Escribir densidades de memoria sería ese mismo error sobre un dato
que alimenta conversiones de volumen a peso — y de ahí, la cantidad que el usuario compra. Lo que
sí se puede hacer sin inventar es fijar la línea base para que un backfill futuro tenga contra qué
compararse y para que un alta masiva sin curar no empeore las tasas en silencio.
"""
from __future__ import annotations

import pytest

#: Tasas medidas el 2026-08-22. Techo = medido + holgura. NO exige mejorar (eso es curación con
#: fuente); exige que no EMPEORE sin que nadie se entere.
_TECHO_SIN_DENSIDAD_PCT = 55        # medido: 166/347 = 47,8%
_TECHO_READY_TO_EAT_NULL = 80       # medido: 64

#: Los cuatro alias que más de una fila reclama, con su veredicto medido.
_ALIAS_AMBIGUOS = {
    "mariscos": "categoría — tres moluscos la reclaman; cualquier elección es arbitraria",
    "mero": "resuelve a la fila específica «Mero», no al filete genérico",
    "nueces": "resuelve a «Nueces mixtas», NO a las almendras",
    "tilapia": "resuelve a la fila específica «Tilapia», no al filete genérico",
}


@pytest.fixture(scope="module")
def filas():
    import shopping_calculator as sc
    rows = sc.get_master_ingredients() or []
    if not rows:
        pytest.skip("catálogo no disponible (sin DB)")
    return rows


@pytest.fixture(scope="module")
def por_alias(filas) -> dict:
    idx: dict[str, list] = {}
    for f in filas:
        for al in (f.get("aliases") or []):
            idx.setdefault(str(al).strip().lower(), []).append(str(f.get("name")))
    return idx


def test_el_catalogo_sigue_siendo_representativo(filas):
    """Sanity: si encogiera, los porcentajes de abajo dejarían de significar lo mismo."""
    assert len(filas) >= 300, f"el catálogo bajó a {len(filas)} filas"


# ── Líneas base que no se rellenan, se vigilan ──────────────────────────────────────────────────

def test_la_falta_de_densidad_no_empeora(filas):
    """Y de paso corrige el diagnóstico: 21 de las filas SIN densidad tienen precio, o sea que son
    del lote dominicano. Tratarlo como «un problema del alta beta» dejaría fuera a una de cada
    ocho."""
    sin = sum(1 for f in filas if f.get("density_g_per_cup") in (None, 0))
    pct = 100 * sin / len(filas)
    assert pct <= _TECHO_SIN_DENSIDAD_PCT, (
        f"{pct:.0f}% del catálogo sin densidad supera el techo de {_TECHO_SIN_DENSIDAD_PCT}%. Si "
        f"el alta es legítima, cura la densidad o sube el techo A SABIENDAS"
    )


def test_la_densidad_que_falta_no_es_solo_del_lote_beta(filas):
    """Ancla de la corrección al plan. Si algún día esto pasa a 0, es que alguien curó el lote
    dominicano — y entonces hay que actualizar la nota, no borrar el caso."""
    con_precio_sin_densidad = [
        str(f.get("name")) for f in filas
        if f.get("density_g_per_cup") in (None, 0) and (f.get("price_per_unit") or 0) > 0
    ]
    assert con_precio_sin_densidad, (
        "ya no hay filas CON precio y sin densidad. El plan decía que esto era exclusivo del lote "
        "beta y era falso; si se curó, actualiza la nota en vez de dejarla mintiendo"
    )


def test_ready_to_eat_sin_declarar_no_empeora(filas):
    """`NULL` en un booleano de cocción no es «no»: es «nadie lo sabe». Quien lo lea como falso
    tratará como cocinable algo que se come crudo, y al revés."""
    nulos = sum(1 for f in filas if f.get("ready_to_eat") is None)
    assert nulos <= _TECHO_READY_TO_EAT_NULL, (
        f"{nulos} filas con `ready_to_eat` sin declarar supera el techo de "
        f"{_TECHO_READY_TO_EAT_NULL}"
    )


# ── La ambigüedad de alias, caracterizada ───────────────────────────────────────────────────────

def test_los_alias_reclamados_por_varias_filas_son_los_conocidos(por_alias):
    """Caracterización. Un alias ambiguo NUEVO tiene que aparecer aquí con su veredicto medido —
    no colarse entre los cuatro que ya sabemos que se resuelven bien."""
    ambiguos = {a for a, duenos in por_alias.items() if len(set(duenos)) > 1}
    assert ambiguos == set(_ALIAS_AMBIGUOS), (
        f"cambiaron los alias reclamados por varias filas.\n"
        f"  conocidos: {sorted(_ALIAS_AMBIGUOS)}\n"
        f"  vistos   : {sorted(ambiguos)}\n"
        f"Si añadiste uno, MIDE a qué resuelve antes de darlo por bueno: «un alias bare es un "
        f"arma» (cuatro veces en Fase 2)"
    )


@pytest.mark.parametrize("alias,esperado", [
    ("mero", "Mero"),
    ("tilapia", "Tilapia"),
    ("nueces", "Nueces mixtas"),
])
def test_el_alias_ambiguo_resuelve_a_la_fila_especifica(alias, esperado, filas):
    """LO QUE DE VERDAD IMPORTA, y lo que refutó el diagnóstico: pese a que dos filas reclaman el
    término, gana la específica. Pedir mero no devuelve un filete genérico, y pedir nueces no
    devuelve almendras."""
    import shopping_calculator as sc
    assert sc.normalize_name(alias) == esperado, (
        f"{alias!r} dejó de resolver a {esperado!r}. Con dos filas reclamándolo, que gane la "
        f"genérica significa servir otra cosa"
    )


def test_la_nevera_no_resuelve_por_un_alias_ambiguo(por_alias):
    """El otro resolutor. `pantry_names_match` descarta lo ambiguo porque cuenta los reclamantes
    ANTES de filtrar — el orden que `P1-PANTRY-NAME-RESOLUTION` documenta como load-bearing.
    Devolver un `True` aquí descontaría de la fila equivocada, en silencio."""
    from constants import pantry_names_match

    for alias in _ALIAS_AMBIGUOS:
        duenos = set(por_alias.get(alias, []))
        # Sólo los que NO son además el nombre exacto de su fila: ahí el match es legítimo.
        for d in duenos:
            if d.strip().lower() == alias:
                continue
            assert pantry_names_match(alias, d) is False, (
                f"pantry_names_match({alias!r}, {d!r}) resolvió pese a que {len(duenos)} filas "
                f"reclaman ese alias: descontaría de la fila equivocada sin avisar"
            )
