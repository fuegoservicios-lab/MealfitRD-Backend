"""[P2-MICROS-BETA-BACKFILL · 2026-08-21] «Desconocido» y «cero medido» son el mismo número.

Las ocho columnas de micros extendidos son `NOT NULL`, así que una fila sin dato guarda `0` — el
mismo valor que una fila donde el micro se midió y salió cero. El micro-closer no puede
distinguirlas, y ante un déficit que sólo existe porque faltan datos **añade portadores** al plan:
comida de verdad para tapar un agujero que no existe.

MEDIDO HOY sobre las 347 filas:

    vitamin_c        151 ceros (44%)      folate            45 (13%)
    vitamin_a        137 (39%)            selenium          37 (11%)
    omega3_ala       112 (32%)            zinc              19 (5%)
    vitamin_k        111 (32%)            vitamin_e         60 (17%)

DOS COSAS QUE LA MEDICIÓN CORRIGE DEL DIAGNÓSTICO:

1. **Muchos de esos ceros son CORRECTOS.** La vitamina C del aceite es cero de verdad; el omega-3
   del azúcar, también. La tasa alta no prueba que falten datos — prueba que no se puede saber cuál
   es cuál, que es el defecto real y es distinto.

2. **Las «filas-cascarón» son DOS, y una es correcta.** La auditoría cita `Sazón con culantro y
   achiote` (kcal 0, macros 0, 17.000 mg de sodio, `nutrition_source='usda'`). Buscando la FORMA
   —kcal 0 y las tres macros 0— salen dos filas, y la otra es **`Sal`**: 0 kcal, 0 macros y 38.758
   mg de sodio por 100 g. Eso no es un cascarón, es sal: NaCl es ~39% sodio en peso, así que la
   fila es exacta. Un guard que hubiera prohibido «kcal 0 con sodio alto» habría acusado al dato
   correcto.

POR QUÉ ESTE P-FIX NO RELLENA NADA. Backfillear micros exige una fuente por fila (USDA, BEDCA) y un
`fdc_id` que apunte de verdad — es la lección de `P1-BEDCA-DEPROXY-ES`: «un `fdc_id` es una
AFIRMACIÓN, no una nota al pie», donde 47 filas compartían id y una daba 404. Escribir cifras de
micronutrientes de memoria sería exactamente ese error, con la agravante de que aquí alimentan un
closer que AÑADE COMIDA al plan.

Lo que sí se puede hacer sin inventar: fijar la línea base. Este fichero pone las tasas medidas y
las dos filas de forma sospechosa con su veredicto, para que un backfill futuro tenga contra qué
compararse y para que una fila-cascarón NUEVA falle en vez de colarse entre el ruido.
"""
from __future__ import annotations

import pytest

_MICROS = (
    "vitamin_k_mcg_per_100g", "selenium_mcg_per_100g", "zinc_mg_per_100g",
    "folate_mcg_dfe_per_100g", "vitamin_a_mcg_rae_per_100g", "vitamin_c_mg_per_100g",
    "vitamin_e_mg_per_100g", "omega3_ala_g_per_100g",
)

# Tasas de ceros medidas el 2026-08-21 sobre 347 filas. Techo = medido + holgura: el test no exige
# mejorar (eso es curación con fuente), exige que no EMPEORE en silencio.
_TECHO_CEROS_PCT = {
    "vitamin_c_mg_per_100g": 50, "vitamin_a_mcg_rae_per_100g": 45,
    "omega3_ala_g_per_100g": 38, "vitamin_k_mcg_per_100g": 38,
    "vitamin_e_mg_per_100g": 23, "folate_mcg_dfe_per_100g": 19,
    "selenium_mcg_per_100g": 17, "zinc_mg_per_100g": 11,
}

# Filas con kcal 0 Y las tres macros 0. Las dos que existen hoy, con su veredicto.
_FORMA_CASCARON = {
    "Sal": "CORRECTA — NaCl no tiene calorías ni macros; 38.758 mg de sodio es ~39% en peso",
    "Sazón con culantro y achiote": "A CURAR — declara `usda` y no trae nada salvo el sodio",
}


@pytest.fixture(scope="module")
def filas():
    import shopping_calculator as sc
    rows = sc.get_master_ingredients() or []
    if not rows:
        pytest.skip("catálogo no disponible (sin DB)")
    return rows


def _cero(v) -> bool:
    try:
        return float(v or 0) == 0.0
    except (TypeError, ValueError):
        return True


# ── La línea base ───────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("col", _MICROS)
def test_la_tasa_de_ceros_no_empeora(filas, col):
    """No exige mejorar —eso es curación con fuente— sino que no empeore EN SILENCIO. Un alta
    masiva sin micros subiría estas tasas y el closer empezaría a tapar más agujeros fantasma sin
    que nadie lo notara."""
    ceros = sum(1 for r in filas if _cero(r.get(col)))
    pct = 100 * ceros / len(filas)
    assert pct <= _TECHO_CEROS_PCT[col], (
        f"{col}: {pct:.0f}% de ceros supera el techo de {_TECHO_CEROS_PCT[col]}%. Si el alta es "
        f"legítima, cura los micros o sube el techo A SABIENDAS — pero no lo dejes pasar"
    )


def test_la_medicion_sigue_siendo_representativa(filas):
    """El guard del guard: si el catálogo encogiera, los porcentajes de arriba dejarían de
    significar lo mismo."""
    assert len(filas) >= 300, f"el catálogo bajó a {len(filas)} filas"


# ── Las filas de forma sospechosa ───────────────────────────────────────────────────────────────

def test_las_filas_sin_kcal_ni_macros_son_exactamente_las_conocidas(filas):
    """Caracterización. Si aparece una NUEVA, falla: sería una fila que parece verificada y no lo
    está, y el ruido de las dos conocidas la habría escondido.

    Si desaparece una, también falla — para que quien la cure actualice la nota en vez de dejarla
    mintiendo."""
    vistas = {
        str(r.get("name"))
        for r in filas
        if _cero(r.get("kcal_per_100g")) and _cero(r.get("protein_g_per_100g"))
        and _cero(r.get("carbs_g_per_100g")) and _cero(r.get("fats_g_per_100g"))
    }
    assert vistas == set(_FORMA_CASCARON), (
        f"las filas sin kcal ni macros cambiaron. Conocidas: {sorted(_FORMA_CASCARON)}. "
        f"Vistas: {sorted(vistas)}"
    )


def test_la_sal_no_se_trata_como_un_error(filas):
    """La corrección al diagnóstico, anclada: un guard que prohibiera «kcal 0 con sodio alto»
    habría acusado al dato CORRECTO. La sal no tiene calorías ni macros por definición."""
    sal = next((r for r in filas if str(r.get("name")) == "Sal"), None)
    assert sal, "desapareció la fila `Sal` del catálogo"
    assert float(sal.get("sodium_mg_per_100g") or 0) > 30000, (
        "el sodio de la sal bajó de 30.000 mg/100g: NaCl es ~39% sodio en peso, así que un valor "
        "menor sería el dato equivocado"
    )


def test_no_se_han_rellenado_micros_sin_procedencia(filas):
    """El error que este P-fix NO comete, anclado en la dirección contraria: si algún día las tasas
    de ceros caen a plomo, que sea porque alguien curó con fuente y actualizó los techos — no
    porque se escribieran cifras de memoria. `P1-BEDCA-DEPROXY-ES`: un `fdc_id` es una AFIRMACIÓN.

    Se comprueba que las filas CON micros declaren procedencia."""
    sin_fuente = [
        str(r.get("name")) for r in filas
        if not _cero(r.get("vitamin_k_mcg_per_100g")) and not str(r.get("nutrition_source") or "").strip()
    ]
    assert not sin_fuente, (
        f"hay filas con micros poblados y SIN `nutrition_source`: {sin_fuente[:10]}. Un micro sin "
        f"procedencia es una cifra que nadie puede re-verificar"
    )
