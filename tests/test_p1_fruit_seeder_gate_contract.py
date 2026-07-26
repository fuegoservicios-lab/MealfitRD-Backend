"""[P1-FRUIT-SEEDER-GATE-CONTRACT · 2026-07-26] El seeder pedía lo que el gate no podía aceptar.

"Fruta repetida el mismo día" era la razón de rechazo DOMINANTE (67% de los planes de la línea base
reintentaban al menos una vez). No era el modelo fallando: la instrucción era **insatisfacible**.

## El contrato estaba roto por los dos lados

    seeder →  UNA fruta por día  (fruit_0, fruit_1, fruit_2)
    gate   →  una fruta dulce DISTINTA por COMIDA dentro del día

Un día con fruta en el desayuno Y en la merienda —la forma más común— no puede cumplir eso desde el
pool. El day-gen repite la única que tiene (el gate rechaza) o improvisa una de fuera (y choca
cross-día). **Caso vivo** (plan del 2026-07-26 07:47, primer rechazo del canario Luna): el pool era
`['Níspero', 'Toronja', 'Limón']` y usó **guineo**, que no estaba asignado. Un modelo mejor se
estrella igual — por eso esto se arregla aquí y no subiendo de modelo.

Y encima el gate no reconocía lo que el seeder asignaba: de las **30** frutas del catálogo veía
**16**. Un pool de 3 salía sin ninguna reconocida el **9%** de las veces y con ≤1 el **44,8%**.
Níspero era invisible: repetirlo no lo veía el gate… pero tampoco el de-dup determinista, que
comparte vocabulario — así que se entregaba el defecto en silencio. Su log lo decía sin adornos:
*"repetición de fruta RESIDUAL tras el de-dup (pool agotado o nombre no-reescribible)"*.

## Las tres mitades del arreglo

1. **Vocabulario**: +6 frutas que el catálogo tiene y el seeder asigna (níspero, guanábana, ciruela,
   durazno, granada, toronja). NO se añaden `naranja` ni `limón`: su exclusión es deliberada
   (ralladura/aderezo, no "la fruta del plato"), ni coco/pasas/dátiles/tamarindo (guarnición) ni
   aguacate (grasa).
2. **Límite de palabra**: `"pina"` ⊂ `"espinaca"` — el gate veía piña en un plato de espinacas, y
   `Espinacas` está en el catálogo. Medido: 2 platos de 12 planes contados como piña; ningún día
   cambiaba de veredicto porque hacía falta espinaca en DOS comidas del mismo día, o sea un rechazo
   falso **latente**. También `"pera"` ⊂ `"temperatura"`.
3. **Seeder**: DOS frutas reconocidas por día, rotando sobre 4 para que la semana use 4 y no 6 (la
   lista de compras importa: el usuario es sensible al presupuesto).

## Por qué el vocabulario más ancho NO produce más rechazos

Suena contraintuitivo y lo medí: sobre 12 planes vivos el gate pasa de rechazar 1 día a 2 (el nuevo
es ciruela ×2 en un día, una repetición REAL que antes se entregaba invisible). Pero el de-dup corre
ANTES del gate y ahora sí la ve: verificado en vivo, reescribe la segunda "Ciruela" a "Lechosa" y el
gate pasa. El resultado es un defecto menos entregado, no un rechazo más.
"""
import pytest

import graph_orchestrator as go
import ai_helpers as ah


# ───────────── 1. límite de palabra: fin de las frutas fantasma ─────────────

@pytest.mark.parametrize("nombre", [
    "Tortilla de Queso Blanco y Espinacas con Casabe",
    "Revoltillo de Pescado blanco con Espinacas",
    "Espinacas salteadas al ajillo",
    "Servir a temperatura ambiente",
])
def test_no_ve_fruta_donde_no_hay(nombre):
    assert go._featured_fruits_in_name(nombre) == set(), nombre


def test_la_subcadena_esta_ahi_y_aun_asi_no_cuenta():
    """Ancla explícita de la clase: si alguien vuelve al `in`, esto falla."""
    from constants import strip_accents as sa
    assert "pina" in sa("espinacas")          # la subcadena SÍ está…
    assert go._featured_fruits_in_name("Espinacas") == set()   # …y no es piña


@pytest.mark.parametrize("nombre,esperado", [
    ("Bowl de Fresas y Yogurt", {"fresa"}),
    ("Batido de Fresa", {"fresa"}),            # singular y plural = MISMA fruta
    ("Piña colada", {"pina"}),
    ("Mangú con salami", set()),               # mangú no es mango
    ("Empanada de pollo", set()),              # empanada no es granada
    ("Jugo de Guanábana", {"guanabana"}),
    ("Ensalada de Mango y Fresas", {"mango", "fresa"}),
])
def test_reconoce_lo_que_debe(nombre, esperado):
    assert go._featured_fruits_in_name(nombre) == esperado


def test_singular_y_plural_son_la_misma_fruta():
    """Con grupo no-capturante `findall` devolvería 'fresas' y 'fresa' como claves distintas y la
    repetición se colaría."""
    plan = {"days": [{"meals": [{"name": "Bowl de Fresas"}, {"name": "Batido de Fresa"}]}]}
    assert go._plan_has_same_day_fruit_repeat(plan) is True


# ───────────── 2. vocabulario alineado con el catálogo ─────────────

@pytest.mark.parametrize("fruta", ["níspero", "guanábana", "ciruela", "durazno", "granada", "toronja"])
def test_las_seis_frutas_que_el_seeder_asigna_ahora_cuentan(fruta):
    assert go._featured_fruits_in_name(f"Merienda de {fruta}")


@pytest.mark.parametrize("no_fruta", ["naranja", "limón", "coco", "pasas", "dátiles", "aguacate",
                                     "tamarindo"])
def test_las_exclusiones_deliberadas_siguen_fuera(no_fruta):
    """`naranja`/`limón` son ralladura y aderezo en varias comidas; el resto es guarnición o grasa.
    Contarlas sobre-flagearía — decisión documentada en el código desde P2-DISH-COHERENCE."""
    assert go._featured_fruits_in_name(f"Plato con {no_fruta}") == set()


def test_el_caso_vivo_del_pool_roto():
    """Pool real del primer rechazo del canario: 2 de las 3 ya cuentan (limón sigue fuera, y es
    correcto: es un condimento)."""
    reconocidas = [f for f in ("Níspero", "Toronja", "Limón") if go._featured_fruits_in_name(f)]
    assert reconocidas == ["Níspero", "Toronja"]


# ───────────── 3. las cuatro superficies comparten matcher ─────────────

def test_las_cuatro_superficies_usan_el_matcher_ssot():
    """Gate, detector de paridad, de-dup y clash fruta↔pescado. Si divergen, el de-dup reescribe
    algo que el gate sigue viendo repetido y se quema el intento — P1-FRUIT-DEDUP-GATE-PARITY ya
    tuvo que cerrar ese fallo una vez."""
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert src.count("_featured_fruits_in_name(") >= 5   # 1 def + 4 usos
    # nadie debe volver a iterar la tupla cruda para contar
    assert "for fr in _FEATURED_FRUITS" not in src
    assert "for f in _FEATURED_FRUITS" not in src


def test_el_dedup_arregla_lo_que_el_gate_detecta():
    """Verificado en vivo sobre el día real (plan cd08ea3c, día 2): ciruela en el desayuno y en la
    merienda. Antes: invisible para ambos → se entregaba. Ahora: el de-dup la reescribe y el gate
    pasa. Por eso ampliar el vocabulario NO sube los rechazos."""
    plan = {"days": [{"meals": [
        {"name": "Panqueques de Harina de Trigo y Canela con Ciruelas",
         "ingredients": ["2 ciruelas"]},
        {"name": "Pechuga de Pollo en Airfryer con Batata", "ingredients": ["150 g de pollo"]},
        {"name": "Tostada Integral con Mantequilla de Maní y Ciruela y Queso Cottage",
         "ingredients": ["1 ciruela"]},
    ]}]}
    assert go._plan_has_same_day_fruit_repeat(plan) is True
    assert go.dedup_featured_fruits_in_plan(plan) >= 1
    assert go._plan_has_same_day_fruit_repeat(plan) is False


# ───────────── 4. el seeder: dos frutas por día ─────────────

def test_dos_frutas_distintas_por_dia():
    slots = ah._rotate_fruit_pairs(["Mango", "Guineo", "Fresas", "Melón"])
    assert len(slots) == 3
    for a, b in slots:
        assert a != b, (a, b)


def test_la_semana_usa_cuatro_frutas_no_seis():
    """2 por día × 3 días serían 6 compras; rotando sobre 4 la lista sube sólo 1 fruta."""
    slots = ah._rotate_fruit_pairs(["Mango", "Guineo", "Fresas", "Melón"])
    assert len({f for par in slots for f in par}) == 4


def test_prioriza_las_que_el_gate_reconoce():
    """Una repetición de níspero era invisible al gate, así que ponerlo primero no ayudaba a
    satisfacerlo. Las reconocidas van delante."""
    slots = ah._rotate_fruit_pairs(["Limón", "Mango", "Guineo"])
    assert slots[0][0] != "Limón"
    assert go._featured_fruits_in_name(slots[0][0])


@pytest.mark.parametrize("pool", [None, [], ["Mango"], ["", "  "]])
def test_sin_pool_utilizable_cae_al_texto_libre(pool):
    """Devuelve None y el caller usa la instrucción libre, en vez de inventar un pool."""
    assert ah._rotate_fruit_pairs(pool) is None


def test_el_prompt_pide_dos_frutas_y_no_deja_placeholders():
    p = ah.get_deterministic_variety_prompt("", {"goal": "gain_muscle"})
    import re
    assert not re.findall(r"\{[a-z_0-9]+\}", p), "placeholder sin rellenar"
    assert p.count("Frutas asignadas al día") == 3
    assert "NUNCA la misma dos veces el mismo día" in p


def test_la_lista_de_relleno_solo_trae_frutas_que_el_gate_ve():
    """`Limón` y `Naranja` estaban en el relleno y gastaban un slot del pool sin poder satisfacer
    nunca 'una fruta distinta por comida'."""
    from pathlib import Path
    src = (Path(ah.__file__).resolve()).read_text(encoding="utf-8")
    i = src.index("_DEFAULT_DR_FRUITS = (")
    bloque = src[i:src.index(")", i)]
    assert "Limón" not in bloque and "Naranja" not in bloque
    for f in ("Lechosa", "Mango", "Guineo"):
        assert f in bloque
