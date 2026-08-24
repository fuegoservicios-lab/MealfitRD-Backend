"""[P2-NEVERA-UNIT-SYSTEM-POR-PAIS · 2026-08-23] La proyección métrica se detenía en la lista
de la compra: la Nevera del usuario beta seguía mostrando libras.

`P1-UNIT-SYSTEM-BY-COUNTRY` proyectó el DISPLAY de la lista (`_project_display_units_for_country`)
y ahí se quedó. La Nevera nace de esa misma lista por `/restock`, pero pinta `quantity` y
`market_container || unit` crudos — y `market_container` es NULL en las 141 filas de
catálogo-país. Ítem real de una corrida de España:

    {"name":"Almejas","market_qty_numeric":4.5,"market_unit":"lbs","display_qty":"2 kg"}

«2 kg» en la lista y «4,5 lbs» en la Nevera, el mismo alimento y la misma sesión.

LA MITAD DE «GUARDA» NO ES UN GAP y este archivo no la toca: que `market_unit` siga siendo
"lbs" es decisión declarada y anclada por `test_p1_unit_system_by_country.py`
(`test_los_campos_que_consume_la_nevera_quedan_intactos`). Convertir el DATO metería gramos
donde la deducción de inventario espera libras. Lo que se proyecta aquí es el DISPLAY.

Y EL SELECTOR ERA DOS TABLAS. `UNIT_OPTIONS` (QPantryBuilder) y `COMMON_PURCHASE_UNITS`
(Pantry) eran la misma lista escrita dos veces, y ya habían drifteado: 'lb' contra 'libra',
'funda' contra 'bolsa', una con 'g' y sin 'kg' y la otra al revés. Ninguna de las dos ofrecía
'ml', y a un español el desplegable le empezaba por libras. Es la clase de fallo que
`P1-DIET-CANON-SSOT` ya pagó con tres tablas de dieta drifteadas.

Cubre:
  A. Paridad del mapa país -> sistema de unidades (frontend vs `COUNTRY_PROFILES`).
  B. Cero segundas tablas: las dos superficies consumen el SSOT.
  C. La fila proyecta cantidad Y unidad del MISMO objeto (medio proyectar es peor que nada).
  D. Las constantes de conversión y el umbral son los del backend.
  E. Control: los valores que el backend produce hoy, que el espejo de vitest replica.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
_SRC = _REPO / "frontend" / "src"
_SSOT = _SRC / "config" / "unitSystem.js"
_PANTRY = _SRC / "pages" / "Pantry.jsx"
_QPB = _SRC / "components" / "assessment" / "questions" / "QPantryBuilder.jsx"


def _sin_comentarios(src: str) -> str:
    """Quita comentarios de JS/JSX respetando cadenas. Los arreglos de este gap llevan encima
    notas que citan las constantes viejas; un guard al que le vale un comentario aprueba el
    defecto que persigue."""
    out = []
    i, n = 0, len(src)
    comilla = None
    while i < n:
        c = src[i]
        if comilla:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(src[i + 1])
                i += 2
                continue
            if c == comilla:
                comilla = None
            i += 1
            continue
        if c in "\"'`":
            comilla = c
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            j = src.find("*/", i + 2)
            i = n if j == -1 else j + 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _limpio(ruta: Path) -> str:
    return _sin_comentarios(ruta.read_text(encoding="utf-8", errors="replace"))


def _mapa_del_frontend() -> dict:
    src = _limpio(_SSOT)
    i = src.index("UNIT_SYSTEM_BY_COUNTRY")
    cuerpo = src[src.index("{", i):src.index("}", i)]
    return dict(re.findall(r"([A-Z]{2})\s*:\s*'(\w+)'", cuerpo))


# -- A. Paridad con el SSOT del backend -------------------------------------------------------

def test_el_mapa_del_frontend_es_espejo_exacto_de_country_profiles():
    """Un país nuevo en `COUNTRY_PROFILES` sin fila aquí caería al fallback 'imperial' EN
    SILENCIO: un mexicano leyendo libras y nadie enterándose. Es el mismo contrato que ya
    tienen COUNTRIES vs COUNTRY_PROFILES."""
    from constants import COUNTRY_PROFILES
    backend = {c: p.get("unit_system", "imperial") for c, p in COUNTRY_PROFILES.items()}
    assert _mapa_del_frontend() == backend


def test_el_fallback_del_frontend_es_el_mismo_que_el_del_backend():
    """Lo desconocido cae a 'imperial' en los dos lados — la conducta de hoy."""
    from constants import unit_system_for_country
    assert unit_system_for_country("Marte") == "imperial"
    assert re.search(r"DEFAULT_UNIT_SYSTEM\s*=\s*'imperial'", _limpio(_SSOT))


# -- B. Cero segundas tablas ------------------------------------------------------------------

@pytest.mark.parametrize("ruta,prohibida", (
    (_PANTRY, "COMMON_PURCHASE_UNITS"),
    (_QPB, "UNIT_OPTIONS"),
))
def test_ninguna_superficie_conserva_su_propia_lista_de_unidades(ruta, prohibida):
    """El guard no mira la GRAFÍA de una copia concreta: exige que la constante local no se
    declare. Volver a declararla es volver a la divergencia."""
    assert not re.search(rf"\b(?:const|let|var)\s+{prohibida}\s*=", _limpio(ruta)), (
        f"{ruta.name} vuelve a declarar {prohibida}: la lista de unidades tiene un solo SSOT"
    )


@pytest.mark.parametrize("ruta", (_PANTRY, _QPB))
def test_las_dos_superficies_piden_las_unidades_al_ssot_con_el_pais(ruta):
    src = _limpio(ruta)
    assert re.search(
        r"import\s*\{[^}]*\bunitOptionsForCountry\b[^}]*\}\s*from\s*'[^']*unitSystem'", src
    ), f"{ruta.name} no importa el SSOT del selector"
    assert re.search(r"unitOptionsForCountry\(\s*formData\?\.country", src), (
        f"{ruta.name} llama al SSOT sin país: ordenaría el selector por el sistema equivocado"
    )


def test_el_ssot_ofrece_las_metricas_que_faltaban_en_las_dos_listas():
    """El síntoma medido: ninguna de las dos listas tenía 'ml', y la del wizard tampoco 'kg'."""
    src = _limpio(_SSOT)
    metric = re.search(r"metric\s*:\s*\[([^\]]*)\]", src).group(1)
    orden = re.findall(r"'([^']+)'", metric)
    assert orden and orden[0] == "kg", f"el sistema métrico no empieza por kg: {orden}"
    assert "ml" in orden and "g" in orden, orden
    assert "libra" in orden, (
        "la proyección amputó 'libra': un español con un alimento YA guardado en libras "
        "perdería la unidad de su propia fila"
    )


# -- C. La fila proyecta cantidad Y unidad, del mismo objeto ----------------------------------

def test_la_fila_de_la_nevera_no_pinta_la_cantidad_cruda():
    """Proyectar sólo la unidad produce la peor de las tres pantallas posibles: el número de
    las libras bajo el rótulo de los kilos."""
    src = _limpio(_PANTRY)
    assert "fmtQty(item.quantity)" not in src, (
        "la fila sigue pintando la cantidad SIN proyectar mientras la unidad sí se proyecta"
    )
    assert src.count("projectMeasureForCountry(") >= 3, (
        "faltan superficies: fila de escritorio, tarjeta móvil y el modal de cantidad exacta"
    )


def test_todas_las_proyecciones_de_la_nevera_reciben_el_pais():
    src = _limpio(_PANTRY)
    llamadas = re.findall(r"projectMeasureForCountry\(([^)]*)\)", src, re.S)
    assert llamadas, "no hay ninguna proyección en Pantry.jsx"
    for arg in llamadas:
        assert "formData?.country" in arg.replace("\n", " "), (
            f"proyección sin país (se comportaría como imperial siempre): {arg!r}"
        )


def test_la_proyeccion_no_alimenta_ninguna_peticion():
    """La mitad de «guarda» que NO es un gap: el display no puede viajar como dato. El commit
    del editor de cantidad exacta y el PATCH de unidad siguen mandando lo guardado."""
    src = _limpio(_PANTRY)
    for m in re.finditer(r"_medida\.qty|_eq\.qty", src):
        linea = src[src.rfind("\n", 0, m.start()) + 1:src.find("\n", m.start())]
        assert "fetchWithAuth" not in linea and "JSON.stringify" not in linea, (
            f"un valor proyectado entra en una petición: {linea.strip()!r}"
        )


# -- D. Las constantes son las del backend ----------------------------------------------------

def test_las_constantes_de_conversion_son_las_del_backend():
    import shopping_calculator as sc
    src = _limpio(_SSOT)
    assert float(re.search(r"G_POR_LB\s*=\s*([\d.]+)", src).group(1)) == sc._G_POR_LB
    assert float(re.search(r"G_POR_OZ\s*=\s*([\d.]+)", src).group(1)) == sc._G_POR_OZ


def test_el_vocabulario_imperial_es_el_mismo_conjunto():
    """Si el backend aprende una unidad nueva y el frontend no, el mismo ítem se proyecta en la
    lista y no en la Nevera: el defecto original, otra vez."""
    import shopping_calculator as sc
    src = _limpio(_SSOT)
    cuerpo = re.search(r"PESO_IMPERIAL\s*=\s*new Set\(\[([^\]]*)\]", src).group(1)
    assert set(re.findall(r"'([^']+)'", cuerpo)) == set(sc._UNIDADES_DE_PESO_IMPERIAL)


def test_el_umbral_de_kg_es_el_mismo():
    """`_etiqueta_metrica` corta en 1000 g. Que las dos superficies redondeen igual es el
    punto: «2 kg» en la lista y «2,04 kg» en la Nevera sería el mismo defecto con otra cara."""
    assert re.search(r"gramos\s*>=\s*1000", _limpio(_SSOT))


# -- E. Control: lo que el backend produce hoy ------------------------------------------------

@pytest.mark.parametrize("qty,unidad,esperado", (
    (4.5, "lbs", "2 kg"),
    (1, "libra", "454 g"),
    (2.2, "lb", "998 g"),
    (8, "oz", "227 g"),
    (40, "oz", "1,1 kg"),
))
def test_los_valores_de_referencia_del_backend_no_se_movieron(qty, unidad, esperado):
    """Si este test se pone rojo, el redondeo del backend cambió y hay que mover el espejo del
    frontend (`unitSystem.p2_nevera_unit_system_por_pais.test.js`) EN EL MISMO cambio."""
    import shopping_calculator as sc
    g = qty * (sc._G_POR_OZ if unidad.startswith(("oz", "onza")) else sc._G_POR_LB)
    assert sc._etiqueta_metrica(g) == esperado
