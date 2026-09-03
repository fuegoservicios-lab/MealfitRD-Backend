"""[P3-AVG-MEAL-HOUR-CIRCULAR · 2026-08-23] La hora media de comida era una media aritmetica de
horas de RELOJ, y su consumidor sumaba el retraso sin volver al reloj.

DOS DEFECTOS, Y EL ORDEN IMPORTA
--------------------------------
1. `db_facts.get_avg_meal_hour` promediaba `[18, 20, 21, 0, 14]` como escalares: 14,6. Esa no es
   ninguna de las horas a las que el usuario come, y ni siquiera esta entre ellas. Una hora de
   reloj es un ANGULO: 23:00 y 01:00 distan 2 h, no 22. Cada cruce de medianoche arrastraba la
   media hacia el MEDIODIA, porque el 0 entraba como "cero" y no como "medianoche". El caso
   extremo medido: `[23.5, 0.5]` -> 12,0 (mediodia) cuando la respuesta es 0,0 (medianoche).

2. `proactive_agent.run_proactive_checks` hacia `nudge_hour = avg_hr + delay_hours` SIN `% 24`, y
   despues comparaba `floor(current_hour_float) == floor(nudge_hour)` contra un reloj que solo
   vale 0..23. Un `nudge_hour` de 24,0 no iguala a nada: ese recordatorio no se envia JAMAS, sin
   log ni error.

EL ARREGLO OBVIO (solo el 1) EMPEORA EL CASO ESPANOL, y por eso el consumidor va primero: la
media circular de una cena espanola `[21, 22, 23, 0]` es 22,5 (correcta) y con delay 1,5 da 24,0
-> fuera del reloj. La media aritmetica rota daba 16,5, que estaba MAL pero caia dentro. Arreglar
la media sin arreglar el consumidor convierte "nudge a la hora equivocada" en "nudge nunca".

VINCULO CON EL PAIS (indirecto pero real): las cenas espanolas (21:00-23:00, con picoteo despues)
cruzan la medianoche mucho mas que las dominicanas (~19:00). Los dos defectos castigan justo a la
poblacion beta.

NO SE PUDO MEDIR CONTRA DATOS REALES: `consumed_meals` tiene 5 filas en toda la base y
`nudge_outcomes` 9, la ultima del 6-ago. Por eso los casos de abajo son inyectados, y por eso el
guard del consumidor evalua la EXPRESION REAL extraida del fuente de produccion en vez de una
replica escrita a mano (leccion `P2-COUNTRY-SETTINGS-TEST-REPLICA`).

ZONA HORARIA DE LA MAQUINA: irrelevante aqui. El huso se INYECTA (`user_tz_offset_min` mockeado) y
las horas de comida entran como datos, no se construyen desde el reloj local del autor.
"""

import ast
import io

import pytest

import db_facts
import proactive_agent


# ===========================================================================
# 1. La media: circular, no aritmetica
# ===========================================================================

def _avg(monkeypatch, horas, tz_offset_min=240):
    """Invoca la funcion REAL con las horas dadas ya en hora local (el SQL hace la conversion;
    aqui se inyecta su resultado). `tz_offset_min` se inyecta explicitamente: este test no puede
    depender de la zona de la maquina que lo corre."""
    filas = [{"hr": int(h), "mn": round((h - int(h)) * 60)} for h in horas]
    monkeypatch.setattr(db_facts, "connection_pool", object(), raising=True)
    monkeypatch.setattr(db_facts, "user_tz_offset_min", lambda uid: tz_offset_min, raising=True)
    monkeypatch.setattr(db_facts, "execute_sql_query", lambda *a, **k: filas, raising=True)
    return db_facts.get_avg_meal_hour("u-1", "Cena")


@pytest.mark.parametrize(
    "horas,aritmetica_rota,circular_correcta,nota",
    [
        # El caso literal del hallazgo.
        ([18, 20, 21, 0, 14], 14.6, 19.57, "la media aritmetica no es ninguna hora real"),
        # Cena espanola con picoteo tras medianoche.
        ([21, 22, 23, 0], 16.5, 22.5, "cena espanola: 16:30 es la merienda, no la cena"),
        ([22, 23, 0, 1], 11.5, 23.5, "cruce de medianoche: la aritmetica cae al mediodia"),
        # El caso extremo: dos horas contiguas a ambos lados de medianoche.
        ([23.5, 0.5], 12.0, 0.0, "23:30 y 00:30 distan 1 h, no 23"),
    ],
)
def test_media_circular_no_aritmetica(monkeypatch, horas, aritmetica_rota, circular_correcta, nota):
    """Revertir a `sum(...)/len(...)` devuelve `aritmetica_rota` y pone esto en rojo."""
    got = _avg(monkeypatch, horas)
    assert got == pytest.approx(circular_correcta, abs=0.02), (
        f"{nota}: para {horas} se obtuvo {got}; la media circular es {circular_correcta} "
        f"y la aritmetica rota daba {aritmetica_rota}."
    )
    assert got != pytest.approx(aritmetica_rota, abs=0.02), (
        f"{nota}: sigue devolviendo la media aritmetica ({aritmetica_rota})."
    )


@pytest.mark.parametrize(
    "horas,esperado",
    [
        ([8, 9, 10], 9.0),
        ([13.0, 13.5, 14.0], 13.5),
        ([19, 19, 20], 19.33),
        ([9.0], 9.0),
    ],
)
def test_sin_cruce_de_medianoche_coincide_con_la_aritmetica(monkeypatch, horas, esperado):
    """El dominicano que cena a las 19:00 no puede notar nada: donde no hay cruce, la media
    circular y la aritmetica coinciden salvo redondeo. Sin este assert, "arreglar" la media
    podria haber movido la hora del nudge de toda la poblacion actual."""
    got = _avg(monkeypatch, horas)
    assert got == pytest.approx(esperado, abs=0.02)
    assert got == pytest.approx(round(sum(horas) / len(horas), 2), abs=0.02)


def test_resultante_nula_devuelve_none_y_no_medianoche(monkeypatch):
    """Comidas antipodales (00:00 y 12:00): la media circular NO EXISTE.

    `atan2(0, 0)` en Python devuelve 0.0 — medianoche, la PEOR hora posible para un nudge. Aqui
    se exige `None`, que es el contrato que el consumidor ya sabe leer (cae a su horario por
    defecto para esa comida)."""
    assert _avg(monkeypatch, [0, 12]) is None
    assert _avg(monkeypatch, [6, 18]) is None


def test_sin_registros_sigue_devolviendo_none(monkeypatch):
    monkeypatch.setattr(db_facts, "connection_pool", object(), raising=True)
    monkeypatch.setattr(db_facts, "user_tz_offset_min", lambda uid: 240, raising=True)
    monkeypatch.setattr(db_facts, "execute_sql_query", lambda *a, **k: [], raising=True)
    assert db_facts.get_avg_meal_hour("u-1", "Cena") is None


@pytest.mark.parametrize(
    "horas",
    [
        [23, 23.5, 0, 0.5],
        [23.9, 23.95],
        [0.0, 0.1],
        [11.9, 12.1],
        [18, 20, 21, 0, 14],
    ],
)
def test_el_resultado_siempre_es_una_hora_de_reloj(monkeypatch, horas):
    """Invariante dura: lo que sale es una hora que un reloj puede mostrar, `[0, 24)`.

    El redondeo a 2 decimales puede empujar 23,999 a 24,0, que no existe."""
    got = _avg(monkeypatch, horas)
    assert got is None or 0.0 <= got < 24.0, f"{horas} -> {got}, que no es una hora de reloj."


def test_el_huso_inyectado_llega_al_sql(monkeypatch):
    """El offset por usuario sigue gobernando la conversion (contrato de `P1-AVG-MEAL-HOUR-SIGN`);
    la media circular no lo puede haber cortocircuitado."""
    capturado = {}

    def _q(query, params, **k):
        capturado["params"] = params
        return [{"hr": 9, "mn": 0}]

    monkeypatch.setattr(db_facts, "connection_pool", object(), raising=True)
    monkeypatch.setattr(db_facts, "user_tz_offset_min", lambda uid: -120, raising=True)
    monkeypatch.setattr(db_facts, "execute_sql_query", _q, raising=True)
    db_facts.get_avg_meal_hour("u-1", "Cena")
    assert capturado["params"][0] == -120 and capturado["params"][1] == -120


# ===========================================================================
# 2. El consumidor: `nudge_hour` tiene que volver al reloj
# ===========================================================================

def _expresion_nudge_hour() -> ast.expr:
    """Extrae del FUENTE DE PRODUCCION la expresion asignada a `nudge_hour`.

    Se evalua el nodo REAL, no una replica escrita a mano: si alguien quita el `% 24`, los tests
    de abajo se ponen rojos porque estan ejecutando su codigo, no una copia mia que seguiria
    diciendo lo que yo escribi (leccion `P2-COUNTRY-SETTINGS-TEST-REPLICA`).
    """
    src = io.open(proactive_agent.__file__, encoding="utf-8").read()
    encontrados = [
        n.value for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "nudge_hour" for t in n.targets)
    ]
    assert len(encontrados) == 1, (
        f"se esperaba UNA asignacion a `nudge_hour` en proactive_agent.py y hay {len(encontrados)}. "
        "Si se duplico, este guard solo vigila una de las dos."
    )
    return encontrados[0]


def _evaluar_nudge_hour(avg_hr: float, delay_hours: float) -> float:
    # `eval` deliberado y acotado, en un test: lo que se evalua NO es entrada externa sino un
    # unico nodo `ast.expr` extraido de `proactive_agent.py`, un fuente de este mismo repo que
    # el interprete ya importo antes de llegar aqui (`import proactive_agent`, arriba). Es
    # justamente el punto del guard: ejecutar la expresion de PRODUCCION en vez de una copia
    # mia. `__builtins__` va vacio y el namespace se limita a los dos operandos, asi que si
    # alguien sustituyera esa linea por algo con efectos, el `eval` fallaria en vez de correrlo.
    # `ast.literal_eval` no sirve: `avg_hr + delay_hours` no es un literal.
    expr = _expresion_nudge_hour()
    codigo = compile(ast.Expression(body=expr), "<proactive_agent:nudge_hour>", "eval")
    return eval(codigo, {"__builtins__": {}}, {"avg_hr": avg_hr, "delay_hours": delay_hours})


@pytest.mark.parametrize("delay_hours", [1.0, 1.5, 2.5])
@pytest.mark.parametrize("avg_hr", [0.0, 8.0, 13.5, 19.0, 21.0, 22.5, 23.0, 23.75])
def test_nudge_hour_siempre_cae_dentro_del_reloj(avg_hr, delay_hours):
    """`current_hour_float` solo vale 0..23. Un `nudge_hour` fuera de ese rango no iguala a nada
    y ese recordatorio no se envia NUNCA, en silencio.

    Quitar el `% 24` pone en rojo las filas de cena tardia — que son exactamente las espanolas."""
    got = _evaluar_nudge_hour(avg_hr, delay_hours)
    assert 0.0 <= got < 24.0, (
        f"avg_hr={avg_hr} + delay={delay_hours} -> nudge_hour={got}, fuera del reloj: "
        "ese nudge no se enviaria jamas."
    )


@pytest.mark.parametrize(
    "avg_hr,delay_hours,esperado",
    [
        (19.0, 1.5, 20.5),    # cena dominicana: sin cambio, sigue dentro del reloj
        (8.0, 1.0, 9.0),      # desayuno: sin cambio
        (22.5, 1.5, 0.0),     # cena espanola: cruza la medianoche en vez de desaparecer
        (23.0, 2.5, 1.5),     # picoteo tardio
        (23.75, 1.0, 0.75),
    ],
)
def test_nudge_hour_cruza_la_medianoche_en_vez_de_desaparecer(avg_hr, delay_hours, esperado):
    """El recordatorio de una comida de las 23:30 es a la 1:00 del dia siguiente, no "nunca".
    Las dos primeras filas fijan que la conducta dominicana NO cambia."""
    assert _evaluar_nudge_hour(avg_hr, delay_hours) == pytest.approx(esperado, abs=1e-9)


def test_la_expresion_usa_ambos_operandos():
    """El `% 24` no puede haberse conseguido tirando el retraso o la media por el camino."""
    expr = _expresion_nudge_hour()
    nombres = {n.id for n in ast.walk(expr) if isinstance(n, ast.Name)}
    assert {"avg_hr", "delay_hours"} <= nombres, (
        f"`nudge_hour` ya no depende de ambos operandos: {sorted(nombres)}"
    )
    assert any(
        isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mod)
        and isinstance(n.right, ast.Constant) and n.right.value == 24
        for n in ast.walk(expr)
    ), "`nudge_hour` ya no vuelve al reloj con `% 24`."


def test_el_formateo_de_la_hora_aguanta_la_medianoche():
    """Con el `% 24`, `nudge_hour` puede valer 0.0. El formateo de `trigger_time_str` ya trataba
    ese caso (`if display_hr == 0: display_hr = 12`), pero antes era codigo inalcanzable: sin
    modulo, `nudge_hour` nunca era 0. Este assert lo fija ahora que SI se alcanza."""
    src = io.open(proactive_agent.__file__, encoding="utf-8").read()
    arbol = ast.parse(src)
    tiene_guarda_medianoche = any(
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "display_hr" for t in n.targets)
        and isinstance(n.value, ast.Constant) and n.value.value == 12
        for n in ast.walk(arbol)
    )
    assert tiene_guarda_medianoche, (
        "desaparecio la normalizacion `display_hr = 12` para la hora 0: con el `% 24` vivo, "
        "un nudge de medianoche se mostraria como '0:00 AM'."
    )
