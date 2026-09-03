"""[P1-TEST-RESIDUE-TELEMETRY · 2026-07-31] El detector de residuo también mira la
telemetría, no solo los usuarios.

POR QUÉ EXISTE
El detector original (`P1-TEST-RESIDUE-DETECTOR`) busca perfiles `e2e-test-%`. Es útil,
pero el residuo que más daño hizo el 31 de julio no tenía perfil que buscar:
`chunk_lesson_telemetry` acumuló 2.237 filas de 447 `user_id` que nunca existieron —
el 94% de la tabla.

Lo que eso rompió, medido:
  1. El cron de flota `_alert_high_synthesized_lesson_ratio` agrega esa tabla SIN
     filtrar por perfil, así que llevaba semanas midiendo la suite de tests en vez del
     producto. Disparó una alerta por ello.
  2. Falseó una cifra usada como diagnóstico: "843 eventos en la semana 2" cuando los
     reales eran 26. El residuo de tests no solo escribe en producción — contamina las
     mediciones que se hacen SOBRE producción, incluidas las que deciden si hay un bug.

Esto NO revisa la decisión de no montar una base aparte (documentada en conftest.py: el
catálogo de 204 alimentos debe ser el REAL). Hace lo que esa misma decisión promete
—"convertir un riesgo silencioso en uno visible"— para el residuo que no se veía.

Los tests EJECUTAN el detector con dobles; ninguno toca la base.
"""
from pathlib import Path

import conftest


def _capturar(monkeypatch, capsys, filas):
    """Corre el detector con una respuesta de DB fabricada y devuelve su stderr."""
    monkeypatch.setattr(conftest, "execute_sql_query", lambda *a, **k: filas)
    conftest._reportar_telemetria_fantasma()
    return capsys.readouterr().err


def _apuntar_a(monkeypatch, es_nonprod):
    """Fija qué responde el predicado de destino (P0-TEST-DB-DUAL-URL)."""
    monkeypatch.setattr(conftest.db_core, "_db_target_is_nonprod",
                        lambda: (es_nonprod, None))


def test_avisa_con_las_cifras_del_incidente(monkeypatch, capsys):
    _apuntar_a(monkeypatch, False)   # producción, como la noche del incidente
    err = _capturar(monkeypatch, capsys, [
        {"tabla": "chunk_lesson_telemetry", "n": 2237, "usuarios": 447},
    ])
    assert "P1-TEST-RESIDUE-TELEMETRY" in err
    assert "2237" in err and "447" in err
    assert "chunk_lesson_telemetry" in err
    assert "PRODUCCIÓN" in err
    # El aviso debe decir POR QUÉ importa, no solo que existe: un número sin
    # consecuencia se ignora. La consecuencia real es que mueve una métrica.
    assert "métrica" in err or "metrica" in err
    # Y debe traer la limpieza lista para copiar.
    #
    # [P1-TEARDOWN-SWEEP · 2026-08-12] Antes se afirmaba la cadena literal
    # `DELETE FROM chunk_lesson_telemetry`, porque el detector solo vigilaba ESA tabla.
    # Ahí estaba su problema: con 7.540 filas huérfanas repartidas en seis tablas, calló
    # sobre cinco. Ahora recorre las que el catálogo dice que tienen `user_id`, así que
    # la receta es una plantilla — el nombre concreto va en la línea del aviso, que este
    # mismo caso ya afirma arriba.
    #
    # Lo que se protege sigue siendo lo mismo: que el aviso no se limite a decir que hay
    # residuo, sino que traiga el DELETE para quitarlo.
    assert "DELETE FROM" in err
    assert "user_profiles" in err  # el predicado que distingue huérfana de legítima


def test_el_barrido_y_el_detector_salen_del_CATALOGO_no_de_una_lista():
    """[P1-TEARDOWN-SWEEP · 2026-08-12] La lección del incidente, en una aserción.

    El teardown limpiaba tres tablas escritas a mano y el detector vigilaba UNA. Ninguno
    de los dos estaba «mal»: estaban escritos cuando esas eran las tablas que había. Lo
    que falló es que una lista a mano no se entera de que el esquema creció, y nadie lo
    supo hasta que aparecieron 7.540 filas huérfanas de 600 dueños fantasma.

    Por eso los dos preguntan ahora al catálogo. Si alguien vuelve a poner una lista fija
    —aunque sea con los nueve nombres correctos de hoy— el problema vuelve el día que
    alguien cree la décima tabla, y otra vez sin avisar.
    """
    fuente = (Path(conftest.__file__).read_text(encoding="utf-8"))

    assert "def _tablas_con_user_id" in fuente, (
        "desapareció el descubrimiento de tablas por catálogo: el teardown vuelve a "
        "depender de una lista que envejece sola"
    )
    # La pregunta que no envejece.
    assert "column_name = 'user_id'" in fuente
    assert "information_schema.columns" in fuente

    # Y ambos consumidores la usan: el teardown para limpiar y el detector para vigilar.
    #
    # Se restan las DEFINICIONES. Un primer intento contaba las apariciones a secas, y la
    # línea `def _tablas_con_user_id()` cuenta como una: con el detector devuelto a una
    # sola tabla el guard seguía en verde, porque la definición le hacía el número. Lo
    # destapó la mutación — un guard que se cuenta a sí mismo mide su propia existencia,
    # no la del código que vigila.
    llamadas = fuente.count("_tablas_con_user_id()") - fuente.count("def _tablas_con_user_id()")
    assert llamadas >= 2, (
        "solo uno de los dos (barrido / detector) usa el catálogo — el otro se quedó "
        "mirando la lista vieja, que es exactamente el estado que dejó pasar el incidente"
    )


def test_contra_un_branch_no_grita_produccion(monkeypatch, capsys):
    """[P1-TEST-RESIDUE-TARGET] El aviso decía "contra la base de producción" SIN
    mirar, y en la primera corrida contra un branch eso ya era falso.

    Sigue reportando —un teardown que no completa conviene saberlo— pero sin la
    alarma ni el DELETE de urgencia: ahí el residuo no toca ninguna métrica. Un
    detector que exagera se aprende a ignorar, y entonces no sirve el día que acierta.
    """
    _apuntar_a(monkeypatch, True)
    err = _capturar(monkeypatch, capsys, [
        {"tabla": "chunk_lesson_telemetry", "n": 103, "usuarios": 15},
    ])
    assert "103" in err
    assert "branch de test" in err
    assert "PRODUCCIÓN" not in err
    assert "DELETE FROM" not in err          # no urge limpiar nada
    assert "Inocuo" in err


def test_calla_cuando_no_hay_residuo(monkeypatch, capsys):
    """Base limpia ⇒ silencio. Un detector que avisa siempre deja de leerse."""
    assert _capturar(monkeypatch, capsys, []) == ""


def test_calla_si_la_db_no_responde(monkeypatch, capsys):
    """Best-effort: sin red, el detector no puede estorbar a la corrida."""
    def _explota(*a, **k):
        raise RuntimeError("sin conexión")
    monkeypatch.setattr(conftest, "execute_sql_query", _explota)
    conftest._reportar_telemetria_fantasma()
    assert capsys.readouterr().err == ""


def test_el_hook_lo_invoca_de_verdad():
    """El detector cuelga de `pytest_sessionfinish`, que es el único nombre que pytest
    llama. Si alguien lo renombra a `pytest_sessionfinish_algo` creyendo que así se
    registra, queda inerte: verde para siempre, vigilando nada.

    Además exige que la llamada vaya ANTES del chequeo de usuarios, que tiene `return`s
    tempranos — colgarla al final la dejaría sin correr justo en el caso normal (sin
    usuarios residuales), que es cuando la telemetría sí puede estar sucia.
    """
    import inspect
    src = inspect.getsource(conftest.pytest_sessionfinish)
    assert "_reportar_telemetria_fantasma()" in src, (
        "P1-TEST-RESIDUE-TELEMETRY regresión: `pytest_sessionfinish` ya no invoca al "
        "detector de telemetría — quedaría como función muerta."
    )
    i_llamada = src.index("_reportar_telemetria_fantasma()")
    i_usuarios = src.index("user_profiles")
    assert i_llamada < i_usuarios, (
        "la llamada debe ir ANTES del chequeo de usuarios: ése retorna temprano cuando "
        "no hay perfiles residuales y se comería el aviso de telemetría."
    )
