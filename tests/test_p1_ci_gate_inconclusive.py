"""[P1-CI-GATE-INCONCLUSIVE · 2026-08-19] El gate del deploy podia dar verde sin
haber ejecutado la suite.

`P2-CI-PYTEST-PARALLEL` bajo el gate de ~20 min a ~9 con xdist a 3 workers, en un
diseno de 3 fases: bulk paralelo (A), cuarentena serial (B) y re-juicio serial de los
fallos del bulk (C). El re-juicio existe porque 47 archivos de la suite stubbean
`sys.modules`, asi que bajo xdist cualquier test puede caer como victima aleatoria de
un vecino de worker; un fallo solo cuenta si falla TAMBIEN en serie.

EL FALLO. La fase A trataba `exit != 0` como «el bulk enumero fallos». No siempre:
xdist aborta con INTERNALERROR (`KeyError: <WorkerController gw0>`, visto 3 veces el
2026-08-19) y ahi la sesion no termina, asi que `cacheprovider` no escribe el cache y
`--lf` se queda vacio o rancio.

Y con el cache vacio, MEDIDO:

    pytest --lf --last-failed-no-failures none   ->   deselecciona todo, exit 0

O sea que la fase C daba verde sin ejecutar un solo test. Ejercitado con un python de
pega contra el `run_ci.ps1` real, el codigo anterior daba PASS en 3 de 6 escenarios en
los que la suite NUNCA corrio entera (`internalerror_siempre`, `internalerror_luego_ok`,
`fallos_con_cache_vacia`). Un falso verde en la puerta de produccion es peor que el
ruido que las 3 fases venian a evitar.

LO QUE ESTE P-FIX ENSENA

1. La senal limpia era el CODIGO DE SALIDA, y no habia que adivinarla: pytest usa 1
   para «hubo fallos» y 3 para «error interno» (medido, no supuesto). Solo el 1 deja
   una lista de fallos en la que confiar. 2 (interrumpido), 3, 4 (uso) y 5 (nada
   coleccionado) son veredictos NO CONCLUYENTES.

2. «No concluyente» no puede colapsar ni a verde ni a rojo: colapsarlo a rojo tumba
   deploys buenos por un aborto transitorio, y a verde es lo que ya pasaba. La salida
   es un reintento y, si insiste, la SERIE COMPLETA — el mismo estandar de verdad del
   gate historico.

3. La captura va a VARIABLE (`Tee-Object -Variable`) y no a fichero. La primera version
   usaba `-FilePath` sobre una ruta fija de %TEMP% y fallaba con «el proceso no puede
   acceder al archivo»: una ruta compartida entre corridas es una clase de bug gratuita
   dentro de un gate. `-Variable` conserva el streaming en vivo (el motivo de usar Tee
   y no una redireccion) y preserva `$LASTEXITCODE`.

NO se bumpea `_LAST_KNOWN_PFIX`: esto no cambia el binario que corre en el VPS. El
marcador existe para detectar deploy lag, y bumpearlo por un script que solo se ejecuta
en la maquina del operador obligaria a un deploy que no entrega nada.

tooltip-anchor: P1-CI-GATE-INCONCLUSIVE
"""
from __future__ import annotations

import io
import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
_GATE = _ROOT / "scripts" / "run_ci.ps1"
_GATE_ESPEJO = _BACKEND / "scripts" / "run_ci.ps1"


def _sql() -> str:
    return io.open(_GATE, encoding="utf-8").read()


def _solo_codigo(s: str) -> str:
    """Quita las lineas que son COMENTARIO ENTERO.

    Existe porque este test cayo en la misma trampa que ya mordio tres veces hoy en
    este repo: la asercion «`Tee-Object -FilePath` no debe aparecer» la disparaba la
    prosa que explica POR QUE no debe aparecer. La respuesta no es reescribir el
    comentario -- un guard que obliga a no documentarse acaba desactivado.

    Solo se filtran las lineas cuyo primer caracter no-blanco es `#`: en PowerShell eso
    es un comentario sin ambiguedad, asi que el filtro NO puede comerse codigo. Un
    filtro mas listo (quitar todo lo que sigue a un `#`) se tragaria un `#` dentro de
    una cadena, y en una asercion de tipo «no debe aparecer» comerse codigo es un
    falso VERDE.
    """
    return "\n".join(l for l in s.splitlines() if not l.lstrip().startswith("#"))


def test_las_dos_copias_ssot_son_byte_identicas():
    """Misma regla que las migraciones (P3-MIGRATIONS-SSOT): el gate vive en los dos
    dirs porque son repos hermanos con remotes propios."""
    a = _GATE.read_bytes()
    b = _GATE_ESPEJO.read_bytes()
    assert a and a == b, "run_ci.ps1 difiere entre workspace-root y backend/"


def test_solo_el_exit_1_deja_una_lista_de_fallos_fiable():
    """El corazon del fix. Si alguien vuelve a escribir `if ($bulkExit -ne 0)` para
    decidir el re-juicio, el aborto de xdist vuelve a colarse como «hubo fallos»."""
    s = _sql()
    assert re.search(r"\$bulkConcluyente\s*=\s*\(\(\$bulkExit -eq 0\)\s*-or\s*\(\$bulkExit -eq 1\)\)", s), (
        "la discriminacion por codigo de salida desaparecio")
    assert "INTERNALERROR" in s, "falta el cinturon por texto para el caso exit=1 tras abortar"
    assert not re.search(r"if \(\$bulkExit -ne 0\) \{", s), (
        "vuelve el `-ne 0`: trata un aborto como una lista de fallos")


def test_un_veredicto_no_concluyente_cae_a_la_serie_completa():
    """No puede colapsar a verde (era el bug) ni a rojo (tumbaria deploys buenos por un
    aborto transitorio): reintento y, si insiste, la suite entera en serie."""
    s = _sql()
    assert re.search(r"foreach \(\$intento in 1, 2\)", s), "falta el reintento"
    assert re.search(r"\$serieCompleta\s*=\s*-not \$bulkConcluyente", s)
    assert 'throw "pytest (serie completa tras bulk no concluyente) fallo' in s


def test_una_fase_c_que_no_re_juzga_nada_no_puede_pasar():
    """Segundo cinturon: el bulk dijo «hubo fallos» pero `--lf` no ejecuto NADA => el
    cache estaba vacio o rancio. Salir verde de ahi es el mismo falso verde sin
    INTERNALERROR de por medio."""
    s = _sql()
    assert re.search(r"\$lfTexto -notmatch", s), "falta la comprobacion de fase C vacua"
    assert "passed|failed|error" in s


def test_la_captura_va_a_variable_y_no_a_un_fichero_compartido():
    """`-FilePath` sobre una ruta fija de %TEMP% se choco consigo mismo. Un fichero
    compartido entre corridas es una clase de bug gratuita dentro de un gate."""
    s = _sql()
    assert "Tee-Object -Variable" in s
    assert "Tee-Object -FilePath" not in _solo_codigo(s), (
        "vuelve la captura a fichero: se choca consigo misma entre corridas")


def test_la_escotilla_serial_sigue_existiendo():
    """MEALFIT_CI_PYTEST_WORKERS=1 es lo que permite desplegar sin que el gate se coma
    ~8 GB cuando hay otra sesion trabajando en la misma maquina.

    Se exige la LECTURA (`$env:...`) sobre el codigo, no la mera mencion del nombre:
    la primera version afirmaba `"MEALFIT_CI_PYTEST_WORKERS" in s` y una mutacion que
    renombraba la variable en el codigo la dejaba VERDE, porque el nombre seguia vivo
    en el comentario que documenta la escotilla. Un test que una linea de prosa puede
    satisfacer no esta comprobando el codigo.
    """
    codigo = _solo_codigo(_sql())
    assert "$env:MEALFIT_CI_PYTEST_WORKERS" in codigo, (
        "el gate ya no LEE la escotilla (mencionarla en un comentario no basta)")
    assert re.search(r'\$workers -eq "1" -or \$workers -eq "0" -or \$workers -eq "serial"', codigo)


def test_documenta_la_medicion_que_lo_destapo():
    """Sin esta nota, «--lf con cache vacio sale 0» hay que re-descubrirlo. Es el hecho
    del que depende todo el diseno."""
    s = _sql()
    assert "DESELECCIONA TODO y sale 0" in s
    assert "3 para \"error interno\"" in s or "3 (interno)" in s


# ─────────── la rama SERIE: la misma distincion, que solo vivia en la paralela ───────────
#
# [P1-CI-SERIE-INCONCLUSIVE · 2026-08-20] `MEALFIT_CI_PYTEST_WORKERS=1` era un
# `pytest -x` pelado: TODO exit != 0 se reportaba como «los tests FALLARON».
#
# El 2026-08-20 la suite murio por falta de memoria (3,7 GB libres de 15,7) y el gate
# dijo exactamente eso. Sali a buscar un test roto que no existia y perdi una corrida
# de 20 minutos. Un veredicto que confunde «aborto» con «hay una regresion» no es solo
# ruido: DIRIGE MAL la investigacion, que es peor que no decir nada.
#
# Aqui no hay a que caer —la serie ya es el modo mas conservador—, asi que se reintenta
# una vez y, si insiste, se falla DICIENDO QUE ABORTO. Fallar sigue siendo lo correcto:
# nunca desplegar sin veredicto. Lo que cambia es que el operador sepa que buscar.
#
# Ejercitado con un python de pega contra el `run_ci.ps1` REAL. El codigo anterior, en
# los mismos 4 escenarios, no distinguia ninguno (motivo vacio) y ademas tumbaba el
# deploy ante un aborto TRANSITORIO, sin reintentar.

def test_la_serie_tambien_discrimina_por_codigo_de_salida():
    s = _sql()
    assert re.search(r"\$serieConcluyente\s*=\s*\(\(\$serieExit -eq 0\)\s*-or\s*\(\$serieExit -eq 1\)\)", s), (
        "la rama serie volvio a tratar cualquier exit != 0 como «fallaron tests»")
    assert re.search(r"foreach \(\$intento in 1, 2\)[\s\S]{0,600}\$serieExit", s), (
        "falta el reintento en la rama serie: un aborto transitorio tumba el deploy")


def test_el_mensaje_de_aborto_no_dice_que_fallaron_tests():
    """Lo que se arregla NO es el veredicto (abortar debe fallar el gate igual), es el
    DIAGNOSTICO. Si el texto vuelve a hablar de tests fallando, el operador vuelve a
    buscar una regresion que no existe."""
    s = _sql()
    m = re.search(r'throw \("pytest \(serie\) NO CONCLUYENTE[\s\S]{0,600}?\)\n', s)
    assert m, "falta el throw especifico del aborto en la rama serie"
    mensaje = m.group(0)
    assert "ABORTO, no fallaron tests" in mensaje
    assert "No busques una regresion todavia" in mensaje
    assert "memoria" in mensaje, "sin la pista de la causa tipica el mensaje no orienta"


def test_una_regresion_real_en_serie_se_llama_por_su_nombre():
    s = _sql()
    # [P1-CI-SERIE-ENUMERA · 2026-08-20] El mensaje crecio: ahora dice CUANTOS fallos
    # hay, porque con `--maxfail` puede haber varios. Se ancla el nucleo, no la frase
    # entera -- un guard atado al texto exacto se rompe cada vez que se mejora el copy,
    # y eso ensena a relajarlo en vez de leerlo.
    assert 'pytest (serie) fallo (exit $serieExit) - regresion real' in s


def test_las_dos_ramas_usan_el_mismo_criterio():
    """Si una rama discrimina y la otra no, el gate miente segun el modo en que corras
    —que es justo lo que paso entre el 19 y el 20 de agosto—."""
    s = _sql()
    assert s.count("-eq 0) -or (") >= 2, (
        "solo una de las dos ramas discrimina por codigo de salida")
    assert s.count("INTERNALERROR") >= 2, (
        "el cinturon por texto falta en una de las dos ramas")


# ─────────── la serie ENUMERA en vez de parar en el primer fallo ───────────
#
# [P1-CI-SERIE-ENUMERA · 2026-08-20] `-x` es `--maxfail=1`. El 2026-08-20 eso costo TRES
# ciclos de deploy seguidos, de 10-17 min cada uno, para descubrir de uno en uno seis
# fallos que estaban ahi desde el principio. Dos de ellos llevaban rotos desde el dia
# ANTERIOR sin que nadie lo supiera, porque el gate paraba antes de llegar a su archivo.
#
# O sea que un gate que enumera de uno en uno no solo es lento: ESCONDE deuda detras del
# primer rojo, y la esconde justo mientras el arbol se sigue tocando.
#
# Tampoco se quita el tope. Sin limite, un fallo en el primer test --un import roto, un
# conftest malo-- obliga a esperar los ~20 min completos para enterarte de algo que se
# sabia en 10 segundos. `--maxfail` es la posicion intermedia, y 10 cubre de sobra una
# tanda normal (aquel dia eran 6).

def test_la_serie_enumera_hasta_un_tope():
    s = _sql()
    assert "--maxfail=$maxfail" in s, (
        "la serie volvio a `-x`: un fallo temprano vuelve a esconder a los demas")
    # Acotado a la rama SERIE: la fase C de la rama paralela conserva `-x` a proposito
    # --re-juzga una lista corta ya conocida, no la suite-- y prohibirlo ahi seria un
    # falso positivo. La primera version de este guard cayo justo en eso.
    ini = s.index('if ($workers -eq "1"')
    fin = s.index('} else {', ini)
    # Y sin comentarios: el `-x` que hay dentro esta en la PROSA que explica por que ya
    # no se usa. Enesima vez del dia que un comentario dispara su propio guard.
    rama_serie = _solo_codigo(s[ini:fin])
    assert "-x" not in rama_serie, "quedo un `-x` en la rama serie: esconde los demas fallos"
    assert "--maxfail=$maxfail" in rama_serie


def test_el_tope_es_configurable_y_razonable():
    s = _sql()
    assert "MEALFIT_CI_PYTEST_MAXFAIL" in _solo_codigo(s), "falta el knob del tope"
    m = re.search(r'if \(-not \$maxfail\) \{ \$maxfail = "(\d+)" \}', s)
    assert m, "el default debe ser un literal legible"
    assert 3 <= int(m.group(1)) <= 50, (
        f"tope {m.group(1)}: uno demasiado bajo vuelve a esconder fallos; uno "
        "demasiado alto paga la suite entera cuando algo esta roto de raiz")


def test_el_mensaje_dice_CUANTOS_fallos_hay():
    """Sin la cuenta, el operador no sabe si relanzar para descubrir el siguiente. Con
    ella --y con la lista completa arriba-- una sola corrida basta."""
    s = _sql()
    assert "regexp" in s.lower() or "[regex]::Matches" in s, "no se cuentan los FAILED"
    assert "no hace falta relanzar" in s
