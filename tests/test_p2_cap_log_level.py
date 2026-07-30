"""[P2-CAP-LOG-LEVEL · 2026-07-29] Un evento de DISEÑO que ocurre cientos de veces no puede ser
WARNING: el nivel deja de significar "mira esto".

Medido en 8 h de producción: **343 de 460 WARNING = 74,6% del journal** eran topes de perecederos
(P5-VEG-CAP 110, P6-LACTEOS-PERISHABLE-CAP 70, P6-CITRUS-CAP 62, P3-HERB-CAP 48…), o sea la
narración de una decisión de producto ya tomada ("no se compran 30 días de tomate fresco de golpe"),
consumida aguas abajo y comunicada al usuario en el display del ítem. Un operador que abre el journal
tras un incidente ve cientos de líneas de una decisión SANA y las 6 señales reales del día quedan
enterradas.

El fix no borra información: baja el per-ítem a INFO (mismo mensaje, mismo marker, greppable igual) y
emite UN warning agregado por corrida con los topes que sí son señal — los que recortan a la mitad o
más, que ya no hablan del tope sino del MENÚ.
"""
from __future__ import annotations

import logging
import os
import pathlib
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "shopping_calculator.py"), encoding="utf-8") as f:
    _SC = f.read()


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.recs = []

    def emit(self, record):
        self.recs.append((record.levelname, record.getMessage()))


@pytest.fixture()
def cap_logs():
    h = _Capture()
    root = logging.getLogger()
    prev = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(h)
    try:
        yield h
    finally:
        root.removeHandler(h)
        root.setLevel(prev)


def test_no_cap_emitter_uses_logging_warning_directly():
    """Anti-regresión estructural: si alguien añade un cap nuevo con `logging.warning`, vuelve el
    74,6%. El canal es `_cap_log`, que decide el nivel."""
    lines = _SC.split("\n")
    offenders = []
    for i, ln in enumerate(lines):
        if ln.rstrip().endswith("logging.warning("):
            blk = "\n".join(lines[i:i + 6])
            if "-CAP]" in blk:
                offenders.append(i + 1)
    assert not offenders, (
        f"emisores de cap todavía en logging.warning (L{offenders}) — usa `_cap_log`, "
        f"que es el que aplica la política de nivel")


def test_per_item_cap_logs_at_info(cap_logs):
    import shopping_calculator as sc
    sc._cap_log("[P5-VEG-CAP] 'Papa' peso cap: 2105g → 750g")
    levels = [lv for lv, m in cap_logs.recs if "P5-VEG-CAP" in m]
    assert levels == ["INFO"], f"el per-ítem debe ser INFO, fue {levels}"


def test_severe_caps_summary_is_one_warning(cap_logs):
    """Lo que SÍ es señal: un tope que recorta >50% habla del MENÚ. Una línea por corrida, no una
    por ítem."""
    import shopping_calculator as sc
    sc.reset_caps_applied_last_run()
    sc._record_cap_applied("Papa", 2105, 750, "P5-VEG-CAP")      # 36% → severo
    sc._record_cap_applied("Rábano", 68, 50, "P5-VEG-CAP")       # 74% → normal
    sc._record_cap_applied("Tomate", 1575, 750, "P5-VEG-CAP")    # 48% → severo
    sc._log_severe_caps_summary()
    warns = [m for lv, m in cap_logs.recs if lv == "WARNING" and "P2-CAP-LOG-LEVEL" in m]
    assert len(warns) == 1, f"exactamente UN warning agregado, hubo {len(warns)}"
    assert "2 tope(s)" in warns[0], f"solo los severos: {warns[0]}"
    assert "Papa" in warns[0] and "Tomate" in warns[0]
    assert "Rábano" not in warns[0], "un recorte del 74% no es señal"


def test_no_summary_when_nothing_is_severe(cap_logs):
    import shopping_calculator as sc
    sc.reset_caps_applied_last_run()
    sc._record_cap_applied("Rábano", 68, 50, "P5-VEG-CAP")
    sc._log_severe_caps_summary()
    assert not [m for lv, m in cap_logs.recs if lv == "WARNING" and "P2-CAP-LOG-LEVEL" in m]


def test_rollback_knob_restores_warning(cap_logs, monkeypatch):
    """`=1.0` devuelve TODO a WARNING (comportamiento previo), sin redeploy."""
    import shopping_calculator as sc
    monkeypatch.setattr(sc, "_CAP_LOG_SEVERE_RATIO", 1.0)
    sc._cap_log("[P6-CITRUS-CAP] 'Limón' peso cap: 900g → 300g")
    assert [lv for lv, m in cap_logs.recs if "P6-CITRUS-CAP" in m] == ["WARNING"]


def test_summary_is_wired_at_the_end_of_the_aggregator():
    """Debe correr cuando `_CAPS_APPLIED_LAST_RUN` ya está completo para ESTE run: después del
    reset de entrada y al final del agregador."""
    i_reset = _SC.index("reset_caps_applied_last_run()\n", _SC.index("def aggregate_and_deduct_shopping_list"))
    i_call = _SC.index("_log_severe_caps_summary()", i_reset)
    i_final = _SC.index("[AGGREGATE FINAL]")
    assert i_reset < i_final < i_call, "el resumen va al final del agregador, tras el log FINAL"


def test_knob_is_registered():
    import shopping_calculator  # noqa: F401
    from knobs import get_knobs_registry_snapshot
    assert "MEALFIT_CAP_LOG_SEVERE_RATIO" in get_knobs_registry_snapshot()

def test_summary_is_deduped_across_aggregator_passes(cap_logs):
    """[v2] `aggregate_and_deduct_shopping_list` corre una vez por variante de lista (semanal /
    quincenal / mensual) sobre el MISMO plan → el resumen salía 2-3 veces idéntico, a veces en el
    mismo segundo (medido en prod 21:28:40 ×2). Repetir la misma línea es la versión pequeña del
    problema que este bloque vino a resolver."""
    import shopping_calculator as sc
    sc._LAST_SEVERE_CAPS_SIG = set()
    sc._LAST_SEVERE_CAPS_AT = 0.0
    sc.reset_caps_applied_last_run()
    sc._record_cap_applied("Tomate", 9100, 3000, "P5-VEG-CAP")
    sc._log_severe_caps_summary()
    sc._log_severe_caps_summary()      # 2ª variante de lista, mismo plan
    sc._log_severe_caps_summary()      # 3ª
    warns = [m for lv, m in cap_logs.recs if lv == "WARNING" and "P2-CAP-LOG-LEVEL" in m]
    assert len(warns) == 1, f"una sola línea por contenido, hubo {len(warns)}"


def test_summary_reappears_when_content_changes(cap_logs):
    """El de-dup es por CONTENIDO, no un mute: otro plan con otros topes vuelve a avisar."""
    import shopping_calculator as sc
    sc._LAST_SEVERE_CAPS_SIG = set()
    sc._LAST_SEVERE_CAPS_AT = 0.0
    sc.reset_caps_applied_last_run()
    sc._record_cap_applied("Tomate", 9100, 3000, "P5-VEG-CAP")
    sc._log_severe_caps_summary()
    sc.reset_caps_applied_last_run()
    sc._record_cap_applied("Cebolla", 4900, 1200, "P5-VEG-CAP")
    sc._log_severe_caps_summary()
    warns = [m for lv, m in cap_logs.recs if lv == "WARNING" and "P2-CAP-LOG-LEVEL" in m]
    assert len(warns) == 2, f"contenido distinto → vuelve a avisar, hubo {len(warns)}"


def test_no_cap_test_captures_only_at_warning():
    """[v2] Blanket: un test que afirma sobre un cap NO puede capturar solo en WARNING.

    Tras bajar los emisores a INFO, `caplog.at_level(logging.WARNING)` deja de ver la línea del
    ítem. Cuatro archivos siguieron en VERDE igualmente — cazaban el resumen agregado, que es
    WARNING y cita dentro el nombre del alimento y el `reason`. El de-dup del resumen puso 5 en
    rojo y destapó los cuatro. **Un test verde por una línea distinta a la que cree mirar no
    protege nada**, y solo se descubre cuando algo no relacionado cambia. Este guard lo hace
    imposible de reintroducir en silencio.
    """
    offenders = []
    for f in sorted(pathlib.Path(__file__).parent.glob("test_*.py")):
        if f.name == pathlib.Path(__file__).name:
            continue
        t = f.read_text(encoding="utf-8", errors="ignore")
        if "at_level(logging.WARNING)" in t and re.search(r"-CAP\]", t):
            offenders.append(f.name)
    assert not offenders, (
        f"capturan en WARNING y afirman sobre caps: {offenders} — el cap per-ítem es INFO desde "
        f"P2-CAP-LOG-LEVEL; usa `at_level(logging.INFO)` o el filtro pasa a depender del resumen "
        f"agregado (verde por la línea equivocada)")



def test_dedup_expires_so_another_plan_is_never_swallowed(cap_logs, monkeypatch):
    """[v2] La firma es un global de MÓDULO: sin TTL, dos planes distintos con un set de topes
    severos idéntico se silenciarían mutuamente para siempre. Improbable, pero el modo de fallo
    sería tragarse la señal de OTRO usuario — lo contrario de lo que persigue este bloque."""
    import shopping_calculator as sc
    sc._LAST_SEVERE_CAPS_SIG = set()
    sc._LAST_SEVERE_CAPS_AT = 0.0
    _t = [1_000.0]
    monkeypatch.setattr(sc._time, "time", lambda: _t[0])
    sc.reset_caps_applied_last_run()
    sc._record_cap_applied("Tomate", 9100, 3000, "P5-VEG-CAP")
    sc._log_severe_caps_summary()
    _t[0] += sc._SEVERE_CAPS_DEDUP_TTL_S / 2      # misma ráfaga → callado
    sc._log_severe_caps_summary()
    warns = [m for lv, m in cap_logs.recs if lv == "WARNING" and "P2-CAP-LOG-LEVEL" in m]
    assert len(warns) == 1, f"dentro del TTL sigue de-duplicando, hubo {len(warns)}"
    _t[0] += sc._SEVERE_CAPS_DEDUP_TTL_S + 1      # otra ráfaga → vuelve a hablar
    sc._log_severe_caps_summary()
    warns = [m for lv, m in cap_logs.recs if lv == "WARNING" and "P2-CAP-LOG-LEVEL" in m]
    assert len(warns) == 2, f"pasado el TTL debe re-emitir, hubo {len(warns)}"


def test_dedup_ttl_knob_is_registered_and_zero_disables():
    import shopping_calculator as sc
    from knobs import get_knobs_registry_snapshot
    assert "MEALFIT_CAP_SUMMARY_DEDUP_TTL_S" in get_knobs_registry_snapshot()
    assert sc._SEVERE_CAPS_DEDUP_TTL_S == 120.0


def test_summary_failure_is_never_silent(cap_logs):
    """[v2] El `except` de este helper era `pass`. Al añadir el TTL referencié `time` en vez de
    `_time` (el módulo lo importa aliaseado) y el NameError se convirtió en "no hay nada severo que
    decir" — indistinguible desde fuera de un plan sano. Un canal de telemetría que se rompe en
    silencio es exactamente el modo de fallo que P2-CAP-LOG-LEVEL vino a atacar.

    (El fallo se inyecta en una fila de `_CAPS_APPLIED_LAST_RUN`, no parcheando `_time`: `_time` ES
    el módulo `time`, del que depende el propio `logging` para estampar cada record — parchearlo
    rompe el canal con el que se comprueba el resultado.)"""
    import shopping_calculator as sc
    sc._LAST_SEVERE_CAPS_SIG = set()
    sc._LAST_SEVERE_CAPS_AT = 0.0
    sc.reset_caps_applied_last_run()

    class _Boom(dict):
        def get(self, *a, **k):
            raise RuntimeError("fila de cap corrupta")
    sc._CAPS_APPLIED_LAST_RUN.append(_Boom())

    sc._log_severe_caps_summary()          # no debe propagar…
    fails = [m for lv, m in cap_logs.recs
             if lv == "WARNING" and "resumen de topes severos falló" in m]
    assert fails, "…pero tampoco callar: la rotura del canal tiene que verse"
    sc.reset_caps_applied_last_run()


def test_v3_dedup_por_alimento_no_por_gramos(cap_logs):
    """[v3] La v2 firmaba el TEXTO (gramos incluidos) y los gramos son justo lo que cambia entre
    variantes de lista: el mismo tope sale escalado ('Puerro 415->200g' / '104->50g' / '208->100g').
    La firma nunca coincidia ⇒ el de-dup era INCAPAZ DE DISPARAR por construccion. Medido en prod:
    53 resumenes en 40 min, 3-4 por plan, habiendo yo afirmado 'como maximo uno por plan'."""
    import shopping_calculator as sc
    sc._LAST_SEVERE_CAPS_SIG = set()
    sc._LAST_SEVERE_CAPS_AT = 0.0
    for _pre, _post in ((415, 200), (104, 50), (208, 100)):     # 3 variantes del MISMO tope
        sc.reset_caps_applied_last_run()
        sc._record_cap_applied("Puerro", _pre, _post, "P3-HERB-CAP")
        sc._log_severe_caps_summary()
    warns = [m for lv, m in cap_logs.recs if lv == "WARNING" and "P2-CAP-LOG-LEVEL" in m]
    assert len(warns) == 1, (
        f"el mismo alimento escalado por variante debe avisar UNA vez, hubo {len(warns)}")


def test_v3_resumen_acumulativo_no_repite_prefijos(cap_logs):
    """Los resumenes de un plan son ACUMULATIVOS (el 2o contiene las entradas del 1o mas otras,
    medido en corr=ef3fec6e). Comparar por SUBCONJUNTO deja pasar solo lo que aporta un alimento
    nuevo, para que el operador vea el resumen mas COMPLETO y no sus prefijos."""
    import shopping_calculator as sc
    sc._LAST_SEVERE_CAPS_SIG = set()
    sc._LAST_SEVERE_CAPS_AT = 0.0

    sc.reset_caps_applied_last_run()
    sc._record_cap_applied("Puerro", 104, 50, "P3-HERB-CAP")
    sc._log_severe_caps_summary()                                # 1: {Puerro}

    sc.reset_caps_applied_last_run()                             # 2: {Puerro} otra vez -> callado
    sc._record_cap_applied("Puerro", 415, 200, "P3-HERB-CAP")
    sc._log_severe_caps_summary()

    sc.reset_caps_applied_last_run()                             # 3: aporta Fresas -> habla
    sc._record_cap_applied("Puerro", 415, 200, "P3-HERB-CAP")
    sc._record_cap_applied("Fresas", 3901, 1814, "P6-FRUITS-PERISHABLE-CAP")
    sc._log_severe_caps_summary()

    warns = [m for lv, m in cap_logs.recs if lv == "WARNING" and "P2-CAP-LOG-LEVEL" in m]
    assert len(warns) == 2, f"prefijo callado, aporte nuevo hablado: esperaba 2, hubo {len(warns)}"
    assert "Fresas" in warns[1]


def test_v3_mismo_alimento_distinta_razon_si_avisa(cap_logs):
    """La clave es (alimento, RAZON): el mismo alimento topado por otro motivo es informacion nueva."""
    import shopping_calculator as sc
    sc._LAST_SEVERE_CAPS_SIG = set()
    sc._LAST_SEVERE_CAPS_AT = 0.0
    sc.reset_caps_applied_last_run()
    sc._record_cap_applied("Yogurt", 6319, 2722, "P6-LACTEOS-PERISHABLE-CAP")
    sc._log_severe_caps_summary()
    sc.reset_caps_applied_last_run()
    sc._record_cap_applied("Yogurt", 6319, 2722, "P6-DAIRY-OTHER-CAP")
    sc._log_severe_caps_summary()
    warns = [m for lv, m in cap_logs.recs if lv == "WARNING" and "P2-CAP-LOG-LEVEL" in m]
    assert len(warns) == 2


def test_v3_la_firma_no_contiene_gramos():
    """Ancla del defecto concreto de la v2: si la firma vuelve a llevar gramos, vuelve a ser inerte."""
    i = _SC.index("def _log_severe_caps_summary")
    body = _SC[i:_SC.index("\ndef ", i + 10)]
    assert '_claves.add(' in body, "la firma debe construirse de claves (alimento|razon)"
    assert '_sig = "|".join(sorted(_sev))' not in body, (
        "la firma volvio a ser el texto completo con gramos -> de-dup inerte por construccion")
    assert "issubset" in body, "falta la comparacion por subconjunto del patron acumulativo"
