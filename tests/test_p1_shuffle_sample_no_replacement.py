"""[P1-SHUFFLE-SAMPLE-NO-REPLACEMENT + P1-P11-SIG-INCLUDE-DAY · 2026-07-30] Dos bugs de producción
que se presentaban como "un test flaky".

`test_chunked_generation` fallaba ~50% de las pasadas del par. La causa NO era el orden de los
tests, ni la base, ni una fixture: es que **`random` nunca se siembra**, y ese ruido destapaba dos
defectos distintos del chunk worker degradado. Correlación 1:1 verificada semilla a semilla.

=== BUG 1 — el reintento muestreaba CON reemplazo ===

    shuffled_day = copy.deepcopy(random.choice(available_days))
    ...
    available_days = [d for d in available_days if d is not shuffled_day]   # NO-OP
    shuffled_day = copy.deepcopy(random.choice(available_days))

`shuffled_day` es un **deepcopy**, así que `d is not shuffled_day` es True para TODOS los elementos:
el candidato rechazado nunca salía del pool y los 3 intentos re-sorteaban de la lista completa —
muestreo con reemplazo donde el comentario del código dice "Remove the failing candidate".

Con 3 días en el pool y 1 solo válido: P(fallar los 3) = (2/3)³ = **29,6%** (medido 29,1% sobre 55
semillas). En ese ~30% el sistema **se rinde teniendo un día válido disponible** y entrega una Edge
Recipe sintética: el usuario ve "Desayuno: 150g Pollo con 100g Arroz" en vez de un día real.

⚠️ Comparar por IDENTIDAD contra algo que acabas de COPIAR nunca casa. Y un fix de 2026-07-28 ya
había curado este mismo síntoma por otro lado (la caché caliente hacía fallar los 3 candidatos):
*se arregló por qué fallaban, no por qué el bucle no llegaba a probar los otros.*

=== BUG 2 — el pre-check de idempotencia saltaba el merge ===

`_p11_meal_signature` comparaba SOLO (nombre, tipo) de las comidas. En la rama degradada `new_days`
son por construcción una permutación de `existing_days`, así que cuando el RNG reproducía el mismo
orden que había en storage, el guard se creía un duplicado y **no mergeaba**: el plan se quedaba en
3 días en vez de 6. P = 1/3 × 1/2 × 1/2 = 1/12 = **8,33%** (medido 8,6%).

El discriminador que faltaba lo tenía delante: el NÚMERO DE DÍA. En el escenario legítimo que el
guard defiende (T1 commiteó y T2 perdió el marker) los días en storage ya llevan su numeración
final y siguen coincidiendo; en el falso positivo storage tiene los días 1-3 y `new_days` son los
4-6.
"""
from __future__ import annotations

import copy
import random
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_CRON = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")


# ═══════════════════════ BUG 1 — muestreo sin reemplazo ═══════════════════════

def test_el_descarte_ya_no_compara_contra_una_copia():
    """La regresión concreta: `d is not shuffled_day` con `shuffled_day` deepcopy."""
    assert "available_days = [d for d in available_days if d is not shuffled_day]" not in _CRON, (
        "volvió el filtro por identidad contra la COPIA — nunca elimina nada y los 3 intentos "
        "re-sortean del pool completo")
    assert "available_days = [d for d in available_days if d is not _shuffle_pick]" in _CRON, (
        "el descarte debe usar el ORIGINAL elegido, no su copia")


def test_el_original_se_guarda_antes_de_copiar():
    i = _CRON.index("_shuffle_pick = random.choice(available_days)")
    seg = _CRON[i:i + 200]
    assert "shuffled_day = copy.deepcopy(_shuffle_pick)" in seg, (
        "la copia debe salir del original guardado, o los dos se desincronizan")
    # y el re-sorteo dentro del while hace lo mismo
    assert _CRON.count("shuffled_day = copy.deepcopy(_shuffle_pick)") >= 2


def test_el_mecanismo_reproducido_en_pequeno():
    """Prueba del defecto sin tocar el worker: con 3 candidatos y 1 válido, el muestreo CON
    reemplazo falla ~30% de las veces y SIN reemplazo acierta siempre. Es toda la diferencia entre
    entregar el día real del usuario y una receta sintética."""
    dias = [{"id": 1}, {"id": 2}, {"id": 3}]
    valido = 3

    def corrida(sin_reemplazo: bool, semilla: int) -> bool:
        random.seed(semilla)
        pool = list(dias)
        pick = random.choice(pool)
        copia = copy.deepcopy(pick)              # el worker trabaja sobre la copia
        for _ in range(3):
            if copia["id"] == valido:
                return True
            pool = [d for d in pool if d is not (pick if sin_reemplazo else copia)]
            if not pool:
                return False
            pick = random.choice(pool)
            copia = copy.deepcopy(pick)
        return copia["id"] == valido

    con = sum(corrida(False, s) for s in range(300))
    sin = sum(corrida(True, s) for s in range(300))
    assert sin == 300, f"sin reemplazo debe encontrar SIEMPRE el válido en 3 intentos, falló {300-sin}"
    assert con < 300, "si el muestreo con reemplazo no falla nunca, este test dejó de medir el bug"
    # (2/3)^3 = 29,6% de fallo teórico; se deja margen amplio para no atarlo a la semilla
    assert 200 <= con <= 260, f"tasa inesperada con reemplazo: {con}/300"


# ═══════════════════════ BUG 2 — la firma incluye el día ═══════════════════════

def test_la_firma_incluye_el_numero_de_dia():
    i = _CRON.index("def _p11_meal_signature")
    body = _CRON[i:_CRON.index("_p11_already_in_storage", i)]
    assert "day.get('day')" in body, (
        "sin el número de día la firma no distingue 'ya mergeado' de 'el shuffle reprodujo la "
        "cola' — y salta el merge, dejando el plan en 3 días en vez de 6")
    assert "if not _meals:" in body and "return ()" in body, (
        "se pierde el sentinel de día vacío, que el filtro de arriba usa para descartar días sin "
        "comidas")


def test_la_firma_discrimina_los_dos_escenarios():
    """Reproducción de la semántica con la misma forma de datos que usa el worker."""
    import textwrap
    ns = {}
    i = _CRON.index("                        def _p11_meal_signature")
    src = _CRON[i:_CRON.index("_p11_already_in_storage", i)]
    # `dedent` en vez de recortar N columnas a mano: la función vive a 24 espacios dentro del
    # worker y cualquier conteo fijo se rompe en cuanto cambie la anidación.
    exec(compile(textwrap.dedent(src).rstrip(), "<sig>", "exec"), ns)
    sig = ns["_p11_meal_signature"]

    def _dia(n, nombres):
        return {"day": n, "meals": [{"name": x, "type": "principal"} for x in nombres]}

    menu = ["Pollo Asado", "Pescado", "Res"]
    # FALSO POSITIVO que causaba el bug: mismas comidas, DIAS distintos (1-3 en storage, 4-6 nuevos)
    storage = [_dia(n, [m]) for n, m in zip((1, 2, 3), menu)]
    nuevos = [_dia(n, [m]) for n, m in zip((4, 5, 6), menu)]
    assert [sig(d) for d in storage] != [sig(d) for d in nuevos], (
        "un shuffle que reproduce el mismo menú NO puede leerse como 'ya mergeado'")

    # ESCENARIO LEGITIMO que el guard defiende: T1 commiteó los dias 4-6 y T2 perdió el marker
    ya_mergeado = [_dia(n, [m]) for n, m in zip((4, 5, 6), menu)]
    assert [sig(d) for d in ya_mergeado] == [sig(d) for d in nuevos], (
        "el guard debe SEGUIR detectando el duplicado real — si esto cae, el fix fue demasiado "
        "lejos y vuelve la duplicación silenciosa")

    assert sig({"day": 1, "meals": []}) == (), "día sin comidas conserva su sentinel"
    assert sig("no soy un dict") == ()
