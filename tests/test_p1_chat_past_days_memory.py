"""[P1-CHAT-PAST-DAYS · 2026-07-27] Test ancla del P-fix.

Spec: backend/docs/chat_past_days_memory.md
"""
import os
import re
import sys
from datetime import date, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chat_history_context import resolve_day_dates, find_plan_day_for_date  # noqa: E402

HOY = date(2026, 7, 27)  # lunes


def _day(name, n, meals=None, fecha=None):
    d = {"day": n, "day_name": name, "meals": meals if meals is not None else []}
    if fecha:
        d["date"] = fecha
    return d


def test_ancla_1_fecha_estampada_gana():
    """Si el día trae `date`, es autoritativa y `inferred` es False."""
    plan = {"days": [_day("Lunes", 1, fecha="2026-07-27"), _day("Martes", 2, fecha="2026-07-28")]}
    rows = resolve_day_dates(plan, HOY)
    assert [r["date"] for r in rows] == [date(2026, 7, 27), date(2026, 7, 28)]
    assert [r["inferred"] for r in rows] == [False, False]


def test_ancla_estampada_en_indice_distinto_de_cero():
    """El ancla estampada puede NO estar en days[0]: la fórmula
    `anchor_date + (i - anchor_idx)` tiene que proyectar hacia ATRÁS y hacia
    ADELANTE desde ella. Sin este guard, un off-by-anchor-index pasaría mudo.
    """
    plan = {"days": [_day("Domingo", 1), _day("Lunes", 2, fecha="2026-07-27"), _day("Martes", 3)]}
    rows = resolve_day_dates(plan, HOY)
    assert [r["date"] for r in rows] == [date(2026, 7, 26), date(2026, 7, 27), date(2026, 7, 28)]
    assert [r["inferred"] for r in rows] == [True, False, True]


def test_ancla_2_weekday_match_para_plan_shifteado():
    """Sin `date`, el ancla es el day_name que coincide con HOY.

    Regresión: `cycle_start_date + i` desplazaría el plan entero, porque tras
    un shift days[0] es HOY y NO el inicio del ciclo.
    """
    plan = {
        "days": [_day("Lunes", 1), _day("Martes", 2), _day("Miércoles", 3)],
        "cycle_start_date": "2026-07-20T10:00:00+00:00",  # una semana antes: trampa
    }
    rows = resolve_day_dates(plan, HOY)
    assert [r["date"] for r in rows] == [date(2026, 7, 27), date(2026, 7, 28), date(2026, 7, 29)]
    assert all(r["inferred"] for r in rows)


def test_ancla_3_cycle_start_cuando_ningun_day_name_es_hoy():
    """Plan que aún no shifteó y cuyo primer día ya pasó."""
    plan = {
        "days": [_day("Sábado", 1), _day("Domingo", 2)],
        "cycle_start_date": "2026-07-25T10:00:00+00:00",  # sábado
    }
    rows = resolve_day_dates(plan, HOY)
    assert [r["date"] for r in rows] == [date(2026, 7, 25), date(2026, 7, 26)]


def test_ancla_prefiere_grocery_start_date_sobre_cycle_start_date():
    """`cycle_start_date` es el ancla INMUTABLE de creación; `grocery_start_date`
    es la que el shift reescribe a HOY en cada rotación, así que es la que sigue
    a days[0]. Plan real 4d2c1111: con cycle (11-jul) el bloque salía VACÍO."""
    plan = {
        "days": [_day("Lunes", 1), _day("Martes", 2)],
        "cycle_start_date": "2026-07-11T10:00:00+00:00",
        "grocery_start_date": "2026-07-27T10:00:00+00:00",
    }
    rows = resolve_day_dates(plan, date(2026, 7, 29))  # HOY deliberadamente != ancla
    assert [r["date"] for r in rows] == [date(2026, 7, 27), date(2026, 7, 28)]
    assert all(r["inferred"] for r in rows)


def test_ancla_grocery_timestamp_se_convierte_a_la_fecha_LOCAL():
    """`grocery_start_date` se persiste en DOS shapes: date-only (marker
    P3-SHIFT-DATEONLY-LOCAL) y timestamp completo. Un plan shifteado a las 22:00
    RD se guarda como 02:00 UTC del día SIGUIENTE — leerlo crudo ancla el plan
    entero un día tarde. Ambos shapes tienen que dar la misma fecha."""
    base = [_day("Lunes", 1), _day("Martes", 2)]
    solo_fecha = resolve_day_dates({"days": base, "grocery_start_date": "2026-07-27"}, HOY)
    con_hora = resolve_day_dates(
        {"days": base, "grocery_start_date": "2026-07-28T02:00:00+00:00"}, HOY)
    assert [r["date"] for r in solo_fecha] == [date(2026, 7, 27), date(2026, 7, 28)]
    assert [r["date"] for r in con_hora] == [r["date"] for r in solo_fecha], (
        "el timestamp de las 22:00 RD (02:00 UTC del 28) debe anclar al 27, no al 28"
    )


def test_ventana_larga_no_lista_dias_FUTUROS_como_pasados():
    """El escaneo por weekday coge la PRIMERA coincidencia; en una ventana >7
    días `day_name` se repite y un desfase de 1 día se vuelve de 7. Plan real
    69f9e03d (26 días vivos) listaba 6 días FUTUROS como 'días que ya pasaron'.
    """
    dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    live = [_day(dias[i % 7], i + 1, [{"meal": "Cena", "name": "X%d" % i, "cals": 400}])
            for i in range(26)]
    plan = {"days": live, "grocery_start_date": "2026-07-27"}
    hoy = date(2026, 7, 27)
    out = build_past_plan_days_block(plan, hoy, days_back=7, max_chars=6000)
    rows = resolve_day_dates(plan, hoy)
    assert all(r["date"] >= hoy for r in rows), "ningún día vivo puede caer en el pasado"
    assert out == "", "sin días pasados el bloque debe estar vacío, no listar futuros"


def test_helper_de_fecha_local_renombrado_sirve_a_los_dos_consumidores():
    """`_consumed_local_date` pasó a `_to_local_date` al empezar a fechar también
    las anclas del plan (`grocery_start_date`/`cycle_start_date`), no solo
    `consumed_at`. El comportamiento es byte-idéntico: 10 chars se devuelven tal
    cual, el timestamp se corre por el offset."""
    import chat_history_context as _chc
    assert not hasattr(_chc, "_consumed_local_date"), \
        "quedó el nombre viejo: hay dos helpers que pueden divergir"
    assert _chc._to_local_date("2026-07-25") == date(2026, 7, 25)
    assert _chc._to_local_date("2026-07-26T02:30:00+00:00") == date(2026, 7, 25)
    assert _chc._to_local_date("2026-07-26T02:30:00+00:00", 0) == date(2026, 7, 26)
    assert _chc._to_local_date(None) is None
    assert _chc._to_local_date("no-soy-fecha") is None


def test_archivados_van_antes_del_primer_dia_vivo():
    plan = {
        "days": [_day("Lunes", 1)],
        "_archived_days": [_day("Sábado", 1), _day("Domingo", 2)],
    }
    rows = resolve_day_dates(plan, HOY)
    assert [r["date"] for r in rows] == [date(2026, 7, 25), date(2026, 7, 26), date(2026, 7, 27)]
    assert [r["archived"] for r in rows] == [True, True, False]


def test_find_plan_day_for_date_encuentra_en_el_archivo():
    plan = {"days": [_day("Lunes", 1)], "_archived_days": [_day("Domingo", 2)]}
    got = find_plan_day_for_date(plan, date(2026, 7, 26), HOY)
    assert got is not None and got["day"]["day_name"] == "Domingo"
    assert find_plan_day_for_date(plan, date(2026, 1, 1), HOY) is None


@pytest.mark.parametrize("basura", [None, "no soy un dict", 42, [], {"days": "tampoco"}])
def test_fail_open_ante_shapes_raras(basura):
    assert resolve_day_dates(basura, HOY) == []


from chat_history_context import (  # noqa: E402
    build_past_diary_block,
    build_past_plan_days_block,
)

_MEALS = [
    {"meal": "Desayuno", "name": "Revoltillo de Tayota con Atún", "cals": 793},
    {"meal": "Almuerzo", "name": "Pulpo al Horno con Croquetas", "cals": 549},
]


def test_bloque_plan_lista_solo_dias_pasados():
    plan = {
        "days": [_day("Lunes", 1, _MEALS)],
        "_archived_days": [_day("Domingo", 1, _MEALS)],
    }
    out = build_past_plan_days_block(plan, HOY, days_back=7, max_chars=3000)
    assert "Domingo 26 jul" in out
    assert "Revoltillo de Tayota con Atún" in out
    assert "793 kcal" in out
    # HOY no es un día pasado: no debe aparecer como tal.
    assert "Lunes 27 jul" not in out


def test_bloque_plan_no_filtra_cantidades_ni_recetas():
    """El índice es barato a propósito: los gramos y los pasos van por la tool."""
    meals = [{"meal": "Cena", "name": "Pescado", "cals": 400,
              "ingredients": ["255g de pescado"], "recipe": ["Paso secreto"]}]
    plan = {"days": [_day("Lunes", 1)], "_archived_days": [_day("Domingo", 1, meals)]}
    out = build_past_plan_days_block(plan, HOY, days_back=7, max_chars=3000)
    assert "255g" not in out
    assert "Paso secreto" not in out
    assert "consultar_dia_del_plan" in out


def test_bloque_plan_marca_fechas_inferidas_con_tilde():
    plan = {"days": [_day("Lunes", 1)], "_archived_days": [_day("Domingo", 1, _MEALS)]}
    out = build_past_plan_days_block(plan, HOY, days_back=7, max_chars=3000)
    assert "~Domingo 26 jul" in out


@pytest.mark.parametrize("cap", [520, 700, 1200, 3000])
def test_bloque_plan_nunca_supera_el_cap(cap):
    """El cap es DURO: el docstring lo promete y la nota de recorte también cuenta."""
    archived = [_day("D%d" % i, i, _MEALS) for i in range(12)]
    plan = {"days": [_day("Lunes", 1)], "_archived_days": archived}
    out = build_past_plan_days_block(plan, HOY, days_back=30, max_chars=cap)
    assert len(out) <= cap, "cap duro violado: %d > %d" % (len(out), cap)


def test_bloque_plan_declara_el_recorte_cuando_recorta():
    """No silent caps: si se cae un día, el prompt lo dice."""
    archived = [_day("D%d" % i, i, _MEALS) for i in range(12)]
    plan = {"days": [_day("Lunes", 1)], "_archived_days": archived}
    apretado = build_past_plan_days_block(plan, HOY, days_back=30, max_chars=520)
    holgado = build_past_plan_days_block(plan, HOY, days_back=30, max_chars=6000)
    assert "omitidos por espacio" in apretado
    assert "omitidos por espacio" not in holgado
    assert apretado.count("\n- ") < holgado.count("\n- ")


def test_bloque_plan_apagado_con_days_back_cero():
    plan = {"days": [_day("Lunes", 1)], "_archived_days": [_day("Domingo", 1, _MEALS)]}
    assert build_past_plan_days_block(plan, HOY, days_back=0, max_chars=3000) == ""


def test_bloque_diario_declara_los_dias_sin_registro():
    """La guarda que impide que el modelo rellene el hueco con el plan."""
    rows = [{"meal_name": "Salami con queso", "meal_type": "desayuno",
             "calories": 820, "consumed_at": "2026-07-25T17:56:00+00:00"}]
    out = build_past_diary_block(rows, HOY, days_back=3, max_chars=3000)
    assert "Salami con queso" in out
    assert "820 kcal" in out
    assert "Domingo 26 jul: SIN REGISTRO" in out
    assert "NUNCA respondas con lo que el plan mandaba" in out


def test_bloque_diario_excluye_hoy():
    """DIARIO DE HOY es su propio bloque; este NO debe duplicarlo."""
    rows = [{"meal_name": "Almuerzo de hoy", "meal_type": "almuerzo",
             "calories": 500, "consumed_at": "2026-07-27T15:00:00+00:00"}]
    out = build_past_diary_block(rows, HOY, days_back=3, max_chars=3000)
    assert "Almuerzo de hoy" not in out


def test_bloque_diario_vacio_sigue_declarando_ignorancia():
    out = build_past_diary_block([], HOY, days_back=2, max_chars=3000)
    assert "SIN REGISTRO" in out
    assert "Domingo 26 jul" in out and "Sábado 25 jul" in out


def test_diario_atribuye_la_comida_al_dia_LOCAL_no_al_utc():
    """Regresión: 10:30pm RD del 25 se guarda como 02:30 UTC del 26. Tomar
    `.date()` del UTC crudo la movía al 26 Y declaraba el 25 'SIN REGISTRO'."""
    rows = [{"meal_name": "Merienda nocturna", "meal_type": "cena",
             "calories": 300, "consumed_at": "2026-07-26T02:30:00+00:00"}]
    out = build_past_diary_block(rows, HOY, days_back=3, max_chars=3000)
    assert "Sábado 25 jul: cena: Merienda nocturna (300 kcal)" in out
    assert "Sábado 25 jul: SIN REGISTRO" not in out
    assert "Domingo 26 jul: SIN REGISTRO" in out


def test_diario_excluye_el_dia_anterior_al_floor():
    """El borde cercano (hoy excluido) ya está fijado; este fija el lejano."""
    rows = [{"meal_name": "Demasiado viejo", "meal_type": "cena",
             "calories": 100, "consumed_at": "2026-07-23T15:00:00+00:00"}]
    out = build_past_diary_block(rows, HOY, days_back=3, max_chars=3000)
    assert "Demasiado viejo" not in out
    assert "Jueves 23 jul" not in out
    assert "Viernes 24 jul: SIN REGISTRO" in out


def test_assemble_respeta_el_cap_aunque_la_nota_de_recorte_ocupe_espacio():
    """Regresión DIRECTA sobre `_assemble`: la nota "(+N ... omitidos)" ocupa
    espacio, y descontarla DESPUÉS de haber gastado el presupuesto desbordaba
    el cap que el docstring promete como duro.

    Se prueba el helper con entradas sintéticas y un BARRIDO de caps, no con
    los bloques reales: así el guard no depende del largo del copy (que cambia)
    y cubre por construcción la franja donde el recorte se activa. Un cap
    concreto tuneado al umbral de hoy caduca a la primera edición del texto.
    """
    from chat_history_context import _assemble
    for cap in range(20, 141):
        out = _assemble("H" * 10, ["L" * 19] * 6, "F" * 10, cap, "test")
        assert len(out) <= cap, "cap duro violado con cap=%d: len=%d" % (cap, len(out))


# --- Estampado de `date`: tests parser-based sobre código de producción ---

_BACKEND = os.path.join(os.path.dirname(__file__), "..")


def _src(rel):
    with open(os.path.join(_BACKEND, rel), encoding="utf-8") as fh:
        return fh.read()


def test_generacion_estampa_date():
    src = _src("graph_orchestrator.py")
    assert 'day["date"] = target_date.date().isoformat()' in src, (
        "graph_orchestrator debe estampar la fecha del día en la generación "
        "(P1-CHAT-PAST-DAYS); `target_date` ya se calcula ahí para el day_name."
    )


def test_shift_api_estampa_date_en_vivos_y_archivados():
    src = _src("routers/plans.py")
    # [P1-CHAT-PAST-DAYS · 2026-07-27] `today` en este scope es un
    # `datetime` (datetime.now(timezone.utc)), NO un `date` — verificado
    # leyendo el código fuente antes de escribir este test (el brief
    # original asumía `date`; se adaptó). `.date()` evita filtrar hora/TZ
    # al JSON persistido.
    assert "day_obj['date'] = target_date.date().isoformat()" in src, "días vivos del shift sin fecha"
    assert "P1-CHAT-PAST-DAYS" in src, "falta el marker en el bloque de archivado"
    assert "_arch_day['date']" in src, "los días archivados deben nacer fechados"


def test_shift_del_cron_usa_la_zona_del_usuario_y_no_un_reloj_utc():
    """[P1-CHAT-PAST-DAYS · 2026-07-28] `_background_shift_plan_for_user` deriva
    `today` de su parámetro `tz_offset`, pero el ÚNICO caller de producción
    (`trigger_background_rolling_refill`) no le pasaba nada → `today` salía de
    `datetime.now(timezone.utc)`. El job corre cada 4 h, así que ~1 tick al día
    cae entre las 00:00 y las 04:00 UTC = 20:00–23:59 RD y cuenta un día de más.

    Evidencia viva (Neon, 2026-07-27, LUNES en RD): los planes `4d2c1111` y
    `69f9e03d` traían ambos `days[0].day_name = "Martes"`.

    Ese `today` es el que estampa `day['date']` y `_arch_day['date']`, y el
    lector las trata como AUTORITATIVAS (`inferred=False`, sin el hedge `~`) — o
    sea, el coach afirmaría como exacta una fecha equivocada. Y los estampados
    archivados no se reescriben nunca (`if not _arch_day.get('date')`).
    """
    src = _src("cron_tasks.py")

    # 1. El default ya no es UTC.
    assert re.search(
        r"def _background_shift_plan_for_user\(\s*user_id: str,\s*tz_offset: int = 240\s*\)", src
    ), ("el default de `tz_offset` debe ser 240 (RD = UTC-4, convención del repo): "
        "con 0, un caller futuro que olvide el parámetro reintroduce UTC en silencio")

    # 2. El caller de producción resuelve la zona y la PASA.
    m = re.search(r"def trigger_background_rolling_refill\(\)[\s\S]+?(?=\ndef |\Z)", src)
    assert m, "no encontré `trigger_background_rolling_refill` en cron_tasks.py"
    cuerpo = m.group(0)
    assert "_get_user_tz_live(" in cuerpo, (
        "el caller debe resolver la zona viva del usuario — es el mismo helper que "
        "usa `routers/plans.py::_resolve_request_tz_offset`"
    )
    assert "fallback_minutes=240" in cuerpo, (
        "el fallback del resolver debe ser 240 (RD), no 0: sin perfil volveríamos a UTC"
    )
    llamada = re.search(r"_background_shift_plan_for_user\(([^)]*)\)", cuerpo)
    assert llamada, "no encontré la llamada al shift dentro del caller"
    args = [a.strip() for a in llamada.group(1).split(",") if a.strip()]
    assert len(args) == 2, (
        "la llamada debe pasar 2 argumentos (uid + offset resuelto), vi %r" % (args,)
    )
    assert args[1] not in ("0", "None"), "el offset pasado no puede ser UTC hardcodeado"


def test_shift_cron_es_gemelo_del_shift_api():
    """El bloque del cron es un duplicado literal en lógica (no en estilo de
    comillas: el renumber loop de cron_tasks.py ya usaba dobles antes de este
    cambio — `day_obj["day_name"]`/`day_obj["day"]` — vs comillas simples en
    plans.py; se respetó la convención local de cada archivo)."""
    src = _src("cron_tasks.py")
    assert "_arch_day['date']" in src
    assert 'day_obj["date"] = target_date.date().isoformat()' in src


def test_formula_de_la_fecha_archivada_es_aritmeticamente_correcta():
    """El substring no basta: `shift_amount - _j - 1` pasaría los tests parser
    igual, y desfecharía por un día TODO lo que el coach cite de días pasados.

    Extrae la fórmula real del source de producción y la evalúa con valores
    concretos. Contrato: tras el slice, days[0] es HOY, así que el ÚLTIMO día
    archivado tiene que caer exactamente en AYER.
    """
    from datetime import datetime as _dt, timedelta as _td

    hoy = _dt(2026, 7, 27, 15, 30)  # datetime, como en producción
    shift_amount = 3
    esperado = ["2026-07-24", "2026-07-25", "2026-07-26"]

    for ruta, patron in (
        ("routers/plans.py", r"_arch_day\['date'\] = \((.+?)\)\.date\(\)\.isoformat\(\)"),
        ("cron_tasks.py", r"_arch_day\['date'\] = \((.+?)\)\.date\(\)\.isoformat\(\)"),
    ):
        src = _src(ruta)
        m = re.search(patron, src)
        assert m, "no encontré la fórmula de fecha archivada en %s" % ruta
        formula = m.group(1)
        # eval() aquí es seguro y necesario: `formula` no es input externo/de
        # usuario, es una expresión aritmética extraída por regex del propio
        # código de producción versionado en este repo (routers/plans.py /
        # cron_tasks.py). El punto del test es precisamente ejecutar ESA
        # expresión (no una reescrita a mano que podría divergir del source
        # real) con valores concretos. `__builtins__` vacío + namespace
        # cerrado a 4 nombres (today/timedelta/shift_amount/_j) bloquea
        # cualquier llamada fuera de la aritmética esperada.
        got = [
            eval(formula, {"__builtins__": {}}, {
                "today": hoy, "timedelta": _td, "shift_amount": shift_amount, "_j": j,
            }).date().isoformat()
            for j in range(shift_amount)
        ]
        assert got == esperado, "%s: la fórmula %r da %r, esperaba %r" % (ruta, formula, got, esperado)
        assert got[-1] == "2026-07-26", "%s: el último archivado debe ser AYER" % ruta


# --- Task 4: cableado en agent.py (stream + no-stream) ---


def test_agent_usa_el_fetcher_multidia_y_conserva_el_de_hoy():
    """`get_consumed_meals_today` solo alimenta DIARIO DE HOY; los días
    anteriores exigen `get_consumed_meals_since` — que ya existía con ~13
    callsites en producción, y el chat era el único que no la usaba.

    También fija que DIARIO DE HOY sigue vivo: el bloque nuevo es ADITIVO, no
    un reemplazo (de él dependen la alerta de micro-adaptación, `_macro_totals_line`
    y la heurística de no re-registrar una foto ya registrada).
    """
    src = _src("agent.py")
    assert "get_consumed_meals_since" in src, "falta el fetcher multi-día"
    assert "get_consumed_meals_today" in src, "DIARIO DE HOY no puede desaparecer"
    assert "build_past_diary_block" in src
    assert "build_past_plan_days_block" in src


def test_agent_inyecta_los_bloques_despues_del_diario_de_hoy():
    """El contrato es el ORDEN EN QUE SE CONCATENA EL PROMPT, no dónde esté
    definida la función en el archivo.

    Se comprueba, en CADA uno de los dos paths (no-stream y stream), que el
    `system_prompt +=` del bloque nuevo cae DESPUÉS del `DIARIO DE HOY` de ese
    mismo path, y después del reordenamiento del prefijo estático (meterlo antes
    invalidaría el prompt-cache del proveedor para todo lo que va detrás).
    """
    src = _src("agent.py")
    hoy = [m.start() for m in re.finditer(r"DIARIO DE HOY: El usuario no ha registrado", src)]
    call = [m.start() for m in re.finditer(r"system_prompt \+= _build_past_days_context\(", src)]
    pref = [m.start() for m in re.finditer(r"if _chat_prompt_static_prefix\(\):", src)]
    assert len(hoy) == 2, "esperaba los 2 paths (no-stream y stream), vi %d" % len(hoy)
    assert len(call) == 2, "el bloque nuevo debe estar en LOS DOS paths, vi %d" % len(call)
    assert len(pref) >= 2, "esperaba el reorden del prefijo estático en ambos paths"
    for i, (h, c, p) in enumerate(zip(hoy, call, pref)):
        assert c > h, "path %d: el bloque de días pasados debe ir DESPUÉS del DIARIO DE HOY" % i
        assert c > p, "path %d: el bloque no puede ir antes del reorden del prefijo estático" % i


def test_agent_reenvia_el_tz_offset_del_cliente_al_bloque_de_diario():
    """El path stream SÍ recibe `tz_offset` del cliente; el bloque de diario lo
    necesita para convertir `consumed_at` (UTC) a la fecha LOCAL del usuario.
    Sin él, una comida de las 10:30pm RD se atribuye al día siguiente y el día
    real se declara 'SIN REGISTRO'."""
    src = _src("agent.py")
    i = src.index("def _build_past_days_context")
    cuerpo = src[i:i + 3000]
    assert "tz_offset_mins=" in cuerpo, "el helper debe reenviar el offset a build_past_diary_block"
    assert re.search(r"_build_past_days_context\([^)]*tz_offset=tz_offset", src), \
        "el path stream debe pasarle el tz_offset del cliente"


def test_tz_offset_se_resuelve_por_is_not_none_nunca_por_truthiness():
    """`tz_offset = 0` es UTC: un offset LEGÍTIMO que además es falsy.

    Un `if tz_offset:` lo trataría como ausente y caería al default 240 (UTC-4),
    desplazando un día las comidas nocturnas de todo usuario en UTC — la misma
    clase de bug que este P-fix existe para cerrar. Este guard lo prohíbe por
    construcción, que es lo que una aserción de substring no consigue.
    """
    src = _src("agent.py")
    i = src.index("def _build_past_days_context")
    cuerpo = src[i:i + 3000]
    assert "tz_offset is not None" in cuerpo, \
        "la resolución del offset debe ser `is not None`, no truthiness"
    assert re.search(r"if\s+tz_offset\s*:", cuerpo) is None, \
        "truthiness sobre tz_offset: el 0 (UTC) se perdería silenciosamente"
    assert re.search(r"tz_offset\s+or\s+240", cuerpo) is None, \
        "`tz_offset or 240` tiene el mismo bug que el truthiness"


# --- Task 5: tool `consultar_dia_del_plan` ---


def test_tool_registrada_y_documentada():
    """P0-AGENT-1: toda tool de `agent_tools` necesita fila en la tabla canónica
    o `test_p2_chat_cleanup.py` falla por paridad bidireccional."""
    src = _src("tools.py")
    assert "def consultar_dia_del_plan" in src
    m = re.search(r"^agent_tools = \[(.+?)\]", src, re.M)
    assert m and "consultar_dia_del_plan" in m.group(1), "falta en agent_tools"
    doc = _src("docs/agent_tools_user_id_table.md")
    assert "consultar_dia_del_plan" in doc


def test_kill_switch_de_la_tool_existe_y_la_retira_de_agent_tools(monkeypatch):
    """[P1-CHAT-PAST-DAYS · 2026-07-28] `MEALFIT_CHAT_PLAN_DAY_TOOL_ENABLED`
    estaba DOCUMENTADO en §4 de la spec pero no implementado: la tool se anexaba
    a `agent_tools` incondicionalmente. Un operador que siguiera la spec durante
    un incidente habría seteado una env var muerta.

    Asimetría deliberada con `MEALFIT_CHAT_PLAN_TOOLS_ENABLED` (default False,
    el knob AÑADE): como este default es True, la tool vive DENTRO de la lista
    literal `agent_tools` —la paridad bidireccional de `test_p2_chat_cleanup.py`
    parsea esa lista contra el doc canónico— y el knob la QUITA.
    """
    src = _src("tools.py")
    assert "MEALFIT_CHAT_PLAN_DAY_TOOL_ENABLED" in src, "el knob de §4 no existe en el código"
    assert "agent_tools = _apply_chat_tool_knobs(agent_tools)" in src, (
        "el gate está definido pero no aplicado a `agent_tools`"
    )

    import tools as _t
    monkeypatch.delenv("MEALFIT_CHAT_PLAN_DAY_TOOL_ENABLED", raising=False)
    assert _t._chat_plan_day_tool_enabled() is True, "el default documentado es True"
    con = {getattr(x, "name", "") for x in _t._apply_chat_tool_knobs(_t.agent_tools)}
    assert "consultar_dia_del_plan" in con, "con el knob en su default la tool debe estar activa"

    monkeypatch.setenv("MEALFIT_CHAT_PLAN_DAY_TOOL_ENABLED", "false")
    assert _t._chat_plan_day_tool_enabled() is False
    sin = {getattr(x, "name", "") for x in _t._apply_chat_tool_knobs(_t.agent_tools)}
    assert "consultar_dia_del_plan" not in sin, "el kill switch no retira la tool"
    assert con - sin == {"consultar_dia_del_plan"}, (
        "el knob solo puede retirar ESA tool, no tocar el resto del set: %r" % (con - sin,)
    )


def test_kill_switch_documentado_en_la_spec():
    doc = _src("docs/chat_past_days_memory.md")
    assert "MEALFIT_CHAT_PLAN_DAY_TOOL_ENABLED" in doc
    assert "_chat_plan_day_tool_enabled" in doc, (
        "la spec debe apuntar a la implementación real del kill switch, no solo nombrarlo"
    )


def _plan_fixture_fechado(meals):
    """Fixture DETERMINISTA: fechas estampadas => el ancla no depende de qué día
    se corra la suite. `_live_anchor` prioriza la fecha estampada sobre el
    weekday-match contra `rd_today()` — sin estampar, un test que corra un
    día distinto al lunes 2026-07-27 (commit day) caduca en silencio."""
    return {
        "days": [_day("Lunes", 1, fecha="2026-07-27")],
        "_archived_days": [_day("Domingo", 1, meals, fecha="2026-07-26")],
    }


def _llamar_tool(plan, fecha):
    import tools as _t
    orig = _t.get_latest_usable_meal_plan
    try:
        _t.get_latest_usable_meal_plan = lambda uid: plan
        return _t.consultar_dia_del_plan.func(user_id="u1", fecha=fecha)
    finally:
        _t.get_latest_usable_meal_plan = orig


def test_tool_devuelve_cantidades_y_receta():
    """Es la razón de existir de la tool: el índice del prompt NO las trae."""
    meals = [{"meal": "Cena", "name": "Pescado Guisado", "cals": 603,
              "ingredients": ["255g de pescado", "1 taza de yuca (150g)"],
              "recipe": ["Sofríe el ajo", "Añade el pescado"]}]
    out = _llamar_tool(_plan_fixture_fechado(meals), "2026-07-26")
    assert "255g de pescado" in out
    assert "Sofríe el ajo" in out
    assert "Pescado Guisado" in out


def test_tool_no_confunde_prescrito_con_consumido():
    meals = [{"meal": "Cena", "name": "X", "cals": 1, "ingredients": [], "recipe": []}]
    out = _llamar_tool(_plan_fixture_fechado(meals), "2026-07-26")
    assert "no es prueba" in out.lower() or "no significa" in out.lower()


@pytest.mark.parametrize("basura", ["ayer", "26/07/2026", "", "2026-13-45", None])
def test_tool_con_fecha_malformada_explica_el_formato_y_no_revienta(basura):
    out = _llamar_tool(_plan_fixture_fechado([]), basura)
    assert "YYYY-MM-DD" in out, "debe decirle al modelo el formato que espera"


def test_tool_sin_plan_activo_lo_dice():
    out = _llamar_tool(None, "2026-07-26")
    assert "plan" in out.lower()
    assert "255g" not in out


def test_tool_con_dia_ausente_ordena_ser_honesto():
    """El mensaje de 'no lo tengo' es load-bearing: sin él el modelo inventa."""
    out = _llamar_tool(_plan_fixture_fechado([]), "2020-01-01")
    assert "verdad" in out.lower() or "no lo tienes" in out.lower()


def test_tool_avisa_cuando_la_fecha_es_inferida_no_estampada():
    """Si la fecha se reconstruyó en vez de venir estampada, la tool tiene que
    decirlo — o el coach afirmaría como exacta una fecha que el sistema no
    garantiza."""
    # [P3-CONSULTAR-DIA-USER-TODAY · 2026-08-22] La costura se movió CON la conducta. Esto
    # parcheaba `chat_history_context.rd_today`, que era de donde la tool sacaba «hoy» — la fecha
    # de RD para todo el mundo, teniendo `user_id` delante. Ahora la deriva del huso del usuario
    # (`tools._local_date_str_for_user`), así que ése es el punto a fijar. La propiedad que este
    # caso defiende —avisar cuando la fecha se INFIRIÓ en vez de venir estampada— no cambia.
    import tools as _tools
    meals = [{"meal": "Cena", "name": "Pescado", "cals": 400,
              "ingredients": ["255g de pescado"], "recipe": ["Sofríe"]}]
    plan = {"days": [_day("Lunes", 1)], "_archived_days": [_day("Domingo", 1, meals)]}
    orig = _tools._local_date_str_for_user
    try:
        _tools._local_date_str_for_user = lambda uid=None: "2026-07-27"  # lunes fijo, no el reloj
        out = _llamar_tool(plan, "2026-07-26")
    finally:
        _tools._local_date_str_for_user = orig
    assert "estimada" in out.lower(), "falta el aviso de fecha inferida"


# --- Fósil "Opción A/B/C" y zona horaria de build_temporal_context ---


def test_fosil_opcion_abc_eliminado_de_los_prompts_base():
    """El fósil de las '3 opciones rotativas' destruía la identidad de día y se
    contradecía con `agent.py` DENTRO DEL MISMO system message."""
    src = _src("prompts/chat_agent.py")
    assert "Opción A" not in src, "queda el fósil en algún prompt base"
    assert "3 opciones distintas" not in src


def test_prohibicion_explicita_se_conserva_en_agent():
    """Lo que se elimina es la REGLA que enseñaba el vocabulario; la
    PROHIBICIÓN que corrige al modelo se queda (la ancla test_p1_chat_today_context)."""
    assert "Nunca digas 'Opción A/B/C'" in _src("agent.py")


def test_temporal_context_respeta_la_fecha_local_del_cliente():
    """Regresión: usaba `datetime.now()` del servidor mientras otro bloque del
    MISMO prompt usaba UTC-4 → 'ayer' podía significar dos días distintos."""
    from prompts.chat_agent import build_temporal_context
    out = build_temporal_context(local_date="2026-07-26")
    assert "26" in out and "Julio" in out
    assert "Domingo" in out
    assert build_temporal_context()  # sin args sigue funcionando


def test_temporal_context_ignora_una_fecha_basura():
    from prompts.chat_agent import build_temporal_context
    assert build_temporal_context(local_date="no-soy-fecha")


def test_temporal_context_trata_el_offset_cero_como_utc_no_como_utc4():
    """`tz_offset = 0` es UTC: legítimo y a la vez falsy. Un truthiness lo
    convertiría en UTC-4 y correría la fecha un día para usuarios en UTC."""
    from prompts.chat_agent import build_temporal_context
    src = _src("prompts/chat_agent.py")
    i = src.index("def build_temporal_context")
    cuerpo = src[i:i + 2000]
    assert "tz_offset is not None" in cuerpo, "resolver por `is not None`, nunca por truthiness"
    assert re.search(r"tz_offset\s+or\s+", cuerpo) is None, "`tz_offset or X` tiene el mismo bug"
    # Con local_date explícito la fecha manda, sea cual sea el offset.
    assert "26" in build_temporal_context(local_date="2026-07-26", tz_offset=0)


@pytest.mark.parametrize("absurdo", [99999999, -99999999, 100000, -5000])
def test_temporal_context_ignora_un_offset_fuera_del_rango_de_husos(absurdo):
    """`getTimezoneOffset()` vive en [-840, 840]. Un valor fuera de ahí no es un
    huso: es un bug del cliente. Sin clamp, `int(99999999)` producía una línea
    afirmando 'Jueves, 9 de Junio de 1836' DENTRO del system prompt.

    [P1-CHAT-PAST-DAYS · 2026-07-28] La aserción se hacía contra los literales
    "2026"/"2025" y por tanto CADUCABA el 2027-01-01 (comprobado con el reloj
    congelado en 2027: daba False). Se ancla al año real del reloj.
    """
    from prompts.chat_agent import build_temporal_context
    out = build_temporal_context(tz_offset=absurdo)
    from datetime import datetime as _dt, timezone as _tz
    anio = str(_dt.now(_tz.utc).year)
    assert anio in out or str(int(anio) - 1) in out, "el año se fue de rango: %r" % out


# El clamp real, tal como está escrito en los dos archivos:
#   prompts/chat_agent.py : `offset_min = _cand if -840 <= _cand <= 840 else 240`
#   agent.py              : `return cand if -840 <= cand <= 840 else default_mins`
# Se ancla la COMPARACIÓN ENCADENADA, no el literal suelto `840`.
_CLAMP_RE = re.compile(r"-840\s*<=\s*[A-Za-z_][A-Za-z0-9_]*\s*<=\s*840")


def test_ambos_sitios_clampan_el_offset_al_rango_de_husos():
    """El mismo `tz_offset` del cliente viaja por dos caminos; los dos clampan.

    [P1-CHAT-PAST-DAYS · 2026-07-28] La versión anterior aseveraba `"840" in src`
    y era VACUA: los DOS archivos escriben `840` dentro del comentario
    explicativo que va encima del clamp, así que borrar la expresión y dejar el
    comentario mantenía el test en verde. Ahora se ancla la expresión.
    """
    for ruta in ("prompts/chat_agent.py", "agent.py"):
        src = _src(ruta)
        assert _CLAMP_RE.search(src), (
            "%s no clampa el tz_offset al rango de husos: falta una comparación "
            "encadenada `-840 <= x <= 840`" % ruta
        )


def test_el_guard_del_clamp_muerde_si_se_borra_la_expresion():
    """Prueba de que el guard de arriba NO es vacuo: se borra la EXPRESIÓN del
    clamp en una copia en memoria del source (dejando intacto el comentario que
    menciona 840, que es lo que hacía pasar la aserción vieja) y el pin tiene
    que fallar."""
    for ruta in ("prompts/chat_agent.py", "agent.py"):
        src = _src(ruta)
        mutilado = _CLAMP_RE.sub("True", src)
        assert "840" in mutilado, (
            "%s: sin el literal 840 en el comentario, este experimento no probaría "
            "que la aserción vieja era vacua" % ruta
        )
        assert not _CLAMP_RE.search(mutilado), (
            "%s: el pin del clamp sigue matcheando tras borrar la expresión — "
            "el guard no muerde" % ruta
        )


# --- Task 7: marker + doc canónica (cierre del P-fix) ---


def test_marker_bumpeado():
    """Supersession-proof: este P-fix, o cualquiera POSTERIOR (fecha ≥).

    `_LAST_KNOWN_PFIX` es last-writer-wins por diseño — su trabajo es detectar
    deploy-lag, no recordar quién fue el último. Exigir el slug propio convierte
    un valor que DEBE cambiar en uno congelado, y deja el test rojo en cuanto
    otro P-fix aterriza en la misma rama (pasó en esta misma sesión con
    `P1-PESCADO-CATCHALL`). Patrón tomado de `test_p2_audit_v5_batch.py`.
    """
    src = _src("app.py")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
    assert m, "no encontré _LAST_KNOWN_PFIX"
    marker = m.group(1)
    if "P1-CHAT-PAST-DAYS" in marker:
        return
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    assert fecha and fecha.group(1) >= "2026-07-28", (
        "marker %r es ANTERIOR a este P-fix: sin bump un operador no puede "
        "confirmar que este fix está vivo en producción." % marker
    )


def test_doc_canonica_existe_y_esta_enlazada():
    doc = _src("docs/chat_past_days_memory.md")
    assert "P1-CHAT-PAST-DAYS" in doc
    assert "MEALFIT_CHAT_HISTORY_DAYS" in doc
    assert "chat_past_days_memory.md" in _src("CLAUDE.md")


def test_doc_no_promete_un_cap_combinado_que_el_codigo_no_aplica():
    """[P1-CHAT-PAST-DAYS · 2026-07-28] El doc llamaba a
    `MEALFIT_CHAT_HISTORY_MAX_CHARS` "cap duro COMBINADO de los dos bloques",
    pero `_assemble` se invoca una vez POR BLOQUE con el mismo valor (su propio
    docstring ya decía "por bloque"). El techo real es 2× el documentado."""
    doc = _src("docs/chat_past_days_memory.md")
    fila = [ln for ln in doc.splitlines() if "MEALFIT_CHAT_HISTORY_MAX_CHARS" in ln and ln.startswith("|")]
    assert fila, "no encontré la fila del knob en la tabla §4"
    assert "combinado" not in fila[0].lower(), (
        "la tabla §4 sigue prometiendo un cap combinado: %r" % fila[0]
    )
    assert "por bloque" in fila[0].lower()
    src = _src("chat_history_context.py")
    assert src.count("return _assemble(") == 2, (
        "si deja de haber exactamente 2 llamadas a `_assemble`, el multiplicador "
        "del techo (2×) que documenta §5 deja de ser cierto"
    )


def test_marker_del_doc_no_va_por_detras_del_codigo():
    """[P1-CHAT-PAST-DAYS · 2026-07-28] §6 del doc mandaba bumpear a
    `P1-CHAT-PAST-DAYS · 2026-07-27` mientras `app.py` ya iba por el 28. El
    marker NUNCA se mueve hacia atrás —es lo que un operador compara contra
    `/health/version` para confirmar el deploy—, así que el que se corrige es el
    DOC.

    Se ancla la instrucción de §6, no la igualdad con `app.py`: el marker vivo
    pertenece al ÚLTIMO P-fix mergeado, que mañana puede ser otro (de hecho
    `P1-PESCADO-CATCHALL` se lo llevó a media sesión). Lo que este guard protege
    es que el doc no dicte una fecha ANTERIOR a la que el código ya tuvo.
    """
    doc = _src("docs/chat_past_days_memory.md")
    i = doc.index("## 6.")
    seis = doc[i:]
    assert "P1-CHAT-PAST-DAYS · 2026-07-28" in seis, "§6 no dicta el marker correcto"
    assert "P1-CHAT-PAST-DAYS · 2026-07-27" not in seis, (
        "§6 sigue mandando bumpear a una fecha que el código ya superó"
    )


# --- Closing fix A: el kill switch retiraba la tool de `agent_tools` pero la
# copia del prompt (dos builders + el footer del bloque) era incondicional ---


def test_plan_day_tool_shim_sigue_el_patron_del_sibling_con_default_true():
    """`_plan_day_tool_enabled()` debe copiar `_plan_tools_enabled()` (import
    inline + `except Exception` fail-safe), pero con default True: el knob real
    (`MEALFIT_CHAT_PLAN_DAY_TOOL_ENABLED`) por defecto es True, así que un import
    fallido NO puede silenciar copy que sí funciona (al revés que el sibling, cuyo
    default seguro es False porque el suyo también lo es)."""
    src = _src("prompts/chat_agent.py")
    i = src.index("def _plan_day_tool_enabled")
    cuerpo = src[i:i + 400]
    assert "from tools import _chat_plan_day_tool_enabled" in cuerpo, (
        "el shim debe importar el getter real dentro de la función, como el sibling"
    )
    assert re.search(r"except Exception:\s*\n\s*return True", cuerpo), (
        "el default seguro de este shim debe ser True, no False"
    )


def test_bullet_y_footer_nombran_la_tool_con_el_knob_en_default(monkeypatch):
    """Con el knob ausente (default True) los DOS builders del prompt y el
    footer del bloque de días pasados deben seguir nombrando la tool."""
    monkeypatch.delenv("MEALFIT_CHAT_PLAN_DAY_TOOL_ENABLED", raising=False)
    from prompts.chat_agent import build_tools_instructions, build_tools_instructions_stream
    for builder in (build_tools_instructions, build_tools_instructions_stream):
        out = builder("11111111-1111-1111-1111-111111111111")
        assert "consultar_dia_del_plan" in out

    plan = {"days": [_day("Lunes", 1)], "_archived_days": [_day("Domingo", 1, _MEALS)]}
    bloque = build_past_plan_days_block(plan, HOY, days_back=7, max_chars=3000)
    assert "consultar_dia_del_plan" in bloque


def test_bullet_y_footer_dejan_de_nombrar_la_tool_con_el_knob_off(monkeypatch):
    """Cierre del bug: con el knob OFF la tool sale de `agent_tools`
    (`test_kill_switch_de_la_tool_existe_y_la_retira_de_agent_tools`), pero la
    copia del prompt era incondicional — el modelo seguía instruido, DOS veces
    por turno, a llamar una tool que ya no tiene. Ninguno de los 3 sitios debe
    nombrarla, y el bloque debe seguir emitiendo sus líneas de días pasados y
    el hedge `~` de fechas inferidas (eso es sobre fechas, no sobre la tool)."""
    monkeypatch.setenv("MEALFIT_CHAT_PLAN_DAY_TOOL_ENABLED", "false")
    from prompts.chat_agent import build_tools_instructions, build_tools_instructions_stream
    for builder in (build_tools_instructions, build_tools_instructions_stream):
        out = builder("11111111-1111-1111-1111-111111111111")
        assert "consultar_dia_del_plan" not in out, "la copia sigue nombrando una tool que ya no existe"
        assert "DÍAS QUE YA PASARON" in out, (
            "el bullet debe seguir diciendo que el índice de días pasados existe, "
            "solo que el detalle no está disponible"
        )

    plan = {"days": [_day("Lunes", 1)], "_archived_days": [_day("Domingo", 1, _MEALS)]}
    bloque = build_past_plan_days_block(plan, HOY, days_back=7, max_chars=3000)
    assert "consultar_dia_del_plan" not in bloque
    assert "Revoltillo de Tayota con Atún" in bloque, "las líneas del día no dependen del knob"
    assert "793 kcal" in bloque
    assert "~Domingo 26 jul" in bloque, "el hedge de fecha inferida es sobre fechas, no sobre la tool"
