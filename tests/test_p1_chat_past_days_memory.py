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
    orig = _t.get_latest_meal_plan
    try:
        _t.get_latest_meal_plan = lambda uid: plan
        return _t.consultar_dia_del_plan.func(user_id="u1", fecha=fecha)
    finally:
        _t.get_latest_meal_plan = orig


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
    import chat_history_context as _chc
    meals = [{"meal": "Cena", "name": "Pescado", "cals": 400,
              "ingredients": ["255g de pescado"], "recipe": ["Sofríe"]}]
    plan = {"days": [_day("Lunes", 1)], "_archived_days": [_day("Domingo", 1, meals)]}
    orig = _chc.rd_today
    try:
        _chc.rd_today = lambda: date(2026, 7, 27)  # lunes fijo, no el reloj real
        out = _llamar_tool(plan, "2026-07-26")
    finally:
        _chc.rd_today = orig
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
