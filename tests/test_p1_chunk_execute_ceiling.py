"""[P1-CHUNK-EXECUTE-CEILING · 2026-08-16] El bloque no puede ejecutarse DESPUÉS
de que haya empezado el tramo que cubre.

El incidente, medido sobre los 3 planes vivos con cola el 2026-08-16:

    plan       reanclado alguna vez   execute_after vs ancla+offset
    e2094da6   sí (08-16 04:00)       +1 día  ← usuario sin menú el 20, el 24, el 28
    f380821a   sí (08-15 07:36)       +1 día
    76a6836d   NO (upd == creado)     exacto en sus 3 chunks

La diferencia entre estar bien y estar mal era UNA columna: `updated_at`.
`_rebase_pending_chunk_offsets_sql` mueve `execute_after` por el mismo delta que
el offset — un movimiento relativo, elegido para preservar la hora local y los
adelantos de `safety_margin`. El precio, que nadie había cobrado: preserva
también el error previo, para siempre, porque el par (offset, execute_after) no
se compara nunca contra el ancla.

Estos tests fijan las dos mitades: la aritmética del techo, y que el SQL lo
aplique como techo (nunca como suelo, nunca por encima del clamp de NOW()).
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent

# El snapshot real del plan e2094da6: el usuario creó el plan a las 20:13 locales
# (RD, UTC-4), así que en UTC ya era el día siguiente.
_SNAPSHOT_REAL = {"form_data": {"_plan_start_date": "2026-08-16T20:13:20", "tzOffset": 240}}


# ─────────────────────────────────────────────────────────────────────────────
# 1. La aritmética del techo
# ─────────────────────────────────────────────────────────────────────────────

def test_el_techo_es_la_medianoche_local_del_primer_dia_que_cubre():
    """Los números exactos del incidente: offset 4 sobre ancla 08-16 ⇒ 08-20."""
    from constants import chunk_execute_after_ceiling

    techo = chunk_execute_after_ceiling(_SNAPSHOT_REAL, 4)
    assert techo == datetime(2026, 8, 20, 4, 30, tzinfo=timezone.utc), (
        "El bloque que cubre 08-20..08-23 debe poder ejecutarse el 20 a las 00:30 "
        "locales. Producción lo tenía el 21: el día 20 se quedaba sin menú."
    )


def test_la_hora_del_dia_no_depende_de_cuando_se_creo_el_plan():
    """`_plan_start_date` trae hora (20:13); el techo la descarta y usa la fecha.

    Si el techo heredara la hora de creación, un plan creado a las 23:50 tendría
    techo a las 23:50 del día que cubre — casi 24 h tarde, el mismo bug con otra
    cara.

    [RE-ANCLADO por P1-CHUNK-ANCHOR-LOCAL-DATE · 2026-08-21] Las dos anclas de la versión previa
    eran 23:50Z y 00:04Z del MISMO día UTC — y con tz=240 pertenecen a días LOCALES distintos
    (19:50 del 16 y 20:04 del 15). El test exigía que dieran el mismo techo, que es exactamente
    la suposición «fecha UTC = fecha del usuario» que este P-fix corrige. Se re-ancla dentro de
    un mismo día local, que es donde la invariante —descartar la HORA— realmente vive."""
    from constants import chunk_execute_after_ceiling

    # Ambas dentro del 16-ago LOCAL para tz=240 (04:00Z del 16 .. 03:59Z del 17).
    tarde = {"form_data": {"_plan_start_date": "2026-08-17T03:50:00+00:00", "tzOffset": 240}}
    temprano = {"form_data": {"_plan_start_date": "2026-08-16T04:04:00+00:00", "tzOffset": 240}}
    assert chunk_execute_after_ceiling(tarde, 4) == chunk_execute_after_ceiling(temprano, 4)


def test_el_dia_cero_es_el_dia_LOCAL_del_usuario_no_el_utc():
    """[P1-CHUNK-ANCHOR-LOCAL-DATE · 2026-08-21] El caso REAL encontrado en la cola viva, y la
    prueba de que el defecto no era sólo de España: chunk con ancla `2026-08-11T03:05Z` y
    tz=240 — el usuario creó su plan a las **23:05 del día 10** hora local. Su día 11 es el 21,
    no el 22.

    Producción lo tenía en `2026-08-22T04:30Z`: un día TARDE, que es literalmente la queja que
    P1-CHUNK-EXECUTE-CEILING documentó («el bloque corría el día DESPUÉS de empezar el tramo que
    cubre»). El techo lo heredaba porque compartía la aritmética equivocada.
    """
    from constants import chunk_execute_after_ceiling

    snap = {"form_data": {"_plan_start_date": "2026-08-11T03:05:07.872334+00:00", "tzOffset": 240}}
    assert chunk_execute_after_ceiling(snap, 11) == datetime(
        2026, 8, 21, 4, 30, tzinfo=timezone.utc
    ), "el día 11 del usuario es el 21 local, no el 22"


def test_el_techo_escala_un_dia_por_offset():
    from constants import chunk_execute_after_ceiling

    base = chunk_execute_after_ceiling(_SNAPSHOT_REAL, 0)
    for offset in (4, 8, 12, 16, 20, 24):
        assert chunk_execute_after_ceiling(_SNAPSHOT_REAL, offset) == base + timedelta(days=offset)


def test_la_tz_desplaza_el_techo():
    """tz=0 ⇒ 00:30 UTC; tz=240 (UTC-4) ⇒ 04:30 UTC. Mismo instante local."""
    from constants import chunk_execute_after_ceiling

    utc = {"form_data": {"_plan_start_date": "2026-08-16T10:00:00", "tzOffset": 0}}
    rd = {"form_data": {"_plan_start_date": "2026-08-16T10:00:00", "tzOffset": 240}}
    assert chunk_execute_after_ceiling(utc, 0) == datetime(2026, 8, 16, 0, 30, tzinfo=timezone.utc)
    assert chunk_execute_after_ceiling(rd, 0) == datetime(2026, 8, 16, 4, 30, tzinfo=timezone.utc)


# [P1-CHUNK-ANCHOR-LOCAL-DATE · 2026-08-21] Los casos que faltaban, y por cuya ausencia este
# fichero dio verde durante todo el defecto: hasta hoy sólo instanciaba `tzOffset ∈ {0, 240}`,
# es decir el hemisferio en el que el bug NO EXISTE. España es el único país beta al este de UTC,
# y ahí la medianoche local cae en el día UTC ANTERIOR — el techo salía 23,5 h adelantado.
# Un guard que sólo prueba el lado que funciona no es un guard, es una coincidencia.
@pytest.mark.parametrize("tz_min,etiqueta", [(-120, "ES verano"), (-60, "ES invierno")])
def test_el_techo_es_correcto_al_este_de_utc(tz_min, etiqueta):
    from constants import chunk_execute_after_ceiling

    # El ancla es el INSTANTE UTC de la medianoche local del 21-ago para ese offset.
    ancla = datetime(2026, 8, 21, tzinfo=timezone.utc) + timedelta(minutes=tz_min)
    snap = {"form_data": {"_plan_start_date": ancla.isoformat(), "tzOffset": tz_min}}
    esperado = datetime(2026, 8, 25, tzinfo=timezone.utc) + timedelta(minutes=tz_min + 30)
    assert chunk_execute_after_ceiling(snap, 4) == esperado, (
        f"{etiqueta}: el techo se adelanta al día que cubre"
    )


@pytest.mark.parametrize(
    "snapshot",
    [
        None,
        {},
        {"form_data": None},
        {"form_data": {}},
        {"form_data": {"_plan_start_date": None}},
        {"form_data": {"_plan_start_date": ""}},
        {"form_data": {"_plan_start_date": "no-es-una-fecha"}},
        "ni siquiera es un dict",
    ],
)
def test_sin_ancla_no_opina(snapshot):
    """None = "no opino", y el caller conserva su comportamiento previo.

    Tratarlo como techo cero mandaría todos los chunks a NOW() a la vez — la
    colisión de dos generaciones sobre el mismo `plan_data` que
    `dias_hasta_su_turno` ya existe para evitar.
    """
    from constants import chunk_execute_after_ceiling

    assert chunk_execute_after_ceiling(snapshot, 4) is None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Funcional: el SQL aplicado como techo, con el suelo de NOW() por encima
# ─────────────────────────────────────────────────────────────────────────────

def _exigir_forma_del_update(sql: str) -> None:
    """El SQL que el fake dice modelar: techo dentro del suelo, en ese orden."""
    plano = re.sub(r"\s+", " ", sql)
    pos_greatest = plano.find("GREATEST(")
    pos_least = plano.find("LEAST(")
    if pos_greatest == -1 or pos_least == -1 or pos_greatest > pos_least:
        raise AssertionError(
            "El UPDATE ya no es `GREATEST(LEAST(movido, techo), suelo)`; el fake "
            f"estaría modelando algo que producción no hace. SQL: {plano[:200]}"
        )


class _FakeCursor:
    """Modela el UPDATE real, incluido el `GREATEST(LEAST(...), suelo)`.

    `execute_after` vive como datetime absoluto (no como delta de días) porque el
    techo es absoluto: un fake que trabajara en días relativos no podría
    distinguir "acotado por el ancla" de "movido por el delta", que es
    exactamente la diferencia que este fix introduce.
    """

    def __init__(self, filas, ahora):
        self.rows = {f["id"]: dict(f) for f in filas}
        self.ahora = ahora
        self._result = []

    @staticmethod
    def _statuses_in(sql):
        return set(re.findall(r"'(pending|stale|pending_user_action|processing)'", sql))

    def execute(self, sql, params=None):
        params = params or ()
        if "SUM(days_count)" in sql:
            self._result = [{
                "en_vuelo": sum(
                    r["days_count"] for r in self.rows.values() if r["status"] == "processing"
                )
            }]
        elif sql.strip().upper().startswith("SELECT"):
            estados = self._statuses_in(sql)
            # Devuelve SOLO lo que el SQL pide. Un fake generoso con las columnas
            # oculta la mutación más silenciosa de todas: quitar
            # `pipeline_snapshot` del SELECT deja el techo en None para siempre y
            # el fix muere sin que nada se ponga rojo.
            pide_snapshot = "pipeline_snapshot" in sql
            self._result = [
                {
                    "id": r["id"],
                    "days_offset": r["days_offset"],
                    "days_count": r["days_count"],
                    **({"pipeline_snapshot": r.get("pipeline_snapshot")} if pide_snapshot else {}),
                }
                for r in sorted(self.rows.values(), key=lambda x: x["week_number"])
                if r["status"] in estados
            ]
        elif sql.strip().upper().startswith("UPDATE"):
            # El fake replica la semántica del UPDATE a mano, así que por sí solo
            # NO puede notar que el SQL cambió: seguiría aplicando el techo
            # aunque producción lo hubiera quitado. Antes de modelar nada,
            # comprueba que el SQL sigue siendo el que dice modelar.
            _exigir_forma_del_update(sql)
            nuevo_offset, delta, techo, dias_turno, chunk_id = params
            fila = self.rows.get(chunk_id)
            if fila and fila["status"] in self._statuses_in(sql):
                movido = fila["execute_after"] - timedelta(days=delta)
                if techo is not None:
                    movido = min(movido, techo)
                fila["execute_after"] = max(movido, self.ahora + timedelta(days=dias_turno))
                fila["days_offset"] = nuevo_offset
        else:  # pragma: no cover
            raise AssertionError(f"SQL inesperado en el fake: {sql[:120]}")

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return self._result


def _cola_del_incidente():
    """La cola real de e2094da6, con el +1 día que producción tenía guardado."""
    return [
        {
            "id": f"w{semana}", "status": "pending", "week_number": semana,
            "days_offset": offset, "days_count": 4,
            # El día DESPUÉS del que debía: el bug.
            "execute_after": datetime(2026, 8, 16, 4, 30, tzinfo=timezone.utc)
                             + timedelta(days=offset + 1),
            "pipeline_snapshot": _SNAPSHOT_REAL,
        }
        for semana, offset in ((3, 4), (4, 8), (5, 12), (6, 16))
    ]


def test_el_techo_recorta_el_bloque_que_llegaba_tarde():
    from cron_tasks import _rebase_pending_chunk_offsets_sql

    ahora = datetime(2026, 8, 16, 4, 0, tzinfo=timezone.utc)
    # live=3 fuerza movimiento (offsets 4/8/12/16 → 3/7/11/15) para que el UPDATE
    # corra: `plan_chunk_offset_moves` omite los que ya están en su sitio.
    cur = _FakeCursor(_cola_del_incidente(), ahora)
    _rebase_pending_chunk_offsets_sql(cur, "e2094da6", live_days_count=3)

    for rid, fila in cur.rows.items():
        from constants import chunk_execute_after_ceiling
        limite = chunk_execute_after_ceiling(_SNAPSHOT_REAL, fila["days_offset"])
        assert fila["execute_after"] <= limite, (
            f"{rid} sigue ejecutándose tras empezar su tramo "
            f"({fila['execute_after']} > {limite}): el usuario se queda sin menú ese día."
        )


def test_el_techo_no_retrasa_a_quien_ya_llegaba_pronto():
    """`safety_margin` adelanta chunks a propósito. El techo acota por ARRIBA:
    un chunk ya adelantado debe conservar su adelanto intacto."""
    from cron_tasks import _rebase_pending_chunk_offsets_sql

    ahora = datetime(2026, 8, 16, 4, 0, tzinfo=timezone.utc)
    adelantado = datetime(2026, 8, 17, 4, 30, tzinfo=timezone.utc)  # muy por debajo del techo
    filas = [{
        "id": "w3", "status": "pending", "week_number": 3, "days_offset": 4, "days_count": 4,
        "execute_after": adelantado, "pipeline_snapshot": _SNAPSHOT_REAL,
    }]
    cur = _FakeCursor(filas, ahora)
    _rebase_pending_chunk_offsets_sql(cur, "e2094da6", live_days_count=4)

    # offset 4 → 4: no se mueve, así que el UPDATE ni corre. El adelanto sobrevive.
    assert cur.rows["w3"]["execute_after"] == adelantado


def test_el_suelo_de_now_manda_sobre_el_techo():
    """Un chunk vencido no puede programarse al pasado por culpa del techo.

    Por eso `LEAST` va DENTRO de `GREATEST` y no al revés: invertirlo dejaría
    `execute_after` en un instante ya pasado y el scheduler lo tomaría todo a la
    vez — dos generaciones escribiendo el mismo `plan_data`.
    """
    from cron_tasks import _rebase_pending_chunk_offsets_sql

    # Ancla muy vieja ⇒ el techo cae en el pasado.
    vieja = {"form_data": {"_plan_start_date": "2026-07-01T10:00:00", "tzOffset": 240}}
    ahora = datetime(2026, 8, 16, 4, 0, tzinfo=timezone.utc)
    filas = [{
        "id": "w3", "status": "pending", "week_number": 3, "days_offset": 9, "days_count": 4,
        "execute_after": datetime(2026, 7, 10, 4, 30, tzinfo=timezone.utc),
        "pipeline_snapshot": vieja,
    }]
    cur = _FakeCursor(filas, ahora)
    _rebase_pending_chunk_offsets_sql(cur, "plan-viejo", live_days_count=0)

    assert cur.rows["w3"]["execute_after"] >= ahora, (
        "El techo empujó un chunk al pasado: el scheduler lo recogería de "
        "inmediato junto a todos sus hermanos vencidos."
    )


def test_sin_snapshot_el_comportamiento_es_el_de_antes():
    """Cobertura parcial no puede cambiar la conducta de los chunks sin ancla."""
    from cron_tasks import _rebase_pending_chunk_offsets_sql

    ahora = datetime(2026, 8, 16, 4, 0, tzinfo=timezone.utc)
    original = datetime(2026, 8, 21, 4, 30, tzinfo=timezone.utc)
    filas = [{
        "id": "w3", "status": "pending", "week_number": 3, "days_offset": 4, "days_count": 4,
        "execute_after": original, "pipeline_snapshot": None,
    }]
    cur = _FakeCursor(filas, ahora)
    _rebase_pending_chunk_offsets_sql(cur, "plan-sin-snap", live_days_count=3)

    # delta = 4 - 3 = 1 día; sin techo, el movimiento relativo manda.
    assert cur.rows["w3"]["execute_after"] == original - timedelta(days=1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Parser-based: el anidamiento del SQL
# ─────────────────────────────────────────────────────────────────────────────

def _cuerpo_del_ejecutor() -> str:
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    ini = src.index("def _rebase_pending_chunk_offsets_sql(")
    return src[ini: src.index("\ndef ", ini + 10)]


def _sql_del_update(cuerpo: str) -> str:
    """El UPDATE sin comentarios ni espacio sobrante.

    Los comentarios se retiran antes de mirar: en este repo son largos por
    diseño y ya han hecho fallar guards por citar el código que vigilan.
    """
    sin_comentarios = "\n".join(
        l for l in cuerpo.splitlines() if not l.strip().startswith("#")
    )
    ini = sin_comentarios.index('"UPDATE plan_chunk_queue "')
    fin = sin_comentarios.index("cursor.execute", ini) if "cursor.execute" in sin_comentarios[ini:] else len(sin_comentarios)
    return re.sub(r"\s+", " ", sin_comentarios[ini:fin])


def test_el_least_esta_anidado_dentro_del_greatest():
    """La PROPIEDAD, no la grafía: el suelo tiene que envolver al techo.

    Se comprueba el orden de apertura (GREATEST antes que LEAST), que es lo que
    distingue "acota tarde pero nunca al pasado" de su inverso peligroso.
    """
    sql = _sql_del_update(_cuerpo_del_ejecutor())
    pos_greatest = sql.find("GREATEST(")
    pos_least = sql.find("LEAST(")
    assert pos_greatest != -1 and pos_least != -1, (
        "Desapareció el clamp de execute_after en el rebase."
    )
    assert pos_greatest < pos_least, (
        "LEAST envuelve a GREATEST: el techo del ancla puede programar un chunk "
        "vencido al pasado, y todos los hermanos vencidos salen a la vez."
    )
    assert sql.find("NOW()", pos_least) > pos_least, (
        "El suelo de NOW() ya no es argumento del GREATEST exterior."
    )


def test_el_techo_llega_desde_el_ssot_de_constants():
    """Que la fórmula no se reescriba aquí: sería una segunda tabla de aritmética.

    Es la lección de `P1-DIET-CANON-SSOT` — tres tablas a mano drifearon y una
    olvidó dos casos.
    """
    cuerpo = _cuerpo_del_ejecutor()
    assert "chunk_execute_after_ceiling" in cuerpo
    assert "from constants import" in cuerpo


def test_el_snapshot_se_selecciona_para_poder_anclar():
    """Si el SELECT deja de traer `pipeline_snapshot`, el techo sale None SIEMPRE
    y el fix muere en silencio: todo seguiría verde, y el bug volvería entero."""
    cuerpo = _cuerpo_del_ejecutor()
    select = cuerpo[cuerpo.index("SELECT id, days_offset"):]
    assert "pipeline_snapshot" in select[:400], (
        "El SELECT de la cadena ya no trae el snapshot: "
        "`chunk_execute_after_ceiling` no puede anclar y el techo es inerte."
    )


# ── [P2-CHUNK-TZ-GUARD-BLIND · 2026-08-21] Husos al ESTE de Greenwich ──────────────────────────
#
# Este fichero instanciaba únicamente offsets 0 y 240 — UTC y República Dominicana, los dos al
# OESTE. Con esa muestra el guard no puede ver el modo de fallo europeo: para un offset NEGATIVO la
# hora local va POR DELANTE de UTC, así que un ancla de última hora de la tarde en UTC ya pertenece
# al día SIGUIENTE en Madrid. Redondear por la fecha UTC adelanta el bloque un día en vez de
# atrasarlo — el error espejo del que se midió en la cola dominicana (día 11 programado para el 22).
#
# Lo destapó `test_tz_offset_chunk_timing::test_timezone_alignment_asia`, que llevaba escrito en su
# propio comentario que su ancla era «midnight local time Manila (day before)» y aun así esperaba
# la fecha UTC. Una muestra que sólo cubre un lado del meridiano no puede acusar al otro.

_TZ_BETA = (
    ("ES invierno", -60),
    ("ES verano", -120),
    ("MX", 360),
    ("CO", 300),
    ("US este", 300),
    ("Manila", -480),
)


@pytest.mark.parametrize("etiqueta,tz", _TZ_BETA)
def test_el_techo_es_la_medianoche_LOCAL_tambien_al_este_de_greenwich(etiqueta, tz):
    """La medianoche local del primer día cubierto, sea cual sea el signo del offset."""
    from constants import chunk_execute_after_ceiling
    from datetime import datetime, timedelta, timezone
    ancla = "2026-08-16T22:30:00+00:00"
    snap = {"form_data": {"_plan_start_date": ancla, "tzOffset": tz}}
    techo = chunk_execute_after_ceiling(snap, 3)
    assert techo is not None, f"{etiqueta}: sin techo (el fix nace inerte para este huso)"
    # Reconstrucción independiente: fecha LOCAL del ancla + 3 días, a medianoche local, en UTC.
    inst = datetime.fromisoformat(ancla)
    fecha_local = (inst - timedelta(minutes=tz)).date() + timedelta(days=3)
    # El «+30 min» es deliberado y está documentado en el SSOT: replica la fórmula del encolado
    # (medianoche local + 30m). Mi primera versión de este test lo omitía y falló contra código
    # CORRECTO en los seis husos a la vez — un desfase constante en todos los signos no es un bug
    # de zona horaria, es un modelo incompleto. Vale la pena dejarlo escrito: la señal de «me
    # equivoco yo» es que el error no depende de la variable que estoy probando.
    esperado = datetime.combine(fecha_local, datetime.min.time(),
                                tzinfo=timezone.utc) + timedelta(minutes=tz + 30)
    assert techo == esperado, (
        f"{etiqueta} (tz={tz}): techo {techo} != medianoche local {esperado}. Con offset negativo "
        f"la hora local va POR DELANTE de UTC y redondear por la fecha UTC adelanta el bloque"
    )


def test_un_ancla_de_noche_en_utc_ya_es_el_dia_siguiente_en_madrid():
    """El caso concreto que la muestra vieja no podía instanciar. 22:30Z del 16 son las 00:30 del
    **17** en Madrid en verano: el día 0 del plan es el 17, no el 16."""
    from constants import chunk_execute_after_ceiling
    from datetime import datetime, timedelta, timezone
    snap = {"form_data": {"_plan_start_date": "2026-08-16T22:30:00+00:00", "tzOffset": -120}}
    techo = chunk_execute_after_ceiling(snap, 0)
    esperado = datetime(2026, 8, 17, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=-120 + 30)
    assert techo == esperado, (
        f"techo {techo}: el ancla se leyó en fecha UTC (16) en vez de en la local de Madrid (17)"
    )
