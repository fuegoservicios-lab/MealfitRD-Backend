# backend/admin_metricas.py
"""[P1-PLAN-LOTE-576 · 2026-09-27] Métricas del panel de administración, capa 1 (spec §2): SOLO agregados, ya
redactados para pintar.

El frontend es un pintor genérico (spec §6): cada bloque trae su título y sus filas con la etiqueta y el valor ya
formateados, así que una métrica nueva es solo backend. Jamás sale un id, un correo, un nombre, texto de un mensaje ni
nombre de un plato. Un bloque que falla no tumba a los demás: sale como `error`.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from db import execute_sql_query

logger = logging.getLogger(__name__)

_VENTANA = "now() - make_interval(days => %s)"

# Revisión final: `generation_status` vive dentro de plan_data, que `/restore-local` deja escribir al cliente — un correo
# o un mensaje acabaría como fila del panel. Solo se pintan los estados que escribe el backend; el resto, «otro».
_ESTADOS_PLAN = frozenset({
    "complete", "complete_partial", "partial", "partial_no_shopping", "active", "generating", "generating_next",
    "in_progress", "paused_by_user", "failed", "abandoned", "degraded_pending_engagement", "expired_pending_pantry",
    "sin estado",
})
# Las etiquetas de código (estado de la cola, función del gasto) las escribe el servidor, pero se pintan igual de
# acotadas: solo [a-z0-9_].
_ETIQUETA_DE_CODIGO = re.compile(r"^[a-z0-9_]{1,64}$")


def _estado_plan(v) -> str:
    v = str(v or "")
    return v if v in _ESTADOS_PLAN else "otro"


def _etiqueta_de_codigo(v) -> str:
    v = str(v or "")
    return v if _ETIQUETA_DE_CODIGO.match(v) else "otro"


def _agrupar(filas: list, clave: str, etiqueta, *sumas: str) -> list:
    """Suma las filas cuya etiqueta segura coincide (dos «otro» son UNA fila), de mayor a menor por la 1.ª suma."""
    grupos: dict = {}
    for r in filas:
        k = etiqueta(r.get(clave))
        g = grupos.setdefault(k, dict.fromkeys(sumas, 0))
        for campo in sumas:
            g[campo] += int(r.get(campo) or 0)
    return sorted(grupos.items(), key=lambda kv: -kv[1][sumas[0]])


def _entero(n) -> str:
    return f"{int(n or 0):,}"


def _pct(x) -> str:
    return "—" if x is None else f"{float(x) * 100:.0f} %"


def _usd(micros) -> str:
    return f"US${(micros or 0) / 1_000_000:,.2f}"


def _seg(ms) -> str:
    return "—" if ms is None else f"{float(ms) / 1000:.1f} s"


def _uno(sql: str, params: tuple = ()) -> dict:
    return execute_sql_query(sql, params, fetch_one=True) or {}


def _todos(sql: str, params: tuple = ()) -> list:
    return execute_sql_query(sql, params, fetch_all=True) or []


def _kpis(bid: str, titulo: str, filas: list) -> dict:
    return {"id": bid, "titulo": titulo, "tipo": "kpis", "filas": [{"etiqueta": e, "valor": v} for e, v in filas]}


def bloque_uso(dias: int) -> dict:
    cuentas = _uno("SELECT COUNT(*) AS n FROM public.user_profiles")
    activas = _uno(
        "SELECT COUNT(DISTINCT u) AS n FROM ("
        f" SELECT user_id::text AS u FROM public.consumed_meals WHERE consumed_at >= {_VENTANA}"
        " UNION SELECT user_id::text FROM public.agent_messages WHERE role = 'user' AND user_id IS NOT NULL"
        f" AND created_at >= {_VENTANA}) a", (dias, dias))
    comidas = _uno(f"SELECT COUNT(*) AS n FROM public.consumed_meals WHERE consumed_at >= {_VENTANA}", (dias,))
    return _kpis("uso", "Uso", [("Cuentas", _entero(cuentas.get("n"))),
                                (f"Activas en {dias} días", _entero(activas.get("n"))),
                                ("Comidas registradas", _entero(comidas.get("n")))])


def bloque_escaner(dias: int) -> dict:
    # Revisión final: «fallido» es solo `error` (no_comida y una compra sin totales son respuestas legítimas), y solo
    # las fotos del ESCÁNER (`purpose='diary'`): las del chat no son este bloque.
    v = _uno("SELECT COUNT(*) AS n, COUNT(*) FILTER (WHERE metadata->>'resultado' = 'error') AS fallidos, "
             "COUNT(*) FILTER (WHERE metadata->>'resultado' = 'no_comida') AS no_comida, "
             "COUNT(*) FILTER (WHERE metadata->>'resultado' = 'sin_totales') AS sin_totales, "
             "percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms) AS p50, "
             "percentile_cont(0.9) WITHIN GROUP (ORDER BY duration_ms) AS p90 "
             "FROM public.pipeline_metrics WHERE node = 'vision_scan_resultado' AND metadata->>'purpose' = 'diary' "
             f"AND created_at >= {_VENTANA}", (dias,))
    s = _uno("SELECT COUNT(*) AS n, "
             "COUNT(*) FILTER (WHERE (metadata->>'corregido')::boolean) AS corregidos, "
             "COUNT(*) FILTER (WHERE (metadata->>'cambiados')::int > 0) AS cambiar, "
             "COUNT(*) FILTER (WHERE (metadata->>'redescrito')::boolean) AS describelo, "
             "COUNT(*) FILTER (WHERE (metadata->>'cantidades_editadas')::int > 0) AS cantidades, "
             "COUNT(*) FILTER (WHERE (metadata->>'dudas_cambiadas')::int > 0) AS dudas, "
             "COUNT(*) FILTER (WHERE (metadata->>'macros_tecleadas')::boolean) AS macros, "
             "percentile_cont(0.5) WITHIN GROUP (ORDER BY (metadata->>'desvio_kcal')::float) AS desvio "
             f"FROM public.pipeline_metrics WHERE node = 'scan_outcome' AND created_at >= {_VENTANA}", (dias,))
    n_v, n_s = int(v.get("n") or 0), int(s.get("n") or 0)

    def frac(k):
        return int(s.get(k) or 0) / n_s if n_s else None

    return _kpis("escaner", "Escáner", [
        ("Fotos analizadas", _entero(n_v)),
        ("Análisis fallidos", _pct(int(v.get("fallidos") or 0) / n_v if n_v else None)),
        ("No era comida", _pct(int(v.get("no_comida") or 0) / n_v if n_v else None)),
        ("Sin totales (compra o etiqueta)", _pct(int(v.get("sin_totales") or 0) / n_v if n_v else None)),
        ("Tiempo de análisis (mediana / p90)", f"{_seg(v.get('p50'))} / {_seg(v.get('p90'))}"),
        ("Platos registrados con el escáner", _entero(n_s)),
        ("Corregidos por el usuario", _pct(frac("corregidos"))),
        ("· cambió un ingrediente", _pct(frac("cambiar"))),
        ("· «Descríbelo»", _pct(frac("describelo"))),
        ("· editó cantidades", _pct(frac("cantidades"))),
        ("· cambió la respuesta a una duda", _pct(frac("dudas"))),
        ("· tecleó las macros", _pct(frac("macros"))),
        ("Desvío mediano de calorías (IA → registrado)", _pct(s.get("desvio"))),
    ])


def bloque_coach(dias: int) -> dict:
    c = _uno("SELECT COUNT(*) FILTER (WHERE role = 'user') AS preguntas, "
             "COUNT(*) FILTER (WHERE role = 'model') AS respuestas, "
             "COUNT(*) FILTER (WHERE feedback = 'up') AS up, COUNT(*) FILTER (WHERE feedback = 'down') AS down "
             f"FROM public.agent_messages WHERE created_at >= {_VENTANA}", (dias,))
    up, down = int(c.get("up") or 0), int(c.get("down") or 0)
    return _kpis("coach", "Coach", [("Mensajes del usuario", _entero(c.get("preguntas"))),
                                    ("Respuestas del coach", _entero(c.get("respuestas"))),
                                    ("👍", _entero(up)), ("👎", _entero(down)),
                                    ("Tasa de 👎", _pct(down / (up + down) if up + down else None))])


def bloque_planes(dias: int) -> dict:
    planes = _todos("SELECT COALESCE(plan_data->>'generation_status', 'sin estado') AS estado, COUNT(*) AS n "
                    f"FROM public.meal_plans WHERE created_at >= {_VENTANA} GROUP BY 1 ORDER BY 2 DESC", (dias,))
    cola = _todos("SELECT status, COUNT(*) AS n FROM public.plan_chunk_queue "
                  f"WHERE created_at >= {_VENTANA} GROUP BY 1 ORDER BY 2 DESC", (dias,))
    alertas = _uno("SELECT COUNT(*) AS n FROM public.system_alerts WHERE resolved_at IS NULL")
    filas = [("Planes creados", _entero(sum(int(r.get("n") or 0) for r in planes)))]
    filas += [(f"· {k}", _entero(g["n"])) for k, g in _agrupar(planes, "estado", _estado_plan, "n")]
    filas += [(f"Bloques en cola: {k}", _entero(g["n"])) for k, g in _agrupar(cola, "status", _etiqueta_de_codigo, "n")]
    filas.append(("Alertas del sistema abiertas", _entero(alertas.get("n"))))
    return _kpis("planes", "Planes", filas)


def bloque_gasto(dias: int) -> dict:
    por_funcion = _todos("SELECT COALESCE(node, 'sin atribuir') AS funcion, COUNT(*) AS llamadas, "
                         "COALESCE(SUM(cost_usd_micros), 0) AS micros FROM public.llm_usage_events "
                         f"WHERE created_at >= {_VENTANA} GROUP BY 1 ORDER BY 3 DESC LIMIT 15", (dias,))
    total = _uno("SELECT COALESCE(SUM(cost_usd_micros), 0) AS micros, COUNT(*) AS n FROM public.llm_usage_events "
                 f"WHERE created_at >= {_VENTANA}", (dias,))
    return {"id": "gasto", "titulo": f"Gasto de IA ({dias} días): {_usd(total.get('micros'))}", "tipo": "tabla",
            "columnas": ["Función", "Llamadas", "Coste"],
            "filas": [[k, _entero(g["llamadas"]), _usd(g["micros"])]
                      for k, g in _agrupar(por_funcion, "funcion", _etiqueta_de_codigo, "micros", "llamadas")]}


def bloque_analizador(_dias: int) -> dict:
    corridas = _todos("SELECT ran_at, model, n, ok, metrics FROM public.analyzer_benchmark_runs "
                      "ORDER BY ran_at DESC LIMIT 10")
    filas = []
    for r in corridas:
        m = r.get("metrics") or {}
        filas.append([r["ran_at"].strftime("%Y-%m-%d %H:%M"), str(r["model"]), f"{r['ok']}/{r['n']}",
                      _pct((m.get("kcal") or {}).get("mediana")), _pct((m.get("proteina_g") or {}).get("mediana")),
                      _pct(m.get("recall_componentes")), "sí" if m.get("valida") else "no"])
    return {"id": "analizador", "titulo": "Banco del analizador", "tipo": "tabla",
            "columnas": ["Fecha (UTC)", "Modelo", "Platos", "Error kcal (mediana)", "Error proteína (mediana)",
                         "Componentes", "Válida"], "filas": filas}


_BLOQUES = (("uso", "Uso", bloque_uso), ("escaner", "Escáner", bloque_escaner), ("coach", "Coach", bloque_coach),
            ("planes", "Planes", bloque_planes), ("gasto", "Gasto de IA", bloque_gasto),
            ("analizador", "Banco del analizador", bloque_analizador))


def metricas(dias: int) -> dict:
    dias = max(1, min(90, int(dias)))
    bloques = []
    for bid, titulo, fn in _BLOQUES:
        try:
            bloques.append(fn(dias))
        except Exception as e:
            logger.warning(f"⚠️ [P1-PLAN-LOTE-576] bloque {bid} no disponible: {e!r}")
            bloques.append({"id": bid, "titulo": titulo, "tipo": "error", "error": "No disponible"})
    return {"dias": dias, "generado": datetime.now(timezone.utc).isoformat(), "bloques": bloques}
