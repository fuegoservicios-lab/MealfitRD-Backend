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
from datetime import datetime, timedelta, timezone

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


# [P1-PLAN-LOTE-620 · 2026-09-27] Los códigos se pintan en español. Un código seguro sin traducción sale tal cual (una
# función nueva del gasto se ve igual, solo que sin nombre bonito); el inseguro ya llegó aquí como «otro».
_ETIQUETA_ESTADO_PLAN = {
    "complete": "Completos", "complete_partial": "Completos con huecos", "partial": "Parciales",
    "partial_no_shopping": "Parciales sin lista", "active": "Activos", "generating": "Generándose",
    "generating_next": "Generando el siguiente bloque", "in_progress": "En curso", "paused_by_user": "Pausados",
    "failed": "Fallidos", "abandoned": "Abandonados", "degraded_pending_engagement": "Degradados (esperan uso)",
    "expired_pending_pantry": "Caducados (esperan la Nevera)", "sin estado": "Sin estado", "otro": "Otro estado",
}
_ETIQUETA_COLA = {
    "pending": "Pendientes", "processing": "Procesándose", "completed": "Completados", "failed": "Fallidos",
    "cancelled": "Cancelados", "pending_user_action": "Esperan al usuario", "otro": "Otro",
}
_ETIQUETA_FUNCION = {
    "day_generator": "Generación de días", "vision_scan": "Escáner de fotos", "planner": "Planificador",
    "reviewer": "Revisor clínico", "chat_call_model": "Coach (chat)", "plan_display_i18n": "Traducción del plan",
    "self_critique": "Autocrítica", "self_critique_correction": "Corrección tras la autocrítica",
    "culinary_judge": "Juez culinario", "surgical_marker": "Corrección puntual", "compressor": "Resumen de la memoria",
    "fact_extractor_extract_facts": "Extractor de hechos",
    "fact_extractor_contradiction_merge": "Contradicciones de hechos",
    "fact_extractor_router": "Clasificador de hechos", "diary_plate_estimate": "Estimación de platos (diario)",
    "diary_freetext_estimate": "Estimación de texto libre (diario)", "otro": "Otro",
}
_GASTO_TOP = 8
_NOTA_LIBRE_PROHIBIDA = re.compile(r"\S*@\S*|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


def _texto_de_nota(v) -> str:
    """Las notas del banco las escribe quien corre el banco, no un usuario; aun así, ni correos ni ids."""
    t = " ".join(_NOTA_LIBRE_PROHIBIDA.sub(" ", str(v or "")).split())
    if len(t) <= 60:
        return t
    corte = t[:60].rsplit(" ", 1)[0] if " " in t[:60] else t[:60]      # en una palabra entera, no a media
    return corte.rstrip(" ,;:") + "…"


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


def _kpis(bid: str, titulo: str, filas: list, nota: str | None = None) -> dict:
    """Cada fila: (etiqueta, valor) o (etiqueta, valor, opciones) con `destacado` (cifra grande) o `nivel` (subfila)."""
    salida = []
    for f in filas:
        fila = {"etiqueta": f[0], "valor": f[1]}
        if len(f) > 2:
            fila.update(f[2])
        salida.append(fila)
    bloque = {"id": bid, "titulo": titulo, "tipo": "kpis", "filas": salida}
    if nota:
        bloque["nota"] = nota
    return bloque


_DESTACADO = {"destacado": True}
_SUBFILA = {"nivel": 1}


def bloque_uso(dias: int) -> dict:
    cuentas = _uno("SELECT COUNT(*) AS n FROM public.user_profiles")
    activas = _uno(
        "SELECT COUNT(DISTINCT u) AS n FROM ("
        f" SELECT user_id::text AS u FROM public.consumed_meals WHERE consumed_at >= {_VENTANA}"
        " UNION SELECT user_id::text FROM public.agent_messages WHERE role = 'user' AND user_id IS NOT NULL"
        f" AND created_at >= {_VENTANA}) a", (dias, dias))
    comidas = _uno(f"SELECT COUNT(*) AS n FROM public.consumed_meals WHERE consumed_at >= {_VENTANA}", (dias,))
    return _kpis("uso", "Uso", [(f"Activas en {dias} días", _entero(activas.get("n")), _DESTACADO),
                                ("Comidas registradas", _entero(comidas.get("n")), _DESTACADO),
                                ("Cuentas", _entero(cuentas.get("n")))])


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
    # [P1-PLAN-LOTE-620] la señal nació el 27-sep: sin esta nota, «1 foto en 30 días» se lee como que nadie escanea
    primero = _uno("SELECT MIN(created_at) AS desde FROM public.pipeline_metrics WHERE node = 'vision_scan_resultado'")
    desde = primero.get("desde")
    nota = None
    if desde is not None and desde > datetime.now(timezone.utc) - timedelta(days=dias):
        nota = f"Se registra desde el {desde:%d-%m-%Y}."

    def frac(k):
        return int(s.get(k) or 0) / n_s if n_s else None

    # [P1-PLAN-LOTE-620] Una fila destacada se pinta aparte, arriba: si «Corregidos» lo fuera, sus cinco razones
    # quedarían colgando de la fila de encima. Las subfilas siguen siempre a su total, que es una fila normal.
    return _kpis("escaner", "Escáner", [
        ("Fotos analizadas", _entero(n_v), _DESTACADO),
        ("Platos registrados con el escáner", _entero(n_s), _DESTACADO),
        ("Análisis fallidos", _pct(int(v.get("fallidos") or 0) / n_v if n_v else None)),
        ("No era comida", _pct(int(v.get("no_comida") or 0) / n_v if n_v else None)),
        ("Sin totales (compra o etiqueta)", _pct(int(v.get("sin_totales") or 0) / n_v if n_v else None)),
        ("Tiempo de análisis (mediana / p90)", f"{_seg(v.get('p50'))} / {_seg(v.get('p90'))}"),
        ("Corregidos por el usuario", _pct(frac("corregidos"))),
        ("Cambió un ingrediente", _pct(frac("cambiar")), _SUBFILA),
        ("«Descríbelo»", _pct(frac("describelo")), _SUBFILA),
        ("Editó cantidades", _pct(frac("cantidades")), _SUBFILA),
        ("Cambió la respuesta a una duda", _pct(frac("dudas")), _SUBFILA),
        ("Tecleó las macros", _pct(frac("macros")), _SUBFILA),
        ("Desvío mediano de calorías (IA → registrado)", _pct(s.get("desvio"))),
    ], nota) | {"ancho": True}   # [P1-PLAN-LOTE-620] el bloque largo, en su propia fila: los cortos llenan la primera


def bloque_coach(dias: int) -> dict:
    c = _uno("SELECT COUNT(*) FILTER (WHERE role = 'user') AS preguntas, "
             "COUNT(*) FILTER (WHERE role = 'model') AS respuestas, "
             "COUNT(*) FILTER (WHERE feedback = 'up') AS up, COUNT(*) FILTER (WHERE feedback = 'down') AS down "
             f"FROM public.agent_messages WHERE created_at >= {_VENTANA}", (dias,))
    up, down = int(c.get("up") or 0), int(c.get("down") or 0)
    return _kpis("coach", "Coach", [("Mensajes del usuario", _entero(c.get("preguntas")), _DESTACADO),
                                    ("Tasa de 👎", _pct(down / (up + down) if up + down else None), _DESTACADO),
                                    ("Respuestas del coach", _entero(c.get("respuestas"))),
                                    ("Valoraciones 👍", _entero(up)), ("Valoraciones 👎", _entero(down))])


def bloque_planes(dias: int) -> dict:
    planes = _todos("SELECT COALESCE(plan_data->>'generation_status', 'sin estado') AS estado, COUNT(*) AS n "
                    f"FROM public.meal_plans WHERE created_at >= {_VENTANA} GROUP BY 1 ORDER BY 2 DESC", (dias,))
    cola = _todos("SELECT status, COUNT(*) AS n FROM public.plan_chunk_queue "
                  f"WHERE created_at >= {_VENTANA} GROUP BY 1 ORDER BY 2 DESC", (dias,))
    alertas = _uno("SELECT COUNT(*) AS n FROM public.system_alerts WHERE resolved_at IS NULL")
    filas = [("Planes creados", _entero(sum(int(r.get("n") or 0) for r in planes)), _DESTACADO),
             ("Alertas del sistema abiertas", _entero(alertas.get("n")), _DESTACADO),
             ("Estado de los planes", "")]
    filas += [(_ETIQUETA_ESTADO_PLAN.get(k, k), _entero(g["n"]), _SUBFILA)
              for k, g in _agrupar(planes, "estado", _estado_plan, "n")]
    filas.append(("Bloques en cola", _entero(sum(int(r.get("n") or 0) for r in cola))))
    filas += [(_ETIQUETA_COLA.get(k, k), _entero(g["n"]), _SUBFILA)
              for k, g in _agrupar(cola, "status", _etiqueta_de_codigo, "n")]
    return _kpis("planes", "Planes", filas)


def bloque_gasto(dias: int) -> dict:
    por_funcion = _todos("SELECT COALESCE(node, 'sin atribuir') AS funcion, COUNT(*) AS llamadas, "
                         "COALESCE(SUM(cost_usd_micros), 0) AS micros FROM public.llm_usage_events "
                         f"WHERE created_at >= {_VENTANA} GROUP BY 1 ORDER BY 3 DESC LIMIT 200", (dias,))
    total = _uno("SELECT COALESCE(SUM(cost_usd_micros), 0) AS micros, COUNT(*) AS n FROM public.llm_usage_events "
                 f"WHERE created_at >= {_VENTANA}", (dias,))
    grupos = _agrupar(por_funcion, "funcion", _etiqueta_de_codigo, "micros", "llamadas")
    # [P1-PLAN-LOTE-620] las 8 que más gastan con nombre legible; el resto en UNA fila, y la proporción para las barras
    filas = [(_ETIQUETA_FUNCION.get(k, k), g["llamadas"], g["micros"]) for k, g in grupos[:_GASTO_TOP]]
    resto = grupos[_GASTO_TOP:]
    if resto:
        filas.append((f"Otras ({len(resto)} funciones)", sum(g["llamadas"] for _, g in resto),
                      sum(g["micros"] for _, g in resto)))
    total_micros = int(total.get("micros") or 0)
    return {"id": "gasto", "titulo": f"Gasto de IA ({dias} días): {_usd(total_micros)}", "tipo": "tabla",
            "columnas": ["Función", "Llamadas", "Coste"],
            "filas": [[nombre, _entero(llamadas), _usd(micros)] for nombre, llamadas, micros in filas],
            "barras": [round(micros / total_micros, 4) if total_micros else 0.0 for _, _, micros in filas]}


def bloque_analizador(_dias: int) -> dict:
    corridas = _todos("SELECT ran_at, model, n, ok, notes, metrics FROM public.analyzer_benchmark_runs "
                      "ORDER BY ran_at DESC LIMIT 10")
    filas = []
    for r in corridas:
        m = r.get("metrics") or {}
        # [P1-PLAN-LOTE-620] qué se midió (la nota de la corrida o, sin ella, el modelo) y si la corrida no vale
        corrida = _texto_de_nota(r.get("notes")) or _texto_de_nota(r.get("model")) or "—"
        if not m.get("valida"):
            corrida += " (no válida)"
        filas.append([r["ran_at"].strftime("%Y-%m-%d %H:%M"), corrida,
                      _pct((m.get("kcal") or {}).get("mediana")), _pct((m.get("proteina_g") or {}).get("mediana")),
                      _pct(m.get("recall_componentes"))])
    return {"id": "analizador", "titulo": "Banco del analizador", "tipo": "tabla",
            "columnas": ["Fecha (UTC)", "Corrida", "kcal", "Proteína", "Componentes"], "filas": filas,
            "nota": "kcal y Proteína: error mediano frente a lo pesado (Nutrition5k); menos es mejor. Componentes: "
                    "los que reconoce. Una corrida sola varía ±2-4 puntos: se decide con la regla pareada."}


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
