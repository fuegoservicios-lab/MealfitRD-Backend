# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1161 · 2026-11-22] Juego 2.0: puntos del día, racha que multiplica, escudos y rangos con divisiones.

Funciones PURAS (sin BD): `progreso_semanas.armar_resumen` las aplica sobre los días ya leídos. Espejo en la app:
`frontend/src/utils/juegoPuntos.js` (paridad con `tests/fixtures/juego_puntos_casos.json`).

- Puntos del día (base): registrar 10 · día completo (≥3 comidas) +5 · calorías en la banda +15 · proteína ≥90 % +10 ·
  agua en meta +5 (solo con el contador de agua encendido).
- Racha = días seguidos REGISTRANDO (no «en meta»); multiplica los puntos de ese día. Hoy sin registrar no la rompe.
- Escudos: cada 7 días de racha, 1 (máx. 2). Un día pasado sin registrar con racha viva gasta uno y la racha sigue.
- Estrellas de la semana por puntos; rangos por puntos totales con 3 divisiones. Nunca se resta ni se baja.
- [P1-PLAN-LOTE-1163] Divisiones como en los videojuegos: 4 por rango y se cuentan HACIA ABAJO (Plata IV es la
  entrada, Plata I la cima). Internamente `division` sigue ascendiendo (1 = la entrada); `romano` es lo que se ve.
  Semana perfecta (3★) = +30 pts de bono. Encima de Diamante, MAESTRO: 18.900 pts + 8 semanas perfectas + una racha
  de 100 días (las tres). Muy difícil a propósito: con un día excelente y racha larga (~60 pts) es casi un año.

Spec: docs/superpowers/specs/2026-11-22-juego-puntos-rachas-design.md (repo raíz).
"""
from __future__ import annotations

import math
from typing import Any, Optional

from informe_medico import estado_del_dia

PUNTOS = {"registro": 10, "completo": 5, "banda": 15, "proteina": 10, "agua": 5}
PUNTOS_PROTEINA_PARCIAL = 5
COMIDAS_DIA_COMPLETO = 3
COMIDAS_AYUNO_COMPLETO = 2
CALORIAS_AYUNO_FRACCION = 0.7
PROTEINA_PARCIAL_FRACCION = 0.7
PROTEINA_FRACCION = 0.9
#: (desde_dias, multiplicador, clave) de mayor a menor.
TRAMOS_RACHA = ((30, 1.6, "leyenda"), (14, 1.45, "imparable"), (7, 1.3, "en_llamas"), (3, 1.15, "encendida"),
                (0, 1.0, None))
ESCUDO_CADA = 7
ESCUDOS_MAX = 2
ESTRELLAS_SEMANA = (85, 170, 265)  # calibrado con replay 22-nov (8 cuentas reales)
RANGOS_PUNTOS = (("bronce", 0), ("plata", 540), ("oro", 1620), ("platino", 4050), ("diamante", 8100))
DIVISIONES = 4
ROMANOS = ("I", "II", "III", "IV")
DIAMANTE_PASO = 2700
HITOS_RACHA_LARGOS = (3, 7, 14, 30, 50, 75, 100, 150, 200, 365)
BONO_SEMANA_PERFECTA = 30
MAESTRO = "maestro"
MAESTRO_PUNTOS = RANGOS_PUNTOS[-1][1] + DIAMANTE_PASO * DIVISIONES   # 18.900: Diamante I completo
MAESTRO_SEMANAS_PERFECTAS = 8
MAESTRO_RACHA = 100


def redondear(x: float) -> int:
    """Mitades hacia arriba (como `Math.round` en la app; `round` de Python es bancario)."""
    return int(math.floor(float(x) + 0.5))


def _num(x: Any) -> float:
    try:
        return float(x or 0)
    except (TypeError, ValueError):
        return 0.0


def logros_del_dia(*, calories: Any, protein: Any, meals_count: Any, meta_kcal: Any, meta_proteina: Any,
                   objetivo: Any, vasos: Any = None, meta_vasos: Any = None) -> dict:
    kcal, comidas = _num(calories), int(_num(meals_count))
    mk = _num(meta_kcal)
    registro = comidas > 0 or kcal > 0
    mp = _num(meta_proteina)
    mv = _num(meta_vasos)
    completo = registro and (
        comidas >= COMIDAS_DIA_COMPLETO
        or (comidas >= COMIDAS_AYUNO_COMPLETO and mk > 0 and kcal >= mk * CALORIAS_AYUNO_FRACCION)
    )
    proteina = registro and mp > 0 and _num(protein) >= mp * PROTEINA_FRACCION
    proteina_parcial = registro and (not proteina) and mp > 0 and _num(protein) >= mp * PROTEINA_PARCIAL_FRACCION
    return {
        "registro": registro,
        "completo": completo,
        "banda": registro and mk > 0 and estado_del_dia(kcal, mk, objetivo) == "en_meta",
        "proteina": proteina,
        "proteina_parcial": proteina_parcial,
        "agua": (registro and _num(vasos) >= mv) if mv > 0 else None,
    }


def puntos_base(logros: dict) -> int:
    if not (logros or {}).get("registro"):
        return 0
    total = sum(PUNTOS[k] for k in PUNTOS if logros.get(k) is True)
    if logros.get("proteina") is not True and logros.get("proteina_parcial") is True:
        total += PUNTOS_PROTEINA_PARCIAL
    return total


def tramo_de_racha(racha: int) -> dict:
    racha = max(0, int(racha or 0))
    for i, (desde, mult, clave) in enumerate(TRAMOS_RACHA):
        if racha >= desde:
            sig = TRAMOS_RACHA[i - 1] if i > 0 else None
            return {"mult": mult, "clave": clave, "siguiente_en": sig[0] if sig else None,
                    "siguiente_mult": sig[1] if sig else None}
    return {"mult": 1.0, "clave": None, "siguiente_en": TRAMOS_RACHA[-2][0], "siguiente_mult": TRAMOS_RACHA[-2][1]}


def recorrer(dias: list[dict], hoy_iso: str) -> dict:
    """`dias` en orden (del día 1 a hoy): {"iso", "registrado", "base"}. Racha, escudos y puntos de cada día."""
    racha = mejor = escudos = total = 0
    usados: list[str] = []
    ganados: list[str] = []
    por_dia: dict[str, dict] = {}
    for d in dias or []:
        iso = str(d.get("iso"))
        pts, mult = 0, 1.0
        if d.get("registrado"):
            racha += 1
            mult = tramo_de_racha(racha)["mult"]
            pts = redondear(int(d.get("base") or 0) * mult)
            if racha % ESCUDO_CADA == 0 and escudos < ESCUDOS_MAX:
                escudos += 1
                ganados.append(iso)
        elif iso == hoy_iso:
            pass                                   # en curso: ni rompe ni gasta
        elif racha > 0 and escudos > 0:
            escudos -= 1
            usados.append(iso)                     # la racha sigue intacta
        else:
            racha = 0
        mejor = max(mejor, racha)
        total += pts
        por_dia[iso] = {"puntos": pts, "mult": mult, "racha": racha}
    return {"por_dia": por_dia, "racha": racha, "mejor_racha": mejor, "escudos": escudos,
            "escudos_usados": usados, "escudos_ganados": ganados, "total": total}


def estrellas_por_puntos(pts: int) -> int:
    return sum(1 for u in ESTRELLAS_SEMANA if int(pts or 0) >= u)


def faltan_para_otra_estrella_pts(pts: int) -> int:
    for u in ESTRELLAS_SEMANA:
        if int(pts or 0) < u:
            return u - int(pts or 0)
    return 0


def escalera() -> list[tuple[str, int, int]]:
    """Todos los peldaños (clave, división, desde) en orden."""
    out: list[tuple[str, int, int]] = []
    for i, (clave, desde) in enumerate(RANGOS_PUNTOS):
        if i + 1 < len(RANGOS_PUNTOS):
            hasta = RANGOS_PUNTOS[i + 1][1]
            for dv in range(DIVISIONES):
                out.append((clave, dv + 1, desde + redondear((hasta - desde) * dv / DIVISIONES)))
        else:
            for dv in range(DIVISIONES):
                out.append((clave, dv + 1, desde + DIAMANTE_PASO * dv))
    return out


def division_romana(division: Any) -> Optional[str]:
    """La división que VE el jugador: 1 (la entrada) → «IV» … 4 (la cima del rango) → «I»."""
    try:
        d = int(division)
    except (TypeError, ValueError):
        return None
    return ROMANOS[DIVISIONES - d] if 1 <= d <= DIVISIONES else None


def _siguiente(sig) -> Optional[dict]:
    return {"clave": sig[0], "division": sig[1], "romano": division_romana(sig[1])} if sig else None


def rango_por_puntos(total: int) -> dict:
    total = max(0, int(total or 0))
    pel = escalera()
    i = 0
    for k, (_, _, desde) in enumerate(pel):
        if total >= desde:
            i = k
    clave, division, desde = pel[i]
    sig = pel[i + 1] if i + 1 < len(pel) else None
    hasta = sig[2] if sig else None
    return {
        "clave": clave, "division": division, "romano": division_romana(division), "desde": desde, "hasta": hasta,
        "siguiente": _siguiente(sig),
        "puntos_para_la_siguiente": (hasta - total) if sig else None,
        "avance": round(min(1.0, (total - desde) / (hasta - desde)), 4) if sig else 1,
    }


def nivel_indice(clave: str, division: int) -> int:
    """Posición del peldaño (0 = Bronce IV) — para comparar subidas. Maestro va después del último."""
    if clave == MAESTRO:
        return indice_maestro()
    for k, (c, d, _) in enumerate(escalera()):
        if c == clave and d == division:
            return k
    return 0


def rango_con_piso(total: int, piso_indice: int = 0) -> dict:
    """Rango por puntos respetando un piso mínimo alcanzado (nunca retrocede de división)."""
    total = max(0, int(total or 0))
    r = rango_por_puntos(total)
    if nivel_indice(r["clave"], r["division"]) >= int(piso_indice or 0):
        return r
    pel = escalera()
    i = min(max(0, int(piso_indice or 0)), len(pel) - 1)
    clave, division, desde = pel[i]
    sig = pel[i + 1] if i + 1 < len(pel) else None
    hasta = sig[2] if sig else None
    return {
        "clave": clave, "division": division, "romano": division_romana(division), "desde": desde, "hasta": hasta,
        "siguiente": _siguiente(sig),
        "puntos_para_la_siguiente": max(0, hasta - total) if sig else None,
        "avance": 0 if sig else 1,
    }


def indice_maestro() -> int:
    return len(escalera())


def maestro_progreso(total: int, semanas_perfectas: int, mejor_racha: int) -> dict:
    """Los tres requisitos de Maestro con lo que lleva el jugador (se cumplen los tres o no hay Maestro)."""
    req = [
        {"clave": "puntos", "actual": max(0, int(total or 0)), "meta": MAESTRO_PUNTOS},
        {"clave": "semanas_perfectas", "actual": max(0, int(semanas_perfectas or 0)), "meta": MAESTRO_SEMANAS_PERFECTAS},
        {"clave": "racha", "actual": max(0, int(mejor_racha or 0)), "meta": MAESTRO_RACHA},
    ]
    for r in req:
        r["ok"] = r["actual"] >= r["meta"]
    cumplidos = sum(1 for r in req if r["ok"])
    return {"conseguido": cumplidos == len(req), "cumplidos": cumplidos, "requisitos": req}


def rango_del_jugador(total: int, semanas_perfectas: int = 0, mejor_racha: int = 0, piso_indice: int = 0) -> dict:
    """El rango completo: divisiones por puntos (con piso) y, en la cima de Diamante, el camino a Maestro."""
    total = max(0, int(total or 0))
    piso = int(piso_indice or 0)
    if piso >= indice_maestro() or maestro_progreso(total, semanas_perfectas, mejor_racha)["conseguido"]:
        return {"clave": MAESTRO, "division": None, "romano": None, "desde": MAESTRO_PUNTOS, "hasta": None,
                "siguiente": None, "puntos_para_la_siguiente": None, "avance": 1}
    r = rango_con_piso(total, piso)
    if r["siguiente"] is None:
        r = dict(r, hasta=MAESTRO_PUNTOS, siguiente={"clave": MAESTRO, "division": None, "romano": None},
                 puntos_para_la_siguiente=max(0, MAESTRO_PUNTOS - total),
                 avance=round(min(1.0, max(0, total - r["desde"]) / (MAESTRO_PUNTOS - r["desde"])), 4))
    return r


def proximo_hito_racha(racha: int) -> Optional[int]:
    r = max(0, int(racha or 0))
    for h in HITOS_RACHA_LARGOS:
        if h > r:
            return h
    return None


def dias_para_escudo(racha: int, escudos: int) -> Optional[int]:
    if int(escudos or 0) >= ESCUDOS_MAX:
        return None
    return ESCUDO_CADA - (max(0, int(racha or 0)) % ESCUDO_CADA)


def hoy_puntos(logros: dict, racha_con_hoy: int) -> dict:
    base = puntos_base(logros)
    mult = tramo_de_racha(racha_con_hoy)["mult"]   # el que tendría hoy (para «registra y ganas ×1,5»)
    return {"puntos": redondear(base * mult), "base": base, "mult": mult, "logros": logros}


def meta_proteina_de(meta: Optional[dict]) -> Optional[float]:
    m = (meta or {}).get("macros") or (meta or {}).get("targetMacros") or {}
    if isinstance(m, dict):
        v = _num(m.get("protein") or m.get("proteina"))
        return v or None
    return None
