# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1161 · 2026-11-22] Juego 2.0: puntos del día, racha que multiplica, escudos y rangos con divisiones.

Funciones PURAS (sin BD): `progreso_semanas.armar_resumen` las aplica sobre los días ya leídos. Espejo en la app:
`frontend/src/utils/juegoPuntos.js` (paridad con `tests/fixtures/juego_puntos_casos.json`).

- Puntos del día (base): registrar 10 · día completo (≥3 comidas) +5 · calorías en la banda +15 · proteína ≥90 % +10 ·
  agua en meta +5 (solo con el contador de agua encendido).
- Racha = días seguidos REGISTRANDO (no «en meta»); multiplica los puntos de ese día. Hoy sin registrar no la rompe.
- Escudos: cada 7 días de racha, 1 (máx. 2). Un día pasado sin registrar con racha viva gasta uno y la racha sigue.
- Estrellas de la semana por puntos; rangos por puntos totales con 3 divisiones. Nunca se resta ni se baja.

Spec: docs/superpowers/specs/2026-11-22-juego-puntos-rachas-design.md (repo raíz).
"""
from __future__ import annotations

import math
from typing import Any, Optional

from informe_medico import estado_del_dia

PUNTOS = {"registro": 10, "completo": 5, "banda": 15, "proteina": 10, "agua": 5}
COMIDAS_DIA_COMPLETO = 3
PROTEINA_FRACCION = 0.9
#: (desde_dias, multiplicador, clave) de mayor a menor.
TRAMOS_RACHA = ((30, 2.0, "leyenda"), (14, 1.75, "imparable"), (7, 1.5, "en_llamas"), (3, 1.2, "encendida"),
                (0, 1.0, None))
ESCUDO_CADA = 7
ESCUDOS_MAX = 2
ESTRELLAS_SEMANA = (120, 200, 280)
RANGOS_PUNTOS = (("bronce", 0), ("plata", 600), ("oro", 1800), ("platino", 4500), ("diamante", 9000))
DIVISIONES = 3
DIAMANTE_PASO = 3000


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
    registro = comidas > 0 or kcal > 0
    mp = _num(meta_proteina)
    mv = _num(meta_vasos)
    return {
        "registro": registro,
        "completo": registro and comidas >= COMIDAS_DIA_COMPLETO,
        "banda": registro and _num(meta_kcal) > 0 and estado_del_dia(kcal, _num(meta_kcal), objetivo) == "en_meta",
        "proteina": registro and mp > 0 and _num(protein) >= mp * PROTEINA_FRACCION,
        "agua": (registro and _num(vasos) >= mv) if mv > 0 else None,
    }


def puntos_base(logros: dict) -> int:
    if not (logros or {}).get("registro"):
        return 0
    return sum(PUNTOS[k] for k in PUNTOS if logros.get(k) is True)


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
        "clave": clave, "division": division, "desde": desde, "hasta": hasta,
        "siguiente": {"clave": sig[0], "division": sig[1]} if sig else None,
        "puntos_para_la_siguiente": (hasta - total) if sig else None,
        "avance": round(min(1.0, (total - desde) / (hasta - desde)), 4) if sig else 1,
    }


def nivel_indice(clave: str, division: int) -> int:
    """Posición del peldaño (0 = Bronce I) — para comparar subidas."""
    for k, (c, d, _) in enumerate(escalera()):
        if c == clave and d == division:
            return k
    return 0


def hoy_puntos(logros: dict, racha_con_hoy: int) -> dict:
    base = puntos_base(logros)
    mult = tramo_de_racha(racha_con_hoy)["mult"] if base else 1.0
    return {"puntos": redondear(base * mult), "base": base, "mult": mult, "logros": logros}


def meta_proteina_de(meta: Optional[dict]) -> Optional[float]:
    m = (meta or {}).get("macros") or (meta or {}).get("targetMacros") or {}
    if isinstance(m, dict):
        v = _num(m.get("protein") or m.get("proteina"))
        return v or None
    return None
