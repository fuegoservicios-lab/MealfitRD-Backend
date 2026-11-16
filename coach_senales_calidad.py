"""[P1-PLAN-LOTE-1027 · A-16] Señales implícitas de calidad del coach (no dependen de que el usuario pulse nada).

Medido 15-nov: 0 valoraciones en 7 días; 186 comidas registradas desde el chat en 30 días y NINGÚN rastro de cuántas
hubo que corregir: `update_consumed_meal` no deja huella y el borrado se lleva la fila. Esto registra, en
`pipeline_metrics` (node `coach_senal_calidad`, sin migración), tres señales:

  · `coach_corrigio`      el coach corrigió una fila del diario (el usuario le dijo que estaba mal);
  · `borrado_en_diario`   el usuario borró a mano en el diario una comida que había anotado el coach (chat/foto);
  · `usuario_dice_error`  el mensaje empieza diciendo que el coach se equivocó («te equivocaste», «eso no es…»).

Sin texto del usuario ni datos de salud en la fila (A-24): solo la señal, los minutos desde el registro y la franja.
`resumen(horas)` da la tasa de «registro corregido» = (coach_corrigio + borrado_en_diario) / registros del coach.
Fail-open: cualquier fallo se traga; nunca rompe la escritura que lo dispara. Knob `MEALFIT_COACH_SENALES_CALIDAD=0`.
"""
from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

NODO = "coach_senal_calidad"
FUENTES_DEL_COACH = ("chat", "photo")

_RE_QUEJA = re.compile(
    r"^\s*(?:no+\s*[,.!]*\s*)?(?:te\s+equivocaste|te\s+confundiste|estas?\s+mal|esta\s+mal|eso\s+no\s+(?:es|fue|era)|"
    r"no\s+(?:es|era|fue)\s+(?:eso|asi)|no\s+comi\s+eso|no\s+me\s+(?:comi|tome)\s+eso|lo\s+(?:anotaste|registraste)\s+mal|"
    r"(?:lo\s+)?pusiste\s+mal|eso\s+esta\s+mal|you(?:'re|\s+are)\s+wrong|that'?s\s+(?:not\s+right|wrong))\b"
)


def activo() -> bool:
    return (os.getenv("MEALFIT_COACH_SENALES_CALIDAD", "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def _plano(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", t or "") if not unicodedata.combining(c)).lower()


def es_queja(texto: str) -> bool:
    """¿El mensaje arranca diciendo que el coach se equivocó?"""
    try:
        return bool(_RE_QUEJA.search(_plano(texto)[:200]))
    except Exception:
        return False


def _minutos_desde(creado: Any) -> Optional[int]:
    try:
        if isinstance(creado, str):
            creado = datetime.fromisoformat(creado.replace("Z", "+00:00"))
        if creado.tzinfo is None:
            creado = creado.replace(tzinfo=timezone.utc)
        return max(0, int((datetime.now(timezone.utc) - creado).total_seconds() // 60))
    except Exception:
        return None


def registrar(user_id: Optional[str], senal: str, session_id: Optional[str] = None, fila: Optional[dict] = None) -> bool:
    """Guarda una señal. `fila` (opcional) es la fila del diario afectada: solo se leen su franja y su antigüedad."""
    if not activo() or not user_id or user_id == "guest":
        return False
    try:
        meta: dict = {"senal": senal}
        if fila:
            meta["franja"] = fila.get("meal_type")
            meta["fuente"] = fila.get("source")
            m = _minutos_desde(fila.get("created_at"))
            if m is not None:
                meta["minutos_desde_registro"] = m
        from db_core import execute_sql_write
        execute_sql_write(
            "INSERT INTO pipeline_metrics (user_id, session_id, node, duration_ms, retries, tokens_estimated, "
            "confidence, metadata) VALUES (%s, %s, %s, 0, 0, 0, 0, %s::jsonb)",
            (user_id, session_id, NODO, json.dumps(meta, ensure_ascii=False)),
        )
        return True
    except Exception as e:
        logger.debug(f"[LOTE-1027 SEÑALES] no registrada {senal}: {type(e).__name__}")
        return False


def borrado_en_diario(user_id: Optional[str], fila: Optional[dict]) -> bool:
    """Tras un borrado manual en el diario: cuenta solo si la comida la había anotado el coach."""
    if not fila or (fila.get("source") or "") not in FUENTES_DEL_COACH:
        return False
    return registrar(user_id, "borrado_en_diario", fila=fila)


def resumen(horas: int = 168) -> dict:
    """Recuentos por señal y la tasa de «registro corregido» en la ventana."""
    from db_core import execute_sql_query
    ventana = f"{max(1, min(720, int(horas)))} hours"
    filas = execute_sql_query(
        "SELECT metadata->>'senal' AS senal, count(*) AS n FROM pipeline_metrics "
        "WHERE node = %s AND created_at > now() - %s::interval GROUP BY 1",
        (NODO, ventana), fetch_all=True,
    ) or []
    cuentas = {str(f["senal"]): int(f["n"]) for f in filas}
    reg = execute_sql_query(
        "SELECT count(*) AS n FROM consumed_meals WHERE source = ANY(%s) AND created_at > now() - %s::interval",
        (list(FUENTES_DEL_COACH), ventana), fetch_one=True,
    ) or {}
    registros = int(reg.get("n") or 0)
    corregidos = cuentas.get("coach_corrigio", 0) + cuentas.get("borrado_en_diario", 0)
    return {
        "horas": int(horas),
        "registros_del_coach": registros,
        "senales": cuentas,
        "tasa_registro_corregido_pct": round(100.0 * corregidos / registros, 1) if registros else None,
    }
