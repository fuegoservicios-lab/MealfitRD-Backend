from fastapi import APIRouter, HTTPException, Depends
import logging
from db_core import execute_sql_query
import json

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/system",
    tags=["system"]
)

# Puedes añadir una dependencia de autenticación de admin aquí después si es necesario
# async def verify_admin(token: str = Header(...)): ...

@router.get("/health")
def get_system_health():
    """
    Meta-Dashboard (Gap 5): Retorna el "Health Status" de la Inteligencia Autónoma.
    Calcula al vuelo las métricas de:
    - Quality Score global
    - Efectividad de Nudges
    - Distribución de abandono causal (Gap 2)
    - Distribución emocional (Gap 4)
    """
    metrics = {
        "nudge_effectiveness": {},
        "abandonment_reasons": {},
        "emotional_distribution": {},
        "average_quality_score": 0.0,
        "users_evaluated": 0
    }
    
    try:
        # 1. Nudge Response Rate Global
        nudge_stats = execute_sql_query(
            "SELECT COUNT(*) as total, SUM(CASE WHEN responded THEN 1 ELSE 0 END) as responded_count FROM nudge_outcomes",
            fetch_one=True
        )
        if nudge_stats and nudge_stats.get("total", 0) > 0:
            total = nudge_stats["total"]
            responded = nudge_stats["responded_count"] or 0
            metrics["nudge_effectiveness"] = {
                "total_sent": total,
                "total_responded": responded,
                "response_rate_percent": round((responded / total) * 100, 2)
            }
            
        # 2. Abandonment Reasons (Gap 2)
        reasons = execute_sql_query(
            "SELECT reason, COUNT(*) as count FROM abandoned_meal_reasons GROUP BY reason ORDER BY count DESC",
            fetch_all=True
        )
        if reasons:
            metrics["abandonment_reasons"] = {row['reason']: row['count'] for row in reasons}
            
        # 3. Emotional State Distribution (Gap 4)
        emotions = execute_sql_query(
            "SELECT response_sentiment, COUNT(*) as count FROM nudge_outcomes WHERE response_sentiment IS NOT NULL GROUP BY response_sentiment ORDER BY count DESC",
            fetch_all=True
        )
        if emotions:
            metrics["emotional_distribution"] = {row['response_sentiment']: row['count'] for row in emotions}
            
        # 4. Average Quality Score de todos los perfiles
        profiles = execute_sql_query(
            "SELECT health_profile->>'quality_history' as qh FROM user_profiles WHERE health_profile->>'quality_history' IS NOT NULL",
            fetch_all=True
        )
        if profiles:
            total_score = 0.0
            count = 0
            for p in profiles:
                try:
                    qh_str = p.get('qh')
                    if qh_str:
                        history = json.loads(qh_str)
                        if history and isinstance(history, list) and len(history) > 0:
                            # Tomamos el score más reciente
                            total_score += float(history[-1])
                            count += 1
                except Exception:
                    continue
            
            if count > 0:
                metrics["average_quality_score"] = round(total_score / count, 2)
                metrics["users_evaluated"] = count

        return {
            "success": True,
            "status": "healthy",
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Error calculando system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))
