import sys
import logging
import os
sys.path.insert(0, os.path.dirname(__file__))

from db_core import execute_sql_write

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_migration():
    logger.info("Añadiendo columnas de precio a master_ingredients...")
    try:
        execute_sql_write("ALTER TABLE master_ingredients ADD COLUMN IF NOT EXISTS price_per_lb NUMERIC DEFAULT 0;")
        execute_sql_write("ALTER TABLE master_ingredients ADD COLUMN IF NOT EXISTS price_per_unit NUMERIC DEFAULT 0;")
        
        # Poblar algunos genéricos (en RD$) para que la prueba funcione (Estimaciones sueltas)
        # Pollo: ~120 RD$/lb
        execute_sql_write("UPDATE master_ingredients SET price_per_lb = 120 WHERE name ILIKE '%pollo%';")
        # Carne de res / Vaca: ~200 RD$/lb
        execute_sql_write("UPDATE master_ingredients SET price_per_lb = 200 WHERE name ILIKE '%res%' OR name ILIKE '%carne%';")
        # Arroz: ~40 RD$/lb
        execute_sql_write("UPDATE master_ingredients SET price_per_lb = 40 WHERE name ILIKE '%arroz%';")
        # Huevos: ~8 RD$/unidad
        execute_sql_write("UPDATE master_ingredients SET price_per_unit = 8 WHERE name ILIKE '%huevo%';")
        # Plátano: ~20 RD$/unidad
        execute_sql_write("UPDATE master_ingredients SET price_per_unit = 20 WHERE name ILIKE '%plátano%' OR name ILIKE '%platano%';")
        # Leche: ~70 RD$/unidad (Litro/Cartón asumiendo unidad)
        execute_sql_write("UPDATE master_ingredients SET price_per_unit = 70 WHERE name ILIKE '%leche%';")
        
        logger.info("Migración completada exitosamente.")
    except Exception as e:
        logger.error(f"Error en migración: {e}")

if __name__ == "__main__":
    run_migration()
