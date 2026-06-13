# Script manual de debug del router de fact-extraction (NO es un test).
# [P0-DEEPSEEK-MIGRATION · 2026-06-12] Guard __main__ añadido: el módulo
# invocaba `should_extract_facts(...)` (LLM EN VIVO) a import-time, así que
# cada colección de pytest disparaba una llamada real al provider.
# Ejecutar a mano: python tests/test_router.py
import asyncio  # noqa: F401  (legacy del script original)

msg = '''[El usuario subió una imagen. Análisis de la imagen: "Una foto de un puré de plátano maduro (mofonguito) con carne glaseada en un plato blanco."]

Mensaje del usuario: Yo me comi esto tambien de desayuno'''

if __name__ == "__main__":
    from fact_extractor import should_extract_facts
    print(should_extract_facts(msg))
