import asyncio
from fact_extractor import should_extract_facts
msg = '''[El usuario subió una imagen. Análisis de la imagen: "Una foto de un puré de plátano maduro (mofonguito) con carne glaseada en un plato blanco."]

Mensaje del usuario: Yo me comi esto tambien de desayuno'''
print(should_extract_facts(msg))
