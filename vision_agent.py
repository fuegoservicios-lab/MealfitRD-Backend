import os
import io
import base64
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from db import save_visual_entry

# Definimos el modelo de salida estructurada para capturar la descripción
class ImageDescription(BaseModel):
    description: str = Field(description="Descripción concisa de los alimentos, ingredientes o comida visible en la imagen.")
    is_food: bool = Field(description="¿Contiene esta imagen comida, ingredientes o una nevera?")
    calories: int = Field(description="Estimación de calorías totales en la imagen. Usa 0 si no es comida.")
    protein: int = Field(description="Estimación de gramos de proteína totales en la imagen. Usa 0 si no es comida.")
    carbs: int = Field(description="Estimación de gramos de carbohidratos totales en la imagen. Usa 0 si no es comida.")
    healthy_fats: int = Field(description="Estimación de gramos de grasas saludables totales en la imagen. Usa 0 si no es comida.")

async def process_image_with_vision(image_bytes: bytes) -> dict:
    """
    Toma los bytes de una imagen, usa Gemini Vision para extraer una descripción
    y determina si contiene alimentos usando structured output.
    """
    try:
        # Iniciamos el modelo Gemini 1.5 Pro (o Flash) que soporta visión nativa
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-pro-preview",
            temperature=0.1,
            google_api_key=os.environ.get("GEMINI_API_KEY")
        ).with_structured_output(ImageDescription)
        
        # Convertimos los bytes a base64 para enviarlo a la API de LangChain/Gemini
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe detalladamente todos los alimentos, ingredientes o platillos que ves en esta imagen. Si es una nevera, lista el contenido visible. Si no hay comida, indícalo. También proporciona una estimación de las calorías, gramos de proteína, gramos de carbohidratos y gramos de grasas saludables (solo el número) totales en la imagen (usa 0 si no es comida)."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        )
        
        response = await llm.ainvoke([message])
        
        description = response.description if response and hasattr(response, 'description') else "Imagen sin descripción clara."
        is_food = response.is_food if response and hasattr(response, 'is_food') else False
        
        if is_food:
            calories = response.calories if hasattr(response, 'calories') else 0
            protein = response.protein if hasattr(response, 'protein') else 0
            carbs = response.carbs if hasattr(response, 'carbs') else 0
            healthy_fats = response.healthy_fats if hasattr(response, 'healthy_fats') else 0
            if calories > 0 or protein > 0:
                description += f" (Estimación: Calorías: {calories}, Proteína: {protein}g, Carbohidratos: {carbs}g, Grasas Saludables: {healthy_fats}g)"
        
        return {
            "description": description,
            "is_food": is_food
        }

    except Exception as e:
        print(f"⚠️ Error procesando imagen con Gemini Vision: {e}")
        return {"description": "Error analizando imagen.", "is_food": False}

def get_multimodal_embedding(text: str) -> list:
    """
    Genera un embedding de la descripción de la imagen usando el modelo text-multilingual-embedding-002
    (recortado a 768 para pgvector local).
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=os.environ.get("GEMINI_API_KEY")
        )
        # Recortamos a 768 dimensiones
        emb = embeddings.embed_query(text)
        return emb[:768]
    except Exception as e:
        print(f"⚠️ Error al generar embedding multimodal: {e}")
        return None

async def async_process_and_save_visual_entry(user_id: str, file_bytes: bytes, image_url: str, user_message: str = ""):
    """
    Procesador en segundo plano (Background Task).
    1. Analiza la imagen con Gemini Vision
    2. Si es comida, saca el embedding de la descripción
    3. Lo guarda en la tabla visual_diary.
    4. Cruce de silos: Extrae hechos nutricionales basados en la foto y el comentario del usuario.
    """
    from fact_extractor import async_extract_and_save_facts

    print("\n-------------------------------------------------------------")
    print("📸 [VISION AGENT] Procesando nueva imagen subida...")
    
    # Paso 1: Visión
    vision_result = await process_image_with_vision(file_bytes)
    
    if not vision_result.get("is_food"):
        print("➡️ La imagen fue ignorada porque no se detectaron alimentos.")
        return

    description = vision_result.get("description", "")
    print(f"✅ Descripción generada: '{description}'")
    
    # Paso 2: Embedding
    embedding = get_multimodal_embedding(description)
    
    if not embedding:
        print("⚠️ No se pudo vectorizar la imagen. Abortando guardado.")
        return
        
    # Paso 3: Base de Datos Visual Diary
    print(f"📦 Guardando entrada visual en la DB (Vector 768d)...")
    save_visual_entry(
        user_id=user_id,
        image_url=image_url,
        description=description,
        embedding=embedding
    )
    print("✅ ¡Imagen registrada en el Diario Visual con éxito!")

    # Paso 4: Cruce de Silos Multimodal (Diario Visual -> Hechos de Usuario)
    # Combinamos lo que dijo el usuario con lo que la IA vio en la foto
    # Ej: "Me cayó pesado esto" + "Plato de mangú con salami, queso frito y cebolla verde"
    combined_context = f"Comentario del usuario sobre su comida actual: '{user_message}'. Lo que estaba comiendo (según análisis de imagen): '{description}'"
    
    print("🔄 [VISION AGENT] Enviando contexto combinado al Extractor de Hechos...")
    # Llamamos al extractor de hechos para que analice esta experiencia y actualice los user_facts
    await async_extract_and_save_facts(user_id, combined_context)
