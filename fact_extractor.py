import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field
from typing import List, Optional

from db import (
    save_user_fact, search_user_facts, delete_user_fact,
    get_user_facts_by_metadata, acquire_fact_lock, release_fact_lock,
    enqueue_pending_fact, dequeue_pending_facts, delete_pending_facts
)


class FactMetadata(BaseModel):
    category: str = Field(description="Una de: 'alergia', 'condicion_medica', 'dieta', 'gusto', 'objetivo'")
    ingrediente_canonico: Optional[str] = Field(description="ID de catálogo universal del ingrediente en minúsculas. OBLIGATORIO usar los IDs definidos en el Catálogo Canónico (ej. 'peanut' para maní/cacahuete). Si no está en el catálogo, usa su traducción al inglés en minúscula y singular.", default=None)
    ingrediente: str = Field(description="Ingrediente principal si aplica, ej: 'mani', 'camarones'. Vacío si no aplica.")
    intensidad: int = Field(description="Intensidad del sentimiento (1 a 5). 1=Rechazo/Odio, 2=No le gusta, 3=Neutral/Info, 4=Le gusta, 5=Le fascina", default=3)

class FactItem(BaseModel):
    fact: str = Field(description="El hecho en sí expresado de forma clara.")
    metadata: FactMetadata = Field(description="Metadatos estructurados para clasificación exacta.")

class FactsModel(BaseModel):
    facts: List[FactItem] = Field(description="Lista de hechos nutricionales puntuales extraídos del mensaje.")

# --- Modelos para Batching de Contradicciones y Fusiones ---

class BatchContradictionItem(BaseModel):
    new_fact: str = Field(description="El texto exacto del nuevo hecho que genera la contradicción.")
    ids_to_delete: List[str] = Field(description="IDs (UUID) de los hechos existentes que este nuevo hecho contradice y deben ser borrados.")

class BatchMergeItem(BaseModel):
    merged_fact: str = Field(description="El hecho fusionado resultante (más completo que los individuales).")
    ids_to_delete: List[str] = Field(description="IDs (UUID) de los hechos existentes redundantes que serán reemplazados por el fusionado.")
    skip_new_fact: str = Field(description="El texto del nuevo hecho que fue absorbido en la fusión (para no guardarlo por separado).")

class BatchContradictionResult(BaseModel):
    contradictions: List[BatchContradictionItem] = Field(
        description="Lista de contradicciones encontradas. Vacía si no hay ninguna."
    )
    merges: List[BatchMergeItem] = Field(
        description="Lista de fusiones de hechos redundantes/complementarios. Vacía si no hay ninguna.",
        default=[]
    )

class RouterResult(BaseModel):
    has_relevant_info: bool = Field(description="True si el mensaje contiene datos médicos, preferencias alimenticias, alergias, síntomas u objetivos. False si es conversación casual (ej: hola, gracias, ok).")
    confidence_score: int = Field(description="Nivel de confianza de tu decisión, de 1 a 10. 10=Completamente seguro. 1=Muy inseguro/confuso (ej. sarcasmo, texto ambiguo).", default=10)

def should_extract_facts(user_message: str) -> bool:
    """Verifica rápidamente si el mensaje vale la pena analizarse para extraer hechos."""
    if not user_message or len(user_message.strip()) < 5:
        return False
        
    prompt = f"""
    Analiza este mensaje y responde True SOLO si el usuario menciona algo sobre:
    - Preferencias alimenticias (gustos, rechazos)
    - Alergias o condiciones médicas
    - Síntomas temporales o estado de salud
    - Objetivos de salud, condición física o peso
    
    Responde False si es conversación general, saludos, agradecimientos o texto sin datos útiles sobre el perfil del usuario.
    
    ES VITAL que también asignes un 'confidence_score' del 1 al 10. Si el mensaje es ambiguo, sarcástico, o no estás 100% seguro de si contiene un hecho médico velado, asigna un score bajo (ej. 1 a 6).
    
    Mensaje: "{user_message}"
    """
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        temperature=0.0,
        google_api_key=os.environ.get("GEMINI_API_KEY")
    ).with_structured_output(RouterResult)
    
    try:
        res = llm.invoke(prompt)
        if not res:
            return False
            
        print(f"🚦 [ROUTER LITE] has_info: {res.has_relevant_info} | confidence: {res.confidence_score}/10")
        
        # Fallback de confianza: Si tiene info, lo pasamos. 
        # Si dice que NO tiene info, pero está inseguro (< 8), forzamos pasarlo al extractor pesado por seguridad.
        if res.has_relevant_info:
            return True
        elif res.confidence_score < 8:
            print(f"⚠️ [ROUTER FALLBACK] Score bajo ({res.confidence_score}/10). Enviando a extractor pesado por si acaso.")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"⚠️ [ROUTER FALLBACK] Error analizando: {e}")
        return True # Fallback seguro: ante la duda, extraemos


DOMINICAN_INGREDIENT_CATALOG = """Catálogo Canónico de Ingredientes y Alérgenos (USA ESTOS IDs SIEMPRE QUE APLIQUE):
- peanut (maní, cacahuete)
- dairy (leche, queso, lactosa, yogur, mantequilla)
- egg (huevo)
- shellfish (mariscos, camarones, langosta, cangrejo, lambí)
- fish (pescado, salmón, atún, bacalao, arenque)
- soy (soja, soya)
- wheat (trigo, gluten, pan, pasta, harina)
- tree_nut (nueces, almendras, cajuil, macadamia)
- chicken (pollo)
- beef (carne de res, vaca, ternera)
- pork (cerdo, chuleta, tocino, chicharrón)
- plantain (plátano verde, plátano maduro, fritos, mangú)
- banana (guineo, guineíto)
- rice (arroz, locrio, moro)
- beans (habichuelas, frijoles, gandules)
- corn (maíz, harina de maíz)
- avocado (aguacate)
- cassava (yuca, casabe)
- potato (papa)
- sweet_potato (batata)
- squash (auyama)
- garlic (ajo)
- onion (cebolla)
- bell_pepper (ají, pimiento, cubanela)
- tomato (tomate)
- cilantro (verdura, cilantro, verdurita)
- oregano (orégano)
- sugar (azúcar)
- salt (sal)
- oats (avena)
- honey (miel)
- coffee (café)
- cocoa (chocolate, cacao)
- coconut (coco)
- pineapple (piña)
- mango (mango)
- papaya (lechosa)
- passion_fruit (chinola)
- soursop (guanábana)
- tamarind (tamarindo)
- strawberry (fresa)
- orange (naranja, china)
- lemon (limón)"""

def extract_facts(user_message: str, recent_history: str = ""):
    """
    Analiza el mensaje del usuario y extrae "hechos" (facts) permanentes
    junto con sus metadatos estructurados (JSON) sobre sus preferencias,
    salud, alergias u objetivos.
    """
    if not user_message or len(user_message.strip()) < 5:
        return []

    print("\n-------------------------------------------------------------")
    print("🔍 [EXTRACTOR DE HECHOS] Analizando mensaje para vectorizar y etiquetar...")
    
    history_context = f"\n    Contexto reciente (últimos mensajes):\n    {recent_history}\n" if recent_history else ""

    prompt = f"""
    Eres un Analista Nutricional que extrae "Hechos" (Facts) de los mensajes de los pacientes y los clasifica.
    Tu objetivo es leer un mensaje y determinar si contiene información útil para el perfil del usuario:
    - Preferencias alimenticias (gustos, rechazos fuertes) -> categoria: 'preferencia' o 'rechazo'
    - Alergias o condiciones crónicas -> categoria: 'alergia' o 'condicion_medica'
    - Síntomas o estados pasajeros (ej. "estómago revuelto esta semana", "estoy resfriado") -> categoria: 'sintoma_temporal'
    - Objetivos o rutinas -> categoria: 'objetivo'
    
    Además, clasifica la 'intensidad' (del 1 al 5) del sentimiento si aplica:
    - 1: Odio absoluto, rechazo frontal ("no soporto el brócoli")
    - 2: Desagrado leve ("no me gusta mucho el pescado")
    - 3: Neutral o dato médico objetivo ("tengo diabetes", "peso 80kg", "ayer comí arroz")
    - 4: Le gusta bastante ("qué buena la pasta")
    - 5: Pasión o adicción ("amo el plátano", "no puedo vivir sin café")

    IMPORTANTE - CANONICALIZACIÓN DE INGREDIENTES:
    Para el campo `ingrediente_canonico`, DEBES usar OBLIGATORIAMENTE uno de los siguientes IDs si el ingrediente coincide, especialmente para alergias y condiciones médicas:
    {DOMINICAN_INGREDIENT_CATALOG}
    Si el ingrediente no está en la lista, usa su nombre en inglés en minúsculas y singular (ej. "broccoli", "apple").

    {history_context}
    Mensaje del usuario: "{user_message}"
    
    Usa el contexto reciente SOLO para entender a qué se refiere el mensaje actual (ej. si dice "y también me dio alergia", el contexto te dirá a qué alimento). Pero EXTRAE hechos principalmente basados en el último mensaje.
    Si el mensaje NO contiene hechos importantes (ej. "hola", "gracias", "ok"), devuelve una lista vacía [].
    Si contiene hechos, extráelos como oraciones cortas, precisas (en tercera persona sobre el usuario).
    Asegúrate de incluir los metadatos de categoría, intensidad, e 'ingrediente' si aplica (y el 'ingrediente_canonico' rigurosamente guiado por el catálogo).
    """

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        temperature=0.1,
        google_api_key=os.environ.get("GEMINI_API_KEY")
    ).with_structured_output(FactsModel)

    try:
        response = llm.invoke(prompt)
        facts = response.facts if response and hasattr(response, 'facts') else []
        if facts:
            print(f"✅ Se encontraron {len(facts)} hechos estructurados.")
        else:
            print("➡️ No se encontraron hechos relevantes.")
        return facts
    except Exception as e:
        print(f"⚠️ Error al extraer hechos: {e}")
        return []

def get_embedding(text: str):
    """Genera un vector embedding usando Gemini text-embedding-004"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=os.environ.get("GEMINI_API_KEY")
        )
        # Recortar a 768 dimensiones (los modelos nuevos de Gemini son Matryoshka y soportan recorte sin perder semántica)
        emb = embeddings.embed_query(text)
        return emb[:768]
    except Exception as e:
        print(f"⚠️ Error al generar embedding: {e}")
        return None

CRITICAL_CATEGORIES = {"condicion_medica", "alergia", "dieta", "objetivo"}

def async_extract_and_save_facts(user_id: str, message: str, recent_history: str = ""):
    """
    Función orquestadora para ser ejecutada en background.
    Extrae hechos estructurados de un mensaje, revisa contradicciones con la DB
    usando BATCHING (una sola llamada a Gemini para todos los hechos),
    borra los obsoletos y guarda los nuevos vectorizados y etiquetados.
    """
    try:
        if not should_extract_facts(message):
            print("⏭️ [ROUTER] Mensaje ignorado. No contiene hechos relevantes para el perfil.")
            return

        import time
        max_retries = 5
        retry_delay = 2
        lock_acquired = False
        
        for attempt in range(max_retries):
            if acquire_fact_lock(user_id):
                lock_acquired = True
                break
            print(f"⚠️ [FACT EXTRACTOR] Extracción en progreso para el usuario {user_id}. Esperando {retry_delay}s ({attempt+1}/{max_retries})...")
            time.sleep(retry_delay)
            
        if not lock_acquired:
            # ====== COLA PERSISTENTE: Nunca perder datos clínicos ======
            enqueue_pending_fact(user_id, message, recent_history)
            print(f"📋 [FACT EXTRACTOR] Mensaje encolado en Supabase para procesamiento posterior.")
            return
            
        fact_items = extract_facts(message, recent_history)
        if not fact_items:
            release_fact_lock(user_id)
            return

        # ================================================================
        # FASE 1: Generar embeddings y buscar similares para TODOS los hechos
        #         + Query directo por categoría para categorías críticas
        # ================================================================
        prepared_facts = []  # Lista de dicts: {item, fact_text, metadata, emb, similar_facts}

        for item in fact_items:
            if isinstance(item, dict):
                fact_text = item.get("fact", "")
                metadata = item.get("metadata", {})
            else:
                fact_text = getattr(item, "fact", "")
                metadata = item.metadata.model_dump() if hasattr(item, 'metadata') else {}
            
            if not fact_text:
                continue
            
        
            emb = get_embedding(fact_text)
            if not emb:
                continue
        
            similar_facts = search_user_facts(user_id, emb, threshold=0.6, limit=5)
        
            # === ANTI FALSOS NEGATIVOS: Query directo por categoría ===
            categoria = metadata.get("category", "") # Changed from 'categoria' to 'category'
            if categoria in CRITICAL_CATEGORIES:
                print(f"🔎 [CATEGORY QUERY] Categoría crítica '{categoria}' detectada. Buscando TODOS los hechos de esta categoría...")
                category_facts = get_user_facts_by_metadata(user_id, "category", categoria) # Changed from 'categoria' to 'category'
            
                if category_facts:
                    # Merge: deduplicar por ID con los resultados del vector search
                    existing_ids = {f.get("id") for f in similar_facts if f.get("id")}
                    new_from_category = [f for f in category_facts if f.get("id") not in existing_ids]
                
                    if new_from_category:
                        print(f"   ➕ {len(new_from_category)} hechos adicionales recuperados por categoría (no encontrados por similitud vectorial).")
                        similar_facts = similar_facts + new_from_category
            # ==========================================================
        
            prepared_facts.append({
                "item": item,
                "fact_text": fact_text,
                "metadata": metadata,
                "emb": emb,
                "similar_facts": similar_facts
            })

        if not prepared_facts:
            return

        # ================================================================
        # FASE 2: Verificar contradicciones en BATCH (una sola llamada LLM)
        # ================================================================
        # Construir el prompt batch solo con hechos que tienen similares
        facts_with_similar = [pf for pf in prepared_facts if pf["similar_facts"]]
    
        ids_to_delete_all = set()  # Acumular todos los IDs a borrar
        merged_facts_to_save = []   # Lista de {merged_fact, metadata} para guardar después
        skipped_new_facts = set()   # Textos de hechos nuevos absorbidos en fusiones

        if facts_with_similar:
            print(f"🔄 [BATCH] Verificando contradicciones para {len(facts_with_similar)} hechos en una sola llamada...")
        
            # Construir secciones del prompt
            sections = []
            for idx, pf in enumerate(facts_with_similar, 1):
                existing_str = "\n".join(
                    [f"    ID: {f['id']} - Hecho: {f['fact']}" for f in pf["similar_facts"] if 'id' in f and 'fact' in f]
                )
                sections.append(
                    f"  NUEVO HECHO #{idx}: \"{pf['fact_text']}\"\n"
                    f"  Hechos existentes relacionados:\n{existing_str}"
                )
        
            all_sections = "\n\n".join(sections)
        
            batch_prompt = f"""
            Analiza TODOS los nuevos hechos y compáralos con sus hechos existentes correspondientes en la memoria del usuario.
            Para cada nuevo hecho, determina:
        
            A) CONTRADICCIÓN: Si el nuevo hecho INVALIDA directamente uno viejo.
               Ejemplo: "no come pescado" vs "ahora le gusta el pescado" → CONTRADICCIÓN.
        
            B) REDUNDANCIA/FUSIÓN: Si el nuevo hecho es COMPLEMENTARIO o REDUNDANTE con uno existente 
               (hablan del mismo tema/ingrediente sin contradecirse).
               Ejemplo: "Le gusta el pollo" + "Amo el pollo asado" → FUSIÓN: "Al usuario le encanta el pollo, especialmente asado".
               Ejemplo: "Es alérgico al maní" + "Tiene alergia a los cacahuetes" → FUSIÓN: "El usuario es alérgico al maní/cacahuetes".
        
            Reglas:
            - Si un hecho nuevo contradice uno viejo, ponlo en "contradictions" con los IDs a borrar.
            - Si un hecho nuevo es redundante o complementario a uno existente (mismo tema), ponlo en "merges":
              crea un "merged_fact" que combine la info de ambos de forma concisa en tercera persona.
              Incluye en "ids_to_delete" los IDs de los hechos viejos que se absorben.
              Incluye en "skip_new_fact" el texto exacto del nuevo hecho (para no guardarlo como duplicado).
            - Si un hecho nuevo es TOTALMENTE INDEPENDIENTE de los existentes, NO lo incluyas en ninguna lista.
            - NO fusiones hechos que hablan de temas distintos (ej: "le gusta el pollo" y "es diabético" son independientes).

            HECHOS A ANALIZAR:
            {all_sections}
            """
        
            llm = ChatGoogleGenerativeAI(
                model="gemini-3.1-flash-lite-preview",
                temperature=0.0,
                google_api_key=os.environ.get("GEMINI_API_KEY")
            ).with_structured_output(BatchContradictionResult)
        
            try:
                response = llm.invoke(batch_prompt)
            
                if response and response.contradictions:
                    for contradiction in response.contradictions:
                        if contradiction.ids_to_delete:
                            print(f"⚠️ [CONTRADICCIÓN] Hecho: \"{contradiction.new_fact}\" → Borrar IDs: {contradiction.ids_to_delete}")
                            ids_to_delete_all.update(contradiction.ids_to_delete)
            
                if response and response.merges:
                    for merge in response.merges:
                        if merge.ids_to_delete and merge.merged_fact:
                            print(f"🔀 [FUSIÓN] \"{merge.merged_fact}\" ← Absorbe IDs: {merge.ids_to_delete}")
                            ids_to_delete_all.update(merge.ids_to_delete)
                            skipped_new_facts.add(merge.skip_new_fact)
                        
                            # Buscar los metadatos del hecho nuevo original para preservar la categoría
                            original_metadata = {}
                            for pf in facts_with_similar:
                                if pf["fact_text"] == merge.skip_new_fact:
                                    original_metadata = pf["metadata"]
                                    break
                        
                            merged_facts_to_save.append({
                                "fact_text": merge.merged_fact,
                                "metadata": original_metadata
                            })
                        
            except Exception as e:
                print(f"⚠️ [Error en validación batch de contradicciones/fusiones]: {e}")

        # ================================================================
        # FASE 3: Borrar contradictorios/redundantes y guardar hechos
        # ================================================================
        # Borrar todos los hechos marcados (contradictorios + absorbidos por fusión)
        if ids_to_delete_all:
            print(f"🗑️ [BATCH] Borrando {len(ids_to_delete_all)} hechos (contradictorios + redundantes)...")
            for f_id in ids_to_delete_all:
                delete_user_fact(f_id)

        # Guardar hechos nuevos (excluyendo los absorbidos en fusiones)
        saved_count = 0
        for pf in prepared_facts:
            if pf["fact_text"] in skipped_new_facts:
                print(f"⏭️ Hecho absorbido en fusión, no se guarda por separado: '{pf['fact_text']}'")
                continue
            save_user_fact(user_id, pf["fact_text"], pf["emb"], metadata=pf["metadata"])
            print(f"📦 Nuevo hecho guardado: '{pf['fact_text']}' | Metadatos: {pf['metadata']}")
            saved_count += 1
        
        # Guardar hechos fusionados (con nuevo embedding)
        merge_count = 0
        for mf in merged_facts_to_save:
            merged_emb = get_embedding(mf["fact_text"])
            if merged_emb:
                save_user_fact(user_id, mf["fact_text"], merged_emb, metadata=mf["metadata"])
                print(f"🔀 Hecho fusionado guardado: '{mf['fact_text']}' | Metadatos: {mf['metadata']}")
                merge_count += 1
        
        total_deleted = len(ids_to_delete_all)
        print(f"✅ [BATCH COMPLETO] {saved_count} nuevos + {merge_count} fusionados guardados, {total_deleted} eliminados (contradictorios + redundantes).")

    except Exception as e:
        import traceback
        print(f"❌ [CRÍTICO] Fallo general en orquestación de hechos: {e}")
        track = traceback.format_exc()
        print(f"Trazabilidad extendida de error: {track}")
    finally:
        # ====== PROCESAR COLA PERSISTENTE (Supabase) antes de liberar el lock ======
        try:
            pending_items = dequeue_pending_facts(user_id)
            
            if pending_items:
                print(f"\n📋 [FACT EXTRACTOR] Procesando {len(pending_items)} mensajes pendientes de la cola (Supabase)...")
                processed_ids = []
                for idx, pending in enumerate(pending_items, 1):
                    try:
                        print(f"   📋 [{idx}/{len(pending_items)}] Procesando: '{pending['message'][:50]}...'")
                        _process_single_extraction(user_id, pending["message"], pending.get("recent_history", ""))
                        processed_ids.append(pending["id"])
                    except Exception as pe:
                        print(f"   ⚠️ Error procesando pendiente #{idx}: {pe}")
                        # Aún así marcarlo como procesado para no re-intentar infinitamente
                        processed_ids.append(pending["id"])
                
                if processed_ids:
                    delete_pending_facts(processed_ids)
                print(f"✅ [FACT EXTRACTOR] Cola pendiente procesada y limpiada de Supabase.")
        except Exception as qe:
            print(f"⚠️ [FACT EXTRACTOR] Error procesando cola pendiente: {qe}")
        
        release_fact_lock(user_id)


def _process_single_extraction(user_id: str, message: str, recent_history: str = ""):
    """
    Procesa una sola extracción de hechos SIN manejar el lock.
    Usado internamente para drenar la cola de pendientes.
    """
    if not should_extract_facts(message):
        return
    
    fact_items = extract_facts(message, recent_history)
    if not fact_items:
        return
    
    # Preparar hechos
    prepared_facts = []
    for item in fact_items:
        if isinstance(item, dict):
            fact_text = item.get("fact", "")
            metadata = item.get("metadata", {})
        else:
            fact_text = getattr(item, "fact", "")
            metadata = item.metadata.model_dump() if hasattr(item, 'metadata') else {}
        
        if not fact_text:
            continue
        
        emb = get_embedding(fact_text)
        if not emb:
            continue
        
        similar_facts = search_user_facts(user_id, emb, threshold=0.6, limit=5)
        
        # Anti falsos negativos: query por categoría
        categoria = metadata.get("category", "")
        if categoria in CRITICAL_CATEGORIES:
            category_facts = get_user_facts_by_metadata(user_id, "category", categoria)
            if category_facts:
                existing_ids = {f.get("id") for f in similar_facts if f.get("id")}
                new_from_category = [f for f in category_facts if f.get("id") not in existing_ids]
                if new_from_category:
                    similar_facts = similar_facts + new_from_category
        
        prepared_facts.append({
            "item": item,
            "fact_text": fact_text,
            "metadata": metadata,
            "emb": emb,
            "similar_facts": similar_facts
        })
    
    if not prepared_facts:
        return
    
    # Guardar directamente (sin batch de contradicciones para pendientes, ya se verificará en la próxima extracción)
    for pf in prepared_facts:
        save_user_fact(user_id, pf["fact_text"], pf["emb"], metadata=pf["metadata"])
        print(f"   📦 Hecho pendiente guardado: '{pf['fact_text']}'")
