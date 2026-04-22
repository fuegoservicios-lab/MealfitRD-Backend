import re

with open('c:/Users/angel/OneDrive/Escritorio/MealfitRD.IA/backend/routers/plans.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add arun_plan_pipeline to imports
content = content.replace('from graph_orchestrator import run_plan_pipeline', 'from graph_orchestrator import run_plan_pipeline, arun_plan_pipeline')

# Find the run_pipeline block in api_analyze_stream
old_block = '''        def run_pipeline():
            try:
                result = run_plan_pipeline(
                    pipeline_data, history, taste_profile,
                    memory_context=memory.get("full_context_str", "") if session_id else "",
                    progress_callback=progress_callback,
                    background_tasks=background_tasks
                )
                pipeline_result["result"] = result
            except Exception as e:
                pipeline_result["error"] = str(e)
                logger.error(f"❌ [SSE PIPELINE ERROR]: {e}")
                traceback.print_exc()
            finally:
                # Señal de fin para que el generador SSE cierre
                try:
                    loop.call_soon_threadsafe(progress_queue.put_nowait, {"event": "_done"})
                except Exception:
                    pass

        pipeline_thread = threading.Thread(target=run_pipeline, daemon=True)
        pipeline_thread.start()'''

new_block = '''        async def run_pipeline():
            try:
                result = await arun_plan_pipeline(
                    pipeline_data, history, taste_profile,
                    memory_context=memory.get("full_context_str", "") if session_id else "",
                    progress_callback=progress_callback,
                    background_tasks=background_tasks
                )
                pipeline_result["result"] = result
            except Exception as e:
                pipeline_result["error"] = str(e)
                logger.error(f"❌ [SSE PIPELINE ERROR]: {e}")
                traceback.print_exc()
            finally:
                # Señal de fin para que el generador SSE cierre
                try:
                    loop.call_soon_threadsafe(progress_queue.put_nowait, {"event": "_done"})
                except Exception:
                    pass

        asyncio.create_task(run_pipeline())'''

if old_block in content:
    content = content.replace(old_block, new_block)
    with open('c:/Users/angel/OneDrive/Escritorio/MealfitRD.IA/backend/routers/plans.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Success')
else:
    print('Block not found')
