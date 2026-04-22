import re

with open('c:/Users/angel/OneDrive/Escritorio/MealfitRD.IA/backend/graph_orchestrator.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Rename run_plan_pipeline to arun_plan_pipeline
content = content.replace('def run_plan_pipeline(', 'async def arun_plan_pipeline(')

# 2. Remove _run_async_in_thread definition and usage
thread_def_start = content.find('    def _run_async_in_thread(coro):')
fallback_def_start = content.find('    def _get_extreme_fallback_plan(nutr: dict, goal: str) -> dict:')
if thread_def_start != -1 and fallback_def_start != -1:
    content = content[:thread_def_start] + content[fallback_def_start:]

# 3. Replace the try block where it executes the graph
exec_block_old = '''    try:
        # Ejecutar asíncronamente con un timeout global
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
            
        if loop and loop.is_running():
            final_state = _run_async_in_thread(asyncio.wait_for(run_graph(), timeout=600))
        else:
            final_state = asyncio.run(asyncio.wait_for(run_graph(), timeout=600))
            
    except Exception as e:'''

exec_block_new = '''    try:
        # Ejecutar asíncronamente con un timeout global (sin saltos de hilo)
        final_state = await asyncio.wait_for(run_graph(), timeout=600)
    except Exception as e:'''

content = content.replace(exec_block_old, exec_block_new)

# 4. Add the sync wrapper at the end of the file
sync_wrapper = '''
def run_plan_pipeline(form_data: dict, history: list = None, taste_profile: str = "", memory_context: str = "", progress_callback=None, previous_ai_error: str = None, background_tasks=None) -> dict:
    """Wrapper síncrono para mantener compatibilidad con cron/callers no-async."""
    import asyncio
    import threading
    import contextvars
    
    coro = arun_plan_pipeline(form_data, history, taste_profile, memory_context, progress_callback, previous_ai_error, background_tasks)
    
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
        
    if loop and loop.is_running():
        ctx = contextvars.copy_context()
        res = [None]
        err = [None]
        
        def _runner():
            try:
                res[0] = asyncio.run(coro)
            except Exception as e:
                err[0] = e
                
        t = threading.Thread(target=ctx.run, args=(_runner,))
        t.start()
        t.join()
        if err[0] is not None:
            raise err[0]
        return res[0]
    else:
        return asyncio.run(coro)
'''

content += sync_wrapper

with open('c:/Users/angel/OneDrive/Escritorio/MealfitRD.IA/backend/graph_orchestrator.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Success')
