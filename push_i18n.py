"""[P1-I18N-PUSH-CRON-ESPANOL · 2026-08-22] Las notificaciones push, en el idioma del usuario.

═══════════════════════════════════════════════════════════════════════════════
QUÉ PASABA
═══════════════════════════════════════════════════════════════════════════════

`P2-I18N-PUSH-SIN-LOCALE` (2026-08-21) tradujo UNA notificación —el nudge del coach— y su
guard sólo abría `proactive_agent.py`. Las otras salían con título Y cuerpo en español duro
desde `cron_tasks.py` y `routers/plans.py`. MEDIDO con AST sobre los call sites: 25 títulos
y 18 cuerpos literales distintos, más 26 dinámicos.

Un usuario con la app en inglés recibía en su pantalla de bloqueo:

    «Tu plan necesita una revisión — Detectamos ingredientes que ya no están en tu nevera.
     Actualízala para que generemos los días siguientes.»

Y es la superficie MENOS perdonable de todas: llega sin que la pidas, se lee de un vistazo
y no hay dónde cambiar el idioma. No es una pantalla que el usuario esté explorando.

═══════════════════════════════════════════════════════════════════════════════
POR QUÉ SE TRADUCE AQUÍ Y NO EN LOS 35 CALL SITES
═══════════════════════════════════════════════════════════════════════════════

`utils_push.send_push_notification` es el cuello de botella por el que pasa TODO push, sin
excepción: `_dispatch_push_notification` es un envoltorio suyo. Traducir ahí ata la
invariante al ACTO —«nada sale de este proceso sin pasar por el idioma del usuario»— en vez
de a 35 llamadas que hay que acordarse de tocar.

Es literalmente la lección que este repo ya pagó dos veces:
  · `P2-DISPLAY-POP-VECINO`: el pop de `_display` colgaba de siete funciones con nombre y el
    octavo re-escritor nacía mintiendo por omisión.
  · `P1-COUNTRY-SYSTEM-F1`: «gatear call sites uno a uno es el agujero, no el cierre».

Un call site nuevo queda cubierto sin wiring. Y `P2-I18N-PUSH-SIN-LOCALE` no se toca: su
título ya llega resuelto y aquí no encuentra clave, así que pasa tal cual.

═══════════════════════════════════════════════════════════════════════════════
LA CLAVE ES EL TEXTO ESPAÑOL
═══════════════════════════════════════════════════════════════════════════════

Misma decisión que el motor del frontend (`P1-I18N-DASHBOARD`), y por la misma razón: el
español no lleva catálogo, y lo que no está traducido cae al español y no a una clave. Un
push con la cadena `push.pantry.empty.title` en la pantalla de bloqueo sería peor que uno
en español.

Consecuencia que hay que conocer: **cambiar el copy en el call site huérfana su traducción
EN SILENCIO** — el push sale en español y nadie se entera. Lo vigila
`test_p1_i18n_push_cron_espanol.py`, que compara los literales de los call sites contra
este catálogo. Si añades un push nuevo, el guard te lo dirá.

Los 26 mensajes DINÁMICOS (f-strings con cifras, variables) caen al español a propósito:
traducir una plantilla compuesta en el call site exigiría reestructurar cada uno, y el
fallback al español es conducta declarada, no fallo. El guard los cuenta para que la deuda
tenga número en vez de ser invisible.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Los cinco idiomas soportados. SSOT de la lista: frontend/src/i18n/locales.js.
_LOCALES = ("es-DO", "en-US", "pt-BR", "fr-FR", "it-IT")

# ── TÍTULOS ────────────────────────────────────────────────────────────────────
_TITULOS = {
    # [P1-I18N-PUSH-GUARD-CIEGO-AL-THREAD · 2026-08-23] Los que escapaban al guard por ir
    # envueltos en `threading.Thread(target=…, kwargs={"title": …})`: el nodo `Call` se
    # llama `Thread`, así que el extractor por AST no los veía y el guard reportaba CERO
    # faltantes mientras salían en español.
    "Renovación pausada": {
        "en-US": "Renewal paused",
        "pt-BR": "Renovação pausada",
        "fr-FR": "Renouvellement en pause",
        "it-IT": "Rinnovo in pausa",
    },
    "Tu plan necesita tu feedback": {
        "en-US": "Your plan needs your feedback",
        "pt-BR": "Seu plano precisa do seu feedback",
        "fr-FR": "Votre plan a besoin de votre avis",
        "it-IT": "Il tuo piano ha bisogno del tuo feedback",
    },
    "Actualiza tu nevera": {
        "en-US": "Update your fridge",
        "pt-BR": "Atualize sua geladeira",
        "fr-FR": "Mettez votre frigo à jour",
        "it-IT": "Aggiorna il tuo Frigo",
    },
    "Confirma tu inventario": {
        "en-US": "Confirm your inventory",
        "pt-BR": "Confirme seu estoque",
        "fr-FR": "Confirmez votre inventaire",
        "it-IT": "Conferma il tuo inventario",
    },
    "Detectamos un cambio de zona horaria": {
        "en-US": "We detected a time zone change",
        "pt-BR": "Detectamos uma mudança de fuso horário",
        "fr-FR": "Nous avons détecté un changement de fuseau horaire",
        "it-IT": "Abbiamo rilevato un cambio di fuso orario",
    },
    "Detectamos un problema con tu plan": {
        "en-US": "We found a problem with your plan",
        "pt-BR": "Encontramos um problema no seu plano",
        "fr-FR": "Nous avons détecté un problème avec votre plan",
        "it-IT": "Abbiamo rilevato un problema con il tuo piano",
    },
    "Generamos tu plan con datos parciales": {
        "en-US": "We built your plan with partial data",
        "pt-BR": "Geramos seu plano com dados parciais",
        "fr-FR": "Nous avons créé votre plan avec des données partielles",
        "it-IT": "Abbiamo creato il tuo piano con dati parziali",
    },
    "Generamos tu próximo bloque": {
        "en-US": "We generated your next block",
        "pt-BR": "Geramos seu próximo bloco",
        "fr-FR": "Nous avons généré votre prochain bloc",
        "it-IT": "Abbiamo generato il tuo prossimo blocco",
    },
    "Generando tu próximo bloque": {
        "en-US": "Generating your next block",
        "pt-BR": "Gerando seu próximo bloco",
        "fr-FR": "Génération de votre prochain bloc",
        "it-IT": "Generazione del tuo prossimo blocco",
    },
    "Loguea tus comidas para tu próximo bloque": {
        "en-US": "Log your meals for your next block",
        "pt-BR": "Registre suas refeições para o próximo bloco",
        "fr-FR": "Enregistrez vos repas pour votre prochain bloc",
        "it-IT": "Registra i tuoi pasti per il prossimo blocco",
    },
    "Necesitamos tu zona horaria": {
        "en-US": "We need your time zone",
        "pt-BR": "Precisamos do seu fuso horário",
        "fr-FR": "Nous avons besoin de votre fuseau horaire",
        "it-IT": "Ci serve il tuo fuso orario",
    },
    "Seguimos con un plato flexible": {
        "en-US": "We're continuing with a flexible dish",
        "pt-BR": "Seguimos com um prato flexível",
        "fr-FR": "Nous continuons avec un plat flexible",
        "it-IT": "Proseguiamo con un piatto flessibile",
    },
    "Tu chunk sigue esperando tu nevera": {
        "en-US": "Your block is still waiting on your fridge",
        "pt-BR": "Seu bloco continua esperando sua geladeira",
        "fr-FR": "Votre bloc attend toujours votre frigo",
        "it-IT": "Il tuo blocco sta ancora aspettando il tuo Frigo",
    },
    "Tu plan está en pausa": {
        "en-US": "Your plan is paused",
        "pt-BR": "Seu plano está pausado",
        "fr-FR": "Votre plan est en pause",
        "it-IT": "Il tuo piano è in pausa",
    },
    "Tu plan está en pausa 🧊": {
        "en-US": "Your plan is paused 🧊",
        "pt-BR": "Seu plano está pausado 🧊",
        "fr-FR": "Votre plan est en pause 🧊",
        "it-IT": "Il tuo piano è in pausa 🧊",
    },
    "Tu plan está esperando": {
        "en-US": "Your plan is waiting",
        "pt-BR": "Seu plano está esperando",
        "fr-FR": "Votre plan attend",
        "it-IT": "Il tuo piano è in attesa",
    },
    "Tu plan necesita actualizarse": {
        "en-US": "Your plan needs an update",
        "pt-BR": "Seu plano precisa ser atualizado",
        "fr-FR": "Votre plan a besoin d'une mise à jour",
        "it-IT": "Il tuo piano ha bisogno di un aggiornamento",
    },
    "Tu plan necesita más ingredientes": {
        "en-US": "Your plan needs more ingredients",
        "pt-BR": "Seu plano precisa de mais ingredientes",
        "fr-FR": "Votre plan a besoin de plus d'ingrédients",
        "it-IT": "Il tuo piano ha bisogno di più ingredienti",
    },
    "Tu plan necesita revisión de ingredientes": {
        "en-US": "Your plan needs an ingredient check",
        "pt-BR": "Seu plano precisa de uma revisão de ingredientes",
        "fr-FR": "Votre plan a besoin d'une vérification des ingrédients",
        "it-IT": "Il tuo piano ha bisogno di un controllo degli ingredienti",
    },
    "Tu plan parece atrasado": {
        "en-US": "Your plan looks behind schedule",
        "pt-BR": "Seu plano parece atrasado",
        "fr-FR": "Votre plan semble en retard",
        "it-IT": "Il tuo piano sembra in ritardo",
    },
    "Tu plan quedó archivado": {
        "en-US": "Your plan has been archived",
        "pt-BR": "Seu plano foi arquivado",
        "fr-FR": "Votre plan a été archivé",
        "it-IT": "Il tuo piano è stato archiviato",
    },
    "Tu plan se generó con poca info": {
        "en-US": "Your plan was built with little information",
        "pt-BR": "Seu plano foi gerado com pouca informação",
        "fr-FR": "Votre plan a été créé avec peu d'informations",
        "it-IT": "Il tuo piano è stato creato con poche informazioni",
    },
    "Tu plan sigue en pausa": {
        "en-US": "Your plan is still paused",
        "pt-BR": "Seu plano continua pausado",
        "fr-FR": "Votre plan est toujours en pause",
        "it-IT": "Il tuo piano è ancora in pausa",
    },
    "Tu plan tiene compras urgentes": {
        "en-US": "Your plan has urgent groceries",
        "pt-BR": "Seu plano tem compras urgentes",
        "fr-FR": "Votre plan a des courses urgentes",
        "it-IT": "Il tuo piano ha una spesa urgente",
    },
    "Tu próximo bloque está esperando": {
        "en-US": "Your next block is waiting",
        "pt-BR": "Seu próximo bloco está esperando",
        "fr-FR": "Votre prochain bloc attend",
        "it-IT": "Il tuo prossimo blocco è in attesa",
    },
    "¡Tu plan está de vuelta! 🧊→▶️": {
        "en-US": "Your plan is back! 🧊→▶️",
        "pt-BR": "Seu plano voltou! 🧊→▶️",
        "fr-FR": "Votre plan est de retour ! 🧊→▶️",
        "it-IT": "Il tuo piano è tornato! 🧊→▶️",
    },
    "⚡ Optimizando tu plan": {
        "en-US": "⚡ Optimizing your plan",
        "pt-BR": "⚡ Otimizando seu plano",
        "fr-FR": "⚡ Optimisation de votre plan",
        "it-IT": "⚡ Ottimizzazione del tuo piano",
    },
}

# ── CUERPOS ────────────────────────────────────────────────────────────────────
_CUERPOS = {
    # [P1-I18N-PUSH-GUARD-CIEGO-AL-THREAD · 2026-08-23] Ver la nota en `_TITULOS`.
    "Actualiza tu nevera para renovar tu plan.": {
        "en-US": "Update your fridge to renew your plan.",
        "pt-BR": "Atualize sua geladeira para renovar seu plano.",
        "fr-FR": "Mettez votre frigo à jour pour renouveler votre plan.",
        "it-IT": "Aggiorna il tuo Frigo per rinnovare il tuo piano.",
    },
    "Necesitamos que registres tus comidas para seguir personalizando tu menú.": {
        "en-US": "We need you to log your meals to keep personalizing your menu.",
        "pt-BR": "Precisamos que você registre suas refeições para continuar personalizando seu cardápio.",
        "fr-FR": "Nous avons besoin que vous enregistriez vos repas pour continuer à personnaliser votre menu.",
        "it-IT": "Abbiamo bisogno che tu registri i tuoi pasti per continuare a personalizzare il tuo menù.",
    },
    "Actualiza 'Mi Nevera' para continuar con el siguiente bloque del plan. Si no, usaremos una opción flexible más adelante.": {
        "en-US": "Update 'My Fridge' to continue with the next block of your plan. Otherwise we'll use a flexible option later on.",
        "pt-BR": "Atualize 'Minha Geladeira' para continuar com o próximo bloco do plano. Caso contrário, usaremos uma opção flexível mais adiante.",
        "fr-FR": "Mettez à jour « Mon frigo » pour continuer avec le prochain bloc du plan. Sinon, nous utiliserons une option flexible plus tard.",
        "it-IT": "Aggiorna «Il mio Frigo» per continuare con il prossimo blocco del piano. Altrimenti useremo un'opzione flessibile più avanti.",
    },
    "Ajustamos tu plan para que coincida con tu hora local actual.": {
        "en-US": "We adjusted your plan to match your current local time.",
        "pt-BR": "Ajustamos seu plano para coincidir com seu horário local atual.",
        "fr-FR": "Nous avons ajusté votre plan pour qu'il corresponde à votre heure locale actuelle.",
        "it-IT": "Abbiamo adattato il tuo piano alla tua ora locale attuale.",
    },
    "Dejamos en pausa los próximos días de tu plan porque no pudimos detectar tu zona horaria. Abre Bioboros y se sincronizará automáticamente para reanudar la generación.": {
        "en-US": "We paused the next days of your plan because we couldn't detect your time zone. Open Bioboros and it will sync automatically to resume generation.",
        "pt-BR": "Pausamos os próximos dias do seu plano porque não conseguimos detectar seu fuso horário. Abra o Bioboros e ele sincronizará automaticamente para retomar a geração.",
        "fr-FR": "Nous avons mis en pause les prochains jours de votre plan car nous n'avons pas pu détecter votre fuseau horaire. Ouvrez Bioboros et il se synchronisera automatiquement pour reprendre la génération.",
        "it-IT": "Abbiamo messo in pausa i prossimi giorni del tuo piano perché non siamo riusciti a rilevare il tuo fuso orario. Apri Bioboros e si sincronizzerà da solo per riprendere la generazione.",
    },
    "El bloque previo no se terminó de marcar a tiempo. Generamos los próximos días con la mejor info disponible — ajústalos en el diario si hace falta.": {
        "en-US": "The previous block wasn't fully logged in time. We generated the next days with the best information available — adjust them in your diary if needed.",
        "pt-BR": "O bloco anterior não terminou de ser marcado a tempo. Geramos os próximos dias com a melhor informação disponível — ajuste no diário se precisar.",
        "fr-FR": "Le bloc précédent n'a pas été entièrement coché à temps. Nous avons généré les jours suivants avec les meilleures informations disponibles — ajustez-les dans le journal si besoin.",
        "it-IT": "Il blocco precedente non è stato completato in tempo. Abbiamo generato i giorni successivi con le migliori informazioni disponibili — modificali nel diario se serve.",
    },
    "Estamos esperando que termines los días anteriores de tu plan para generar los siguientes. Loguea tus comidas o tócalas en el diario.": {
        "en-US": "We're waiting for you to finish the earlier days of your plan before generating the next ones. Log your meals or tap them in your diary.",
        "pt-BR": "Estamos esperando você terminar os dias anteriores do plano para gerar os próximos. Registre suas refeições ou toque nelas no diário.",
        "fr-FR": "Nous attendons que vous terminiez les jours précédents de votre plan pour générer les suivants. Enregistrez vos repas ou touchez-les dans le journal.",
        "it-IT": "Stiamo aspettando che tu finisca i giorni precedenti del piano per generare i successivi. Registra i tuoi pasti o toccali nel diario.",
    },
    "Estamos generando tu próximo bloque sin info de qué comiste — márcanos lo que comes para mejorar.": {
        "en-US": "We're generating your next block without knowing what you ate — log your meals so we can do better.",
        "pt-BR": "Estamos gerando seu próximo bloco sem saber o que você comeu — marque suas refeições para melhorarmos.",
        "fr-FR": "Nous générons votre prochain bloc sans savoir ce que vous avez mangé — enregistrez vos repas pour que nous fassions mieux.",
        "it-IT": "Stiamo generando il tuo prossimo blocco senza sapere cosa hai mangiato — registra i pasti per farlo meglio.",
    },
    "Estamos generando tu siguiente bloque, pero el aprendizaje histórico tuvo un problema de datos. Si notas comidas repetidas, regenera tu plan.": {
        "en-US": "We're generating your next block, but the historical learning hit a data problem. If you notice repeated meals, regenerate your plan.",
        "pt-BR": "Estamos gerando seu próximo bloco, mas o aprendizado histórico teve um problema de dados. Se notar refeições repetidas, gere o plano de novo.",
        "fr-FR": "Nous générons votre prochain bloc, mais l'apprentissage historique a rencontré un problème de données. Si vous voyez des repas répétés, régénérez votre plan.",
        "it-IT": "Stiamo generando il tuo prossimo blocco, ma l'apprendimento storico ha avuto un problema di dati. Se noti pasti ripetuti, rigenera il piano.",
    },
    "Estamos terminando de ajustar los últimos detalles de tu plan. Estará listo en breve.": {
        "en-US": "We're finishing the last details of your plan. It'll be ready shortly.",
        "pt-BR": "Estamos terminando de ajustar os últimos detalhes do seu plano. Estará pronto em breve.",
        "fr-FR": "Nous terminons les derniers détails de votre plan. Il sera prêt sous peu.",
        "it-IT": "Stiamo rifinendo gli ultimi dettagli del tuo piano. Sarà pronto a breve.",
    },
    "No pudimos generar los próximos días con los ingredientes que tienes. Actualiza tu nevera para continuar.": {
        "en-US": "We couldn't generate the next days with the ingredients you have. Update your fridge to continue.",
        "pt-BR": "Não conseguimos gerar os próximos dias com os ingredientes que você tem. Atualize sua geladeira para continuar.",
        "fr-FR": "Nous n'avons pas pu générer les prochains jours avec les ingrédients que vous avez. Mettez votre frigo à jour pour continuer.",
        "it-IT": "Non siamo riusciti a generare i prossimi giorni con gli ingredienti che hai. Aggiorna il tuo Frigo per continuare.",
    },
    "Refresca tu nevera para continuar tu plan en tu nueva zona horaria.": {
        "en-US": "Refresh your fridge to continue your plan in your new time zone.",
        "pt-BR": "Atualize sua geladeira para continuar seu plano no novo fuso horário.",
        "fr-FR": "Actualisez votre frigo pour continuer votre plan dans votre nouveau fuseau horaire.",
        "it-IT": "Aggiorna il tuo Frigo per continuare il piano nel nuovo fuso orario.",
    },
    "Sigue guardado en tu Historial. Cuando quieras volver, genera uno nuevo — tu cuenta y tus datos están intactos.": {
        "en-US": "It's still saved in your History. Whenever you want to come back, generate a new one — your account and your data are intact.",
        "pt-BR": "Continua salvo no seu Histórico. Quando quiser voltar, gere um novo — sua conta e seus dados estão intactos.",
        "fr-FR": "Il reste enregistré dans votre Historique. Quand vous voudrez revenir, générez-en un nouveau — votre compte et vos données sont intacts.",
        "it-IT": "Resta salvato nella tua Cronologia. Quando vorrai tornare, generane uno nuovo — il tuo account e i tuoi dati sono intatti.",
    },
    "Tu Nevera está vacía, así que congelamos tu plan — tus días NO corren. Agrega tus alimentos y todo se reanuda solo.": {
        "en-US": "Your Fridge is empty, so we froze your plan — your days are NOT running. Add your food and everything resumes on its own.",
        "pt-BR": "Sua Geladeira está vazia, então congelamos seu plano — seus dias NÃO estão correndo. Adicione seus alimentos e tudo é retomado sozinho.",
        "fr-FR": "Votre frigo est vide, nous avons donc gelé votre plan — vos jours NE défilent PAS. Ajoutez vos aliments et tout reprend tout seul.",
        "it-IT": "Il tuo Frigo è vuoto, quindi abbiamo congelato il piano — i tuoi giorni NON scorrono. Aggiungi i tuoi alimenti e tutto riparte da solo.",
    },
    "Tu chunk seguía en pausa por nevera vacía. Lo reintentaremos con un plato flexible para no bloquear tu plan.": {
        "en-US": "Your block was still paused because your fridge was empty. We'll retry it with a flexible dish so your plan isn't blocked.",
        "pt-BR": "Seu bloco continuava pausado por geladeira vazia. Vamos tentar de novo com um prato flexível para não travar seu plano.",
        "fr-FR": "Votre bloc était toujours en pause parce que votre frigo était vide. Nous le réessaierons avec un plat flexible pour ne pas bloquer votre plan.",
        "it-IT": "Il tuo blocco era ancora in pausa perché il Frigo era vuoto. Riproveremo con un piatto flessibile per non bloccare il piano.",
    },
    "Tu nevera cambió mucho durante la generación. Confirma su contenido para continuar.": {
        "en-US": "Your fridge changed a lot during generation. Confirm its contents to continue.",
        "pt-BR": "Sua geladeira mudou muito durante a geração. Confirme o conteúdo para continuar.",
        "fr-FR": "Votre frigo a beaucoup changé pendant la génération. Confirmez son contenu pour continuer.",
        "it-IT": "Il tuo Frigo è cambiato molto durante la generazione. Conferma il contenuto per continuare.",
    },
    "Tu nevera necesita reposición para que el plan siga variado": {
        "en-US": "Your fridge needs restocking to keep your plan varied",
        "pt-BR": "Sua geladeira precisa de reposição para o plano continuar variado",
        "fr-FR": "Votre frigo a besoin d'être réapprovisionné pour que le plan reste varié",
        "it-IT": "Il tuo Frigo ha bisogno di rifornimento perché il piano resti vario",
    },
    "Tu nevera no se está sincronizando ahora mismo. Generamos los próximos días con la última versión disponible — revísalos cuando vuelva la sincronización.": {
        "en-US": "Your fridge isn't syncing right now. We generated the next days with the latest version available — review them when syncing is back.",
        "pt-BR": "Sua geladeira não está sincronizando agora. Geramos os próximos dias com a última versão disponível — revise quando a sincronização voltar.",
        "fr-FR": "Votre frigo ne se synchronise pas en ce moment. Nous avons généré les prochains jours avec la dernière version disponible — vérifiez-les au retour de la synchronisation.",
        "it-IT": "Il tuo Frigo non si sta sincronizzando in questo momento. Abbiamo generato i prossimi giorni con l'ultima versione disponibile — controllali quando torna la sincronizzazione.",
    },
    "Tu próximo bloque parece atrasado. Verifica que la zona horaria de tu perfil sea correcta para que podamos generarlo.": {
        "en-US": "Your next block looks behind schedule. Check that your profile's time zone is correct so we can generate it.",
        "pt-BR": "Seu próximo bloco parece atrasado. Verifique se o fuso horário do seu perfil está correto para podermos gerá-lo.",
        "fr-FR": "Votre prochain bloc semble en retard. Vérifiez que le fuseau horaire de votre profil est correct pour que nous puissions le générer.",
        "it-IT": "Il tuo prossimo blocco sembra in ritardo. Controlla che il fuso orario del profilo sia corretto così possiamo generarlo.",
    },
    "Tus restricciones actuales no permiten reusar los días previos del plan. Regenera el plan para que se adapte a tus alergias y preferencias actuales.": {
        "en-US": "Your current restrictions don't allow reusing the earlier days of your plan. Regenerate it so it matches your current allergies and preferences.",
        "pt-BR": "Suas restrições atuais não permitem reaproveitar os dias anteriores do plano. Gere o plano de novo para que se adapte às suas alergias e preferências atuais.",
        "fr-FR": "Vos restrictions actuelles ne permettent pas de réutiliser les jours précédents du plan. Régénérez-le pour qu'il corresponde à vos allergies et préférences actuelles.",
        "it-IT": "Le tue restrizioni attuali non permettono di riusare i giorni precedenti del piano. Rigeneralo perché si adatti alle tue allergie e preferenze attuali.",
    },
}

# Un solo diccionario: título y cuerpo se resuelven igual y nada impide que una cadena
# sirva de ambos. Separarlos en dos tablas obligaría a saber cuál es cuál en el punto de
# traducción, que es justo lo que NO sabe el cuello de botella.
_CATALOGO: dict = {}
_CATALOGO.update(_TITULOS)
_CATALOGO.update(_CUERPOS)


def translate_push_text(texto, locale) -> str:
    """El texto en el idioma del usuario, o el español si no hay traducción.

    Fail-open TOTAL: cualquier forma inesperada devuelve la entrada tal cual. Una
    notificación en español es una degradación; una que no sale, o que sale con una clave
    técnica, es un fallo.
    """
    if not isinstance(texto, str) or not texto:
        return texto
    if not isinstance(locale, str) or locale == "es-DO" or locale not in _LOCALES:
        return texto
    try:
        return _CATALOGO.get(texto, {}).get(locale) or texto
    except Exception:  # noqa: BLE001
        return texto


def push_catalog_keys() -> set:
    """Las claves vivas del catálogo. La usa el guard para comparar contra los call sites."""
    return set(_CATALOGO)
