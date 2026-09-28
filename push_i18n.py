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
    # [P1-PLAN-LOTE-228 · 2026-09-25] El fin de la generación con la app cerrada (`aviso_plan_listo.py`).
    "Tu plan está listo 🎉": {
        "en-US": "Your plan is ready 🎉",
        "pt-BR": "Seu plano está pronto 🎉",
        "fr-FR": "Votre plan est prêt 🎉",
        "it-IT": "Il tuo piano è pronto 🎉",
    },
    "No pudimos terminar tu plan": {
        "en-US": "We couldn't finish your plan",
        "pt-BR": "Não conseguimos terminar seu plano",
        "fr-FR": "Nous n’avons pas pu terminer votre plan",
        "it-IT": "Non siamo riusciti a completare il tuo piano",
    },
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
    # [P1-PLAN-LOTE-228 · 2026-09-25] Ver `_TITULOS`.
    "Toca para verlo.": {
        "en-US": "Tap to see it.",
        "pt-BR": "Toque para ver.",
        "fr-FR": "Touchez pour le voir.",
        "it-IT": "Tocca per vederlo.",
    },
    "Toca para intentarlo de nuevo.": {
        "en-US": "Tap to try again.",
        "pt-BR": "Toque para tentar de novo.",
        "fr-FR": "Touchez pour réessayer.",
        "it-IT": "Tocca per riprovare.",
    },
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
# [P1-PLAN-LOTE-645 · 2026-09-27] Los 25 textos que el guard no veía: el ALIAS del import en el `target=` del
# Thread (cinco «Tu plan necesita una revisión») y los textos guardados antes en una variable. Más las cuatro
# frases completas de `_build_zero_log_push_payload`, que arma el cuerpo por partes: la traducción busca el
# texto ENTERO que sale, así que la pieza sola no traduciría nada.
_LOTE_645 = {
    'Tu plan necesita una revisión': {
        "en-US": 'Your plan needs a review',
        "pt-BR": 'Seu plano precisa de uma revisão',
        "fr-FR": 'Votre plan a besoin d’une révision',
        "it-IT": 'Il tuo piano ha bisogno di una revisione',
    },
    'Tu plan necesita atención': {
        "en-US": 'Your plan needs attention',
        "pt-BR": 'Seu plano precisa de atenção',
        "fr-FR": 'Votre plan demande votre attention',
        "it-IT": 'Il tuo piano ha bisogno di attenzione',
    },
    'Tu plan necesita regenerarse': {
        "en-US": 'Your plan needs to be regenerated',
        "pt-BR": 'Seu plano precisa ser gerado de novo',
        "fr-FR": 'Votre plan doit être régénéré',
        "it-IT": 'Il tuo piano va rigenerato',
    },
    'No pudimos generar tu plan': {
        "en-US": "We couldn't generate your plan",
        "pt-BR": 'Não conseguimos gerar seu plano',
        "fr-FR": 'Nous n’avons pas pu générer votre plan',
        "it-IT": 'Non siamo riusciti a generare il tuo piano',
    },
    'Loguea tus comidas para continuar': {
        "en-US": 'Log your meals to continue',
        "pt-BR": 'Registre suas refeições para continuar',
        "fr-FR": 'Enregistrez vos repas pour continuer',
        "it-IT": 'Registra i tuoi pasti per continuare',
    },
    'Loguea más comidas para que el plan aprenda': {
        "en-US": 'Log more meals so your plan can learn',
        "pt-BR": 'Registre mais refeições para o plano aprender',
        "fr-FR": 'Enregistrez plus de repas pour que le plan apprenne',
        "it-IT": 'Registra più pasti perché il piano impari',
    },
    'Tu plan se está generando sin tu feedback': {
        "en-US": 'Your plan is being generated without your feedback',
        "pt-BR": 'Seu plano está sendo gerado sem o seu feedback',
        "fr-FR": 'Votre plan se génère sans vos retours',
        "it-IT": 'Il tuo piano si sta generando senza il tuo feedback',
    },
    'Tu próximo bloque espera más feedback': {
        "en-US": 'Your next block is waiting for more feedback',
        "pt-BR": 'Seu próximo bloco espera mais feedback',
        "fr-FR": 'Votre prochain bloc attend plus de retours',
        "it-IT": 'Il tuo prossimo blocco aspetta più feedback',
    },
    'Pool atómico degradado: lost-updates posibles': {
        "en-US": 'Atomic pool degraded: lost updates possible',
        "pt-BR": 'Pool atômico degradado: lost-updates possíveis',
        "fr-FR": 'Pool atomique dégradé : lost-updates possibles',
        "it-IT": 'Pool atomico degradato: lost-update possibili',
    },
    'Detectamos datos inválidos en la fecha de inicio de tu plan. Tócalo para regenerarlo con tu nevera actual.': {
        "en-US": "We found invalid data in your plan's start date. Tap to regenerate it with your current fridge.",
        "pt-BR": 'Detectamos dados inválidos na data de início do seu plano. Toque para gerá-lo de novo com sua geladeira atual.',
        "fr-FR": 'Nous avons détecté des données invalides dans la date de début de votre plan. Touchez pour le régénérer avec votre frigo actuel.',
        "it-IT": 'Abbiamo rilevato dati non validi nella data di inizio del tuo piano. Tocca per rigenerarlo con il tuo frigo attuale.',
    },
    'Detectamos ingredientes que ya no están en tu nevera. Actualízala para que generemos los días siguientes.': {
        "en-US": 'Some ingredients are no longer in your fridge. Update it so we can generate the next days.',
        "pt-BR": 'Detectamos ingredientes que não estão mais na sua geladeira. Atualize-a para gerarmos os próximos dias.',
        "fr-FR": 'Certains ingrédients ne sont plus dans votre frigo. Mettez-le à jour pour que nous générions les jours suivants.',
        "it-IT": 'Alcuni ingredienti non sono più nel tuo frigo. Aggiornalo così generiamo i prossimi giorni.',
    },
    'Detectamos que tu historial reciente no tiene suficiente información para generar el siguiente bloque. Ábrelo para que lo revisemos juntos.': {
        "en-US": "Your recent history doesn't have enough information to generate the next block. Open it so we can review it together.",
        "pt-BR": 'Seu histórico recente não tem informação suficiente para gerar o próximo bloco. Abra-o para revisarmos juntos.',
        "fr-FR": 'Votre historique récent ne contient pas assez d’informations pour générer le prochain bloc. Ouvrez-le pour que nous le revoyions ensemble.',
        "it-IT": 'La tua cronologia recente non ha abbastanza informazioni per generare il prossimo blocco. Aprilo per rivederlo insieme.',
    },
    'Detectamos un problema al continuar tu plan. Ábrelo para regenerarlo con tu nevera actual.': {
        "en-US": 'We ran into a problem continuing your plan. Open it to regenerate it with your current fridge.',
        "pt-BR": 'Houve um problema ao continuar seu plano. Abra-o para gerá-lo de novo com sua geladeira atual.',
        "fr-FR": 'Un problème est survenu en poursuivant votre plan. Ouvrez-le pour le régénérer avec votre frigo actuel.',
        "it-IT": 'C’è stato un problema nel continuare il tuo piano. Aprilo per rigenerarlo con il tuo frigo attuale.',
    },
    'Detectamos un problema con el historial de tu plan. Ábrelo para que lo revisemos juntos.': {
        "en-US": "We found a problem with your plan's history. Open it so we can review it together.",
        "pt-BR": 'Detectamos um problema no histórico do seu plano. Abra-o para revisarmos juntos.',
        "fr-FR": 'Nous avons détecté un problème dans l’historique de votre plan. Ouvrez-le pour que nous le revoyions ensemble.',
        "it-IT": 'Abbiamo rilevato un problema nella cronologia del tuo piano. Aprilo per rivederlo insieme.',
    },
    'Detectamos un problema con la fecha de inicio de tu plan. Ábrelo para que lo revisemos juntos.': {
        "en-US": "We found a problem with your plan's start date. Open it so we can review it together.",
        "pt-BR": 'Detectamos um problema com a data de início do seu plano. Abra-o para revisarmos juntos.',
        "fr-FR": 'Nous avons détecté un problème avec la date de début de votre plan. Ouvrez-le pour que nous le revoyions ensemble.',
        "it-IT": 'Abbiamo rilevato un problema con la data di inizio del tuo piano. Aprilo per rivederlo insieme.',
    },
    'Detectamos un problema técnico con tu plan que impide continuar generando los próximos días. Tócalo para regenerarlo con tu nevera actual.': {
        "en-US": 'A technical problem is keeping us from generating the next days of your plan. Tap to regenerate it with your current fridge.',
        "pt-BR": 'Um problema técnico impede continuar gerando os próximos dias do seu plano. Toque para gerá-lo de novo com sua geladeira atual.',
        "fr-FR": 'Un problème technique empêche de générer les prochains jours de votre plan. Touchez pour le régénérer avec votre frigo actuel.',
        "it-IT": 'Un problema tecnico impedisce di generare i prossimi giorni del tuo piano. Tocca per rigenerarlo con il tuo frigo attuale.',
    },
    'Hubo un problema y tu plan no llegó a generarse. Abre Bioboros y vuelve a generarlo con tu nevera actual.': {
        "en-US": "Something went wrong and your plan wasn't generated. Open Bioboros and generate it again with your current fridge.",
        "pt-BR": 'Houve um problema e seu plano não foi gerado. Abra o Bioboros e gere-o de novo com sua geladeira atual.',
        "fr-FR": 'Un problème est survenu et votre plan n’a pas été généré. Ouvrez Bioboros et générez-le à nouveau avec votre frigo actuel.',
        "it-IT": 'C’è stato un problema e il tuo piano non è stato generato. Apri Bioboros e generalo di nuovo con il tuo frigo attuale.',
    },
    'Loguea las comidas que hiciste estos días — el siguiente bloque del plan se ajusta a partir de eso.': {
        "en-US": 'Log the meals you had these days — the next block of your plan adjusts to them.',
        "pt-BR": 'Registre as refeições que você fez nestes dias — o próximo bloco do plano se ajusta a partir disso.',
        "fr-FR": 'Enregistrez les repas de ces derniers jours — le prochain bloc du plan s’ajuste à partir de là.',
        "it-IT": 'Registra i pasti di questi giorni — il prossimo blocco del piano si adatta a partire da lì.',
    },
    'No hemos visto qué comiste de tu plan actual. Loguea tus comidas en el diario para que el siguiente bloque aprenda de ti.': {
        "en-US": "We haven't seen what you ate from your current plan. Log your meals in the diary so the next block learns from you.",
        "pt-BR": 'Ainda não vimos o que você comeu do seu plano atual. Registre suas refeições no diário para o próximo bloco aprender com você.',
        "fr-FR": 'Nous n’avons pas vu ce que vous avez mangé de votre plan actuel. Enregistrez vos repas dans le journal pour que le prochain bloc apprenne de vous.',
        "it-IT": 'Non abbiamo visto cosa hai mangiato del tuo piano attuale. Registra i tuoi pasti nel diario perché il prossimo blocco impari da te.',
    },
    'No pudimos completar parte de tu plan automáticamente. Abre Bioboros y regenera tu plan para que volvamos a generarlo con tu nevera actual.': {
        "en-US": "We couldn't complete part of your plan automatically. Open Bioboros and regenerate your plan so we can generate it again with your current fridge.",
        "pt-BR": 'Não conseguimos completar parte do seu plano automaticamente. Abra o Bioboros e gere seu plano de novo com sua geladeira atual.',
        "fr-FR": 'Nous n’avons pas pu compléter automatiquement une partie de votre plan. Ouvrez Bioboros et régénérez votre plan avec votre frigo actuel.',
        "it-IT": 'Non siamo riusciti a completare automaticamente parte del tuo piano. Apri Bioboros e rigenera il piano con il tuo frigo attuale.',
    },
    'No pudimos confirmar tu zona horaria, así que tu plan se pausó para no generar días desfasados. Tócalo para regenerarlo con tu nevera actual.': {
        "en-US": "We couldn't confirm your time zone, so your plan was paused to avoid generating days out of sync. Tap to regenerate it with your current fridge.",
        "pt-BR": 'Não conseguimos confirmar seu fuso horário, então seu plano foi pausado para não gerar dias defasados. Toque para gerá-lo de novo com sua geladeira atual.',
        "fr-FR": 'Nous n’avons pas pu confirmer votre fuseau horaire : votre plan a été mis en pause pour ne pas générer de jours décalés. Touchez pour le régénérer avec votre frigo actuel.',
        "it-IT": 'Non siamo riusciti a confermare il tuo fuso orario, quindi il piano è in pausa per non generare giorni sfasati. Tocca per rigenerarlo con il tuo frigo attuale.',
    },
    'Tu plan se pausó porque no pudimos reconstruir el historial de aprendizaje de los días previos. Tócalo para regenerarlo con tu nevera actual.': {
        "en-US": "Your plan was paused because we couldn't rebuild the learning history of the previous days. Tap to regenerate it with your current fridge.",
        "pt-BR": 'Seu plano foi pausado porque não conseguimos reconstruir o histórico de aprendizado dos dias anteriores. Toque para gerá-lo de novo com sua geladeira atual.',
        "fr-FR": 'Votre plan a été mis en pause car nous n’avons pas pu reconstruire l’historique d’apprentissage des jours précédents. Touchez pour le régénérer avec votre frigo actuel.',
        "it-IT": 'Il tuo piano è in pausa perché non siamo riusciti a ricostruire la cronologia di apprendimento dei giorni precedenti. Tocca per rigenerarlo con il tuo frigo attuale.',
    },
    'Un día programado de tu plan quedó fuera del rango actual. Tócalo para regenerarlo con tu nevera actual.': {
        "en-US": 'A scheduled day of your plan fell outside the current range. Tap to regenerate it with your current fridge.',
        "pt-BR": 'Um dia programado do seu plano ficou fora do intervalo atual. Toque para gerá-lo de novo com sua geladeira atual.',
        "fr-FR": 'Un jour programmé de votre plan est sorti de la plage actuelle. Touchez pour le régénérer avec votre frigo actuel.',
        "it-IT": 'Un giorno programmato del tuo piano è finito fuori dall’intervallo attuale. Tocca per rigenerarlo con il tuo frigo attuale.',
    },
    'Llevas varios bloques sin registrar comidas. Loguea en el diario para que los siguientes se ajusten a ti': {
        "en-US": "You've gone several blocks without logging meals. Log in the diary so the next ones adjust to you",
        "pt-BR": 'Você está há vários blocos sem registrar refeições. Registre no diário para os próximos se ajustarem a você',
        "fr-FR": 'Vous n’avez pas enregistré de repas depuis plusieurs blocs. Enregistrez-les dans le journal pour que les suivants s’ajustent à vous',
        "it-IT": 'Da diversi blocchi non registri i pasti. Registrali nel diario perché i prossimi si adattino a te',
    },
    'Llevas varios bloques sin registrar comidas. Loguea en el diario para que los siguientes se ajusten a ti.': {
        "en-US": "You've gone several blocks without logging meals. Log in the diary so the next ones adjust to you.",
        "pt-BR": 'Você está há vários blocos sem registrar refeições. Registre no diário para os próximos se ajustarem a você.',
        "fr-FR": 'Vous n’avez pas enregistré de repas depuis plusieurs blocs. Enregistrez-les dans le journal pour que les suivants s’ajustent à vous.',
        "it-IT": 'Da diversi blocchi non registri i pasti. Registrali nel diario perché i prossimi si adattino a te.',
    },
    "Llevas varios bloques sin registrar comidas. Loguea en el diario para que los siguientes se ajusten a ti, o elige 'Continuar sin registrar' en el banner para que generemos los próximos días con tu nevera actual.": {
        "en-US": "You've gone several blocks without logging meals. Log in the diary so the next ones adjust to you, or choose 'Continue without logging' in the banner and we'll generate the next days with your current fridge.",
        "pt-BR": "Você está há vários blocos sem registrar refeições. Registre no diário para os próximos se ajustarem a você, ou escolha 'Continuar sem registrar' no aviso para gerarmos os próximos dias com sua geladeira atual.",
        "fr-FR": 'Vous n’avez pas enregistré de repas depuis plusieurs blocs. Enregistrez-les dans le journal pour que les suivants s’ajustent à vous, ou choisissez « Continuer sans enregistrer » dans le bandeau pour que nous générions les prochains jours avec votre frigo actuel.',
        "it-IT": "Da diversi blocchi non registri i pasti. Registrali nel diario perché i prossimi si adattino a te, oppure scegli 'Continua senza registrare' nel banner e genereremo i prossimi giorni con il tuo frigo attuale.",
    },
    'Tu siguiente bloque está en pausa porque no tenemos registro de tus comidas. ': {
        "en-US": 'Your next block is paused because we have no record of your meals. ',
        "pt-BR": 'Seu próximo bloco está pausado porque não temos registro das suas refeições. ',
        "fr-FR": 'Votre prochain bloc est en pause car nous n’avons aucun enregistrement de vos repas. ',
        "it-IT": 'Il tuo prossimo blocco è in pausa perché non abbiamo registrato i tuoi pasti. ',
    },
    "Tu siguiente bloque está en pausa porque no tenemos registro de tus comidas. Abre el diario para loguear, o tap 'Continuar sin registrar' para que generemos los próximos días con tu nevera actual.": {
        "en-US": "Your next block is paused because we have no record of your meals. Open the diary to log them, or tap 'Continue without logging' and we'll generate the next days with your current fridge.",
        "pt-BR": "Seu próximo bloco está pausado porque não temos registro das suas refeições. Abra o diário para registrar, ou toque em 'Continuar sem registrar' para gerarmos os próximos dias com sua geladeira atual.",
        "fr-FR": 'Votre prochain bloc est en pause car nous n’avons aucun enregistrement de vos repas. Ouvrez le journal pour les enregistrer, ou touchez « Continuer sans enregistrer » pour que nous générions les prochains jours avec votre frigo actuel.',
        "it-IT": "Il tuo prossimo blocco è in pausa perché non abbiamo registrato i tuoi pasti. Apri il diario per registrarli, oppure tocca 'Continua senza registrare' e genereremo i prossimi giorni con il tuo frigo attuale.",
    },
    'Tu siguiente bloque está en pausa porque no tenemos registro de tus comidas. Abre el diario y loguea lo que hayas comido para que aprenda de ti.': {
        "en-US": "Your next block is paused because we have no record of your meals. Open the diary and log what you've eaten so it can learn from you.",
        "pt-BR": 'Seu próximo bloco está pausado porque não temos registro das suas refeições. Abra o diário e registre o que você comeu para ele aprender com você.',
        "fr-FR": 'Votre prochain bloc est en pause car nous n’avons aucun enregistrement de vos repas. Ouvrez le journal et enregistrez ce que vous avez mangé pour qu’il apprenne de vous.',
        "it-IT": 'Il tuo prossimo blocco è in pausa perché non abbiamo registrato i tuoi pasti. Apri il diario e registra cosa hai mangiato perché impari da te.',
    },
}

_CATALOGO: dict = {}
_CATALOGO.update(_TITULOS)
_CATALOGO.update(_CUERPOS)
_CATALOGO.update(_LOTE_645)


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
