// Paste this into the browser console at http://localhost:5173
// It will call the recalculate API for household sizes 1-6 and log the comparison

(async () => {
    const { data: sessionData } = await window.__supabase?.auth?.getSession?.() 
        || (await import('/src/supabaseClient.js')).supabase.auth.getSession();
    
    // Try to get supabase from the window or module
    let supabase;
    try {
        const mod = await import('/src/supabaseClient.js');
        supabase = mod.supabase || mod.default;
    } catch(e) {
        console.error('Could not import supabase client');
        return;
    }
    
    const { data } = await supabase.auth.getSession();
    const token = data?.session?.access_token;
    const userId = data?.session?.user?.id;
    
    if (!token) {
        console.error('No auth token found!');
        return;
    }
    
    console.log('User ID:', userId);
    console.log('Token obtained, testing scaling...\n');
    
    const API = 'http://localhost:3001';
    const KEYWORDS = ['pechuga', 'pavo', 'yogurt', 'lechosa', 'aguacate', 'arroz', 'pollo', 'cebolla', 'tomate'];
    
    for (const household of [1, 3, 4, 5, 6]) {
        try {
            const res = await fetch(`${API}/api/recalculate-shopping-list`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    user_id: userId,
                    householdSize: household,
                    groceryDuration: 'weekly'
                })
            });
            const result = await res.json();
            
            if (result.success && result.plan_data) {
                const items = result.plan_data.aggregated_shopping_list_weekly || [];
                console.log(`\n${'='.repeat(60)}`);
                console.log(`HOUSEHOLD: ${household} persona(s) | Total items: ${items.length}`);
                console.log(`${'='.repeat(60)}`);
                
                items.forEach(item => {
                    const name = (item.name || '').toLowerCase();
                    if (KEYWORDS.some(kw => name.includes(kw))) {
                        console.log(`  ${(item.name || '').padEnd(30)} -> ${(item.display_qty || '').padEnd(25)} | ${item.display_string}`);
                    }
                });
            } else {
                console.log(`[${household}p] Error:`, result);
            }
        } catch(e) {
            console.error(`[${household}p] Exception:`, e);
        }
    }
    
    console.log('\n✅ Done! Check backend logs for 🔬 [RAW LBS] debug output.');
})();
