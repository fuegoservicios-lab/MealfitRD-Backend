-- MIGRATION: Distributed Lock para Shopping List
-- Ejecuta este script en el SQL Editor de tu Dashboard de Supabase.

CREATE TABLE IF NOT EXISTS shopping_locks (
    user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    locked_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Habilitar RLS (Row Level Security) para mayor seguridad
ALTER TABLE shopping_locks ENABLE ROW LEVEL SECURITY;

-- Políticas de seguridad: Permitir a los usuarios y al backend manejar sus propios locks
CREATE POLICY "Allow users to read their own locks" ON shopping_locks FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Allow users to insert their own locks" ON shopping_locks FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Allow users to delete their own locks" ON shopping_locks FOR DELETE USING (auth.uid() = user_id);
CREATE POLICY "Allow users to update their own locks" ON shopping_locks FOR UPDATE USING (auth.uid() = user_id);

-- En caso de usar una Service Role Key desde el backend (que salta RLS), 
-- las políticas anteriores son buena práctica aunque no obligatorias si todo se maneja desde el backend con Service Key.
