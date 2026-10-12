ALTER TABLE consumed_meals ADD COLUMN IF NOT EXISTS nutrition jsonb;
COMMENT ON COLUMN consumed_meals.nutrition IS 'Declared label nutrients and explicit consumed servings; null means unavailable';
