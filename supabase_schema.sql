
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. PROFILES TABLE (Linked to Supabase Auth auth.users)
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT UNIQUE NOT NULL,
    full_name TEXT,
    role TEXT NOT NULL DEFAULT 'USER' CHECK (role IN ('USER', 'ADMIN')),
    institution TEXT DEFAULT 'BCA Department',
    preferred_ai_provider TEXT DEFAULT 'gemini',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. DATASETS TABLE (Stores uploaded dataset metadata per user)
CREATE TABLE IF NOT EXISTS public.datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    raw_rows INTEGER NOT NULL CHECK (raw_rows >= 0),
    cleaned_rows INTEGER NOT NULL CHECK (cleaned_rows >= 0),
    column_count INTEGER NOT NULL CHECK (column_count > 0),
    missing_values_count INTEGER NOT NULL DEFAULT 0,
    duplicates_removed INTEGER NOT NULL DEFAULT 0,
    numeric_columns JSONB NOT NULL DEFAULT '[]'::jsonb,
    categorical_columns JSONB NOT NULL DEFAULT '[]'::jsonb,
    detected_mapping JSONB NOT NULL DEFAULT '{}'::jsonb,
    summary_stats JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. DATASET ANALYSES & ML RUNS TABLE
CREATE TABLE IF NOT EXISTS public.dataset_analyses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    dataset_id UUID REFERENCES public.datasets(id) ON DELETE SET NULL,
    dataset_name TEXT NOT NULL,
    task_type TEXT NOT NULL CHECK (task_type IN ('classification', 'regression')),
    target_column TEXT NOT NULL,
    feature_columns JSONB NOT NULL DEFAULT '[]'::jsonb,
    model_name TEXT NOT NULL,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    feature_importance JSONB NOT NULL DEFAULT '[]'::jsonb,
    train_size INTEGER NOT NULL,
    test_size INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 4. PREDICTIONS HISTORY TABLE
CREATE TABLE IF NOT EXISTS public.predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    dataset_id UUID REFERENCES public.datasets(id) ON DELETE SET NULL,
    model_name TEXT NOT NULL,
    task_type TEXT NOT NULL CHECK (task_type IN ('classification', 'regression')),
    input_features JSONB NOT NULL,
    predicted_score NUMERIC,
    predicted_label TEXT NOT NULL,
    performance_category TEXT NOT NULL,
    risk_level TEXT NOT NULL CHECK (risk_level IN ('LOW RISK', 'MEDIUM RISK', 'HIGH RISK')),
    risk_reasons JSONB NOT NULL DEFAULT '[]'::jsonb,
    confidence_score NUMERIC,
    important_factors JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 5. AI INSIGHTS HISTORY TABLE
CREATE TABLE IF NOT EXISTS public.ai_insights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    prediction_id UUID REFERENCES public.predictions(id) ON DELETE SET NULL,
    dataset_id UUID REFERENCES public.datasets(id) ON DELETE SET NULL,
    insight_type TEXT NOT NULL CHECK (insight_type IN ('prediction_explanation', 'dataset_chat', 'executive_summary')),
    provider_used TEXT NOT NULL,
    model_used TEXT,
    prompt_context JSONB NOT NULL DEFAULT '{}'::jsonb,
    response_content JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 6. REPORTS METADATA TABLE
CREATE TABLE IF NOT EXISTS public.reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    dataset_id UUID REFERENCES public.datasets(id) ON DELETE SET NULL,
    prediction_id UUID REFERENCES public.predictions(id) ON DELETE SET NULL,
    report_title TEXT NOT NULL,
    model_used TEXT,
    risk_level TEXT,
    summary_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 7. ACTIVITY LOGS TABLE
CREATE TABLE IF NOT EXISTS public.activity_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE,
    action TEXT NOT NULL,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- INDEXES FOR PERFORMANCE
-- =============================================================================
CREATE INDEX IF NOT EXISTS idx_datasets_user_id ON public.datasets(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analyses_user_id ON public.dataset_analyses(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_user_id ON public.predictions(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ai_insights_user_id ON public.ai_insights(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reports_user_id ON public.reports(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_activity_logs_user_id ON public.activity_logs(user_id, created_at DESC);

-- =============================================================================
-- AUTOMATIC PROFILE CREATION TRIGGER ON SUPABASE AUTH SIGNUP
-- =============================================================================
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (id, email, full_name, role)
    VALUES (
        NEW.id,
        NEW.email,
        COALESCE(NEW.raw_user_meta_data->>'full_name', split_part(NEW.email, '@', 1)),
        COALESCE(NEW.raw_user_meta_data->>'role', 'USER')
    )
    ON CONFLICT (id) DO NOTHING;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- =============================================================================
-- ROW LEVEL SECURITY (RLS) POLICIES FOR STRICT USER ISOLATION
-- =============================================================================
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.datasets ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.dataset_analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ai_insights ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.activity_logs ENABLE ROW LEVEL SECURITY;

-- Profiles: Users can read/update their own profile; Admins can view all profiles
CREATE POLICY "Users can view own profile" ON public.profiles
    FOR SELECT USING (auth.uid() = id OR EXISTS (
        SELECT 1 FROM public.profiles p WHERE p.id = auth.uid() AND p.role = 'ADMIN'
    ));

CREATE POLICY "Users can insert own profile" ON public.profiles
    FOR INSERT WITH CHECK (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON public.profiles
    FOR UPDATE USING (auth.uid() = id);

-- Datasets: Strict user ownership isolation
CREATE POLICY "Users manage own datasets" ON public.datasets
    FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

-- Dataset Analyses: Strict user ownership isolation
CREATE POLICY "Users manage own analyses" ON public.dataset_analyses
    FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

-- Predictions: Strict user ownership isolation
CREATE POLICY "Users manage own predictions" ON public.predictions
    FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

-- AI Insights: Strict user ownership isolation
CREATE POLICY "Users manage own ai_insights" ON public.ai_insights
    FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

-- Reports: Strict user ownership isolation
CREATE POLICY "Users manage own reports" ON public.reports
    FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

-- Activity Logs: Users view/insert own logs; Admins can view all logs
CREATE POLICY "Users insert own activity_logs" ON public.activity_logs
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users view own activity_logs" ON public.activity_logs
    FOR SELECT USING (auth.uid() = user_id OR EXISTS (
        SELECT 1 FROM public.profiles p WHERE p.id = auth.uid() AND p.role = 'ADMIN'
    ));
