# StudentAI — AI-Powered Student Performance Analyzer & Predictor

**BCA 3rd Semester 100-Mark Project**  
*An AI-Powered Academic Performance Intelligence & Predictive Analytics SaaS Platform*

---

## 1. Problem Statement
Educational institutions collect vast quantities of student academic records—attendance percentages, study hours, internal assessments, assignment marks, and previous semester scores—yet typically analyze this data retrospectively after final examinations conclude. Traditional spreadsheets or CRUD student portals cannot automatically clean messy educational datasets, identify complex non-linear patterns driving student failure, quantify individual academic risk before final exams, or translate statistical model outputs into clear academic interventions.

**StudentAI** solves this problem by combining an automated **Scikit-learn Machine Learning pipeline** with a **Multi-Provider Generative AI explanation engine** and **Supabase PostgreSQL cloud persistence**.

---

## 2. Project Objectives
1. **Ingest & Validate Real Datasets:** Accept arbitrary educational CSV files and automatically detect canonical columns (`attendance`, `study_hours`, `assignment_score`, `internal_marks`, `previous_score`, `final_score`, `result`, `course`, `semester`, `subject`, `gender`).
2. **Automate Data Cleaning:** Handle missing values, duplicate records, numeric string conversions, and domain/IQR outliers while reporting exact before-and-after counts.
3. **Deliver Interactive Visual Analytics:** Provide 10 dynamic Plotly charts and 6 real-time KPI cards computed strictly from the uploaded dataset.
4. **Train Leakage-Free Machine Learning Models:** Support both **Classification** (`Random Forest Classifier`, `Logistic Regression`) and **Regression** (`Random Forest Regressor`, `Linear Regression`) using Scikit-learn `Pipeline` and `ColumnTransformer` objects fitted strictly on the training split.
5. **Perform Academic Risk Stratification:** Classify students into `LOW RISK`, `MEDIUM RISK`, or `HIGH RISK` based on real model outputs and academic indicators with transparent justifications.
6. **Provide Grounded Generative AI Explanations:** Route structured ML predictions and dataset statistics through **OpenAI**, **Google Gemini**, or **OpenRouter** with automatic single-provider failover—ensuring the ML model is always the source of numerical predictions while Generative AI explains the results.
7. **Ensure Multi-User Cloud Persistence:** Store accounts, profiles, dataset metadata, ML analyses, predictions, AI insights, reports, and activity logs in **Supabase PostgreSQL** with Row Level Security (RLS).

---

## 3. Core Architecture

```text
USER UPLOADS DATASET (CSV)
        ↓
DATA VALIDATION & SMART COLUMN DETECTION (utils/validators.py)
        ↓
DATA CLEANING & OUTLIER CONTROL (utils/data_cleaner.py)
        ↓
EXPLORATORY DATA ANALYSIS & VISUAL ANALYTICS (pages/analytics.py)
        ↓
LEAKAGE-FREE SCIKIT-LEARN PIPELINE (ml/preprocessing.py)
        ↓
MODEL TRAINING & TEST-SET EVALUATION (ml/classification.py, ml/regression.py, ml/evaluation.py)
        ↓
STUDENT PERFORMANCE PREDICTION (ml/prediction.py)
        ↓
ACADEMIC RISK STRATIFICATION (utils/helpers.py)
        ↓
GENERATIVE AI ROUTER EXPLANATION (services/ai_router.py -> OpenAI / Gemini / OpenRouter)
        ↓
PDF & CSV EXECUTIVE REPORT GENERATION (utils/report_generator.py)
```

### Project Directory Structure
```text
student-performance-ai/
├── app.py                      # Main entry point, authentication & sidebar router
├── requirements.txt            # Clean Python dependencies
├── README.md                   # Complete documentation & BCA Viva guide
├── supabase_schema.sql         # PostgreSQL relational schema & RLS policies
├── .env.example                # Safe placeholder template for environment variables
├── .gitignore                  # Excludes .env, secrets, and cache files
├── .streamlit/
│   └── config.toml             # Theme & sidebar navigation settings
├── pages/
│   ├── dashboard.py            # 6 KPI cards + 6 core analytics charts + empty state
│   ├── upload_dataset.py       # CSV upload, smart column mapping & cleaning pipeline
│   ├── data_explorer.py        # Multi-column filtering & descriptive statistics
│   ├── analytics.py            # 10-chart deep visual analytics suite
│   ├── ml_models.py            # Classification & regression studio + Viva guide
│   ├── prediction.py           # Interactive student predictor & risk stratification
│   ├── ai_insights.py          # AI prediction explanation & 'Ask AI About Dataset'
│   ├── reports.py              # ReportLab PDF report & CSV prediction export
│   └── settings.py             # AI Router config, Supabase diagnostics & Admin view
├── services/
│   ├── supabase_service.py     # Supabase Auth & PostgreSQL REST/SDK integration
│   ├── openai_service.py       # OpenAI Chat Completions service
│   ├── gemini_service.py       # Google Gemini API service
│   ├── openrouter_service.py   # OpenRouter configurable model service
│   └── ai_router.py            # Primary -> Fallback AI router with safe degradation
├── ml/
│   ├── preprocessing.py        # Train/test split & ColumnTransformer pipeline
│   ├── classification.py       # Random Forest Classifier & Logistic Regression
│   ├── regression.py           # Linear Regression & Random Forest Regressor
│   ├── evaluation.py           # Accuracy, Precision, Recall, F1, CM, MAE, RMSE, R²
│   └── prediction.py           # Single-student inference & probability extraction
├── utils/
│   ├── validators.py           # CSV validation & smart regex column detection
│   ├── data_cleaner.py         # Missing value imputation, deduplication & IQR clipping
│   ├── helpers.py              # KPI calculator, risk engine & Viva definitions
│   └── report_generator.py     # ReportLab PDF builder & CSV exporter
├── assets/
│   └── styles.css              # Modern SaaS typography, cards & risk badges
├── data/
│   └── sample_students.csv     # Realistic 300+ student dataset for testing
└── tests/
    └── test_suite.py           # Automated end-to-end verification suite
```

---

## 4. Technology Stack

| Technology | Role in Project | Why It Was Chosen |
| :--- | :--- | :--- |
| **Python 3.10+** | Core Programming Language | Industry standard for data science, machine learning, and rapid backend integration. |
| **Streamlit** | Full-Stack Web UI Framework | Builds reactive, data-driven analytics SaaS interfaces entirely in Python with built-in session state. |
| **Pandas** | Tabular Data Processing | Provides `DataFrame` operations for CSV parsing, missing value imputation, duplicate removal, and filtering. |
| **NumPy** | Numerical Computing | Powers fast vectorized array math, statistical quantiles, and linear algebra calculations. |
| **Scikit-learn** | Machine Learning Engine | Provides standardized `Pipeline`, `ColumnTransformer`, classification/regression estimators, and evaluation metrics. |
| **Plotly** | Interactive Data Visualization | Renders responsive histograms, scatter plots, box plots, pie charts, and correlation heatmaps with tooltips. |
| **Supabase PostgreSQL** | Cloud Relational Database & Auth | Provides managed PostgreSQL persistence, JWT authentication, and Row Level Security (RLS) across restarts/deployments. |
| **OpenAI / Gemini / OpenRouter** | Generative AI Interpretation | Translates structured numerical ML outputs and feature importances into actionable natural-language recommendations. |
| **ReportLab** | PDF Report Engine | Generates downloadable multi-section executive PDF reports directly in memory (`io.BytesIO`). |
| **python-dotenv** | Configuration Security | Loads API keys and database credentials securely from `.env` without hardcoding secrets in source code. |

---

## 5. Database Architecture (Supabase PostgreSQL)
- **Primary Database:** **Supabase PostgreSQL** is the application's single persistent database. Local SQLite is **not** used anywhere in the project.
- **Why Supabase PostgreSQL:** PostgreSQL is a robust relational database. Supabase provides managed cloud PostgreSQL together with JWT authentication (`auth.users`) and Row Level Security (RLS). Because data is stored in the cloud, user accounts, uploaded dataset metadata, trained model metrics, predictions, AI insights, and reports persist across application restarts and Streamlit Community Cloud redeployments.
- **Relational Tables (`supabase_schema.sql`):**
  1. `public.profiles` — Stores user profile, institution, and role (`USER` or `ADMIN`), linked via foreign key to `auth.users(id)`.
  2. `public.datasets` — Stores dataset metadata, row counts, cleaning summary, and column mappings per user.
  3. `public.dataset_analyses` — Stores trained ML model configurations, train/test sizes, evaluation metrics, and feature importances.
  4. `public.predictions` — Stores individual student prediction inputs, predicted scores/outcomes, risk levels, and rationales.
  5. `public.ai_insights` — Stores generative AI explanations and dataset Q&A transcripts.
  6. `public.reports` — Stores generated PDF report metadata snapshots.
  7. `public.activity_logs` — Audit trail of user actions (`USER_LOGIN`, `DATASET_UPLOADED`, `MODEL_TRAINED`, `PREDICTION_GENERATED`, `REPORT_GENERATED`).
- **User Isolation:** Every table includes a `user_id UUID REFERENCES public.profiles(id)` column protected by PostgreSQL Row Level Security policies (`auth.uid() = user_id`), ensuring users can never see another user's private records.

---

## 6. Machine Learning Algorithms
1. **Random Forest Classifier (`Classification`):** An ensemble of decision trees trained via bootstrap aggregation (bagging) and random feature subsets. Captures non-linear interactions across attendance, study hours, and internal marks to classify `Pass` vs `Fail` outcomes and outputs feature importances and class probabilities.
2. **Logistic Regression (`Classification`):** A linear probabilistic classifier that models the log-odds of passing using the logistic sigmoid function. Provides well-calibrated class probabilities and interpretable feature weights.
3. **Random Forest Regressor (`Regression`):** Averages predictions across multiple decision trees to predict continuous `final_score` values (`0–100`) while minimizing variance and overfitting.
4. **Linear Regression (`Regression`):** Fits a linear hyperplane using Ordinary Least Squares (OLS) to model the direct additive effect of each academic indicator on the student's final score.

---

## 7. Generative AI Integration & AI Router
- **Separation of Concerns:** The **Scikit-learn ML model** is always the sole source of numerical predictions, probabilities, and evaluation metrics. **Generative AI** acts strictly as the interpretation layer.
- **Single-Provider Execution & Automatic Fallback (`services/ai_router.py`):**
  - On each request, `AIRouter` calls only **one** provider (the configured Primary Provider: `Gemini`, `OpenAI`, or `OpenRouter`).
  - If the primary provider fails (e.g., network timeout, quota limit, or missing key), `AIRouter` safely catches and sanitizes the error (never exposing API keys) and routes the request to the configured **Fallback Provider**.
  - If all external AI APIs are unreachable or unconfigured, `AIRouter` gracefully degrades to its built-in deterministic statistical interpreter so the ML application remains 100% functional.

---

## 8. Expected Dataset Format
You can upload any educational CSV file. Smart Column Detection automatically recognizes variations of:
- `student_id` (or `Roll_No`, `UID`)
- `attendance` (or `Attendance`, `attendance_percentage`)
- `study_hours` (or `studyHours`, `Study Hours`)
- `assignment_score` (or `Assignment Marks`, `assignments`)
- `internal_marks` (or `Internal Score`, `midterm_score`)
- `previous_score` (or `Previous Marks`, `prev_score`)
- `final_score` (or `Final Score`, `total_marks`, `percentage`)
- `result` (or `Result`, `Pass_Fail`, `status`)
- Optional categorical/temporal columns: `course`, `semester`, `subject`, `gender`, `assessment_date`.

A ready-to-use demonstration dataset is included at `data/sample_students.csv`.

---

## 9. Step-by-Step Supabase Setup
1. **Create a Supabase Project:** Go to [https://supabase.com](https://supabase.com), sign in, and click **New Project**.
2. **Obtain Supabase URL & Anon Key:** In your Supabase Dashboard, navigate to **Project Settings → API**. Copy your **Project URL** (`SUPABASE_URL`) and **anon public key** (`SUPABASE_ANON_KEY`).
3. **Configure Environment Variables:** Copy `.env.example` to `.env` and paste your `SUPABASE_URL` and `SUPABASE_ANON_KEY`.
4. **Create Database Tables & RLS Policies:** Open the **SQL Editor** in your Supabase Dashboard, paste the entire contents of `supabase_schema.sql`, and click **Run**.
5. **Configure Authentication:** In Supabase Dashboard, go to **Authentication → Providers → Email**. For quick classroom testing without SMTP setup, toggle **Confirm email** to **OFF** so newly registered accounts can log in immediately.
6. **Verify Row Level Security:** In **Authentication → Policies**, verify that RLS is enabled on `profiles`, `datasets`, `dataset_analyses`, `predictions`, `ai_insights`, `reports`, and `activity_logs`.

---

## 10. Local Setup & Running Instructions

### Step 1: Unzip and Enter the Project Directory
```bash
cd student-performance-ai
```

### Step 2: Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### Step 3: Configure `.env`
Copy `.env.example` to `.env` and fill in your credentials:
```env
SUPABASE_URL=YOUR_SUPABASE_URL
SUPABASE_ANON_KEY=YOUR_SUPABASE_ANON_KEY
OPENAI_API_KEY=YOUR_KEY
GEMINI_API_KEY=YOUR_KEY
OPENROUTER_API_KEY=YOUR_KEY
OPENROUTER_MODEL=openai/gpt-4o-mini
```

### Step 4: Run the Application
```bash
streamlit run app.py
```

---

## 11. Deployment Instructions (Streamlit Community Cloud)
1. Push your `student-performance-ai` project to a GitHub repository (ensure `.env` is excluded via `.gitignore`).
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io) and click **Create app**.
3. Select your repository, branch, and set **Main file path** to `app.py`.
4. Click **Advanced settings → Secrets** and add your environment variables in TOML format:
   ```toml
   SUPABASE_URL = "https://your-project-id.supabase.co"
   SUPABASE_ANON_KEY = "your-supabase-anon-key"
   OPENAI_API_KEY = "your-openai-key"
   GEMINI_API_KEY = "your-gemini-key"
   OPENROUTER_API_KEY = "your-openrouter-key"
   OPENROUTER_MODEL = "openai/gpt-4o-mini"
   ```
5. Click **Deploy**. Your cloud-persistent SaaS application is now live!

---

## 12. BCA Viva-Voce Questions & Answers

1. **Why Python?**  
   Python has a clear, readable syntax and the strongest ecosystem of open-source libraries for data analysis (`pandas`, `numpy`), machine learning (`scikit-learn`), interactive visualization (`plotly`), and web deployment (`streamlit`).
2. **Why Pandas?**  
   Pandas provides the `DataFrame` structure, enabling fast CSV ingestion, missing-value detection/imputation, duplicate removal, data type conversion, and relational slicing.
3. **Why NumPy?**  
   NumPy provides contiguous C-optimized multidimensional arrays and vectorized mathematical operations (means, standard deviations, quantiles, matrix operations) used underneath Pandas and Scikit-learn.
4. **Why Scikit-learn?**  
   Scikit-learn provides a consistent, modular API (`fit`, `transform`, `predict`) with `Pipeline` and `ColumnTransformer` objects that prevent data leakage during train/test evaluation.
5. **Why Random Forest?**  
   Random Forest is an ensemble learning method that builds multiple decision trees on bootstrap samples and averages their predictions, capturing non-linear relationships while resisting overfitting and providing feature importance scores.
6. **Why Logistic Regression?**  
   Logistic Regression models binary classification probabilities (`Pass` vs `Fail`) using a linear combination of features passed through the logistic sigmoid function, serving as a strong, interpretable baseline.
7. **Why Linear Regression?**  
   Linear Regression estimates continuous numeric outcomes (`final_score`) by minimizing the sum of squared residuals between actual and predicted marks.
8. **What is Train/Test Split?**  
   Dividing the dataset into two mutually exclusive subsets—typically 80% for training the model parameters and 20% held back as unseen test data to objectively measure generalization performance.
9. **What is Overfitting?**  
   Overfitting occurs when a model memorizes noise and specific patterns in the training set (achieving near-100% training score) but fails to generalize to unseen test data.
10. **What is Accuracy?**  
    The ratio of correctly predicted instances (both True Positives and True Negatives) to the total number of instances in the test set: $\frac{TP + TN}{TP + TN + FP + FN}$.
11. **What is Precision?**  
    The proportion of positive predictions that were actually correct: $\frac{TP}{TP + FP}$. High precision means few false alarms.
12. **What is Recall?**  
    The proportion of actual positive cases that the model successfully identified: $\frac{TP}{TP + FN}$.
13. **What is F1 Score?**  
    The harmonic mean of Precision and Recall: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$, providing a balanced metric even when classes are imbalanced.
14. **What is MAE (Mean Absolute Error)?**  
    The average of the absolute differences between actual values and predicted values: $\frac{1}{n}\sum |y_i - \hat{y}_i|$.
15. **What is RMSE (Root Mean Squared Error)?**  
    The square root of the average squared differences between actual and predicted values: $\sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}$. Because errors are squared before averaging, RMSE penalizes large errors more heavily than MAE.
16. **What is $R^2$ (Coefficient of Determination)?**  
    A statistical measure between $-\infty$ and $1.0$ representing the proportion of variance in the dependent target variable that is predictable from the independent features. An $R^2$ of $0.85$ means 85% of the variation in scores is explained by the model.
17. **Why use Generative AI in this project?**  
    While Scikit-learn outputs raw numbers (e.g., predicted score `46.2`, feature importance `0.34`), Generative AI translates those numbers into human-readable academic explanations, strengths, and personalized study recommendations for students and faculty.
18. **What is an API?**  
    An Application Programming Interface (API) is a structured communication contract that allows our Python application to send requests to external cloud services (Supabase PostgreSQL, OpenAI, Google Gemini, OpenRouter) over HTTPS and receive JSON responses.
19. **Why use Supabase PostgreSQL?**  
    PostgreSQL is a relational database. Supabase provides managed PostgreSQL together with backend authentication services. Cloud persistence makes it suitable for deployed multi-user applications so data remains available after application restarts and redeployments, making it far more suitable than a local SQLite file.
20. **How is the dataset processed?**  
    The uploaded CSV is validated for structure and size, inspected for missing values and duplicates, cleaned via whitespace stripping, duplicate removal, numeric parsing, median/mode imputation, and IQR outlier clipping, and then mapped to canonical academic roles.
21. **How does prediction work?**  
    User inputs are placed into a single-row DataFrame matching the training schema, transformed through the fitted `ColumnTransformer` (imputing and scaling using training-set statistics), and passed to the trained estimator's `predict` (and `predict_proba`) method.
22. **How does AI differ from ML in this project?**  
    Machine Learning (`Scikit-learn`) performs mathematical training and numerical inference on tabular student records. Generative AI (`OpenAI` / `Gemini` / `OpenRouter`) never computes or invents numerical predictions; it receives the structured ML output and generates natural-language explanations and academic recommendations.

---

## 13. Screenshots Placeholder
*(Insert screenshots of your running application here before printing your project report)*
- `[Screenshot 1: StudentAI Login & Registration Interface]`
- `[Screenshot 2: Dataset Upload, Smart Column Mapping & Data Cleaning Summary]`
- `[Screenshot 3: Academic Intelligence Dashboard & KPI Cards]`
- `[Screenshot 4: 10-Chart Visual Analytics Suite]`
- `[Screenshot 5: ML Model Training, Confusion Matrix & Feature Importance]`
- `[Screenshot 6: Student Performance Predictor & Risk Stratification]`
- `[Screenshot 7: Generative AI Insights & Ask AI About Dataset]`
- `[Screenshot 8: Generated PDF Report Preview]`

---

## 14. Future Scope & Limitations
- **Future Scope:**
  - Semester-over-semester longitudinal student cohort tracking across multiple academic years.
  - Automated email alerts to faculty advisors when a batch upload flags `HIGH RISK` students.
  - Integration with institutional ERP/LMS APIs (such as Moodle or Google Classroom).
- **Known Limitations:**
  - Model accuracy depends on the quality and sample size of the uploaded CSV dataset (minimum 12+ valid rows required for train/test splitting; 100+ rows recommended).
  - Live cloud database persistence requires valid `SUPABASE_URL` and `SUPABASE_ANON_KEY` credentials in `.env` or Streamlit Secrets.
