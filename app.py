from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
from keras.models import load_model

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Student Performance Predictor | AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. PROJECT PATHS
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "ann_model_compatible.keras"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"

# ==========================================
# 3. CUSTOM CSS & GLASSMORPHISM STYLING
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* Global Font & Background */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 15% 15%, #1e1b4b 0%, #090d16 45%, #05080f 100%);
    color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Header Banner */
.hero-box {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.14) 0%, rgba(168, 85, 247, 0.08) 100%);
    border: 1px solid rgba(129, 140, 248, 0.25);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 2rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
}

.badge-pill {
    display: inline-block;
    background: rgba(99, 102, 241, 0.2);
    color: #a5b4fc;
    border: 1px solid rgba(129, 140, 248, 0.35);
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.main-title {
    font-size: 40px;
    font-weight: 800;
    background: linear-gradient(90deg, #ffffff 0%, #c7d2fe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
    line-height: 1.2;
}

.subtitle {
    color: #94a3b8;
    font-size: 16px;
    max-width: 650px;
    margin: 0 auto;
}

/* Glass Cards */
.glass-card {
    background: rgba(30, 41, 59, 0.45);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 18px;
    padding: 24px;
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
}

.section-header {
    font-size: 19px;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Result Showcase */
.result-card {
    background: linear-gradient(145deg, rgba(49, 46, 129, 0.65) 0%, rgba(15, 23, 42, 0.85) 100%);
    border: 1px solid rgba(99, 102, 241, 0.5);
    border-radius: 22px;
    padding: 32px 24px;
    text-align: center;
    box-shadow: 0 20px 40px -15px rgba(79, 70, 229, 0.35);
    position: relative;
    overflow: hidden;
}

.result-label {
    color: #c7d2fe;
    font-size: 14px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}

.result-score {
    font-size: 62px;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.1;
    margin: 10px 0;
    text-shadow: 0 4px 20px rgba(99, 102, 241, 0.5);
}

.score-max {
    font-size: 24px;
    color: #94a3b8;
    font-weight: 600;
}

.grade-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 700;
    margin-top: 10px;
}

/* Sidebar Tech Tags */
.tech-tag {
    display: inline-block;
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid rgba(148, 163, 184, 0.2);
    color: #e2e8f0;
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    margin: 4px 4px 4px 0;
}

/* Form Submit Button */
div.stFormSubmitButton > button {
    width: 100%;
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
    color: #ffffff;
    border: none;
    border-radius: 12px;
    padding: 14px 20px;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 0.3px;
    transition: all 0.25s ease;
    box-shadow: 0 8px 20px -6px rgba(99, 102, 241, 0.6);
}

div.stFormSubmitButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 25px -6px rgba(139, 92, 246, 0.75);
    color: #ffffff;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(148, 163, 184, 0.1);
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. LOAD MODEL AND PREPROCESSOR
# ==========================================
@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH, compile=False)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor

try:
    model, preprocessor = load_artifacts()
except Exception as e:
    st.error("⚠️ Unable to load the ANN model or preprocessor. Please verify your `/models` directory.")
    st.exception(e)
    st.stop()

# ==========================================
# 5. SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("## 🎓 Student AI")
    st.caption("Deep Learning Academic Analytics")
    st.markdown("---")
    
    st.markdown("#### 📌 About this Project")
    st.markdown(
        "<p style='color:#94a3b8; font-size:14px; line-height:1.6;'>"
        "This application leverages a trained <b>Artificial Neural Network (ANN)</b> "
        "to estimate a student's average examination score based on demographic "
        "and academic preparation indicators."
        "</p>",
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    st.markdown("#### 🛠️ Technology Stack")
    st.markdown("""
        <div>
            <span class="tech-tag">🧠 Artificial Neural Network</span>
            <span class="tech-tag">🔶 TensorFlow / Keras</span>
            <span class="tech-tag">⚙️ Scikit-learn</span>
            <span class="tech-tag">🐍 Python</span>
            <span class="tech-tag">📊 Pandas</span>
            <span class="tech-tag">🚀 Streamlit</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("💡 **Tip:** Completing a test preparation course typically has a strong positive impact on predicted scores.")

# ==========================================
# 6. HERO HEADER
# ==========================================
st.markdown("""
<div class="hero-box">
    <div class="badge-pill">Deep Learning Inference Engine</div>
    <div class="main-title">🎓 Student Performance Predictor</div>
    <div class="subtitle">
        Configure student profile parameters below to generate an instant AI-powered 
        average examination score estimation.
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 7. MAIN WORKSPACE (TWO-COLUMN LAYOUT)
# ==========================================
left_col, right_col = st.columns([1.25, 1], gap="large")

with left_col:
    st.markdown('<div class="section-header">📝 Student Profile Configuration</div>', unsafe_allow_html=True)
    
    with st.form("student_prediction_form"):
        c1, c2 = st.columns(2)
        
        with c1:
            gender = st.selectbox(
                "Gender",
                ["female", "male"],
                format_func=lambda x: x.capitalize()
            )
            
            race = st.selectbox(
                "Race / Ethnicity",
                ["group A", "group B", "group C", "group D", "group E"],
                format_func=lambda x: x.replace("group", "Group")
            )
            
            parental_education = st.selectbox(
                "Parental Level of Education",
                [
                    "some high school",
                    "high school",
                    "some college",
                    "associate's degree",
                    "bachelor's degree",
                    "master's degree"
                ],
                format_func=lambda x: x.title()
            )

        with c2:
            lunch = st.selectbox(
                "Lunch Type",
                ["standard", "free/reduced"],
                format_func=lambda x: x.title()
            )
            
            test_preparation = st.selectbox(
                "Test Preparation Course",
                ["none", "completed"],
                format_func=lambda x: "Completed" if x == "completed" else "None"
            )
            
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.caption("All 5 features are preprocessed before feeding into the ANN input layer.")

        st.markdown("---")
        submitted = st.form_submit_button("🚀 Predict Student Performance")

# ==========================================
# 8. PREDICTION & RESULTS DISPLAY
# ==========================================
with right_col:
    st.markdown('<div class="section-header">📊 AI Prediction Output</div>', unsafe_allow_html=True)

    if submitted:
        # Construct DataFrame with exact column names expected by preprocessor
        input_data = pd.DataFrame({
            "gender": [gender],
            "race/ethnicity": [race],
            "parental level of education": [parental_education],
            "lunch": [lunch],
            "test preparation course": [test_preparation]
        })

        try:
            processed_data = preprocessor.transform(input_data)
            prediction = model.predict(processed_data, verbose=0)
            predicted_score = max(0.0, min(100.0, float(prediction[0][0])))

            # Determine Performance Tier
            if predicted_score >= 85:
                tier_text = "🌟 Excellent Performance (Grade A)"
                badge_bg = "rgba(16, 185, 129, 0.2)"
                badge_color = "#6ee7b7"
                badge_border = "rgba(16, 185, 129, 0.4)"
            elif predicted_score >= 70:
                tier_text = "✅ Good Performance (Grade B)"
                badge_bg = "rgba(59, 130, 246, 0.2)"
                badge_color = "#93c5fd"
                badge_border = "rgba(59, 130, 246, 0.4)"
            elif predicted_score >= 55:
                tier_text = "📘 Average Performance (Grade C)"
                badge_bg = "rgba(245, 158, 11, 0.2)"
                badge_color = "#fcd34d"
                badge_border = "rgba(245, 158, 11, 0.4)"
            else:
                tier_text = "⚠️ Needs Academic Support"
                badge_bg = "rgba(239, 68, 68, 0.2)"
                badge_color = "#fca5a5"
                badge_border = "rgba(239, 68, 68, 0.4)"

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Estimated Average Exam Score</div>
                <div class="result-score">{predicted_score:.2f}<span class="score-max"> / 100</span></div>
                <div class="grade-badge" style="background:{badge_bg}; color:{badge_color}; border:1px solid {badge_border};">
                    {tier_text}
                </div>
                <p style="color:#94a3b8; font-size:13px; margin-top:16px; margin-bottom:0;">
                    Generated by trained TensorFlow/Keras ANN Model
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
            st.progress(predicted_score / 100.0)

            with st.expander("🔍 Inspect Encoded Input Summary", expanded=False):
                st.dataframe(input_data, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error("Prediction failed. Please verify the preprocessor and model compatibility.")
            st.exception(e)
    else:
        # Empty State Placeholder Card
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 55px 24px; border-style: dashed;">
            <div style="font-size: 42px; margin-bottom: 12px;">🧠</div>
            <div style="font-size: 18px; font-weight: 700; color: #e2e8f0; margin-bottom: 8px;">
                Ready for Inference
            </div>
            <p style="color: #94a3b8; font-size: 14px; max-width: 320px; margin: 0 auto;">
                Select the student's attributes on the left and click 
                <b>"Predict Student Performance"</b> to calculate the expected score.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 9. FOOTER
# ==========================================
st.markdown("""
<div class="footer">
    <b>Student Performance Predictor</b> • Powered by Artificial Neural Networks (TensorFlow & Streamlit)<br>
    <span style="font-size: 12px;">Disclaimer: Predictions are statistical estimates generated for educational and analytical purposes.</span>
</div>
""", unsafe_allow_html=True)