import streamlit as st
import joblib
import numpy as np
import time

# ================= 1. PAGE CONFIGURATION =================
st.set_page_config(
    page_title="AutoJudge",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= 2. CUSTOM CSS (MODERN DARK THEME) =================
st.markdown("""
<style>
    /* Main Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at top left, #1b202b, #0e1117);
        color: #e6e6e6;
    }
    
    /* Dark Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #11151c;
        border-right: 1px solid #1f2937;
    }

    /* Input Fields (Glassmorphism) */
    .stTextArea textarea {
        background-color: rgba(255, 255, 255, 0.03);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        font-size: 14px;
        transition: all 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #4f8bf9;
        box-shadow: 0 0 15px rgba(79, 139, 249, 0.2);
        background-color: rgba(255, 255, 255, 0.05);
    }
    .stTextArea label {
        color: #8b949e !important;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Primary Button */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 14px 28px;
        font-size: 16px;
        font-weight: 600;
        border-radius: 10px;
        width: 100%;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5);
    }
    
    /* Result Card Styling */
    .result-box {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        backdrop-filter: blur(20px);
        box-shadow: 0 20px 50px -12px rgba(0, 0, 0, 0.5);
        animation: fadeIn 0.6s ease-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Sidebar Step Cards */
    .step-card {
        padding: 12px;
        margin-bottom: 12px;
        border-left: 3px solid #3b82f6;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 0 8px 8px 0;
    }
    .step-title { font-weight: 700; font-size: 14px; color: #fff; margin: 0; }
    .step-desc { font-size: 12px; color: #9ca3af; margin: 4px 0 0 0; }
</style>
""", unsafe_allow_html=True)

# ================= 3. MODEL LOADING & LOGIC =================

@st.cache_resource
def load_models():
    tfidf = joblib.load("data/processed/tfidf.pkl")
    clf = joblib.load("data/processed/classifier.pkl")
    reg = joblib.load("data/processed/regressor.pkl")
    return tfidf, clf, reg

tfidf, clf, reg = load_models()

# FULL Keyword List (Must match features.py)
ALGO_KEYWORDS = [
    "dp", "dynamic programming", "graph", "tree", "dfs", "bfs",
    "shortest path", "dijkstra", "bellman", "segment tree", "fenwick",
    "binary search", "two pointers", "sliding window", "subarray", 
    "subsequence", "subcontiguous", "greedy", "backtracking", "recursion",
    "bitmask", "bit manipulation", "heap", "priority queue", 
    "union find", "disjoint set", "topological", "math", "number theory",
    "modulo", "combinatorics"
]

def make_features(text):
    text = text.lower()
    tf = tfidf.transform([text]).toarray()
    extra = np.array([[
        len(text), 
        len(text.split()), 
        text.count("\n"),
        sum(text.count(c) for c in "+-*/%=<>"),
        sum(k in text for k in ALGO_KEYWORDS)
    ]])
    return np.hstack([tf, extra])

def map_score_by_class(raw_score, difficulty):
    norm = min(1, max(0, (raw_score - 3) / 3))
    if difficulty == "easy": return round(1 + norm * 3, 2)
    elif difficulty == "medium": return round(4 + norm * 3, 2)
    else: return round(7 + norm * 3, 2)

# ================= 4. SIDEBAR LAYOUT =================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=60)
    st.markdown("<h2 style='margin-top: -10px;'>AutoJudge</h2>", unsafe_allow_html=True)
    
    st.markdown("### Workflow")
    
    # Simple Steps
    st.markdown("""
<div class="step-card">
<p class="step-title">1. Input</p>
<p class="step-desc">Paste problem details.</p>
</div>
<div class="step-card">
<p class="step-title">2. Analyze</p>
<p class="step-desc">AI extracts patterns.</p>
</div>
<div class="step-card">
<p class="step-title">3. Predict</p>
<p class="step-desc">Get class & score.</p>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("v2.3 • Production Ready")

# ================= 5. MAIN INTERFACE =================
st.markdown("""
<div style='text-align: center; margin-bottom: 40px;'>
    <h1 style='font-size: 3.5rem; background: -webkit-linear-gradient(#fff, #999); -webkit-background-clip: text; margin-bottom: 0;'>⚡ AutoJudge</h1>
    <p style='color: #9ca3af; font-size: 1.2rem; margin-top: 10px;'>
        Instant ML-based difficulty prediction for coding problems.
    </p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3, gap="medium")
with c1: desc = st.text_area("Problem Description", height=200, placeholder="Paste the problem story here...")
with c2: inp = st.text_area("Input Format", height=200, placeholder="e.g. First line contains T...")
with c3: out = st.text_area("Output Format", height=200, placeholder="e.g. Print the maximum sum...")

st.markdown("<br>", unsafe_allow_html=True)

_, col_btn, _ = st.columns([1, 1, 1])
with col_btn:
    run = st.button("RUN ANALYSIS 🚀")

# ================= 6. PREDICTION LOGIC =================
if run:
    full_text = f"{desc}\n{inp}\n{out}"
    
    if len(full_text.strip()) < 20:
        st.toast("⚠️ Input is too short to analyze.", icon="⚠️")
    else:
        with st.spinner('🔍 Analyzing algorithms and complexity...'):
            time.sleep(0.6)
            
            # Feature & Prediction
            X = make_features(full_text)
            c_id = clf.predict(X)[0]
            diff = ["easy", "medium", "hard"][c_id]
            score = map_score_by_class(reg.predict(X)[0], diff)

        # Color Mapping
        diff_color_map = {
            "easy": "#4ade80",
            "medium": "#facc15",
            "hard": "#f87171"
        }
        accent_color = diff_color_map[diff]
        
        # FINAL RESULT HTML (Clean, No Confidence Badge)
        result_html = f"""
<div style="margin-top: 20px;"></div>
<div class="result-box">
<p style="color: #8b949e; text-transform: uppercase; letter-spacing: 2px; font-size: 12px; margin-bottom: 15px;">
    Predicted Difficulty
</p>
<h1 style="color: {accent_color}; font-size: 56px; margin: 0; text-transform: capitalize; letter-spacing: -1px;">
    {diff}
</h1>

<div style="margin-top: 30px; display: inline-block; background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05)); padding: 15px 35px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.1);">
    <div style="font-size: 14px; color: #9ca3af; margin-bottom: 5px;">DIFFICULTY SCORE</div>
    <span style="font-size: 36px; font-weight: 800; color: white;">{score}</span>
    <span style="font-size: 20px; color: #9ca3af; font-weight: 500;"> / 10</span>
</div>

<p style="color: #666; font-size: 12px; margin-top: 25px; font-style: italic;">
    * Note: Predictions are based on textual analysis and may vary from human judgment.
</p>
</div>
"""
        st.markdown(result_html, unsafe_allow_html=True)