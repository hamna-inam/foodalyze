#!/usr/bin/env python3
"""
Foodalyze - CalorieMama-Style Multi-Page App
Modern, Polished Design with Multiple Pages
"""

import streamlit as st
import requests
from PIL import Image
import json
from datetime import datetime

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Foodalyze - AI Nutrition Tracker",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state FIRST before anything else
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"
if "detection_result" not in st.session_state:
    st.session_state.detection_result = None
if "qa_result" not in st.session_state:
    st.session_state.qa_result = None
if "current_question" not in st.session_state:
    st.session_state.current_question = ""
if "meal_history" not in st.session_state:
    st.session_state.meal_history = []
if "daily_stats" not in st.session_state:
    st.session_state.daily_stats = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}

# ============================================================================
# MODERN STYLING - CALORIEMAMA INSPIRED (FIXES APPLIED HERE)
# ============================================================================
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #003d99 0%, #002966 100%);
        font-size: 10px;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
        font-size: 18px !important; 

    }

    .sidebar-title {
        font-size: 40px;
        font-weight: 700;
        color: #fff;
        margin-bottom: 2rem;
        text-align: center;
        letter-spacing: -0.5px;
    }
    
    .nav-button {
        background: rgba(255, 255, 255, 0.1);
        color: #fff;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.75rem;
        border: none;
        cursor: pointer;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        text-align: left;
        font-size: 10px; 
    }
    
    .nav-button:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateX(4px);
    }
    
    .nav-button.active {
        background: linear-gradient(90deg, #0052cc 0%, #003d99 100%);
        box-shadow: 0 4px 15px rgba(0, 82, 204, 0.3);
    }
    
    /* Main container */
    .main {
        max-width: 1400px;
        margin: 0 auto;
        padding: 0 !important;
    }
    
    /* Header */
    .header-gradient {
        background: linear-gradient(135deg, #003d99 0%, #002966 100%);
        padding: 3rem 2rem;
        color: white;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 10px 30px rgba(0, 61, 153, 0.2);
        margin-bottom: 2rem;
    }
    
    .header-title {
        font-size: 56px;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
        letter-spacing: -1px;
    }
    
    .header-subtitle {
        font-size: 18px;
        opacity: 0.95;
        font-weight: 400;
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        margin-bottom: 1.5rem;
    }
    
    .card p {
        font-size: 18px !important;
        line-height: 1.8 !important;
    }
    
    .card:hover {
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.12);
        transform: translateY(-4px);
    }
    
    .card-title {
        font-size: 24px;
        font-weight: 700;
        color: #2d3436;
        margin: 0 0 1.5rem 0;
    }
    
    /* Metrics Grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .metric {
        background: linear-gradient(135deg, #003d99 0%, #0052cc 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 61, 153, 0.2);
    }
    
    .metric-value {
        font-size: 36px;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 14px;
        opacity: 0.9;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Upload Zone */
    .upload-box {
        border: 3px dashed #003d99;
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        background: linear-gradient(135deg, #003d9908 0%, #005ccc08 100%);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .upload-box:hover {
        border-color: #002966;
        background: linear-gradient(135deg, #003d9915 0%, #005ccc15 100%);
    }
    
    .upload-icon {
        font-size: 64px;
        margin-bottom: 1rem;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .upload-text {
        font-size: 18px;
        font-weight: 600;
        color: #2d3436;
        margin-bottom: 0.5rem;
    }
    
    .upload-subtext {
        font-size: 16px;
        color: #666;
    }
    
    /* Detection Results */
    .detection-card {
        background: linear-gradient(135deg, #e8f0fe 0%, #f0f4ff 100%);
        border-left: 5px solid #003d99;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .detection-card:hover {
        border-left-color: #002966;
        box-shadow: 0 4px 15px rgba(0, 61, 153, 0.15);
    }
    
    .food-name {
        font-size: 18px;
        font-weight: 700;
        color: #2d3436;
        margin: 0 0 0.75rem 0;
        text-transform: capitalize;
    }
    
    .food-info {
    /* FIX: Switched from grid to flexbox to make badges sit closely together */
    display: flex; 
    gap: 1rem; /* Use the gap property to maintain consistent spacing between badges */
    align-items: center; /* Keep badges vertically aligned */
    font-size: 14px;
    }
    
    .info-badge {
        background: white;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        border-left: 3px solid #667eea;
        font-weight: 600;
        font-size: 16px;
        color: #2d3436;
    }
    
    .calorie-badge {
        background: linear-gradient(135deg, #1a73e8 0%, #1f9ef5 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: 700;
        font-size: 16px;
        display: inline-block;
    }
    
    .confidence-badge {
        background: linear-gradient(135deg, #34a853 0%, #4db8ff 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: 700;
        font-size: 16px;
        display: inline-block;
    }
    
    /* Q&A Styling */
    .qa-answer {
        background: linear-gradient(135deg, #e8f0fe 0%, #f0f4ff 100%);
        border-left: 5px solid #003d99;
        padding: 2rem;
        border-radius: 12px;
        line-height: 1.9;
        color: #2d3436;
        font-size: 17px;
    }
    
    .sample-q {
        background: white;
        border: 2px solid #e0e0e0;
        color: #003d99;
        padding: 1.25rem;
        border-radius: 10px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-bottom: 0.75rem;
        width: 100%;
        text-align: left;
        font-size: 16px;
    }
    [data-testid="stSidebarContent"] ~ div .card .stButton button > div > span {
    padding-top: 1.25rem !important;
    padding-bottom: 1.25rem !important;
}
    .sample-q:hover {
        border-color: #003d99;
        background: #f8f9fa;
        transform: translateX(4px);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #003d99 0%, #0052cc 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.875rem 2rem !important;
        font-weight: 700 !important;
        font-size: 30px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 61, 153, 0.3) !important;
        min-height: 48px !important;
    }
    
    .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(0, 61, 153, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Text inputs (FIXED: Larger font for Q&A input) */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border: 2px solid #e0e0e0 !important;
        border-radius: 10px !important;
        padding: 0.75rem !important;
        /* Increased font size for better visibility */
        font-size: 18px !important; 
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Stats section */
    .stats-box {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .stat-row:last-child {
        border-bottom: none;
    }
    
    .stat-label {
        font-weight: 600;
        color: #666;
        font-size: 14px;
    }
    
    .stat-value {
        font-size: 18px;
        font-weight: 700;
        color: #003d99;
    }
    
    /* Home page styles */
    .home-hero {
        background: linear-gradient(135deg, #003d99 0%, #0052cc 100%);
        color: white;
        padding: 4rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .home-hero h1 {
        font-size: 56px;
        font-weight: 800;
        margin-bottom: 1rem;
        letter-spacing: -1px;
    }
    
    .home-hero p {
        font-size: 24px;
        opacity: 0.95;
        max-width: 600px;
        margin: 0 auto 2rem;
        line-height: 1.8;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 2rem;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 61, 153, 0.1);
        text-align: center;
        transition: all 0.3s ease;
        border-top: 4px solid #003d99;
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 30px rgba(26, 115, 232, 0.2);
    }
    
    .feature-icon {
        font-size: 48px;
        margin-bottom: 1rem;
    }
    
    .feature-card h3 {
        font-size: 20px;
        font-weight: 700;
        color: #2d3436;
        margin-bottom: 0.75rem;
    }
    
    .feature-card p {
        color: #666;
        font-size: 18px;
        line-height: 1.8;
    }
    
    .cta-button {
        background: linear-gradient(135deg, #1a73e8 0%, #1f9ef5 100%);
        color: white;
        padding: 1rem 2.5rem;
        border-radius: 10px;
        font-weight: 700;
        font-size: 16px;
        text-decoration: none;
        display: inline-block;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(26, 115, 232, 0.3);
    }
    
    .cta-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(26, 115, 232, 0.4);
    }
    
    /* Hide streamlit elements */
    #MainMenu {display: none;}
    footer {display: none;}
    header {display: none;}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: transparent;
        border-bottom: 2px solid #e0e0e0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border-bottom: 4px solid transparent !important;
        color: #666 !important;
        font-weight: 700 !important;
        font-size: 15px !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #667eea !important;
        border-bottom-color: #667eea !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
API_BASE_URL = "http://13.61.104.51:8000"
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"
ASK_ENDPOINT = f"{API_BASE_URL}/ask"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

# ============================================================================
# SESSION STATE
# ============================================================================
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "detection_result" not in st.session_state:
    st.session_state.detection_result = None
if "meal_history" not in st.session_state:
    st.session_state.meal_history = []
if "daily_stats" not in st.session_state:
    st.session_state.daily_stats = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">🍎 Foodalyze</div>', unsafe_allow_html=True)
    
    pages = ["🏠 Home", "📸 Scanner", "📊 Dashboard", "💬 Nutrition AI", "📚 Food Database", "⚙️ Settings"]
    
    for page in pages:
        if st.button(page, use_container_width=True, key=f"nav_{page}"):
            st.session_state.page = page
    
    st.divider()
    
    with st.expander("📊 Today's Summary"):
        today_cal = st.session_state.daily_stats["calories"]
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
            <p style="margin: 0; font-size: 13px; color: #666;">Total Calories</p>
            <p style="margin: 0; font-size: 20px; font-weight: 700; color: #003d99;">{today_cal:.0f} cal</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
            <p style="margin: 0; font-size: 13px; color: #666;">Protein</p>
            <p style="margin: 0; font-size: 20px; font-weight: 700; color: #003d99;">{st.session_state.daily_stats['protein']:.0f}g</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
            <p style="margin: 0; font-size: 13px; color: #666;">Carbs</p>
            <p style="margin: 0; font-size: 20px; font-weight: 700; color: #003d99;">{st.session_state.daily_stats['carbs']:.0f}g</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px;">
            <p style="margin: 0; font-size: 13px; color: #666;">Fat</p>
            <p style="margin: 0; font-size: 20px; font-weight: 700; color: #003d99;">{st.session_state.daily_stats['fat']:.0f}g</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE: HOME
# ============================================================================
if st.session_state.page == "🏠 Home":
    st.markdown("""
    <div class="home-hero">
        <h1>🍎 Foodalyze</h1>
        <p>Your AI-Powered Nutrition & Food Analysis Companion</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">📸</div>
            <h3>Smart Food Scanner</h3>
            <p>Take a photo of your meal and instantly get calorie counts and nutritional breakdown powered by AI</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3>Track Progress</h3>
            <p>Monitor your daily macros, calories, and nutrition goals with an intuitive dashboard</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">💬</div>
            <h3>AI Nutritionist</h3>
            <p>Ask personalized nutrition questions and get expert advice powered by advanced AI</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📚</div>
            <h3>Food Database</h3>
            <p>Browse thousands of foods with complete nutritional information at your fingertips</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("")
    with col2:
        if st.button(" Get Started", use_container_width=True):
            st.session_state.page = "📸 Scanner"
            st.rerun()
    with col3:
        st.markdown("")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
            <h2 class="card-title">Why Foodalyze?</h2>
            <p><strong> Instant Analysis</strong> - Get nutrition facts in seconds</p>
            <p><strong> AI-Powered</strong> - Advanced computer vision & LLM technology</p>
            <p><strong> Easy to Use</strong> - Simple, intuitive interface</p>
            <p><strong> Comprehensive</strong> - Track calories, macros, and micronutrients</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h2 class="card-title">How It Works</h2>
            <p><strong>1. Capture</strong> - Take a photo of your meal</p>
            <p><strong>2. Analyze</strong> - AI identifies all food items</p>
            <p><strong>3. Track</strong> - Log to your daily nutrition</p>
            <p><strong>4. Learn</strong> - Get insights and recommendations</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE: SCANNER
# ============================================================================
elif st.session_state.page == "📸 Scanner":
    st.markdown("""
    <div class="header-gradient">
        <h1 class="header-title">📸 Meal Scanner</h1>
        <p class="header-subtitle">Point your camera at food and instantly analyze its nutrition</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        
        st.markdown('<h2 class="card-title">Upload Your Meal</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Select a photo",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            key="meal_upload"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            # Deprecation fix: Changed use_column_width to use_container_width
            st.image(image, use_container_width=True, caption="Your meal")
            
            if st.button("🔍 Analyze Meal", use_container_width=True, key="analyze"):
                with st.spinner("🔄 Analyzing your meal..."):
                    try:
                        response = requests.post(
                            PREDICT_ENDPOINT,
                            files={"file": ("image.jpg", uploaded_file.getvalue())},
                            timeout=60 # Increased timeout for robustness
                        )
                        if response.status_code == 200:
                            st.session_state.detection_result = response.json()
                            st.success("✅ Analysis complete!")
                        else:
                            st.error(f"❌ Analysis failed. Status code: {response.status_code}. Response: {response.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Connection Error: Could not connect to the analysis backend. Please ensure the service is running at http://localhost:8000.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.markdown("""
            <div class="upload-box">
                <div class="upload-icon">📷</div>
                <div class="upload-text">Select an image file above</div>
                <div class="upload-subtext">JPG, PNG, or WebP • Supports multiple foods</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.detection_result:
            result = st.session_state.detection_result
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h2 class="card-title">Results</h2>', unsafe_allow_html=True)
            
            # Metrics
            col_m1, col_m2, col_m3 = st.columns(3)
            total_cal = sum([d.get("calories_estimate", 0) or 0 for d in result.get("detections", [])])
            
            with col_m1:
                st.markdown(f"""
                <div class="metric">
                    <div class="metric-value">{result.get('num_detections', 0)}</div>
                    <div class="metric-label">Items</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m2:
                st.markdown(f"""
                <div class="metric">
                    <div class="metric-value">{total_cal:.0f}</div>
                    <div class="metric-label">Calories</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m3:
                avg_conf = (sum([d.get('confidence', 0) for d in result.get('detections', [])])/max(len(result.get('detections', [])), 1)*100) if result.get('detections') else 0
                st.markdown(f"""
                <div class="metric">
                    <div class="metric-value">{avg_conf:.0f}%</div>
                    <div class="metric-label">Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Detections
            st.markdown("**Detected Foods:**")
            for detection in result.get("detections", []):
                st.markdown(f"""
                <div class="detection-card">
                    <div class="food-name">✓ {detection.get('class_name', 'Unknown').replace('_', ' ')}</div>
                    <div class="food-info">
                        <div class="info-badge">📊 {detection.get('portion_desc', 'N/A')}</div>
                        <div style="display: inline-block;"><span class="calorie-badge">{detection.get('calories_estimate', 0):.0f} cal</span></div>
                        <div style="display: inline-block;"><span class="confidence-badge">{detection.get('confidence', 0)*100:.1f}% match</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("➕ Add to Daily Log", use_container_width=True):
                st.session_state.daily_stats["calories"] += total_cal
                st.session_state.meal_history.append({
                    "time": datetime.now().strftime("%H:%M"),
                    "items": result.get("detections", []),
                    "total_cal": total_cal
                })
                st.success("Added to your daily log! (Note: Macro tracking is a placeholder.)")
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
        
            """, unsafe_allow_html=True)

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================
elif st.session_state.page == "📊 Dashboard":
    st.markdown("""
    <div class="header-gradient">
        <h1 class="header-title">📊 Nutrition Dashboard</h1>
        <p class="header-subtitle">Track your daily nutrition goals and progress</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric">
            <div class="metric-value">{st.session_state.daily_stats['calories']:.0f}</div>
            <div class="metric-label">Calories</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric">
            <div class="metric-value">{st.session_state.daily_stats['protein']:.0f}g</div>
            <div class="metric-label">Protein</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric">
            <div class="metric-value">{st.session_state.daily_stats['carbs']:.0f}g</div>
            <div class="metric-label">Carbs</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric">
            <div class="metric-value">{st.session_state.daily_stats['fat']:.0f}g</div>
            <div class="metric-label">Fat</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.5, 1])
    with col2:
        if st.session_state.meal_history:
            for meal in reversed(st.session_state.meal_history):
                st.markdown(f"""
                <div class="stats-box">
                    <div class="stat-row">
                        <span class="stat-label">{meal['time']}</span>
                        <span class="stat-value">{meal['total_cal']:.0f} cal</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No meals logged yet. Start by scanning a meal!")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col1:
        st.markdown('<h2 class="card-title">🎯 Goals</h2>', unsafe_allow_html=True)
        st.metric("Daily Goal", "2000 cal")
        st.metric("Progress", f"{min(100, st.session_state.daily_stats['calories']/2000*100):.0f}%")
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# PAGE: NUTRITION AI
# ============================================================================
elif st.session_state.page == "💬 Nutrition AI":
    st.markdown("""
    <div class="header-gradient">
        <h1 class="header-title">💬 Ask Your Nutritionist</h1>
        <p class="header-subtitle">Get personalized nutrition advice powered by AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.3], gap="large")
    
    with col1:
        st.markdown('<h2 class="card-title">💡 Sample Questions</h2>', unsafe_allow_html=True)
        
        samples = [
            "How many calories in butter chicken?",
            "Is paneer healthy for weight loss?",
            "Best high-protein Indian dishes?",
            "Healthy swaps for samosa?",
            "Nutritional benefits of dal?",
            "How to eat biryani healthily?"
        ]
        
        for sample in samples:
            if st.button(sample, use_container_width=True, key=f"sample_{sample}"):
                st.session_state.current_question = sample
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h2 class="card-title">Your Question</h2>', unsafe_allow_html=True)
        
        user_question = st.text_area(
            "Ask anything about nutrition",
            value=st.session_state.current_question, 
            height=120,
            placeholder="What's your nutrition question?",
            label_visibility="collapsed"
        )
        
        if st.button("🚀 Get Answer", use_container_width=True):
            if user_question.strip():
                with st.spinner("Consulting AI nutritionist..."):
                    try:
                        response = requests.post(
                            ASK_ENDPOINT,
                            json={"text": user_question},
                            timeout=120
                        )
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.qa_result = result
                            st.session_state.current_question = ""
                        else:
                            st.error(f"Try again. Status code: {response.status_code}. Response: {response.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Connection Error: Could not connect to the AI backend. Please ensure the service is running at http://localhost:8000.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please ask a question")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    if "qa_result" in st.session_state and st.session_state.qa_result is not None:
        result = st.session_state.qa_result
        st.markdown('<h2 class="card-title">✨ Response</h2>', unsafe_allow_html=True)
        st.markdown(f'<div class="qa-answer">{result.get("answer", "No response")}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# PAGE: FOOD DATABASE
# ============================================================================
elif st.session_state.page == "📚 Food Database":
    st.markdown("""
    <div class="header-gradient">
        <h1 class="header-title">📚 Food Database</h1>
        <p class="header-subtitle">Browse and search common foods with nutrition info</p>
    </div>
    """, unsafe_allow_html=True)
    
    search_term = st.text_input("🔍 Search foods", placeholder="e.g., chicken, rice, apple...")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sample database
    food_db = {
        "Chicken Breast": {"calories": 165, "protein": 31, "carbs": 0, "fat": 3.6},
        "Brown Rice": {"calories": 111, "protein": 2.6, "carbs": 23, "fat": 0.9},
        "Paneer": {"calories": 265, "protein": 28, "carbs": 3.6, "fat": 17},
        "Butter": {"calories": 717, "protein": 0.9, "carbs": 0, "fat": 81},
        "Milk": {"calories": 61, "protein": 3.2, "carbs": 4.8, "fat": 3.3},
    }
    
    if search_term:
        results = {k: v for k, v in food_db.items() if search_term.lower() in k.lower()}
    else:
        results = food_db
    
    if results:
        for food_name, nutrition in results.items():
            st.markdown(f"""
            <div class="detection-card">
                <div class="food-name">{food_name}</div>
                <div class="food-info">
                    <div class="info-badge">🔥 {nutrition['calories']} cal</div>
                    <div class="info-badge">🥚 {nutrition['protein']}g protein</div>
                    <div class="info-badge">🌾 {nutrition['carbs']}g carbs</div>
                    <div class="info-badge">🧈 {nutrition['fat']}g fat</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# PAGE: SETTINGS
# ============================================================================
elif st.session_state.page == "⚙️ Settings":
    st.markdown("""
    <div class="header-gradient">
        <h1 class="header-title">⚙️ Settings</h1>
        <p class="header-subtitle">Customize your Foodalyze experience</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="card-title">👤 Profile</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Full Name", "")
        st.number_input("Age", 18, 100)
        st.selectbox("Gender", ["Male", "Female", "Other"])
    with col2:
        st.number_input("Height (cm)", 140, 220)
        st.number_input("Weight (kg)", 40, 200)
        st.selectbox("Activity Level", ["Sedentary", "Light", "Moderate", "Active"])
        
    st.markdown('<h2 class="card-title">🎯 Daily Goals</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Daily Calorie Goal", 1500, 3500, 2000)
        st.number_input("Protein Goal (g)", 40, 250, 150)
    with col2:
        st.number_input("Carbs Goal (g)", 100, 500, 250)
        st.number_input("Fat Goal (g)", 30, 150, 65)
        
    if st.button("💾 Save Settings", use_container_width=True):
        st.success("Settings saved successfully!")
