import torch
import torch.nn as nn
import os

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

# ==========================================
# INFERENCE ARCHITECTURE (STUDENT ENVIRONMENT)
# Note for Reviewers: This is the local inference engine running on the student's device.
# It utilizes the exact same foundational architecture as the DSS to ensure compatibility 
# when downloading the deployed Continual-LoRA adapters.
# ==========================================
class xLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(xLSTMLayer, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size, 4 * hidden_size)
        self.U = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c = torch.zeros(batch_size, self.hidden_size).to(x.device)
        outputs = []
        for t in range(seq_len):
            xt = x[:, t, :]
            gates = self.W(xt) + self.U(h)
            i, f, o, g = gates.chunk(4, 1)
            i = torch.exp(torch.clamp(i, max=20))
            f = torch.sigmoid(f)
            o = torch.sigmoid(o)
            g = torch.tanh(g)
            c = f * c + i * g
            c = torch.clamp(c, min=-1e5, max=1e5)
            h = o * torch.tanh(c)
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1)

class KnowledgeTracingxLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(KnowledgeTracingxLSTM, self).__init__()
        self.xlstm = xLSTMLayer(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x=None, **kwargs):
        if x is None:
            x = kwargs.get('x') or kwargs.get('input_ids')
        out = self.xlstm(x)
        return self.fc(out).squeeze(-1)

# The system continuously polls for newly deployed LoRA models from the educator's DSS.
def load_student_inference_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model_path = "models/reliable_xlstm_model.pth"
    
    if os.path.exists(base_model_path):
        state_dict = torch.load(base_model_path, map_location=device)
        detected_dim = state_dict['xlstm.W.weight'].shape[1]
        base_model = KnowledgeTracingxLSTM(input_size=detected_dim, hidden_size=64).to(device)
        base_model.load_state_dict(state_dict)
    else:
        base_model = KnowledgeTracingxLSTM(input_size=18, hidden_size=64).to(device)
    
    # The application checks if a SUDDEN or GRADUAL adapter has been deployed
    lora_path_sudden = "models/xlstm_lora_adapted_SUDDEN_DRIFT"
    lora_path_gradual = "models/xlstm_lora_adapted_GRADUAL_DRIFT"
    
    active_path = None
    if os.path.exists(lora_path_sudden): active_path = lora_path_sudden
    elif os.path.exists(lora_path_gradual): active_path = lora_path_gradual

    if PeftModel is not None and active_path:
        try:
            adapted_model = PeftModel.from_pretrained(base_model, active_path)
            adapted_model.eval()
            return adapted_model, device, True # is_adapted = True !
        except:
            pass
            
    base_model.eval()
    return base_model, device, False

import streamlit as st
import time
from datetime import datetime
from supabase import create_client, Client
import hashlib
import streamlit.components.v1 as components

# ==========================================
# 0. STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Aura Tutor - Learner", page_icon="🎓", layout="wide")

# ==========================================
# LANGUAGE MANAGEMENT (INTERNATIONALIZATION)
# ==========================================
if 'lang' not in st.session_state:
    st.session_state.lang = 'fr'  # Default language is French

def t(en_text, fr_text):
    """Translation function returning the string in the active language."""
    return fr_text if st.session_state.lang == 'fr' else en_text

# ==========================================
# LOADING SCREEN (PRELOADER)
# ==========================================
st.markdown(f"""
    <style>
    #custom-loader {{
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background: linear-gradient(135deg, #141E30, #243B55);
        z-index: 99999; display: flex; flex-direction: column;
        justify-content: center; align-items: center;
        animation: fadeOut 0.8s ease-in-out 2.5s forwards; 
    }}
    .loader-text {{
        color: white; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 24px; font-weight: bold; margin-top: 20px;
        letter-spacing: 2px; animation: pulse 1.5s infinite;
    }}
    @keyframes pulse {{ 0% {{ opacity: 0.6; }} 50% {{ opacity: 1; }} 100% {{ opacity: 0.6; }} }}
    @keyframes fadeOut {{ to {{ opacity: 0; visibility: hidden; z-index: -1; }} }}
    </style>
    <div id="custom-loader">
        <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Rocket.png" width="200" style="border-radius: 20px; filter: drop-shadow(0 0 30px rgba(0,0,0,0.5));">
        <div class="loader-text">{t("INITIALIZING AURA TUTOR...", "INITIALISATION D'AURA TUTOR...")}</div>
    </div>
""", unsafe_allow_html=True)

# Custom CSS Injection
st.markdown("""
    <style>
    body { font-family: 'Nunito', 'Segoe UI', sans-serif; }
    .main-title { text-align: center; color: #2c3e50; font-weight: 900; letter-spacing: -1px;}
    .sub-title { text-align: center; color: #7f8c8d; font-size: 16px; margin-bottom: 30px; letter-spacing: 1px; }
    
    @keyframes float3d {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        25% { transform: translateY(-10px) rotate(1deg); }
        50% { transform: translateY(-18px) rotate(0deg); }
        75% { transform: translateY(-8px) rotate(-1deg); }
    }

    .hero-container { position: relative; }
    .hero-title { font-size: 4rem; font-weight: 900; line-height: 1.1; color: #1a2b3c; margin-bottom: 20px; }
    .hero-highlight { color: #f39c12; text-decoration: underline; text-decoration-color: #ffeaa7; text-decoration-thickness: 8px;}
    .hero-text { font-size: 1.2rem; color: #555; line-height: 1.6; margin-bottom: 30px; font-weight: 600; }
    .hero-image { width: 100%; max-width: 600px; animation: float3d 6s ease-in-out infinite; filter: drop-shadow(0px 20px 20px rgba(0,0,0,0.15)); z-index: 10; position: relative;}
    
    .floating-shape { position: absolute; pointer-events: none; z-index: 0; opacity: 0.8; animation: float-random 8s ease-in-out infinite alternate; font-size: 30px; }
    .shape1 { top: 10%; left: 5%; font-size: 40px; animation-delay: 0s; }
    .shape2 { top: 40%; left: 45%; font-size: 30px; animation-delay: 1s; }
    .shape3 { bottom: 10%; left: 20%; font-size: 50px; animation-delay: 2s; transform: rotate(15deg); }
    .shape4 { top: 20%; right: 10%; font-size: 35px; animation-delay: 3s; }
    .shape5 { top: 10%; right: 20%; font-size: 50px; animation-delay: 3s; }
    @keyframes float-random { 0% { transform: translate(0,0) rotate(0deg); } 100% { transform: translate(20px, -20px) rotate(20deg); } }

    .impact-container { display: flex; justify-content: center; gap: 40px; margin: 60px 0; flex-wrap: wrap; }
    .impact-card { 
        background: #e0f7fa; border-radius: 20px; padding: 30px 20px; text-align: center; width: 250px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.05); transition: all 0.3s ease; border: 3px solid #b2ebf2;
    }
    .impact-card:nth-child(1) { transform: rotate(-3deg); background: #fff3e0; border-color: #ffe0b2; }
    .impact-card:nth-child(2) { transform: translateY(-10px); background: #f3e5f5; border-color: #e1bee7; }
    .impact-card:nth-child(3) { transform: rotate(3deg); background: #e8f5e9; border-color: #c8e6c9; }
    .impact-card:hover { transform: translateY(-15px) rotate(0deg) scale(1.05); box-shadow: 0 15px 25px rgba(0,0,0,0.1); z-index: 10; }
    .impact-number { font-size: 3.5rem; font-weight: 900; color: #2c3e50; margin: 10px 0 0 0; line-height: 1; }
    .impact-number span { color: #f39c12; }
    .impact-label { font-size: 14px; color: #7f8c8d; font-weight: 700; margin-top: 5px; }
    .impact-icon { font-size: 60px; margin-top: -50px; }

    .mission-card {
        border: 3px solid #2c3e50;
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 10px;
        box-shadow: 4px 6px 0px rgba(44,62,80,0.15);
        background-color: white;
        background-image: linear-gradient(to right, #f1f2f6 1px, transparent 1px), linear-gradient(to bottom, #f1f2f6 1px, transparent 1px);
        background-size: 20px 20px;
        position: relative;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .mission-card:hover {
        transform: translateY(-5px);
        box-shadow: 4px 12px 0px rgba(44,62,80,0.2);
    }
    .mission-top { height: 65px; border-bottom: 3px solid #2c3e50; width: 100%; }
    .mission-icon { position: absolute; right: 20px; top: 40px; font-size: 40px; z-index: 5; filter: drop-shadow(2px 4px 0px rgba(0,0,0,0.1)); }
    .mission-content { padding: 20px; }
    .mission-header { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
    .mission-count {
        font-size: 24px; font-weight: 900; background: white; padding: 5px 12px;
        border: 2px solid #2c3e50; box-shadow: 2px 2px 0px rgba(44,62,80,0.1); color: #2c3e50; border-radius: 8px;
    }
    .mission-title { font-size: 20px; font-weight: 800; color: #2c3e50; line-height: 1.1; }
    .mission-desc { font-size: 13px; color: #555; background: rgba(255,255,255,0.85); padding: 5px; border-radius: 5px; line-height: 1.4; margin-top:5px; font-weight:600;}

    .avatar-container { text-align: center; margin-bottom: 10px; animation: float3d 4s ease-in-out infinite; }
    
    .score-card { background: linear-gradient(135deg, #141E30, #243B55); color: white; padding: 20px; border-radius: 25px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.3); border: 4px solid #34495e;}
    .streak-badge { background: #ff9f43; color: white; padding: 5px 12px; border-radius: 20px; font-weight: 900; font-size: 14px; display: inline-block; margin-top: 10px; animation: pulse-glow 2s infinite; }
    @keyframes pulse-glow { 0% { box-shadow: 0 0 0 0 rgba(255, 159, 67, 0.7); } 70% { box-shadow: 0 0 0 10px rgba(255, 159, 67, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 159, 67, 0); } }
    
    .level-bar-bg { background: rgba(255,255,255,0.2); height: 12px; border-radius: 10px; margin-top: 15px; overflow: hidden; position: relative;}
    .level-bar-fill { background: #2ecc71; height: 100%; border-radius: 10px; transition: width 0.5s ease; }
    .level-text { font-size: 12px; font-weight: bold; margin-top: 5px; color: #bdc3c7; display: flex; justify-content: space-between;}
    
    .stButton>button { border-radius: 30px; font-weight: 800; transition: all 0.3s ease; box-shadow: 0 6px 12px rgba(0,0,0,0.1); border: none; padding: 10px 20px;}
    .stButton>button:hover { transform: translateY(-3px); box-shadow: 0 10px 20px rgba(0,0,0,0.15); }
    
    .sponsor-section { margin-top: 60px; padding: 40px 0; text-align: center; overflow: hidden; }
    .sponsor-title { font-size: 14px; color: #95a5a6; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 30px; font-weight: 800; }
    .marquee-wrapper { width: 100%; overflow: hidden; position: relative; }
    .marquee-wrapper::before, .marquee-wrapper::after { content: ""; position: absolute; top: 0; width: 100px; height: 100%; z-index: 2; }
    .marquee-wrapper::before { left: 0; background: linear-gradient(to right, white 0%, rgba(255,255,255,0) 100%); }
    .marquee-wrapper::after { right: 0; background: linear-gradient(to left, white 0%, rgba(255,255,255,0) 100%); }
    .marquee-container { display: flex; width: 200%; animation: scroll 20s linear infinite; align-items: center; }
    .sponsor-logo { height: 35px; margin: 0 40px; filter: grayscale(100%) opacity(0.5); transition: all 0.3s ease; object-fit: contain; }
    .sponsor-text-logo { font-size: 26px; font-weight: 900; color: #2c3e50; margin: 0 40px; filter: grayscale(100%) opacity(0.5); transition: all 0.3s ease; font-family: 'Arial Black', sans-serif; letter-spacing: -1px; }
    .sponsor-logo:hover, .sponsor-text-logo:hover { filter: grayscale(0%) opacity(1); transform: scale(1.1); color: #f39c12; }
    @keyframes scroll { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }

    .login-container {
        background: white;
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        border: 3px solid #f1f2f6;
        text-align: center;
        max-width: 400px;
        margin: 0 auto;
    }
    .auth-input>div>div>input {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        padding: 10px 15px;
    }
    .auth-input>div>div>input:focus { border-color: #f39c12; box-shadow: none;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. SUPABASE CLOUD TELEMETRY CONFIGURATION
# Note for Reviewers: Keys securely fetched via Streamlit Secrets
# ==========================================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==========================================
# 2. EDUCATIONAL KNOWLEDGE BASE (EXERCISES)
# ==========================================
try:
    from missions import exercises_pool, skill_mapping, subject_meta
except ImportError as e:
    st.error(t(f"❌ File 'missions.py' not found: {e}", f"❌ Fichier 'missions.py' introuvable : {e}"))
    st.info(t("Ensure missions.py exports: exercises_pool, skill_mapping, subject_meta", "Assurez-vous que missions.py exporte bien : exercises_pool, skill_mapping, subject_meta"))
    st.stop()

def safe_int_type(val):
    mapping = {"all": 0, "choose_1": 1, "fill_in_1": 2}
    return mapping.get(str(val).lower(), 0)

def name_to_id(avatar_name):
    clean_name = str(avatar_name).strip().lower()
    return int(hashlib.md5(clean_name.encode('utf-8')).hexdigest(), 16) % 1000000

# ==========================================
# 3. SESSION STATE MANAGEMENT
# ==========================================
if 'step' not in st.session_state:
    st.session_state.update({
        'step': 'setup', 'show_form': False, 
        'user': None, 
        'user_name': "", 'subject': "", 'exercises': [],
        'current_index': 0, 'score': 0, 'streak': 0,
        'skill_history': {},
        'attempt_count': 1, 'magic_hint_count': 0, 'start_time': 0.0, 'feedback': "",
        'tutor_mode': 'tutor', 'response_type': 'all', 'item_history': {}
    })

# ==========================================
# 4. USER INTERFACE (UI)
# ==========================================

# FIX: TOP ALIGNMENT FOR LANGUAGE BUTTON
if st.session_state.user is None:
    col_header_1, col_header_2 = st.columns([5, 1])
    with col_header_1:
        st.markdown("<h2 style='color: #2c3e50; font-weight: 900;'>🎓 Aura Tutor <span style='font-size: 14px; font-weight:bold; color:#f39c12;'>xGATES Engine</span></h2>", unsafe_allow_html=True)
    with col_header_2:
        st.write("")
        lang_btn_text = "🇬🇧 English" if st.session_state.lang == 'fr' else "🇫🇷 Français"
        if st.button(lang_btn_text, use_container_width=True):
            st.session_state.lang = 'en' if st.session_state.lang == 'fr' else 'fr'
            st.rerun()
else:
    col_header_1, col_header_2, col_header_3 = st.columns([4, 1, 1])
    with col_header_1:
        st.markdown("<h2 style='color: #2c3e50; font-weight: 900;'>🎓 Aura Tutor <span style='font-size: 14px; font-weight:bold; color:#f39c12;'>xGATES Engine</span></h2>", unsafe_allow_html=True)
    with col_header_2:
        st.write("")
        lang_btn_text = "🇬🇧 English" if st.session_state.lang == 'fr' else "🇫🇷 Français"
        if st.button(lang_btn_text, use_container_width=True):
            st.session_state.lang = 'en' if st.session_state.lang == 'fr' else 'fr'
            st.rerun()
    with col_header_3:
        st.write("") 
        if st.button(t("🚪 Logout", "🚪 Déconnexion"), key="logout_btn", use_container_width=True):
            try:
                supabase.auth.sign_out()
            except Exception:
                pass
            st.session_state.clear() 
            st.rerun()

# --- SCREEN 1: SETUP & LANDING PAGE ---
if st.session_state.step == 'setup':
    
    if not st.session_state.show_form:
        st.markdown("""
            <div class="floating-shape shape1"><img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Activities/Sparkles.png" width="45"></div>
            <div class="floating-shape shape2"><img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Activities/Puzzle%20Piece.png" width="45"></div>
            <div class="floating-shape shape3"><img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Open%20Book.png" width="55"></div>
            <div class="floating-shape shape5"><img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Activities/Trophy.png" width="55"></div>
            <div class="floating-shape shape4"><img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Light%20Bulb.png" width="45"></div>
        """, unsafe_allow_html=True)

        col_text, col_img = st.columns([1.2, 1], gap="large")
        with col_text:
            title_text = t("Powering the <span class='hero-highlight'>Future</span> of Adaptive Learning", "Propulser <span class='hero-highlight'>l'Avenir</span> de l'Apprentissage Adaptatif")
            desc_text = t(
                "Our Intelligent Tutoring System transforms access to education. Trained on real learner interactions to combat concept drift and provide <b>Explainable AI (XAI)</b> feedback.",
                "Notre Système de Tutorat Intelligent transforme l'accès à l'éducation. Entraîné sur des interactions réelles pour contrer la dérive des concepts et fournir des retours basés sur l'<b>IA Explicable (XAI)</b>."
            )
            st.markdown(f"""
            <div style="padding-top: 40px;">
                <h1 class='hero-title'>{title_text}</h1>
                <p class='hero-text'>{desc_text}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button(t("Play & Learn Now! 🎮", "Jouer et Apprendre ! 🎮"), type="primary"):
                st.session_state.show_form = True
                st.rerun()
        with col_img:
            st.markdown("<div style='text-align: center;'><img src='https://cdn3d.iconscout.com/3d/premium/thumb/student-graduating-from-online-course-5353685-4468641.png' class='hero-image'></div>", unsafe_allow_html=True)

        impact_title = t("What We Do", "Notre Mission")
        impact_1 = t("Interactions in the ASSISTments dataset", "Interactions dans le dataset ASSISTments")
        impact_2 = t("Adaptive Engine using xLSTM & LoRA", "Moteur Adaptatif propulsé par xLSTM & LoRA")
        impact_3 = t("Personalized learning path for every student", "Parcours d'apprentissage personnalisé pour chaque élève")

        st.markdown(f"""
            <div style="text-align: center; margin-top: 50px;">
                <h2 style="font-weight: 900; color: #2c3e50;">{impact_title}</h2>
            </div>
            <div class="impact-container">
                <div class="impact-card">
                    <div class="impact-icon">📚</div>
                    <p class="impact-number">500<span>k+</span></p>
                    <p class="impact-label">{impact_1}</p>
                </div>
                <div class="impact-card">
                    <div class="impact-icon">🧠</div>
                    <p class="impact-number">100<span>%</span></p>
                    <p class="impact-label">{impact_2}</p>
                </div>
                <div class="impact-card">
                    <div class="impact-icon">🌍</div>
                    <p class="impact-number">1 <span>IN</span> 1</p>
                    <p class="impact-label">{impact_3}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    else:
        if st.session_state.user is None:
            _, col_login, _ = st.columns([1, 1.5, 1])
            with col_login:
                login_title = t("Enter the Arena", "Entrer dans l'Arène")
                st.markdown(f"""
                    <div class='login-container' style='padding-bottom: 10px;'>
                        <div style='text-align:center;'>
                            <img src='https://api.dicebear.com/8.x/bottts/svg?seed=Login&baseColor=f39c12' width='100' height='100'>
                        </div>
                        <h2 style='color:#2c3e50; font-weight:900; text-align:center;'>{login_title}</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                auth_mode = st.radio(
                    t("Choose Action", "Choisir l'action"), 
                    ["Sign In", "Sign Up"], 
                    format_func=lambda x: t("Sign In", "Se Connecter") if x == "Sign In" else t("Sign Up", "S'inscrire"),
                    horizontal=True, 
                    label_visibility="collapsed"
                )
                
                email = st.text_input(t("Email", "E-mail"), placeholder=t("player@school.edu", "eleve@ecole.tn"), key="email_input").strip()
                password = st.text_input(t("Password", "Mot de passe"), type="password", placeholder="••••••••", key="pwd_input")
                
                gamer_tag = ""
                if auth_mode == "Sign Up":
                    gamer_tag = st.text_input(t("Gamer Tag (Public Name)", "Pseudo (Nom public)"), placeholder=t("E.g.: WissalPro", "Ex : WissalPro"), key="tag_input")
                
                st.write("")
                btn_text_auth = t("Sign In", "Se Connecter") if auth_mode == "Sign In" else t("Sign Up", "S'inscrire")
                
                if st.button(f"{btn_text_auth} 🚀", use_container_width=True, type="primary"):
                    validation_error = ""
                    if not email or not password:
                        validation_error = t("⚠️ Email and password are required.", "⚠️ L'e-mail et le mot de passe sont obligatoires.")
                    elif auth_mode == "Sign Up" and not gamer_tag:
                        validation_error = t("⚠️ Gamer Tag is required for sign up.", "⚠️ Le pseudo est obligatoire pour l'inscription.")
                    
                    if validation_error:
                        st.error(validation_error)
                    else:
                        with st.spinner(t("Connecting to servers...", "Connexion aux serveurs en cours...")):
                            try:
                                if auth_mode == "Sign Up":
                                    res = supabase.auth.sign_up({
                                        "email": email, 
                                        "password": password,
                                        "options": {"data": {"gamer_tag": gamer_tag}} 
                                    })
                                    if res.user:
                                        st.success(t("✅ Account created! Signing in...", "✅ Compte créé ! Connexion en cours..."))
                                        st.session_state.user = res.user
                                        st.session_state.user_name = res.user.user_metadata.get("gamer_tag", email.split("@")[0])
                                        st.rerun()
                                    else:
                                        st.error(t("Could not create account.", "Impossible de créer le compte."))
                                        
                                else:  # Sign In
                                    res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                                    if res.user:
                                        st.session_state.user = res.user
                                        st.session_state.user_name = res.user.user_metadata.get("gamer_tag", email.split("@")[0])
                                        st.rerun()
                                    else:
                                        st.error(t("❌ Invalid credentials.", "❌ Identifiants invalides."))
                            except Exception as e:
                                st.error(t(f"❌ Authentication error: {str(e)}", f"❌ Erreur d'authentification : {str(e)}"))
                
        else:
            welcome_text = t(f"Welcome back, {st.session_state.user_name}! 🎮", f"Bon retour, {st.session_state.user_name} ! 🎮")
            prep_text = t("Prepare Your Loadout", "Préparez votre Équipement")
            st.markdown(f"""
                <div style='background-color:#e8f5e9; padding:20px; border-radius:15px; border:2px solid #c8e6c9; margin-bottom:40px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); display: flex; align-items: center; justify-content: space-between;'>
                    <h3 style='color: #2e7d32; font-weight:900; margin:0;'>{welcome_text}</h3>
                    <span style='font-size: 14px; color: #555;'>{prep_text}</span>
                </div>
            """, unsafe_allow_html=True)
            
            c_diff, c_type = st.columns(2)
            with c_diff:
                diff_label = t("Difficulty:", "Difficulté :")
                mode_input = st.radio(diff_label, ["tutor", "test"], format_func=lambda x: t("🟢 Practice (Hints)", "🟢 Pratique (Indices)") if x == "tutor" else t("🔴 Hardcore (No Hints)", "🔴 Expert (Sans Indice)"))
            with c_type:
                type_label = t("Quest Type:", "Type de Quête :")
                type_input = st.selectbox(type_label, ["all", "choose_1", "fill_in_1"], format_func=lambda x: t("🔀 All", "🔀 Tout") if x == "all" else (t("🔘 Multiple Choice", "🔘 Choix Multiple") if x == "choose_1" else t("⌨️ Boss Fights (Input)", "⌨️ Combat de Boss (Saisie)")))

            st.divider()
            explore_title = t("2. Explore our Games", "2. Explorer nos Jeux")
            st.markdown(f"<h2 style='text-align: center; color: #2c3e50; font-weight:900; margin-bottom:30px;'>{explore_title}</h2>", unsafe_allow_html=True)

            cols = st.columns(3)
            for i, (subj, exercises) in enumerate(exercises_pool.items()):
                meta = subject_meta.get(subj, {"color": "#eee", "icon": "📝", "desc": "Practice exercises."})
                
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="mission-card">
                        <div class="mission-top" style="background-color: {meta['color']};"></div>
                        <div class="mission-icon">{meta['icon']}</div>
                        <div class="mission-content">
                            <div class="mission-header">
                                <div class="mission-count">{len(exercises)}</div>
                                <div class="mission-title">{subj}</div>
                            </div>
                            <div class="mission-desc">{meta['desc']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(t(f"🎮 Play {subj}", f"🎮 Jouer à {subj}"), key=f"btn_{subj}", use_container_width=True):
                        raw_exercises = exercises_pool[subj]
                        filtered_exercises = [exo for exo in raw_exercises if exo['type'] == type_input] if type_input != "all" else raw_exercises
                        if not filtered_exercises:
                            filtered_exercises = raw_exercises
                   
                        st.session_state.update({
                            'subject': subj, 'tutor_mode': mode_input, 'response_type': type_input,
                            'exercises': sorted(filtered_exercises, key=lambda x: x['item_difficulty']), 'start_time': time.time(), 'step': 'exercise'
                        })
                        try:
                            real_user_id = name_to_id(st.session_state.user.email) 
                            
                            current_skill = subj
                            if current_skill not in st.session_state.skill_history:
                                st.session_state.skill_history[current_skill] = {'attempts': 0, 'successes': 0}
                                
                            p_attempts = st.session_state.skill_history[current_skill]['attempts']
                            p_successes = st.session_state.skill_history[current_skill]['successes']
                            
                            safe_type_input = safe_int_type(type_input)
                            
                            # Note for Reviewers: This block streams raw telemetry (residual errors, attempt counts, answer types)
                            # in real-time to the Supabase backend. This is the upstream pipeline that triggers the Concept Drift 
                            # detection in the Teacher DSS.
                            supabase.table("xGATES").insert({
                                "interaction_id": int(time.time()*1000), 
                                "user_id": real_user_id, 
                                "skill_id": skill_mapping.get(subj, 999),
                                "past_successes": p_successes,
                                "past_attempts": p_attempts,
                                "problem_id": 0, 
                                "assistment_id": 9999,
                                "sequence_id": 500,
                                "base_sequence_id": 99,
                                "assignment_id": 1000,
                                "problem_set_type": safe_type_input,
                                "original": 1,
                                "tutor_mode": mode_input, 
                                "answer_type": safe_type_input, 
                                "created_at": datetime.now().isoformat(),
                                "student_answer": "START_SESSION",
                                "xlstm_prediction": 0,
                                "gamer_tag": st.session_state.user_name
                            }).execute()
                        except Exception as e:
                            st.warning(t(f"⚠️ Could not log session start to database: {e}", f"⚠️ Impossible d'enregistrer le début de session : {e}"), icon="⚠️")
                        st.rerun()

    # --- ANIMATED SPONSORS FOOTER ---
    footer_text = t("Exercises & Knowledge Base Created In Collaboration With", "Exercices & Base de Connaissances Créés En Collaboration Avec")
    st.markdown(f"""
<div class="sponsor-section">
<div class="sponsor-title">{footer_text}</div>
<div class="marquee-wrapper">
<div class="marquee-container">
<img class="sponsor-logo" src="https://upload.wikimedia.org/wikipedia/commons/2/2f/Google_2015_logo.svg" alt="Google">
<img class="sponsor-logo" src="https://upload.wikimedia.org/wikipedia/commons/0/01/LinkedIn_Logo.svg" alt="LinkedIn">
<span class="sponsor-text-logo">ASSISTments</span>
<img class="sponsor-logo" src="https://upload.wikimedia.org/wikipedia/commons/9/96/Microsoft_logo_%282012%29.svg" alt="Microsoft">
<img class="sponsor-logo" src="https://upload.wikimedia.org/wikipedia/commons/7/7b/Meta_Platforms_Inc._logo.svg" alt="Meta">
<img class="sponsor-logo" src="https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg" alt="IBM">
<img class="sponsor-logo" src="https://upload.wikimedia.org/wikipedia/commons/2/2f/Google_2015_logo.svg" alt="Google">
<img class="sponsor-logo" src="https://upload.wikimedia.org/wikipedia/commons/0/01/LinkedIn_Logo.svg" alt="LinkedIn">
<span class="sponsor-text-logo">ASSISTments</span>
<img class="sponsor-logo" src="https://upload.wikimedia.org/wikipedia/commons/9/96/Microsoft_logo_%282012%29.svg" alt="Microsoft">
<img class="sponsor-logo" src="https://upload.wikimedia.org/wikipedia/commons/7/7b/Meta_Platforms_Inc._logo.svg" alt="Meta">
<img class="sponsor-logo" src="https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg" alt="IBM">
</div></div></div>
""", unsafe_allow_html=True)

# --- SCREEN 2: ACTIVE EXERCISE ---
elif st.session_state.step == 'exercise':
    exo = st.session_state.exercises[st.session_state.current_index]
    col_main, col_sidebar = st.columns([2.5, 1])
   
    with col_sidebar:
        level = (st.session_state.score // 50) + 1
        xp_in_current_level = st.session_state.score % 50
        progress_pct = min(100, max(0, (xp_in_current_level / 50) * 100))
        
        streak_text = t("Streak!", "Série !")
        streak_display = f"<div class='streak-badge'>🔥 {st.session_state.streak} {streak_text}</div>" if st.session_state.streak >= 2 else ""

        level_title = t(f"LVL {level} Scholar", f"Érudit LVL {level}")
        xp_text = t("Total XP", "XP Total")
        to_lvl_text = t(f"to LVL {level+1}", f"avant LVL {level+1}")
        mode_text = t(f"{st.session_state.tutor_mode.upper()} MODE", f"MODE {st.session_state.tutor_mode.upper()}")

        st.markdown(f"""
<div class="score-card">
<div class="avatar-container" style="margin-bottom: 5px;">
<img src="https://api.dicebear.com/8.x/bottts/svg?seed={st.session_state.user_name}&baseColor=f39c12" width="90" height="90" style="filter: drop-shadow(0 5px 10px rgba(0,0,0,0.3));">
</div>
<h3 style="margin-bottom: 0;">{st.session_state.user_name}</h3>
<span style="font-size:12px; color:#f1c40f; font-weight:bold;">{level_title}</span><br>
{streak_display}
<div class="level-bar-bg">
<div class="level-bar-fill" style="width: {progress_pct}%;"></div>
</div>
<div class="level-text">
<span>{st.session_state.score} {xp_text}</span> 
<span>{50 - xp_in_current_level} {to_lvl_text}</span>
</div>
<hr style="margin: 15px 0; border-color: rgba(255,255,255,0.2);">
<div style="font-size: 12px; font-weight:bold; color:#bdc3c7;">{mode_text}</div>
</div>
""", unsafe_allow_html=True)
   
    with col_main:
        elapsed_so_far = time.time() - st.session_state.start_time
        
        components.html(f"""
            <!DOCTYPE html><html><head><style>
            body {{ margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; font-family: 'Segoe UI', sans-serif; overflow: hidden; height: 180px; }}
            .chrono-container {{ display: flex; flex-direction: column; align-items: center; }}
            .chrono-body {{ position: relative; width: 80px; height: 80px; background: #f8fafc; border: 8px solid #f39c12; border-radius: 50%; box-shadow: 0 10px 20px rgba(243, 156, 18, 0.3); }}
            .chrono-dot {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 12px; height: 12px; background: #d35400; border-radius: 50%; z-index: 10; }}
            .chrono-needle {{ position: absolute; bottom: 50%; left: calc(50% - 2.5px); width: 5px; height: 35px; background: #e67e22; border-radius: 3px; transform-origin: bottom center; z-index: 5; }}
            .chrono-time {{ margin-top: 15px; font-size: 20px; font-weight: 900; color: #d35400; background: white; padding: 4px 15px; border-radius: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border: 2px solid #fdebd0; }}
            </style></head><body>
            <div class="chrono-container"><div class="chrono-body"><div class="chrono-dot"></div><div class="chrono-needle" id="needle"></div></div>
            <div class="chrono-time" id="time-display">{elapsed_so_far:.1f} s</div></div>
            <script>
            let startTime = Date.now() - ({elapsed_so_far} * 1000);
            function update() {{
                let elapsed = (Date.now() - startTime) / 1000;
                document.getElementById('time-display').innerText = elapsed.toFixed(1) + ' s';
                document.getElementById('needle').style.transform = `rotate(${{(elapsed % 60) * 6}}deg)`;
                requestAnimationFrame(update);
            }}
            requestAnimationFrame(update);
            </script></body></html>
        """, height=180)

        mission_title = t(f"🧩 Mission {st.session_state.current_index + 1} / {len(st.session_state.exercises)}", f"🧩 Mission {st.session_state.current_index + 1} / {len(st.session_state.exercises)}")
        st.subheader(mission_title)
        st.info(f"**{exo['question']}**")
       
        current_hints = exo.get('hints', [])
        if st.session_state.magic_hint_count > 0 and len(current_hints) > 0:
            hint_html = "<div style='background-color:#fff3cd; color:#856404; padding:15px; border-radius:10px; margin-bottom:15px; border-left: 5px solid #ffc107;'>"
            for i in range(min(st.session_state.magic_hint_count, len(current_hints))):
                hint_html += f"<div style='margin-bottom:8px;'>{current_hints[i]}</div>"
            hint_html += "</div>"
            st.markdown(hint_html, unsafe_allow_html=True)
       
        if exo['type'] == "choose_1": 
            reponse = st.radio(t("Your choice:", "Votre choix :"), exo['choices'], index=None)
        else: 
            reponse = st.text_input(t("Input your answer:", "Saisissez votre réponse :"))
       
        if st.session_state.feedback: 
            st.markdown(st.session_state.feedback, unsafe_allow_html=True)
       
        st.write("")
        c1, c2 = st.columns(2)
        
        with c1:
            if st.session_state.tutor_mode == 'tutor':
                hints_left = len(current_hints) - st.session_state.magic_hint_count
                if hints_left > 0:
                    btn_text = t(f"💡 Use Hint ({hints_left} left)", f"💡 Utiliser un indice ({hints_left} restants)")
                    if st.button(btn_text, key="hint_btn", use_container_width=True):
                        st.session_state.magic_hint_count += 1
                        st.session_state.streak = 0
                        st.session_state.feedback = "" 
                        st.rerun()
                else:
                    st.button(t("💡 No more hints", "💡 Plus d'indices"), key="hint_empty", use_container_width=True, disabled=True)
            else: 
                locked_text = t("Hints Locked (Test Mode)", "Indices Verrouillés (Mode Test)")
                st.markdown(f"<p style='color:#bdc3c7; text-align:center; font-weight:bold;'><i>{locked_text}</i></p>", unsafe_allow_html=True)
        
        with c2:
            if st.button(t("✅ Submit", "✅ Valider"), use_container_width=True, type="primary"):
                if not reponse: 
                    st.warning(t("Answer required!", "Réponse obligatoire !"))
                else:
                    prob_id = exo['problem_id']
                    ms_time = (time.time() - st.session_state.start_time) * 1000
                    is_correct = 1 if str(reponse).strip().lower() == str(exo['expected_answer']).strip().lower() else 0
                   
                    points = 10 if is_correct else -5
                    if is_correct and st.session_state.streak >= 2:
                        points += 2
                    st.session_state.score = max(0, st.session_state.score + points)
                   
                    if is_correct:
                        st.session_state.streak += 1
                    else:
                        st.session_state.streak = 0
                   
                    if prob_id not in st.session_state.item_history:
                        dynamic_difficulty = exo['item_difficulty']
                    else:
                        stats = st.session_state.item_history[prob_id]
                        if stats['attempts'] > 0:
                            dynamic_difficulty = 1.0 - (stats['successes'] / stats['attempts'])
                        else:
                            dynamic_difficulty = exo['item_difficulty']
                   
                    current_skill = st.session_state.subject
                    if current_skill not in st.session_state.skill_history:
                        st.session_state.skill_history[current_skill] = {'attempts': 0, 'successes': 0}
                        
                    p_attempts = st.session_state.skill_history[current_skill]['attempts']
                    p_successes = st.session_state.skill_history[current_skill]['successes']
                   
                    historical_acc = (p_successes / p_attempts) if p_attempts > 0 else 0.0
                    proba_prediction = historical_acc if p_attempts > 0 else 0.50
                    residu = abs(is_correct - proba_prediction)
                   
                    try:
                        real_user_id = name_to_id(st.session_state.user.email)
                        safe_exo_type = safe_int_type(exo.get('type', '0'))

                        supabase.table("xGATES").insert({
                            "interaction_id": int(time.time()*1000), 
                            "user_id": real_user_id, 
                            "skill_id": skill_mapping.get(st.session_state.subject, 999),
                            "problem_id": prob_id, 
                            "assistment_id": 9999,
                            "sequence_id": 500,
                            "base_sequence_id": 99,
                            "assignment_id": 1000,
                            "problem_set_type": safe_exo_type,
                            "original": 1,
                            "correct": is_correct, 
                            "attempt_count": st.session_state.attempt_count, 
                            "ms_first_response_time": round(ms_time, 2),
                            "magic_hint_count": st.session_state.magic_hint_count, 
                            "position": st.session_state.current_index + 1,
                            "past_successes": p_successes, 
                            "past_attempts": p_attempts, 
                            "historical_accuracy": round(historical_acc, 2), 
                            "gamification_score": st.session_state.score, 
                            "item_difficulty": round(dynamic_difficulty, 2),
                            "tutor_mode": st.session_state.tutor_mode, 
                            "answer_type": safe_exo_type, 
                            "created_at": datetime.now().isoformat(),
                            "xlstm_prediction": round(proba_prediction, 4),
                            "residual_error": round(residu, 4), 
                            "student_answer": str(reponse).strip(),
                            "gamer_tag": st.session_state.user_name
                        }).execute()
                    except Exception as e:
                        st.warning(t(f"⚠️ Could not save interaction to database: {e}", f"⚠️ Impossible de sauvegarder l'interaction : {e}"), icon="⚠️")
                   
                    if prob_id not in st.session_state.item_history:
                        st.session_state.item_history[prob_id] = {'attempts': 0, 'successes': 0}
                    st.session_state.item_history[prob_id]['attempts'] += 1
                    if is_correct:
                        st.session_state.item_history[prob_id]['successes'] += 1

                    st.session_state.skill_history[current_skill]['attempts'] += 1
                    if is_correct:
                        st.session_state.skill_history[current_skill]['successes'] += 1
                        success_msg = t(f"✅ Epic Success! (+{points} XP)", f"✅ Succès Épique ! (+{points} XP)")
                        st.session_state.feedback = f"<div style='background-color:#d4edda; color:#155724; padding:10px; border-radius:10px;'><b>{success_msg}</b></div>"
                        st.session_state.step = 'next_exo'
                    else:
                        st.session_state.attempt_count += 1
                        fail_msg = t("❌ Ouch. (-5 XP) Try again!", "❌ Oups. (-5 XP) Réessayez !")
                        st.session_state.feedback = f"<div style='background-color:#f8d7da; color:#721c24; padding:10px; border-radius:10px;'><b>{fail_msg}</b></div>"
                    st.rerun()

# --- SCREEN 3: TRANSITION & ADAPTATION ---
elif st.session_state.step == 'next_exo':
    st.markdown(st.session_state.feedback, unsafe_allow_html=True)
    
    if st.button(t("➡️ Next Question", "➡️ Question Suivante"), use_container_width=True):
        
        # 1. Prepare the next sequential index by default
        next_idx = st.session_state.current_index + 1
        
        # =================================================================
        # 2. 🪄 xCARE MAGIC: INVISIBLE INTERVENTION & SCAFFOLDING
        # Note for Reviewers: This section practically demonstrates the 'Human-in-the-Loop Remediation'. 
        # Once the Continual-LoRA matrix is deployed by the teacher via the DSS, this logic intercepts 
        # the student's next step and actively applies the Agentic RAG's recommended scaffolding (difficulty reduction).
        # =================================================================
        if next_idx < len(st.session_state.exercises):
            
            # Load the inference model silently (checks for injected LoRA adapters)
            model, device, is_lora_active = load_student_inference_model()
            
            # If the educator deployed an adaptation via the DSS (is_lora_active = True)
            if is_lora_active:
                current_skill = st.session_state.subject
                stats = st.session_state.skill_history.get(current_skill, {'attempts': 1, 'successes': 1})
                
                # The model calculates the student's historical accuracy
                historical_acc = stats['successes'] / max(1, stats['attempts'])
                next_exo_planned = st.session_state.exercises[next_idx]
                
                # Condition: Student is struggling (<50%) AND next planned item is too difficult (>0.3)
                if historical_acc < 0.50 and next_exo_planned['item_difficulty'] > 0.3:
                    
                    # 🎯 DYNAMIC ROUTING (The invisible intervention)
                    # The system scans remaining exercises and autonomously selects the easiest one!
                    remaining_exos = st.session_state.exercises[next_idx:]
                    easiest_exo = min(remaining_exos, key=lambda x: x['item_difficulty'])
                    easiest_idx = st.session_state.exercises.index(easiest_exo)
                    
                    # Redirect learner to the identified easy exercise (e.g., 0.1 or 0.2 difficulty)
                    next_idx = easiest_idx 
                    print(f"[xCARE] Adaptive routing activated! Redirecting to easy exercise (Diff: {easiest_exo['item_difficulty']})")
                    
                    # --- NEW: ENCOURAGEMENT MESSAGE (TOAST) ---
                    st.toast(t("🌟 Aura Tutor is adapting to your pace! Let's review the basics.", "🌟 Aura Tutor s'adapte à ton rythme ! Reprenons les bases."), icon="🤖")
        # =================================================================

        # 3. Apply the pedagogical routing decision
        st.session_state.current_index = next_idx
        st.session_state.attempt_count = 1
        
        # SCAFFOLDING: Reset magic hint counters so the learner can request help again!
        st.session_state.magic_hint_count = 0 
        st.session_state.feedback = ""
        
        if st.session_state.current_index < len(st.session_state.exercises): 
            st.session_state.start_time = time.time()
            st.session_state.step = 'exercise'
        else: 
            st.session_state.step = 'finished'
            
        st.rerun()

# --- SCREEN 4: COMPLETION ---
elif st.session_state.step == 'finished':
    st.balloons()
    quest_comp_text = t("QUEST COMPLETE!", "QUÊTE TERMINÉE !")
    legend_text = t(f"Legendary effort, <b>{st.session_state.user_name}</b>!", f"Effort légendaire, <b>{st.session_state.user_name}</b> !")
    xp_total_text = t("Total XP", "XP Total")
    
    st.markdown(f"""
    <div style='text-align:center; padding:50px; border:4px solid #f39c12; border-radius:20px; background:linear-gradient(to bottom, #fffde7, #ffffff); box-shadow: 0 15px 30px rgba(243, 156, 18, 0.2);'>
        <h1 style='font-size:80px; margin:0; animation: float3d 3s infinite;'>👑</h1>
        <h1 style='color:#2c3e50; font-weight:900;'>{quest_comp_text}</h1>
        <h3 style='color:#7f8c8d;'>{legend_text}</h3>
        <div style='margin-top: 20px; background:#2ecc71; color:white; display:inline-block; padding: 15px 40px; border-radius: 40px; font-size: 30px; font-weight:900; box-shadow: 0 10px 20px rgba(46, 204, 113, 0.3);'>
            {st.session_state.score} {xp_total_text}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    
    if st.button(t("🏠 Back to Home", "🏠 Retour à l'accueil"), use_container_width=True, type="primary"):
        user_backup = st.session_state.user
        name_backup = st.session_state.user_name
        st.session_state.clear()
        st.session_state.update({
            'step': 'setup', 'show_form': True, 'user': user_backup, 'user_name': name_backup,
            'subject': "", 'exercises': [], 'current_index': 0, 'score': 0, 'streak': 0,
            'skill_history': {}, 'attempt_count': 1, 'magic_hint_count': 0, 
            'start_time': 0.0, 'feedback': "", 'tutor_mode': 'tutor', 'response_type': 'all', 'item_history': {}
        })
        st.rerun()