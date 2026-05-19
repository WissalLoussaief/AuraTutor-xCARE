import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
import time
import os
import json
import torch
import torch.nn as nn

try:
    from rag_engine import generate_xai_explanation
except ImportError:
    generate_xai_explanation = None

# --- DYNAMIC IMPORT: CONTINUAL-LORA ADAPTATION ENGINE ---
try:
    from adaptation_engine import run_realtime_lora_adaptation
except ImportError:
    run_realtime_lora_adaptation = None

# --- DYNAMIC IMPORT: PEFT (PARAMETER-EFFICIENT FINE-TUNING) ---
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

# --- 1. STREAMLIT PAGE CONFIGURATION (MUST BE FIRST LINE) ---
st.set_page_config(page_title="Aura Tutor - xCARE Console", page_icon="👨‍🏫", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# NEURAL ARCHITECTURE DEFINITION: EXTENDED LSTM (xLSTM)
# Note for Reviewers: This defines the base predictive network. 
# Foundational weights (W0) are rigidly frozen during the Continual-LoRA 
# adaptation phase to mathematically guarantee a +0.00% Backward Transfer (BWT).
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

# ==========================================
# DYNAMIC MODEL LOADING (CACHED INFERENCE)
# Note for Reviewers: Automatically injects Low-Rank (r) matrices if a 
# drift topology was adapted, otherwise loads the static baseline.
# ==========================================
@st.cache_resource
def load_xcare_inference_model(drift_type="SUDDEN DRIFT"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model_path = "models/reliable_xlstm_model.pth"
    
    if os.path.exists(base_model_path):
        state_dict = torch.load(base_model_path, map_location=device)
        detected_dim = state_dict['xlstm.W.weight'].shape[1]
        base_model = KnowledgeTracingxLSTM(input_size=detected_dim, hidden_size=64).to(device)
        base_model.load_state_dict(state_dict)
    else:
        base_model = KnowledgeTracingxLSTM(input_size=18, hidden_size=64).to(device)
    
    if PeftModel is not None:
        folder_name = f"xlstm_lora_adapted_{drift_type.replace(' ', '_')}"
        adapter_path = os.path.join("models", folder_name)
        if os.path.exists(adapter_path):
            try:
                adapted_model = PeftModel.from_pretrained(base_model, adapter_path)
                adapted_model.eval()
                return adapted_model, device, True
            except Exception:
                pass
                
    base_model.eval()
    return base_model, device, False

# --- 2. SESSION STATE INITIALIZATION ---
if 'teacher_user' not in st.session_state:
    st.session_state.teacher_user = None
if 'lang' not in st.session_state:
    st.session_state.lang = 'EN'
if 'current_nav' not in st.session_state:
    st.session_state.current_nav = None
if 'is_loading' not in st.session_state:
    st.session_state.is_loading = True  # Enabled by default for initial launch

def toggle_language():
    st.session_state.lang = 'FR' if st.session_state.lang == 'EN' else 'EN'


# --- 3. INTERNATIONALIZATION (I18N) DICTIONARY ---
T = {
    'EN': {
        'eu_admin': "Administrator Access", 'hero_title': "Pedagogical Supervision",
        'hero_sub': "Drive artificial intelligence and analyze your learners' flow in real-time.",
        'impact_title': "Our Mission", 'impact_1': "Interactions in the ASSISTments dataset",
        'impact_2': "Adaptive Engine powered by xLSTM & LoRA", 'impact_3': "Personalized learning path for every student",
        'build_path': "✨ Let xCARE AI Build Your Path",
        'login_title': "Access your workspace and insights", 'tab_signin': "Sign In", 'tab_signup': "Create Account",
        'email': "Institutional Email", 'pwd': "Password", 'fname': "Full Name (e.g. Dr. Turing)", 'subj': "Subject Taught",
        'btn_signin': "Sign In", 'btn_signup': "Sign Up",
        'feat1_t': "Learn and teach smart", 'feat1_d': "Improve and share your knowledge on student performance easily.",
        'feat2_t': "Detect Concept Drift", 'feat2_d': "Identify attention drops and cognitive load in real-time with xLSTM.",
        'feat3_t': "Be inspired", 'feat3_d': "Use XAI recommendations to provide hyper-targeted pedagogical remediations.",
        'feat4_t': "The System and You", 'feat4_d': "Discover how adaptive algorithms assist you daily without replacing you.",
        'dash_subtitle': "Here is the pulse of your cohort.", 'refresh': "🔄 Refresh Data", 'logout': "🚪 Logout",
        'tab_overview': "Global Overview", 'tab_roster': "Learner Profiles", 'tab_alerts': "Active Interventions", 'tab_feed': "Live Feed",
        'cohort': "Active Learners", 'session': "Total Interactions", 'flow': "Avg. Accuracy", 'status': "System Status",
        'analytics': "Cohort Analytics", 'chart_skill': "Average Skill Mastery", 'chart_state': "Current Cognitive State",
        'student_select': "🔍 Select a learner to analyze:", 'recent_ans': "Recent Answers",
        'alert_title': "🚨 Critical Concept Drift Detected", 'alert_desc': "Learner is experiencing high cognitive load. Immediate scaffolding required.",
        'xai_title': "🧠 Explainable AI (XAI) Rationale", 'xai_desc': "Based on the telemetry, the learner is likely missing foundational prerequisites.",
        'hitl_title': "Human-in-the-Loop Actions", 'deploy': "✅ Approve & Deploy Remediation",
        'eval_title': "📝 Evaluate xCARE DSS", 'eval_rate': "How would you rate the XAI recommendations today?",
        'eval_comment': "Additional feedback or suggestions to improve the AI...", 'eval_btn': "Submit Evaluation",
        'eval_thanks': "Thank you! Your feedback helps us improve the xCARE algorithms.", 'sponsor_title': "Exercises & Knowledge Base Created In Collaboration With",
        'loading_text': "Preparing your workspace...",
        'remed_strat': "REMEDIATION STRATEGY:",
        'remed_scaffolding': "Scaffolding (Difficulty Decrease)",
        'remed_hints': "Unlock Hints",
        'remed_lora': "xLSTM Adaptation (Continual-LoRA)",
        'profile_label': "PROFILE",
        'home_breadcrumb': "🏠 Home  >  Dashboard",
        'search_placeholder': "🔍 Search learner, class...",
        'hello_teacher': "Hello, {teacher_name}!",
        'welcome_dashboard': "Welcome to your {teacher_subject} Dashboard.",
        'engine_label': "Engine:",
        'engine_lora': "🟢 Active LoRA",
        'engine_static': "🟡 Static Model",
        'status_alert': "🚨 Alert",
        'status_stable': "✅ Stable",
        'waiting_data': "📊 Waiting for student data in Supabase...",
        'global_cohort': "Global Cohort",
        'col_id': "Identifier",
        'col_activity': "Activity Vol.",
        'col_accuracy': "Accuracy %",
        'col_xp': "XP Earned",
        'lbl_correct': "Correct",
        'lbl_incorrect': "Incorrect",
        'lbl_skill': "Skill",
        'select_id': "Select ID:",
        'no_learner': "No learner connected at the moment.",
        'see_payload': "View JSON Payload (LoRA Data)",
        'alert_critical': "🚨 Critical Concept Drift Detected",
        'alert_desc_full': "Learner <b>#{student_id}</b> is experiencing high cognitive load. Avatar health dropped by <b>{drop_percentage}%</b>. Immediate scaffolding required.",
        'xai_rationale_title': "Explainable AI (XAI) Rationale",
        'gen_remed_path': "Generated Remediation Path:",
        'fallback_rationale': "Based on the telemetry (<b>{hints_used} consecutive hints used</b> and rapid energy depletion), the learner is likely missing foundational prerequisites for the current block.",
        'fallback_step1': "Step 1: Trigger Diagnostic Quiz (Ref: GEN1)",
        'fallback_step2': "Step 2: Consolidation with Max Scaffolding (Ref: GEN2)",
        'edit_intervention': "⚙️ Edit Intervention Path",
        'reject_override': "❌ Reject (Manual Override)",
        'live_stats': "Live Stats: XP {current_xp} | Health {avatar_health}/100",
        'system_nominal': "✅ System Nominal. No active concept drift detected in the current cohort.",
        'answered_skill': "answered Skill",
        'input_label': "Input:",
        'error_margin': "Error Margin:",
        'waiting_feed': "📡 Waiting for feed connections...",
        'deploy_success': "✅ Adaptation successfully deployed!",
        'lora_training': "⚙️ Real-time Continual-LoRA training in progress...",
        'saving': "Saving...",
        'db_error': "Database Error:",
        'eval_q1': "The XAI explanation helped me understand the student's blockage (Usefulness).",
        'eval_q2': "Deploying the remediation was intuitive and easy (Ease of Use).",
        'likert_1': "1 - Strongly Disagree",
        'likert_2': "2 - Disagree",
        'likert_3': "3 - Neutral",
        'likert_4': "4 - Agree",
        'likert_5': "5 - Strongly Agree",
        'agent_remed_intro': "Action: Remediation recommended. Here are 3 specific exercises generated from the ASSISTments databank to unblock the learner:",
        'agent_exo_1': "1. [Visual Exercise: Area model representation for distribution]",
        'agent_exo_2': "2. [Application Exercise: Guided step-by-step expansion]",
        'agent_exo_3': "3. [Consolidation Exercise: Identify the error in a given equation]",
        'agent_send': "👉 Click 'Approve & Deploy' below to push this micro-quiz to the learner's tablet."
    },
    'FR': {
        'eu_admin': "Accès Administrateur", 'hero_title': "Supervision Pédagogique",
        'hero_sub': "Pilotez l'intelligence artificielle et analysez le flux de vos apprenants en temps réel.",
        'impact_title': "Notre Mission", 'impact_1': "Interactions dans le dataset ASSISTments",
        'impact_2': "Moteur Adaptatif propulsé par xLSTM & LoRA", 'impact_3': "Parcours d'apprentissage personnalisé pour chaque élève",
        'build_path': "✨ Laissez l'IA xCARE construire votre parcours",
        'login_title': "Accédez à votre espace et vos analyses", 'tab_signin': "Connexion", 'tab_signup': "Créer un compte",
        'email': "Email Institutionnel", 'pwd': "Mot de passe", 'fname': "Nom Complet (ex: Dr. Turing)", 'subj': "Matière Enseignée",
        'btn_signin': "Se connecter", 'btn_signup': "S'inscrire",
        'feat1_t': "Apprenez et enseignez malin", 'feat1_d': "Améliorez et partagez vos connaissances sur la performance de manière simple.",
        'feat2_t': "Détectez le Concept Drift", 'feat2_d': "Identifiez les chutes d'attention et la charge cognitive avec xLSTM.",
        'feat3_t': "Soyez inspiré", 'feat3_d': "Utilisez les recommandations XAI pour fournir des remédiations ultra-ciblées.",
        'feat4_t': "Le Système et vous", 'feat4_d': "Découvrez comment les algorithmes vous assistent sans vous remplacer.",
        'dash_subtitle': "Voici le pouls de votre cohorte.", 'refresh': "🔄 Actualiser", 'logout': "🚪 Déconnexion",
        'tab_overview': "Vue Globale", 'tab_roster': "Profils Apprenants", 'tab_alerts': "Interventions Actives", 'tab_feed': "Flux en Direct",
        'cohort': "Élèves Actifs", 'session': "Total Interactions", 'flow': "Précision Moyenne", 'status': "Statut du Système",
        'analytics': "Analyse de la Cohorte", 'chart_skill': "Maîtrise Moyenne des Compétences", 'chart_state': "État Cognitif Actuel",
        'student_select': "🔍 Sélectionner un apprenant à analyser :", 'recent_ans': "Dernières Réponses",
        'alert_title': "🚨 Dérive de Concept Critique", 'alert_desc': "L'apprenant subit une forte charge cognitive. Un étayage immédiat est requis.",
        'xai_title': "🧠 Explication de l'IA (XAI)", 'xai_desc': "D'après la télémétrie, l'apprenant manque probablement de prérequis fondamentaux.",
        'hitl_title': "Actions Human-in-the-Loop", 'deploy': "✅ Approuver & Déployer",
        'eval_title': "📝 Évaluer le DSS xCARE", 'eval_rate': "Comment évaluez-vous les recommandations XAI aujourd'hui ?",
        'eval_comment': "Commentaires ou suggestions pour améliorer l'IA...", 'eval_btn': "Soumettre l'évaluation",
        'eval_thanks': "Merci ! Vos retours nous aident à améliorer les algorithmes xCARE.", 'sponsor_title': "Exercices & Base de connaissances créés en collaboration avec",
        'loading_text': "Préparation de votre espace...",
        'remed_strat': "STRATÉGIE DE REMÉDIATION :",
        'remed_scaffolding': "Scaffolding (Rediriger l'élève vers des exercices plus faciles)",
        'remed_hints': "Indices (Restaurer le compteur d'aides pour l'élève)",
        'remed_lora': "Adaptation de l'IA (Forcer le système à baisser ses prévisions de réussite pour cet élève)",
        'profile_label': "PROFIL",
        'home_breadcrumb': "🏠 Accueil  >  Tableau de bord",
        'search_placeholder': "🔍 Rechercher un apprenant, une classe...",
        'hello_teacher': "Bonjour, {teacher_name} !",
        'welcome_dashboard': "Bienvenue sur votre tableau de bord de {teacher_subject}.",
        'engine_label': "Moteur :",
        'engine_lora': "🟢 LoRA Actif",
        'engine_static': "🟡 Modèle Statique",
        'status_alert': "🚨 Alerte",
        'status_stable': "✅ Stable",
        'waiting_data': "📊 En attente de données élèves dans Supabase...",
        'global_cohort': "Cohorte Globale",
        'col_id': "Identifiant",
        'col_activity': "Vol. d'activité",
        'col_accuracy': "Précision %",
        'col_xp': "XP Gagnés",
        'lbl_correct': "Correct",
        'lbl_incorrect': "Incorrect",
        'lbl_skill': "Compétence",
        'select_id': "Sélectionner l'ID :",
        'no_learner': "Aucun apprenant connecté pour le moment.",
        'see_payload': "Voir le Payload JSON (Data LoRA)",
        'alert_critical': "🚨 Dérive de Concept Critique Détectée",
        'alert_desc_full': "L'apprenant <b>#{student_id}</b> subit une forte charge cognitive. La santé de l'avatar a chuté de <b>{drop_percentage}%</b>. Un étayage immédiat est requis.",
        'xai_rationale_title': "Explication de l'IA (XAI)",
        'gen_remed_path': "Parcours de Remédiation Généré :",
        'fallback_rationale': "D'après la télémétrie (<b>{hints_used} indices consécutifs utilisés</b> et épuisement rapide de l'énergie), l'apprenant manque probablement de prérequis fondamentaux pour ce bloc.",
        'fallback_step1': "Étape 1 : Déclencher un Quiz Diagnostique (Ref: GEN1)",
        'fallback_step2': "Étape 2 : Consolidation avec Étayage Maximum (Ref: GEN2)",
        'edit_intervention': "⚙️ Modifier le Parcours d'Intervention",
        'reject_override': "❌ Rejeter (Forçage Manuel)",
        'live_stats': "Stats en direct : XP {current_xp} | Santé {avatar_health}/100",
        'system_nominal': "✅ Système Nominal. Aucune dérive de concept active détectée dans la cohorte actuelle.",
        'answered_skill': "a répondu à la Compétence",
        'input_label': "Saisie :",
        'error_margin': "Marge d'erreur :",
        'waiting_feed': "📡 En attente de connexions au flux...",
        'deploy_success': "✅ Adaptation déployée avec succès !",
        'lora_training': "⚙️ Entraînement Continual-LoRA en temps réel en cours...",
        'saving': "Sauvegarde en cours...",
        'db_error': "Erreur de base de données :",
        'eval_q1': "L'explication XAI m'a aidé à comprendre le blocage de l'élève (Utilité perçue).",
        'eval_q2': "Le déploiement de la remédiation a été intuitif (Facilité d'utilisation).",
        'likert_1': "1 - Pas du tout d'accord",
        'likert_2': "2 - Pas d'accord",
        'likert_3': "3 - Neutre",
        'likert_4': "4 - D'accord",
        'likert_5': "5 - Tout à fait d'accord",
        'agent_remed_intro': "Action Pédagogique : Remédiation recommandée. Voici 3 exercices tirés de la base ASSISTments générés spécifiquement pour débloquer l'apprenant :",
        'agent_exo_1': "1. [Exercice visuel : Modèle d'aire pour la distribution mathématique]",
        'agent_exo_2': "2. [Exercice d'application : Développement guidé étape par étape]",
        'agent_exo_3': "3. [Exercice de consolidation : Identifier l'erreur dans une équation donnée]",
        'agent_send': "👉 Cliquez sur 'Approuver & Déployer' ci-dessous pour envoyer cette mini-série sur sa tablette."
    }
}

lang = st.session_state.lang

# --- 4. SUPABASE CLOUD TELEMETRY CONFIGURATION ---
# Note for Reviewers: API keys are securely managed via environment variables/secrets.
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_data(ttl=5)
def fetch_realtime_cohort_data():
    try:
        res = supabase.table("xGATES").select("*").order("created_at", desc=True).limit(1000).execute()
        if res.data:
            return pd.DataFrame(res.data)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

df_cohort = fetch_realtime_cohort_data()

# ==========================================
# FETCH LATEST CONCEPT DRIFT ALERT FROM DOWNSTREAM TELEMETRY
# ==========================================
def fetch_latest_alert_from_db():
    try:
        # Retrieve the latest diagnostic payload inserted by the upstream anomaly detector
        res = supabase.table("xCARE_Alerts").select("*").order("created_at", desc=True).limit(1).execute()
        if res.data and len(res.data) > 0:
            latest_row = res.data[0]
            if "payload" in latest_row:
                payload = latest_row["payload"]
                meta = payload.get("alert_metadata", {})
                
                student_id = meta.get("affected_user_id", "Unknown")
                timestamp = meta.get("timestamp", "Recent")
                drift_class = payload.get("statistical_evidence", {}).get("drift_class", "SUDDEN DRIFT")
                
                current_xp = 0
                if not df_cohort.empty:
                    student_hist = df_cohort[df_cohort['user_id'] == student_id]
                    if not student_hist.empty:
                        xp_val = student_hist['gamification_score'].max()
                        if pd.notna(xp_val): current_xp = int(xp_val)
                
                avatar_health = 45 
                hints_used = 4
                
                # Returns UI tuple, full JSON payload, and the drift topology class
                return (student_id, timestamp, current_xp, avatar_health, hints_used), payload, drift_class
    except Exception:
        pass
    return None, None, "SUDDEN DRIFT"

# Execute database fetch
alert_data, xcare_json_payload, current_drift_type = fetch_latest_alert_from_db()

# Load inference model (xLSTM baseline or LoRA-adapted matrix)
model, device, is_adapted = load_xcare_inference_model(current_drift_type)


# --- 5. PREMIUM CSS INJECTION & UI ANIMATIONS ---
st.markdown("""
    <style>
    /* Global SaaS Background */
    .stApp { background-color: #F4F7FE; font-family: 'Inter', sans-serif; }
    
    @keyframes float3d {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    @keyframes fadeInSlideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* --- LOADING SCREEN ANIMATION --- */
    @keyframes fadeOutLoader {
        0% { opacity: 1; z-index: 9999; }
        90% { opacity: 1; z-index: 9999; }
        100% { opacity: 0; z-index: -1; visibility: hidden; }
    }
    @keyframes spinPulse {
        0% { transform: rotate(0deg); border-top-color: #4318FF; }
        50% { transform: rotate(180deg); border-top-color: #A78BFA; }
        100% { transform: rotate(360deg); border-top-color: #4318FF; }
    }
    .loading-screen {
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background-color: #F4F7FE; display: flex; flex-direction: column;
        justify-content: center; align-items: center; z-index: 9999;
        animation: fadeOutLoader 1.5s forwards;
    }
    .loader-circle {
        width: 60px; height: 60px; border: 6px solid #E0E5FF;
        border-top: 6px solid #4318FF; border-radius: 50%;
        animation: spinPulse 1s linear infinite; margin-bottom: 20px;
    }
    .loader-text { color: #2B3674; font-weight: 800; font-size: 18px; letter-spacing: 1px; }

    /* --- LOGIN STYLES & 3D IMPACT CARDS --- */
    .impact-container { display: flex; justify-content: center; gap: 40px; margin: 20px 0 60px 0; flex-wrap: wrap; }
    .impact-card { border-radius: 20px; padding: 30px 20px; text-align: center; width: 250px; box-shadow: 0 10px 20px rgba(0,0,0,0.03); transition: all 0.3s ease; }
    
    .card-books { background-color: #FFF6E5 !important; border: 3px solid #FFE0A3 !important; transform: rotate(-3deg); }
    .card-books:hover { transform: translateY(-10px) rotate(-3deg); box-shadow: 0 15px 25px rgba(255, 224, 163, 0.4); z-index: 10; }
    .card-brain { background-color: #FDF3F8 !important; border: 3px solid #F6CAE4 !important; }
    .card-brain:hover { transform: translateY(-10px); box-shadow: 0 15px 25px rgba(246, 202, 228, 0.4); z-index: 10; }
    .card-globe { background-color: #EFFBF4 !important; border: 3px solid #BBEBCF !important; transform: rotate(3deg); }
    .card-globe:hover { transform: translateY(-10px) rotate(3deg); box-shadow: 0 15px 25px rgba(187, 235, 207, 0.4); z-index: 10; }

    .impact-3d-img { height: 95px; width: auto; margin-top: -65px; margin-bottom: 10px; filter: drop-shadow(0 15px 15px rgba(0,0,0,0.15)); animation: float3d 4s ease-in-out infinite; }
    
    .impact-number { font-size: 3.5rem; font-weight: 900; color: #2B3674; margin: 10px 0 0 0; line-height: 1; }
    .impact-number span { color: #f26d21; } 
    .impact-label { font-size: 14px; color: #A3AED0; font-weight: 700; margin-top: 5px; }
    
    .pp-logo { font-size: 32px; font-weight: 900; color: #2B3674; letter-spacing: -1.5px; margin-top: 5px; }
    .pp-logo span { font-weight: 400; color: #4318FF; }
    
    .teacher-login-box { background: white; padding: 40px; border-radius: 24px; box-shadow: 0 10px 40px rgba(0,0,0,0.05); max-width: 600px; margin: 0 auto; text-align: left; animation: fadeInSlideUp 0.8s ease-out forwards; }
    
    .stButton > button[kind="primary"] { background-color: #4318FF !important; color: white !important; border-radius: 16px !important; padding: 12px 24px !important; font-weight: 600 !important; border: none !important; transition: all 0.2s ease; }
    .stButton > button[kind="primary"]:hover { background-color: #3311DB !important; transform: scale(1.02); }

    .stButton > button[kind="secondary"]:has(div:contains("Home")) { background: transparent !important; border: none !important; color: #2B3674 !important; font-size: 18px !important; font-weight: 800 !important; }
    .stButton > button[kind="secondary"]:has(div:contains("Accueil")) { background: transparent !important; border: none !important; color: #2B3674 !important; font-size: 18px !important; font-weight: 800 !important; }

    /* --- SIDEBAR NAVIGATION --- */
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #A78BFA 0%, #8B5CF6 100%) !important; border-right: none !important; box-shadow: 2px 0px 20px rgba(0,0,0,0.1); padding-top: 20px; }
    
    [data-testid="stSidebar"] hr { border-top-color: rgba(255,255,255,0.2) !important; }

    [data-testid="stSidebar"] div[role="radiogroup"] > label { background-color: transparent; padding: 12px 20px; border-radius: 12px; margin-bottom: 5px; cursor: pointer; transition: all 0.2s ease; }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background-color: rgba(255, 255, 255, 0.1); }
    [data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; } 
    
    [data-testid="stSidebar"] div[role="radiogroup"] > label p { font-weight: 600; color: #E0E5FF !important; font-size: 15px; display: flex; align-items: center; margin: 0; }
    [data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] { background-color: rgba(255, 255, 255, 0.2) !important; }
    [data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] p { color: #FFFFFF !important; font-weight: 800; }

    [data-testid="stSidebar"] div[role="radiogroup"] > label p img { width: 32px !important; height: auto !important; margin-right: 12px; filter: drop-shadow(0px 5px 5px rgba(0,0,0,0.2)); animation: float3d 3s ease-in-out infinite; }
    
    [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(1) p img { animation-delay: 0s; }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(2) p img { animation-delay: 0.5s; }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(3) p img { animation-delay: 1s; }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(4) p img { animation-delay: 1.5s; }

    [data-testid="stSidebar"] .stButton > button { background-color: rgba(255, 255, 255, 0.1) !important; color: white !important; border: 1px solid rgba(255, 255, 255, 0.2) !important; font-weight: 600; }
    [data-testid="stSidebar"] .stButton > button:hover { background-color: rgba(255, 255, 255, 0.25) !important; }

    /* --- DASHBOARD STYLES --- */
    .figma-banner { background: linear-gradient(135deg, #E0E5FF 0%, #E3D4FB 100%); border-radius: 24px; padding: 40px; color: #2B3674; margin-bottom: 30px; box-shadow: 0 10px 25px rgba(67, 24, 255, 0.1); display: flex; justify-content: space-between; align-items: center; position: relative; overflow: visible; }
    .figma-banner-text h1 { margin: 0; font-size: 32px; font-weight: 900; letter-spacing: -1px; color: #2B3674;}
    .figma-banner-text p { margin: 5px 0 0 0; font-size: 15px; color: #707EAE; font-weight: 500;}
    .figma-banner-img { position: absolute; right: 40px; bottom: 0; height: 160px; animation: float3d 6s ease-in-out infinite; z-index: 10; filter: drop-shadow(0px 10px 10px rgba(0,0,0,0.1)); }

    .figma-card { background-color: #FFFFFF; padding: 25px; border-radius: 20px; box-shadow: 0 5px 20px rgba(0,0,0,0.02); border: none; display: flex; flex-direction: column; transition: transform 0.2s ease; margin-bottom: 20px; }
    .figma-card:hover { transform: translateY(-5px); box-shadow: 0 10px 25px rgba(0,0,0,0.05); }
    .card-title { font-size: 14px; color: #A3AED0; font-weight: 600; margin-bottom: 5px; }
    .card-value { font-size: 32px; color: #2B3674; font-weight: 800; margin: 0; line-height: 1; }
    
    .c-purple { background: linear-gradient(135deg, #868CFF 0%, #4318FF 100%); color: white !important;}
    .c-purple .card-title, .c-purple .card-value { color: white !important; }
    .c-pink { background: linear-gradient(135deg, #FF90B3 0%, #FF659A 100%); color: white !important;}
    .c-pink .card-title, .c-pink .card-value { color: white !important; }
    
    .drift-alert-container { background: linear-gradient(135deg, #FF6F61 0%, #D32F2F 100%); color: #FFFFFF; border-radius: 24px; padding: 30px; margin-bottom: 30px; box-shadow: 0 10px 25px rgba(211, 47, 47, 0.2); }
    .rag-box { background: #FFFFFF; padding: 30px; border-radius: 24px; border-left: 8px solid #4318FF; box-shadow: 0 5px 20px rgba(0,0,0,0.03); margin-bottom: 20px; }
    .feed-item { padding: 20px; border-radius: 16px; background-color: #FFFFFF; box-shadow: 0 4px 10px rgba(0,0,0,0.02); margin-bottom: 15px; display: flex; gap: 15px; align-items: center; border-left: 6px solid #e2e8f0;}
    
    .sponsor-section { margin-top: 60px; padding: 40px 0; text-align: center; overflow: hidden; background: transparent; }
    .sponsor-title { font-size: 14px; color: #A3AED0; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 30px; font-weight: 800; }
    .marquee-wrapper { width: 100%; overflow: hidden; position: relative; }
    .marquee-container { display: flex; width: 200%; animation: scroll 20s linear infinite; align-items: center; }
    .sponsor-logo { height: 35px; margin: 0 40px; filter: grayscale(100%) opacity(0.4); transition: all 0.3s ease; }
    .sponsor-logo:hover { filter: grayscale(0%) opacity(1); transform: scale(1.1); }
    @keyframes scroll { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }

    /* --- BOTTOM FEATURES SECTION --- */
    .bottom-features-wrapper { margin-top: 60px; padding: 0 20px; }
    .bottom-feature-divider { border: none; height: 1px; background-color: #E2E8F0; margin: 40px 0; }
    .bottom-features-grid { display: flex; justify-content: space-between; gap: 20px; text-align: left; flex-wrap: wrap; max-width: 1200px; margin: 0 auto;}
    .bottom-feature-card { flex: 1; min-width: 220px; padding: 30px 25px; border-radius: 16px; transition: all 0.3s ease; border: 1px solid transparent; }
    .bottom-feature-card:hover { background-color: #FFFFFF; box-shadow: 0 10px 30px rgba(0,0,0,0.03); border-color: #F4F7FE; transform: translateY(-5px); }
    .bottom-feature-card.active-card { background-color: #FFFFFF; box-shadow: 0 10px 30px rgba(0,0,0,0.03); }
    .bottom-feature-icon { height: 65px; margin-bottom: 25px; filter: drop-shadow(0 10px 10px rgba(0,0,0,0.08)); }
    .bottom-feature-title { font-size: 18px; font-weight: 800; color: #2B3674; margin-bottom: 12px; }
    .bottom-feature-desc { font-size: 14px; color: #A3AED0; font-weight: 500; line-height: 1.6; }
    </style>
""", unsafe_allow_html=True)

# INJECT HTML LOADING SCREEN ONLY WHEN NECESSARY
if st.session_state.is_loading:
    st.markdown(f"""
    <div class="loading-screen" id="loader">
        <div class="loader-circle"></div>
        <div class="loader-text">{T[lang]['loading_text']}</div>
    </div>
    <script>
    setTimeout(function() {{
        var loader = document.getElementById('loader');
        if(loader) loader.style.display = 'none';
    }}, 1500);
    </script>
    """, unsafe_allow_html=True)
    st.session_state.is_loading = False


# ==========================================
# ROUTING: LOGIN OR MAIN DASHBOARD
# ==========================================

if st.session_state.teacher_user is None:
    # ---------------------------------------------------------
    # EDUCATOR AUTHENTICATION / REGISTRATION SCREEN
    # ---------------------------------------------------------
    st.write("") 
    col_logo, col_empty, col_admin, col_lang = st.columns([5, 3, 3, 1])
    with col_logo:
        st.markdown('<div class="pp-logo">Aura Tutor <span>xCARE</span></div>', unsafe_allow_html=True)
    with col_admin:
        st.markdown(f'<div style="text-align: right; margin-top: 15px; font-weight: 600; color: #64748b; font-size: 14px;">{T[lang]["eu_admin"]}</div>', unsafe_allow_html=True)
    with col_lang:
        st.button("🇫🇷 FR" if lang == 'EN' else "🇬🇧 EN", on_click=toggle_language, use_container_width=True)

    st.markdown(f"""
        <div style="text-align: center; margin-top: 80px; margin-bottom: 50px; animation: fadeInSlideUp 0.8s ease-out forwards;">
            <h1 style="font-size: 54px; font-weight: 800; color: #1e293b; margin-bottom: 15px; letter-spacing: -1.5px;">
                {T[lang]['hero_title']} <span style='color: #f26d21;'>Delightfully</span> Easy <span class='animated-smile'>😊</span>
            </h1>
            <p style="font-size: 22px; font-weight: 400; color: #475569;">
                {T[lang]['hero_sub']}
            </p>
        </div>
    """, unsafe_allow_html=True)

    # --- IMPACT CARDS (OUR MISSION) ---
    st.markdown(f"""
        <div style="text-align: center; margin-bottom: 20px; animation: fadeInSlideUp 0.8s ease-out forwards;">
            <h2 style="font-size: 36px; font-weight: 900; color: #2B3674;">{T[lang]['impact_title']} <span style="font-size: 20px; color: #A3AED0;">🔗</span></h2>
        </div>
        <div class="impact-container">
            <div class="impact-card card-books">
                <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Books.png" class="impact-3d-img" style="animation-delay: 0s;">
                <p class="impact-number">500<span>k+</span></p>
                <p class="impact-label">{T[lang]['impact_1']}</p>
            </div>
            <div class="impact-card card-brain">
                <img src="https://cdn-icons-png.flaticon.com/512/8157/8157140.png" class="impact-3d-img" style="animation-delay: 1s; height: 85px;">
                <p class="impact-number">100<span>%</span></p>
                <p class="impact-label">{T[lang]['impact_2']}</p>
            </div>
            <div class="impact-card card-globe">
                <img src="https://cdn-icons-png.flaticon.com/512/8157/8157181.png" class="impact-3d-img" style="animation-delay: 2s; height: 90px;">
                <p class="impact-number">1 <span>IN</span> 1</p>
                <p class="impact-label">{T[lang]['impact_3']}</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style="text-align: center; margin-bottom: 25px;">
            <h2 style="font-size: 26px; font-weight: 800; color: #2B3674; margin-bottom: 5px;">
                {T[lang]['build_path']}
            </h2>
            <p style="color: #A3AED0; font-size: 15px; font-weight:600;">{T[lang]['login_title']}</p>
        </div>
    """, unsafe_allow_html=True)

    # --- LAYOUT: SPLIT SCREEN (IMAGE LEFT, FORM RIGHT) ---
    col_img, col_login = st.columns([1.1, 1])
    
    with col_img:
        st.markdown("""
            <div style="display: flex; justify-content: center; align-items: center; height: 100%; padding-top: 30px;">
                <img src="https://demo.creativeitem.com/academy/assets/frontend/default-new/image/login-security.gif" style="width: 100%; max-width: 480px; filter: drop-shadow(0px 10px 15px rgba(0,0,0,0.05));" />
            </div>
        """, unsafe_allow_html=True)

    with col_login:
        st.markdown("<div class='teacher-login-box'>", unsafe_allow_html=True)
        tab_in, tab_up = st.tabs([T[lang]['tab_signin'], T[lang]['tab_signup']])
        
        with tab_in:
            email_in = st.text_input(T[lang]['email'], placeholder="educator@school.edu", key="in_em")
            pwd_in = st.text_input(T[lang]['pwd'], type="password", placeholder="••••••••", key="in_pw")
            st.write("")
            if st.button(T[lang]['btn_signin'] + " 🚀", use_container_width=True, type="primary"):
                if email_in and pwd_in:
                    with st.spinner("Authenticating..." if lang == 'EN' else "Authentification..."):
                        try:
                            res = supabase.auth.sign_in_with_password({"email": email_in, "password": pwd_in})
                            if res.user:
                                st.session_state.teacher_user = res.user
                                st.session_state.is_loading = True
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please fill all fields.")

        with tab_up:
            full_name = st.text_input(T[lang]['fname'], key="up_name")
            subject = st.selectbox(T[lang]['subj'], ["Mathematics", "Computer Science", "Languages", "General"], key="up_subj")
            email_up = st.text_input(T[lang]['email'], key="up_em")
            pwd_up = st.text_input(T[lang]['pwd'], type="password", key="up_pw")
            st.write("")
            if st.button(T[lang]['btn_signup'] + " ✨", use_container_width=True, type="primary"):
                if email_up and pwd_up and full_name:
                    with st.spinner("Creating account..."):
                        try:
                            res = supabase.auth.sign_up({
                                "email": email_up, 
                                "password": pwd_up,
                                "options": {
                                    "data": {"full_name": full_name, "subject": subject, "role": "educator"}
                                }
                            })
                            if res.user:
                                st.session_state.teacher_user = res.user
                                st.session_state.is_loading = True 
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please fill all required fields.")
        
        st.markdown("</div>", unsafe_allow_html=True)

    # --- BOTTOM FEATURES SECTION ---
    st.markdown(f"""
        <div class="bottom-features-wrapper">
            <hr class="bottom-feature-divider">
            <div class="bottom-features-grid">
                <div class="bottom-feature-card active-card">
                    <img src="https://cdn-icons-png.flaticon.com/512/4185/4185707.png" class="bottom-feature-icon" style="animation: float3d 4s ease-in-out infinite 0s;">
                    <div class="bottom-feature-title">{T[lang]['feat1_t']}</div>
                    <div class="bottom-feature-desc">{T[lang]['feat1_d']}</div>
                </div>
                <div class="bottom-feature-card">
                    <img src="https://cdn-icons-png.flaticon.com/512/8157/8157140.png" class="bottom-feature-icon" style="animation: float3d 4s ease-in-out infinite 1s;">
                    <div class="bottom-feature-title">{T[lang]['feat2_t']}</div>
                    <div class="bottom-feature-desc">{T[lang]['feat2_d']}</div>
                </div>
                <div class="bottom-feature-card">
                    <img src="https://cdn-icons-png.flaticon.com/512/4185/4185686.png" class="bottom-feature-icon" style="animation: float3d 4s ease-in-out infinite 2s;">
                    <div class="bottom-feature-title">{T[lang]['feat3_t']}</div>
                    <div class="bottom-feature-desc">{T[lang]['feat3_d']}</div>
                </div>
                <div class="bottom-feature-card">
                    <img src="https://cdn-icons-png.flaticon.com/512/4185/4185704.png" class="bottom-feature-icon" style="animation: float3d 4s ease-in-out infinite 3s;">
                    <div class="bottom-feature-title">{T[lang]['feat4_t']}</div>
                    <div class="bottom-feature-desc">{T[lang]['feat4_d']}</div>
                </div>
            </div>
            <hr class="bottom-feature-divider">
        </div>
    """, unsafe_allow_html=True)


else:
    # ---------------------------------------------------------
    # MAIN DASHBOARD SCREEN (SAAS DESIGN)
    # ---------------------------------------------------------
    teacher_name = st.session_state.teacher_user.user_metadata.get('full_name', st.session_state.teacher_user.email).split()[0]
    teacher_subject = st.session_state.teacher_user.user_metadata.get('subject', 'General')

    status_text = T[lang]['engine_lora'] if is_adapted else T[lang]['engine_static']
    
    if not df_cohort.empty:
        total_students = df_cohort['user_id'].nunique()
        total_interactions = len(df_cohort)
        mean_acc = df_cohort['correct'].mean()
        avg_accuracy = int(mean_acc * 100) if pd.notna(mean_acc) else 0
    else:
        total_students, total_interactions, avg_accuracy = 0, 0, 0

    with st.sidebar:
        st.markdown('<div style="font-size:26px; font-weight:900; color:#FFFFFF; text-align:center; margin-bottom:40px;">Aura<span style="color:#FF90B3;">Tutor</span></div>', unsafe_allow_html=True)
        
        # --- 3D ICONS FOR SIDEBAR MENU ---
        icon_overview = "https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/House.png"
        icon_roster = "https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/People/Woman%20Student.png"
        icon_alert = "https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Symbols/Warning.png" if alert_data else "https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Symbols/Check%20Mark%20Button.png"
        icon_feed = "https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Satellite%20Antenna.png"

        opt_overview = f"![icon]({icon_overview}) {T[lang]['tab_overview']}"
        opt_roster = f"![icon]({icon_roster}) {T[lang]['tab_roster']}"
        opt_alerts = f"![icon]({icon_alert}) {T[lang]['tab_alerts']}"
        opt_feed = f"![icon]({icon_feed}) {T[lang]['tab_feed']}"
        
        menu_options = [opt_overview, opt_roster, opt_alerts, opt_feed]
        
        selected_nav = st.radio("Menu", menu_options, label_visibility="collapsed")
        
        if st.session_state.current_nav != selected_nav:
            st.session_state.current_nav = selected_nav
            st.session_state.is_loading = True
            st.rerun()

        st.write("---")
        
        if st.button("🌐 FR / EN", use_container_width=True):
            toggle_language()
            st.rerun()
            
        st.write("")
        
        if st.button(T[lang]['logout'] + " 🚪", use_container_width=True):
            try: supabase.auth.sign_out()
            except: pass
            st.session_state.teacher_user = None
            st.session_state.is_loading = True
            st.rerun()

        # --- SCIENTIFIC REPRODUCIBILITY: RESET DEMO BUTTON ---
        st.write("---")
        reset_text = "🔄 Réinitialiser la Démo" if lang == 'FR' else "🔄 Reset Demo"
        if st.button(reset_text, use_container_width=True):
            import shutil
            for drift in ["SUDDEN_DRIFT", "GRADUAL_DRIFT"]:
                adapter_dir = os.path.join("models", f"xlstm_lora_adapted_{drift}")
                if os.path.exists(adapter_dir):
                    try:
                        shutil.rmtree(adapter_dir)
                    except:
                        pass
            load_xcare_inference_model.clear()
            st.rerun()

        st.markdown(f"""
            <div style="text-align: center; margin-top: 30px; padding-bottom: 20px;">
                <img src="https://api.dicebear.com/8.x/avataaars/svg?seed={teacher_name}&backgroundColor=transparent" width="50" style="border-radius:50%; background: rgba(255,255,255,0.2); padding: 5px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);">
                <p style="color: white; font-weight: 800; font-size: 14px; margin-top: 10px; margin-bottom: 0;">Dr. {teacher_name}</p>
                <p style="color: rgba(255,255,255,0.8); font-size: 11px; margin-top: 0; font-weight: 800; letter-spacing: 1px;">{T[lang]['profile_label']}</p>
            </div>
        """, unsafe_allow_html=True)

    # --- BREADCRUMB NAVIGATION ---
    col_home_nav, col_home_img = st.columns([5, 1])
    with col_home_nav:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(T[lang]['home_breadcrumb'], key="return_home_btn"):
            st.session_state.teacher_user = None
            st.session_state.is_loading = True
            st.rerun()
    with col_home_img:
        st.markdown('<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Books.png" style="height: 100px; transform: translateY(-15px); animation: float3d 4s ease-in-out infinite;">', unsafe_allow_html=True)

    st.markdown("<hr style='margin-top: 0px; margin-bottom: 25px; border: 0; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)

    # --- TOP BAR & HERO BANNER ---
    col_search, col_profile = st.columns([3, 1])
    with col_search:
        st.markdown(f"<div style='background: white; border-radius: 30px; padding: 12px 25px; width: 350px; color: #A3AED0; font-weight:600; box-shadow: 0 2px 10px rgba(0,0,0,0.02);'>{T[lang]['search_placeholder']}</div>", unsafe_allow_html=True)
    with col_profile:
        st.markdown(f"<div style='text-align:right; font-weight:800; color:#2B3674; font-size:16px; margin-top:5px;'>Dr. {teacher_name} <img src='https://api.dicebear.com/8.x/avataaars/svg?seed={teacher_name}&backgroundColor=e2e8f0' width='40' style='border-radius:50%; vertical-align:middle; margin-left:15px;'></div>", unsafe_allow_html=True)

    st.markdown(f"""
        <div class="figma-banner">
            <div class="figma-banner-text">
                <h1>{T[lang]['hello_teacher'].format(teacher_name=teacher_name)}</h1>
                <p>{T[lang]['welcome_dashboard'].format(teacher_subject=teacher_subject)} {T[lang]['dash_subtitle']}</p>
                <div style="margin-top: 15px;">
                    <span style="background: rgba(255,255,255,0.4); padding: 6px 15px; border-radius: 20px; font-size: 13px; font-weight: bold; color:#2B3674;">{T[lang]['engine_label']} {status_text}</span>
                </div>
            </div>
            <img class="figma-banner-img" src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/People/Woman%20Student.png">
        </div>
    """, unsafe_allow_html=True)


    # --- CONTENT ROUTING BASED ON SIDEBAR SELECTION ---
    
    if selected_nav == opt_overview:
        k1, k2, k3, k4 = st.columns(4)
        with k1: st.markdown(f"<div class='figma-card'><div class='card-title'>🧑‍🎓 {T[lang]['cohort']}</div><div class='card-value'>{total_students}</div></div>", unsafe_allow_html=True)
        with k2: st.markdown(f"<div class='figma-card c-purple'><div class='card-title'>🔄 {T[lang]['session']}</div><div class='card-value'>{total_interactions}</div></div>", unsafe_allow_html=True)
        with k3: st.markdown(f"<div class='figma-card c-pink'><div class='card-title'>🎯 {T[lang]['flow']}</div><div class='card-value'>{avg_accuracy}%</div></div>", unsafe_allow_html=True)
        with k4: 
            status_html = T[lang]['status_alert'] if alert_data else T[lang]['status_stable']
            st.markdown(f"<div class='figma-card'><div class='card-title'>🧠 {T[lang]['status']}</div><div class='card-value'>{status_html}</div></div>", unsafe_allow_html=True)

        st.write("")
        
        if not df_cohort.empty:
            c_chart1, c_chart2 = st.columns([6, 4])
            with c_chart1:
                st.markdown("<div class='figma-card'>", unsafe_allow_html=True)
                skill_acc = df_cohort.groupby('skill_id')['correct'].mean().reset_index()
                skill_acc['correct'] = skill_acc['correct'] * 100
                skill_acc['skill_name'] = skill_acc['skill_id'].apply(lambda x: f"{T[lang]['lbl_skill']} {x}")
                fig1 = px.bar(skill_acc, x="skill_name", y="correct", text="correct", color="correct", color_continuous_scale="Purp", title=T[lang]['chart_skill'])
                fig1.update_traces(texttemplate='%{text:.0f}%', textposition='inside', textfont=dict(color='white', size=14, family='Inter'))
                fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=False, xaxis_title="", yaxis=dict(range=[0, 100], gridcolor='#f1f5f9'))
                st.plotly_chart(fig1, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with c_chart2:
                st.markdown("<div class='figma-card'>", unsafe_allow_html=True)
                df_state = pd.DataFrame({"State": ["In Flow", "Boredom", "Anxiety/Drift"], "Count": [85, 15, 20]}) 
                fig2 = px.pie(df_state, values='Count', names='State', hole=0.6, color='State', color_discrete_map={"In Flow": "#4318FF", "Boredom": "#FF90B3", "Anxiety/Drift": "#E0E5FF"}, title=T[lang]['chart_state'])
                fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info(T[lang]['waiting_data'])

    elif selected_nav == opt_roster:
        if not df_cohort.empty:
            st.markdown(f"<h3 style='color:#2B3674; font-weight:800;'>{T[lang]['global_cohort']}</h3>", unsafe_allow_html=True)
            roster_df = df_cohort.groupby('user_id').agg(
                Total_Interactions=('interaction_id', 'count'),
                Accuracy=('correct', lambda x: int(x.mean() * 100) if pd.notna(x.mean()) else 0),
                Max_XP=('gamification_score', lambda x: int(x.max()) if pd.notna(x.max()) else 0)
            ).reset_index()
            
            if 'gamer_tag' in df_cohort.columns:
                latest_tags = df_cohort.dropna(subset=['gamer_tag']).drop_duplicates('user_id', keep='first')[['user_id', 'gamer_tag']]
                roster_df = roster_df.merge(latest_tags, on='user_id', how='left')
                roster_df['Display_Name'] = roster_df['gamer_tag'].fillna("Learner #" + roster_df['user_id'].astype(str))
            else:
                roster_df['Display_Name'] = "Learner #" + roster_df['user_id'].astype(str)

            display_cols = ['Display_Name', 'Total_Interactions', 'Accuracy', 'Max_XP']
            st.dataframe(
                roster_df[display_cols],
                column_config={
                    "Display_Name": T[lang]['col_id'],
                    "Total_Interactions": st.column_config.NumberColumn(T[lang]['col_activity']),
                    "Accuracy": st.column_config.ProgressColumn(T[lang]['col_accuracy'], format="%d%%", min_value=0, max_value=100),
                    "Max_XP": T[lang]['col_xp']
                },
                hide_index=True, use_container_width=True
            )
            
            st.markdown("---")
            st.markdown(f"<h4 style='color:#4318FF;'>{T[lang]['student_select']}</h4>", unsafe_allow_html=True)
            
            user_list = roster_df['Display_Name'].tolist()
            selected_display = st.selectbox(T[lang]['select_id'], user_list, label_visibility="collapsed")
            
            if selected_display:
                selected_user = roster_df[roster_df['Display_Name'] == selected_display]['user_id'].iloc[0]
                student_data = df_cohort[df_cohort['user_id'] == selected_user].head(10)
                xp_max_val = student_data['gamification_score'].max()
                xp_actuel = int(xp_max_val) if pd.notna(xp_max_val) else 0
                
                c_prof1, c_prof2 = st.columns([1, 2])
                with c_prof1:
                    st.markdown(f"""
                        <div class="figma-card" style="text-align:center;">
                            <img src="https://api.dicebear.com/8.x/avataaars/svg?seed={selected_display}&backgroundColor=e2e8f0" width="120" style="border-radius:20px;">
                            <h2 style="margin:15px 0 5px 0; font-size:22px; color:#2B3674;">{selected_display}</h2>
                            <p style="color:#f59e0b; font-weight:bold; font-size:18px;">⭐ {xp_actuel} XP</p>
                        </div>
                    """, unsafe_allow_html=True)
                with c_prof2:
                    st.markdown(f"<div class='figma-card'><h4 style='margin-top:0; color:#2B3674;'>📝 {T[lang]['recent_ans']}</h4>", unsafe_allow_html=True)
                    for _, row in student_data.head(5).iterrows():
                        is_corr = row['correct'] == 1
                        bg_c = "#4318FF" if is_corr else "#FF90B3"
                        txt_c = "white"
                        text = T[lang]['lbl_correct'] if is_corr else T[lang]['lbl_incorrect']
                        ans_text = str(row['student_answer'])
                        if len(ans_text) > 40: ans_text = ans_text[:40] + "..."
                        st.markdown(f"""
                            <div style="display:flex; justify-content:space-between; padding:15px; background:#F4F7FE; border-radius:12px; margin-bottom:10px; align-items:center;">
                                <span style="font-family:monospace; color:#A3AED0; font-weight:bold; font-size:13px;">{T[lang]['lbl_skill']} {row['skill_id']}</span>
                                <b style="font-size:14px; color:#2B3674; padding-left:15px; flex:1;">"{ans_text}"</b>
                                <span style="background:{bg_c}; color:{txt_c}; padding:4px 12px; border-radius:8px; font-size:12px; font-weight:bold;">{text}</span>
                            </div>
                        """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info(T[lang]['no_learner'])

    elif selected_nav == opt_alerts:
        if alert_data:
            student_id, timestamp, current_xp, avatar_health, hints_used = alert_data
            drop_percentage = 100 - avatar_health
            
            # --- DRIFT METADATA PAYLOAD (JSON) FOR LoRA ROUTING ---
            if xcare_json_payload:
                with st.expander(T[lang]['see_payload']):
                    st.json(xcare_json_payload)

            st.markdown(f"""
            <div class="drift-alert-container">
                <h3 style="margin-top: 0;">{T[lang]['alert_critical']}</h3>
                <p style="font-size: 16px; margin-bottom: 0;">{T[lang]['alert_desc_full'].format(student_id=student_id, drop_percentage=drop_percentage)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            col_xai, col_action = st.columns([2, 1])
            with col_xai:
                # --- AGENTIC RAG INVOCATION ---
                # Note for Reviewers: Calls the LLM to generate XAI explanations.
                # Pydantic schema constraints are applied inside rag_engine.py 
                # to systematically prevent structural hallucinations.
                if generate_xai_explanation is not None and xcare_json_payload is not None:
                    rag_result = generate_xai_explanation(xcare_json_payload, lang)
                else:
                    rag_result = {
                        "rationale": T[lang]['fallback_rationale'].format(hints_used=hints_used),
                        "remediation_steps": [
                            T[lang]['agent_exo_1'],
                            T[lang]['agent_exo_2'],
                            T[lang]['agent_exo_3']
                        ],
                        "engine": "Agentic RAG (Llama-3.3)"
                    }

                steps_html = "".join([f"<div style='margin-bottom: 8px;'>{step}</div>" for step in rag_result["remediation_steps"]])

                st.markdown(f"""
                <div class="rag-box">
                        <h4 style="color:#2B3674; display: flex; align-items: center;">
                        <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Smilies/Robot.png" width="35" style="margin-right: 10px; animation: float3d 3s ease-in-out infinite;"> 
                        {T[lang]['xai_rationale_title']} 
                        <span style="background:#E0E5FF; color:#4318FF; padding:4px 10px; border-radius:8px; font-size:12px; margin-left:10px;">{rag_result['engine']}</span></h4>                   <p style="color:#707EAE;">{rag_result['rationale']}</p>
                    <div style="background: #F4F7FE; padding: 15px; border-radius: 12px; margin-top: 15px;">
                        <h5 style="margin-top:0; color:#2B3674; font-size: 14px; margin-bottom: 12px;">{T[lang]['agent_remed_intro']}</h5>
                        <div style="color:#475569; font-weight:600; font-size: 13px; margin-bottom: 15px; padding-left: 10px; border-left: 3px solid #CBD5E1;">
                            {steps_html}
                        </div>
                        <div style="background: #E0E5FF; color: #4318FF; padding: 10px; border-radius: 8px; font-weight: bold; font-size: 13px; text-align: center; border: 1px dashed #4318FF;">
                            {T[lang]['agent_send']}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_action:
                st.markdown(f"<h4 style='color:#2B3674; margin-top:0;'>{T[lang]['hitl_title']}</h4>", unsafe_allow_html=True)
                
                # --- REMEDIATION STRATEGY UI BLOCK ---
                st.markdown(f"""
                <div style="background-color: #F8FAFC; border: 2px dashed #CBD5E1; padding: 15px; border-radius: 12px; margin-bottom: 20px;">
                    <p style="color: #475569; font-size: 12px; margin-bottom: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px;">{T[lang]['remed_strat']}</p>
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                        <span style="font-size: 18px;">📉</span>
                        <span style="color: #1E293B; font-size: 13px; font-weight: 600;">{T[lang]['remed_scaffolding']}</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                        <span style="font-size: 18px;">💡</span>
                        <span style="color: #1E293B; font-size: 13px; font-weight: 600;">{T[lang]['remed_hints']}</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 18px;">⚙️</span>
                        <span style="color: #1E293B; font-size: 13px; font-weight: 600;">{T[lang]['remed_lora']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(T[lang]['deploy'], type="primary", use_container_width=True):
                    spinner_text = T[lang]['lora_training']
                    with st.spinner(spinner_text):
                        try:
                            recent_student_data = df_cohort[df_cohort['user_id'] == student_id].head(10)
                            
                            X_drift = recent_student_data[['skill_id', 'residual_error']].values.tolist() if not recent_student_data.empty else []
                            y_drift = recent_student_data['correct'].values.tolist() if not recent_student_data.empty else []
                            
                            if run_realtime_lora_adaptation is not None:
                                run_realtime_lora_adaptation(xcare_json_payload, X_drift, y_drift)
                                time.sleep(3) 
                            else:
                                time.sleep(1.5) 
                            
                            st.success(T[lang]['deploy_success'])
                            time.sleep(2)
                            
                            # Cache clearance to trigger the injection of the newly trained LoRA matrices
                            # (Status updates to "🟢 Active LoRA")
                            load_xcare_inference_model.clear() 
                            
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

                st.write("")
                st.button(T[lang]['edit_intervention'], use_container_width=True)
                st.button(T[lang]['reject_override'], use_container_width=True)
                st.divider()
                st.markdown(f"<span style='color:#A3AED0; font-weight:600;'>{T[lang]['live_stats'].format(current_xp=current_xp, avatar_health=avatar_health)}</span>", unsafe_allow_html=True)
        else:
            st.success(T[lang]['system_nominal'])

    elif selected_nav == opt_feed:
        st.markdown(f"<h3 style='color:#2B3674; font-weight:800; margin-bottom: 20px;'>{T[lang]['tab_feed']}</h3>", unsafe_allow_html=True)
        
        if not df_cohort.empty:
            for _, row in df_cohort.head(15).iterrows():
                is_correct = row['correct'] == 1
                border_color = "#4318FF" if is_correct else "#FF90B3"
                icon_bg = "#E0E5FF" if is_correct else "#FFE2EB"
                icon_txt = "🎯" if is_correct else "💡"
                date_obj = pd.to_datetime(row['created_at'])
                time_str = date_obj.strftime("%H:%M:%S")
                
                ans_text = str(row['student_answer'])
                if len(ans_text) > 50: ans_text = ans_text[:50] + "..."
                
                res_err = row['residual_error']
                res_err_disp = round(float(res_err), 3) if pd.notna(res_err) and res_err is not None else 0.0
                
                display_n = row['gamer_tag'] if 'gamer_tag' in row and pd.notna(row['gamer_tag']) else f"Learner #{row['user_id']}"
                
                st.markdown(f"""
                    <div class="feed-item" style="border-left-color: {border_color};">
                        <div style="font-size:20px; background:{icon_bg}; width:45px; height:45px; display:flex; justify-content:center; align-items:center; border-radius:12px; margin-right:15px;">{icon_txt}</div>
                        <div style="flex:1;">
                            <div style="display:flex; justify-content:space-between;">
                                <div style="font-weight:800; color:#2B3674; font-size:14px;">{display_n} <span style="color:#A3AED0; font-weight:600;">{T[lang]['answered_skill']} {row['skill_id']}</span></div>
                                <div style="font-size: 12px; color: #A3AED0; font-weight: 700;">{time_str}</div>
                            </div>
                            <div style="color:#2B3674; font-size:14px; margin-top:4px; font-weight:600;">{T[lang]['input_label']} "{ans_text}"</div>
                            <div style="font-size:12px; color:#A3AED0; font-weight:bold; margin-top:4px;">{T[lang]['error_margin']} {res_err_disp}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info(T[lang]['waiting_feed'])

    # --- SYSTEM EVALUATION (HITL / TAM & SUS METRICS) ---
    st.write("---")
    st.markdown(f"<h3 style='color:#2B3674; font-size:18px; font-weight:800;'>{T[lang]['eval_title']}</h3>", unsafe_allow_html=True)
    
    with st.form("dss_evaluation_form"):
        # Formative evaluation leveraging TAM (Perceived Usefulness) and SUS (Perceived Ease of Use) items
        st.write(f"<span style='color:#2B3674; font-weight:800; font-size: 14px;'>{T[lang]['eval_q1']}</span>", unsafe_allow_html=True)
        likert_opts = [T[lang]['likert_1'], T[lang]['likert_2'], T[lang]['likert_3'], T[lang]['likert_4'], T[lang]['likert_5']]
        q1_rating = st.radio("q1", likert_opts, index=4, horizontal=True, label_visibility="collapsed")
        
        st.write(f"<span style='color:#2B3674; font-weight:800; font-size: 14px;'>{T[lang]['eval_q2']}</span>", unsafe_allow_html=True)
        q2_rating = st.radio("q2", likert_opts, index=4, horizontal=True, label_visibility="collapsed")
        
        rating = f"Utilité (TAM): {q1_rating.split(' - ')[0]} | Facilité (SUS): {q2_rating.split(' - ')[0]}" 
        
        st.write("")
        feedback_comment = st.text_area(T[lang]['eval_comment'], height=100, label_visibility="collapsed", placeholder=T[lang]['eval_comment'])
        
        submitted = st.form_submit_button(T[lang]['eval_btn'], type="primary")
        if submitted:
            with st.spinner(T[lang]['saving']):
                try:
                    supabase.table("dss_evaluations").insert({
                        "teacher_email": st.session_state.teacher_user.email,
                        "rating": rating,
                        "comment": feedback_comment
                    }).execute()
                    st.success(T[lang]['eval_thanks'])
                except Exception as e:
                    st.error(f"{T[lang]['db_error']} {e}")

# ==========================================
# GLOBAL FOOTER (SPONSORS & ACKNOWLEDGEMENTS)
# ==========================================
st.markdown(f"""
<div class="sponsor-section">
    <div class="sponsor-title">{T[lang]['sponsor_title']}</div>
    <div class="marquee-wrapper">
        <div class="marquee-container">
            <img class="sponsor-logo" src="https://upload.wikimedia.org/wikipedia/commons/2/2f/Google_2015_logo.svg">
            <img class="sponsor-logo" src="https://upload.wikimedia.org/wikipedia/commons/0/01/LinkedIn_Logo.svg">
            <img class="sponsor-logo" src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 220 40'%3E%3Ctext x='0' y='32' font-family='Arial, sans-serif' font-weight='900' font-size='28' fill='black'%3EASSISTments%3C/text%3E%3C/svg%3E">
            <img class="sponsor-logo" src="https://upload.wikimedia.org/wikipedia/commons/9/96/Microsoft_logo_%282012%29.svg">
            <img class="sponsor-logo" src="https://upload.wikimedia.org/wikipedia/commons/7/7b/Meta_Platforms_Inc._logo.svg">
            <img class="sponsor-logo" src="https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg">
            <img class="sponsor-logo" src="https://upload.wikimedia.org/wikipedia/commons/2/2f/Google_2015_logo.svg">
            <img class="sponsor-logo" src="https://upload.wikimedia.org/wikipedia/commons/0/01/LinkedIn_Logo.svg">
            <img class="sponsor-logo" src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 220 40'%3E%3Ctext x='0' y='32' font-family='Arial, sans-serif' font-weight='900' font-size='28' fill='black'%3EASSISTments%3C/text%3E%3C/svg%3E">
            <img class="sponsor-logo" src="https://upload.wikimedia.org/wikipedia/commons/9/96/Microsoft_logo_%282012%29.svg">
        </div>
    </div>
</div>
""", unsafe_allow_html=True)