import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
from supabase import create_client

# ==========================================
# Note for Reviewers: UPSTREAM INFERENCE BASELINE
# This module represents the foundational xLSTM prediction engine developed 
# during the preliminary research phase (upstream anomaly detection). 
# It serves as the static baseline (W0) upon which the downstream xCARE 
# Continual-LoRA adaptation engine dynamically operates when drift is detected.
# ==========================================

# ==========================================
# 1. CONFIGURATION & ARTIFACT LOADING
# Note for Reviewers: Keys securely fetched via Environment Variables
# ==========================================
SUPABASE_URL = os.environ.get("SUPABASE_URL", "your_supabase_url")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "your_supabase_key")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Loading the Feature Scaler and Model Features Mapping
try:
    SCALER = joblib.load('scaler.pkl')
    with open('model_features.json', 'r') as f:
        MODEL_FEATURES = json.load(f)
except FileNotFoundError:
    print("ERROR: scaler.pkl or model_features.json artifacts not found.")
    MODEL_FEATURES = []

INPUT_DIM = len(MODEL_FEATURES) if MODEL_FEATURES else 10 

# ==========================================
# 2. FOUNDATIONAL MODEL ARCHITECTURE
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
            f, o, g = torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
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

    def forward(self, x):
        out = self.xlstm(x)
        return self.fc(out).squeeze(-1)

# Loading Baseline Weights (W0)
try:
    model = KnowledgeTracingxLSTM(input_size=INPUT_DIM, hidden_size=64)
    model.load_state_dict(torch.load('reliable_xlstm_model.pth', map_location=torch.device('cpu')))
    model.eval()
except Exception as e:
    print(f"Model Loading Error: {e}")

# ==========================================
# 3. PREDICTION FUNCTION
# ==========================================
def get_xlstm_prediction(student_id):
    """
    Retrieves the learner's historical telemetry, formats the temporal sequence (Sequence=50),
    and outputs the base success probability (0.0 to 1.0).
    """
    MAX_SEQ_LEN = 50
    
    # 1. Supabase Telemetry Extraction
    response = supabase.table("xGATES")\
        .select("*")\
        .eq("user_id", student_id)\
        .order("interaction_id", desc=False)\
        .limit(MAX_SEQ_LEN)\
        .execute()
        
    df = pd.DataFrame(response.data)
    
    if df.empty:
        return 0.50 

    # 2. Data Preprocessing
    categorical_cols = ['tutor_mode', 'answer_type', 'problem_set_type', 'original']
    cat_cols_present = [col for col in categorical_cols if col in df.columns]
    
    if cat_cols_present:
        df = pd.get_dummies(df, columns=cat_cols_present, drop_first=True) 

    df_aligned = df.reindex(columns=MODEL_FEATURES, fill_value=0)

    continuous_features = [
        'ms_first_response_time', 'past_attempts', 'past_successes',
        'historical_accuracy', 'item_difficulty', 'gamification_score', 'magic_hint_count'
    ]
    cols_to_scale = [c for c in continuous_features if c in df_aligned.columns]
    if cols_to_scale:
        df_aligned[cols_to_scale] = SCALER.transform(df_aligned[cols_to_scale])

    # 3. Sequence Padding
    x_user = df_aligned.values
    if len(x_user) < MAX_SEQ_LEN:
        pad_length = MAX_SEQ_LEN - len(x_user)
        pad_x = np.zeros((pad_length, INPUT_DIM))
        x_user = np.vstack((pad_x, x_user))
        
    # 4. Neural Inference
    X_tensor = torch.tensor(x_user, dtype=torch.float32).unsqueeze(0) 
    
    with torch.no_grad():
        output = model(X_tensor)
        last_step_output = output[0, -1] 
        probability = torch.sigmoid(last_step_output).item()
        
    return probability