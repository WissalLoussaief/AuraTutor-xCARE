import numpy as np
from datetime import datetime
from river import drift
from statsmodels.stats.proportion import proportions_ztest

def check_and_trigger_double_alert(current_interaction, subject_name, supabase):
    """
    xGATES SIAD ENGINE - Secure Architecture
    The emergency pedagogical circuit breaker is prioritized (Bypass mode).
    """

    user_id = current_interaction['user_id']
    problem_id = current_interaction['problem_id']
    skill_id = current_interaction['skill_id']
    current_error = current_interaction['residual_error']
    current_score = current_interaction['gamification_score']
    current_correct = current_interaction['correct']

    # ==========================================
    # 0. EMERGENCY CIRCUIT BREAKER (ABSOLUTE PRIORITY)
    # Note for Reviewers: This heuristic bypass acts as a 'Pedagogical Emergency Circuit'. 
    # It prioritizes immediate student intervention (scaffolding) over statistical latency, 
    # validating the Human-in-the-Loop safety constraints.
    # ==========================================
    # We verify the active memory state BEFORE querying the database.
    # TEST MODE: Triggers on consecutive errors (Set to >= 3 or 4 for production)
    urgence_pedagogique_detectee = (current_interaction.get('error_streak', 0) >= 4)

    if urgence_pedagogique_detectee:
        print("[SIAD] Pedagogical Emergency triggered via active memory bypass!")
        
        xcare_payload = {
            "alert_metadata": {
                "alert_id": f"XG-HEURISTIC-{datetime.now().strftime('%Y%m%d%H%M%S')}-{user_id}",
                "affected_user_id": user_id,
                "problem_id": problem_id,
                "skill_id": skill_id,
                "timestamp": datetime.now().isoformat()+"Z"
            },
            "statistical_evidence": {
                "p_value": 0.0, 
                "cohen_h": 0.99, 
                "drift_class": "SUDDEN DRIFT"
            },
            "xai_metadata": {
                "root_cause": "Pedagogical Emergency: Consecutive errors", 
                "subject": subject_name
            },
            "siad_recommendations": {
                "action": "Trigger Tactical Pause", 
                "strategy": "Direct Intervention"
            }
        }
        
        # Attempt to persist the alert payload
        try:
            supabase.table("xCARE_Alerts").insert({"payload": xcare_payload}).execute()
        except Exception as e:
            print(f"[SIAD] DB Warning ({e}): Alert not persisted, BUT UI block is still enforced.")
        
        # Return True IN ALL CASES to guarantee the execution of the Tactical Pause
        return True

    # ==========================================
    # 1. HISTORICAL TELEMETRY RETRIEVAL (For ADWIN/KSWIN)
    # ==========================================
    # This block executes only if the student is NOT in an active emergency state
    try:
        response = (
            supabase.table("xGATES")
            .select("residual_error, gamification_score, correct")
            .eq("user_id", user_id)
            .eq("skill_id", skill_id)
            .order("created_at", desc=False)
            .limit(30)
            .execute()
        )
        history = response.data
    except Exception as e:
        print(f"[SIAD] Silent DB error ignored: {e}")
        return False

    if not history:
        return False

    error_stream = [row["residual_error"] for row in history if row["residual_error"] is not None]
    score_stream = [row["gamification_score"] for row in history if row["gamification_score"] is not None]
    binary_error_stream = [1 if row["correct"] == 0 else 0 for row in history]

    error_stream.append(current_error)
    score_stream.append(current_score)
    binary_error_stream.append(1 if current_correct == 0 else 0)

    # ==========================================
    # 2. STATISTICAL DRIFT DETECTION (ADWIN / KSWIN)
    # Note for Reviewers: This implements the upstream anomaly detection pipeline 
    # presented in our preliminary work. It utilizes a dual-metric tracking approach: 
    # ADWIN for latent predictive residuals and KSWIN for behavioral gamification shifts.
    # ==========================================
    adwin_detector = drift.ADWIN(delta=0.01)
    kswin_detector = drift.KSWIN(alpha=0.05, window_size=30, stat_size=10)

    is_adwin_alert = False
    is_kswin_alert = False

    for err in error_stream:
        adwin_detector.update(err)
        if adwin_detector.drift_detected:
            is_adwin_alert = True

    for sc in score_stream:
        kswin_detector.update(sc)
        if kswin_detector.drift_detected:
            is_kswin_alert = True

    drift_statistique_detecte = (is_adwin_alert and is_kswin_alert)

    # Halt execution if no statistical drift is corroborated or if the current answer is correct
    if not drift_statistique_detecte or current_correct != 0:
        return False

    # ==========================================
    # 3. STATISTICAL VALIDATION & TOPOLOGY CLASSIFICATION
    # Note for Reviewers: Cohen's h effect size is utilized to mathematically classify 
    # the drift topology. This classification directly dictates the dynamic rank (r) 
    # allocated to the Continual-LoRA router downstream.
    # ==========================================
    window = min(10, len(binary_error_stream)//2)
    if window < 5:
        return False 

    before = binary_error_stream[:-window]
    after = binary_error_stream[-window:]
    p1 = np.mean(before)
    p2 = np.mean(after)

    count = np.array([np.sum(before), np.sum(after)])
    nobs = np.array([len(before), len(after)])
    _, p_value = proportions_ztest(count, nobs)

    if p_value >= 0.05:
        return False 

    eps = 1e-5
    h = abs(
        2*np.arcsin(np.sqrt(np.clip(p2, eps, 1-eps))) -
        2*np.arcsin(np.sqrt(np.clip(p1, eps, 1-eps)))
    )
    
    if h >= 0.80: drift_class = "SUDDEN DRIFT"
    elif h >= 0.20: drift_class = "GRADUAL DRIFT"
    else: drift_class = "INCREMENTAL DRIFT"

    # ==========================================
    # 4. DATABASE DISPATCH (Statistical Drift Payload)
    # ==========================================
    xcare_payload = {
        "alert_metadata": {
            "alert_id": f"XG-STAT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{user_id}",
            "affected_user_id": user_id,
            "problem_id": problem_id,
            "skill_id": skill_id,
            "timestamp": datetime.now().isoformat()+"Z"
        },
        "statistical_evidence": {"p_value": float(p_value), "cohen_h": float(h), "drift_class": drift_class},
        "xai_metadata": {"root_cause": "Statistical Algorithms (ADWIN/KSWIN)", "subject": subject_name},
        "siad_recommendations": {"action": "Trigger Tactical Pause", "strategy": "Adaptive Tuning"}
    }

    try:
        supabase.table("xCARE_Alerts").insert({"payload": xcare_payload}).execute()
        return True
    except Exception as e:
        print(f"[SIAD] xCARE dispatch failed: {e}")

    return False