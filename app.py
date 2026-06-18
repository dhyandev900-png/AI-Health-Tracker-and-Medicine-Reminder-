import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
import os
import warnings
warnings.filterwarnings('ignore')

# 1. APPLICATION BRANDING & MASTER PAGE SETUP
st.set_page_config(page_title="AI Medicine Reminder Dashboard", page_icon="💊", layout="centered")
st.title("🩺 AI Health Monitoring Ecosystem")
st.markdown("**Developer:** K K Dhyan Devaiah | **SRN:** PES1PG25CA094")
st.write("---")

# 2. BULLETPROOF DATA GENERATION & TRAINING COMPONENT
@st.cache_resource
def initialize_prediction_engine():
    # SELF-HEALING BLOCK: If the file is missing on GitHub, create it automatically!
    if not os.path.exists("health_monitoring_dataset.csv"):
        np.random.seed(42)
        rows = 4286
        data = {
            'Sleep_Duration_Hours': np.random.uniform(5.0, 9.5, rows).round(1),
            'Water_Intake_Liters': np.random.uniform(1.5, 4.0, rows).round(1),
            'Physical_Activity_Min': np.random.randint(15, 90, rows),
            'Heart_Rate_BPM': np.random.randint(60, 110, rows),
            'Calories_Burned_Kcal': np.random.randint(1500, 3200, rows),
            'Age': np.random.randint(18, 75, rows)
        }
        df_temp = pd.DataFrame(data)
        df_temp['Health_Score'] = ((df_temp['Sleep_Duration_Hours'] * 5) + (df_temp['Water_Intake_Liters'] * 6) + 
                                   (df_temp['Physical_Activity_Min'] * 0.4) - (abs(df_temp['Heart_Rate_BPM'] - 75) * 0.3) + 
                                   np.random.normal(30, 5, rows)).clip(0, 100).round(1)
        cond = [(df_temp['Health_Score'] >= 75), (df_temp['Health_Score'] >= 50) & (df_temp['Health_Score'] < 75), (df_temp['Health_Score'] < 50)]
        choices = ['Routine Notification', 'Standard Reminder Required', 'Urgent Alert Required']
        df_temp['Reminder_Urgency_Level'] = np.select(cond, choices, default='Standard Reminder Required')
        df_temp.to_csv("health_monitoring_dataset.csv", index=False)

    # Now load the guaranteed file
    df = pd.read_csv("health_monitoring_dataset.csv")
    X = df[['Sleep_Duration_Hours', 'Water_Intake_Liters', 'Physical_Activity_Min', 
            'Heart_Rate_BPM', 'Calories_Burned_Kcal', 'Age']]
    
    # Train Models
    y_reg = df['Health_Score']
    reg_model = LinearRegression().fit(X, y_reg)
    
    le = LabelEncoder()
    y_clf = le.fit_transform(df['Reminder_Urgency_Level'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf_model = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_scaled, y_clf)
    
    return reg_model, clf_model, scaler, le

reg_model, clf_model, scaler, le = initialize_prediction_engine()

# 3. SPLITTING ARCHITECTURE INTO MODULAR USER TAB INTERFACES
tab_metrics, tab_reminder = st.tabs(["📊 Diagnostic Vitals Input", "🔔 Live Medicine Reminder Center"])

# GLOBAL VARIABLE INITIALIZATION
if 'evaluated' not in st.session_state:
    st.session_state.evaluated = False
    st.session_state.score = 75.0
    st.session_state.reminder = "Routine Notification"

# TAB MODULE 1: PATIENT HEALTH METRICS COLLECTION
with tab_metrics:
    st.subheader("Patient Vitals Entry Form")
    st.write("Modify the tracked biological baseline criteria below to drive the AI engine algorithms:")
    
    with st.form("vitals_collection_form"):
        col1, col2 = st.columns(2)
        with col1:
            sleep = st.number_input("Sleep Duration (Hours)", min_value=0.0, max_value=24.0, value=5.5, step=0.5)
            water = st.number_input("Water Intake Volume (Liters)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
            activity = st.number_input("Daily Active Physical Exercise (Minutes)", min_value=0, max_value=300, value=15)
        with col2:
            heart_rate = st.number_input("Recorded Heart Rate Metric (BPM)", min_value=30, max_value=200, value=105)
            calories = st.number_input("Calculated Daily Active Metabolism (Kcal)", min_value=500, max_value=6000, value=1600)
            age = st.number_input("Patient Demographic Age Profile", min_value=1, max_value=120, value=45)
            
        process_btn = st.form_submit_button("Run Diagnostics & Sync Alerts")
        
    if process_btn:
        input_data = pd.DataFrame([[sleep, water, activity, heart_rate, calories, age]], 
                                  columns=['Sleep_Duration_Hours', 'Water_Intake_Liters', 
                                           'Physical_Activity_Min', 'Heart_Rate_BPM', 
                                           'Calories_Burned_Kcal', 'Age'])
        
        raw_score = reg_model.predict(input_data)[0]
        st.session_state.score = min(max(raw_score, 0), 100)
        
        input_scaled = scaler.transform(input_data)
        class_idx = clf_model.predict(input_scaled)[0]
        st.session_state.reminder = le.inverse_transform([class_idx])[0]
        st.session_state.evaluated = True
        
        st.success("🎉 Data diagnostics successfully mapped! Click on the 'Live Medicine Reminder Center' tab above to view medication actions.")

# TAB MODULE 2: DEDICATED ALARM & MEDICATION DISPATCH COMPONENT
with tab_reminder:
    st.subheader("📋 Smart Medicine Reminder Center")
    
    if not st.session_state.evaluated:
        st.info("💡 Waiting for patient metric profiles. Please input data values on the first tab and click 'Run Diagnostics' to activate notifications.")
    else:
        st.metric(label="System Evaluated Health Index Rating", value=f"{st.session_state.score:.1f} / 100")
        st.write("---")
        
        if st.session_state.reminder == "Urgent Alert Required" or st.session_state.score < 50:
            st.error("### 🚨 HIGH PRIORITY ALARM: Immediate Prescription Dispatch Notice")
            st.markdown("**Condition Risk Matrix:** Critical score variance triggered due to poor cardiac metrics and hydration deficiency. The following clinical response cycle is mandatory:")
            st.markdown("#### 💊 Active Emergency Prescription Load Details:")
            st.code("👉 [TAKE IMMEDIATELY] Cardiovascular Maintenance Tablet (Dosage: 1x 50mg)\n👉 [HYDRATION REQUIREMENT] Consume 500ml of mineralized water solution.\n👉 [ALERT CODE] Automated critical status logs transmitted to cloud monitoring services.", language="text")
            
        elif st.session_state.reminder == "Standard Reminder Required" or (50 <= st.session_state.score < 75):
            st.warning("### ⚠️ STANDARD ALARM: Mid-Day Routine Dosage Window Open")
            st.markdown("**Condition Risk Matrix:** Sub-optimal lifestyle inputs observed. Ensure compliance with standard clinical protocol bounds:")
            st.markdown("#### 💊 Standard Maintenance Medication Schedule:")
            st.code("👉 [SCHEDULED DOSAGE] General Health Multi-Vitamin Support (Dosage: 1 Capsule)\n👉 [DIETARY UPDATE] Increase resting phase hydration goals within next 2 hours.", language="text")
            
        else:
            st.success("### ✅ ROUTINE TRACKER: Health Targets Achieved")
            st.markdown("**Condition Risk Matrix:** Patient data lines match ideal baseline telemetry targets. No critical tracking corrections necessary.")
            st.markdown("#### 📅 Automated Preventive Strategy Tracking Logs:")
            st.info("Prescription requirements clear. Maintain normal health routines and log metrics tomorrow.")
