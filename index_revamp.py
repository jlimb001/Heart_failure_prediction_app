import streamlit as st
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the saved model with error handling
try:
    with open('rf_best_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'rf.pkl' not found. Please ensure the model file is in the directory.")
    st.stop()

def main():
    st.title("❤️ Heart Failure Prediction App")

    st.header("Disclaimer and Instructions")
    with st.expander("View Disclaimer & Instructions"):
        st.write("""
        **Disclaimer:** This app is for educational purposes only and should not be used as a substitute for professional medical advice.
        **Instructions:** Fill in the required details and click **'Predict Heart Failure Risk'** to get the results.
        """)
    
    with st.form("user_input_form"):
        st.header("🔍 Personal Information")
        age = st.slider('Age', min_value=0, max_value=120, value=30)
        sex = st.radio('Sex', ['Male', 'Female'])

        st.header("❤️ Cardiovascular Health")
        cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3], 
                          format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}[x])
        trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=0, max_value=300)
        chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=0, max_value=600)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
        hypertension = st.radio('Do you have Hypertension?', ['No', 'Yes'])
        
        st.header("📊 ECG & Exercise Test Results")
        restecg = st.selectbox('Resting ECG Results', [0, 1, 2], 
                               format_func=lambda x: {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}[x])
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=250)
        exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
        oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, step=0.1)
        slope = st.selectbox('Slope of Peak Exercise ST Segment', [0, 1, 2], 
                             format_func=lambda x: {0: 'Unsloping', 1: 'Flat', 2: 'Downsloping'}[x])
        
        thal = st.selectbox('Thalassemia', [0, 1, 2, 3], 
                                format_func=lambda x: {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect", 3: "Unknown"}[x], key='form_thal') # Removed the feature as XGB model only accept 15 features

        st.header("🛠 Lifestyle & Other Factors")
        bmi = st.number_input('Body Mass Index (BMI)', min_value=10.0, max_value=50.0, step=0.1)
        smoking = st.radio('Smoking Status', ['Non-smoker', 'Smoker'])
        # alcohol = st.radio('Alcohol Consumption', ['No', 'Yes'])
        # physical_activity = st.radio('Regular Physical Activity', ['No', 'Yes'])
        diabetes = st.radio('Diabetes', ['No', 'Yes'])
        # family_history = st.radio('Family History of Heart Disease', ['No', 'Yes'])
        # stress_level = st.slider('Stress Level (1-10)', min_value=1, max_value=10, key='form_stress_level', help= '1 being the lowest and 10 being the highest')
        submitted = st.form_submit_button("Predict Heart Failure Risk")
        
        if submitted:
            # Convert categorical inputs to numerical
            sex_num = 0 if sex == 'Male' else 1
            fbs_num = 1 if fbs == 'Yes' else 0
            exang_num = 1 if exang == 'Yes' else 0
            smoking_num = 1 if smoking == 'Smoker' else 0
            #alcohol_num = 1 if alcohol == 'Yes' else 0
            #physical_activity_num = 1 if physical_activity == 'Yes' else 0
            diabetes_num = 1 if diabetes == 'Yes' else 0
            #family_history_num = 1 if family_history == 'Yes' else 0
            # hypertension_num = 1 if hypertension == 'Yes' else 0
            
            # Prepare data for prediction (ensuring exactly 20 features)
            input_data = np.array([[age, sex_num, cp, trestbps, chol, fbs_num, restecg, thalach, exang_num, oldpeak, 
                                    slope, thal, bmi, smoking_num, diabetes_num]])
            
            # Make prediction
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0][1]
            
            st.subheader("📢 Prediction Results:")
            if prediction[0] == 0:
                st.success("🟢 Low risk of heart failure. Continue maintaining a healthy lifestyle!")
            else:
                st.error("🔴 High risk of heart failure. Please consult a healthcare professional.")
            
            st.info(f"**Probability of heart failure: {probability:.2%}**")
            
            st.write("\n**Note:** This prediction is based on the provided information and is not a medical diagnosis. Always consult a healthcare professional.")
    
        # Feature importance visualization (if available)
        if hasattr(model, 'feature_importances_'):
            st.subheader("🔍 Feature Importance")
            fig, ax = plt.subplots()
            features = ['Age', 'Sex', 'CP', 'Trestbps', 'Chol', 'FBS', 'RestECG', 'Thalach', 'Exang', 'Oldpeak', 'Slope']
            sns.barplot(x=model.feature_importances_[:len(features)], y=features, ax=ax)
            ax.set_title("Feature Importance in Prediction")
            st.pyplot(fig)

main()

# import streamlit as st
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import StandardScaler

# # Load the saved model
# try:
#     with open('rf.pkl', 'rb') as file:
#         model = pickle.load(file)
# except FileNotFoundError:
#     st.error("Model file 'rf.pkl' not found. Please ensure the model file is in the directory.")
#     st.stop()

# # Function to clear form inputs
# def clear_form():
#     for key in list(st.session_state.keys()):
#         if key.startswith("form_"):
#             del st.session_state[key]

# def main():
#     st.title('❤️ Heart Failure Prediction Dashboard')
    
#     # Input form
#     with st.form("user_input_form"):
#         age = st.slider('Age (yrs)', 0, 120, 30, key="form_age") 
#         sex = st.radio('Sex', ['Male', 'Female'], key="form_sex")
#         cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3], key="form_cp")
#         trestbps = st.number_input('Resting BP (mm Hg)', 0, 300, key="form_trestbps")
#         chol = st.number_input('Serum Cholesterol (mg/dl)', 0, 600, key="form_chol")
#         fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'], key="form_fbs")
#         restecg = st.selectbox('Resting ECG Results', [0, 1, 2], key="form_restecg")
#         thalach = st.number_input("Max Heart Rate Achieved", 0, 250, key="form_thalach")
#         exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'], key="form_exang")
#         oldpeak = st.number_input('ST Depression', 0.0, 10.0, 0.1, key="form_oldpeak")
#         slope = st.selectbox('ST Segment Slope', [0, 1, 2], key="form_slope")
        
#         submitted = st.form_submit_button('Predict Heart Failure Risk')
#         clear_button = st.form_submit_button('Clear Form')
    
#     if clear_button:
#         clear_form()
#         st.rerun()
    
#     if submitted:
#         sex_num = 0 if sex == 'Male' else 1
#         fbs_num = 1 if fbs == 'Yes' else 0
#         exang_num = 1 if exang == 'Yes' else 0
        
#         # Create a full 20-feature array (fill missing with zeros)
#         full_input = np.zeros(20)
#         input_data = np.array([[age, sex_num, cp, trestbps, chol, fbs_num, restecg, thalach, exang_num, oldpeak, slope]])
#         full_input[:input_data.shape[1]] = input_data
        
#         # Scale input
#         scaler = StandardScaler()
#         full_input_scaled = scaler.fit_transform([full_input])
        
#         # Model prediction
#         prediction = model.predict(full_input_scaled)
#         probability = model.predict_proba(full_input_scaled)[0][1]
        
#         st.subheader('📢 Prediction Results:')
#         if prediction[0] == 0:
#             st.success('🟢 Low risk of heart failure.')
#         else:
#             st.error('🔴 High risk of heart failure.')
#         st.info(f'Probability of heart failure: {probability:.2%}')
        
#         # Additional Visualization
#         # st.subheader("📊 Risk Distribution")
#         # fig, ax = plt.subplots()
#         # labels = ['Low Risk', 'High Risk']
#         # values = [1 - probability, probability]
#         # ax.bar(labels, values, color=['skyblue', 'skyblue'])
#         # ax.set_ylabel('Probability')
#         # ax.set_xlabel('Heart Failure Risk Distribution')

#         # st.pyplot(fig)
        
#         # Feature importance visualization (if available)
#         if hasattr(model, 'feature_importances_'):
#             st.subheader("🔍 Feature Importance")
#             fig, ax = plt.subplots()
#             features = ['Age', 'Sex', 'CP', 'Trestbps', 'Chol', 'FBS', 'RestECG', 'Thalach', 'Exang', 'Oldpeak', 'Slope']
#             sns.barplot(x=model.feature_importances_[:len(features)], y=features, ax=ax)
#             ax.set_title("Feature Importance in Prediction")
#             st.pyplot(fig)

# main()
