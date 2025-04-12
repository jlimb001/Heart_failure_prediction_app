import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the saved model
try:
    with open('best_xgb01.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'best_xgb.pkl' not found. Please ensure the model file is in the directory.")
    st.warning()

def clear_form():
    for key in st.session_state.keys():
        if key.startswith('form_'):
            del st.session_state[key]

def main():
    st.title('❤️Heart Failure Prediction App')

    st.header('Disclaimer and Instruction')
    with st.expander('## View Disclaimer & Instruction'):
        st.write("""
        ### Disclaimer: ###
        This app is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

        ### How to Use This App: ###
        1. Fill in all the required information accurately.
        2. If you're unsure about any values, consult your most recent medical records or ask your healthcare provider.
        3. After entering all information, click the 'Predict Heart Failure Risk' button at the bottom of the page.
        4. The app will provide a risk assessment based on the information you've entered.
        5. Remember, this is a predictive tool and not a definitive diagnosis. Always consult with a healthcare professional for proper medical advice.

        ### Privacy Notice: ###
        This app does not store any of the information you enter. All calculations are done locally in your browser.
        """)

    st.write("#### Please fill in the following information to assess your heart failure risk:")


    # Input form for user data
    with st.form('user_input_form'):
        st.header('🔍 Personal Information')
        age = st.number_input('Age', min_value= 0, max_value= 120, step=1, key='form_age', help='Enter age in years') 
        sex = st.radio('Sex', ['Male', 'Female'], key='form_sex')

        st.header(' ❤️ Cardiovascular Health')
        st.subheader('Chest pain type')
        st.write("Chest pain is a common symptom of heart problems. Select the type that best describes your experience:")
        cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3], format_func = lambda x: {0:"Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}[x], key='form_cp')
            
        st.subheader("Resting Blood Pressure (mm Hg)")
        st.write("Normal range: Below 120/80 mm Hg")
        st.write("High: 140/90 mm Hg or higher")
        trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value = 0, max_value = 300, key='form_trestbps')

        st.subheader("Serum Cholesterol (mg/dl)")
        st.write('Serum cholesterol is the amount of cholesterol in your blood.')
        chol = st.number_input('Serum Cholesterol (mg/dl)', min_value = 0, max_value = 600, key='form_chol')

        st.subheader("Fasting Blood Sugar (FBS)")
        st.write("Fasting blood sugar measures your blood glucose level after not eating or drinking for at least 8 hours.")
        st.write("This test is important for diagnosing diabetes and prediabetes.")
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'], format_func=lambda x: 'Yes' if x == 'Yes' else 'No', key='form_fbs')

        st.header('📊 ECG Results')
        st.write('ECG (Electrocardiogram) measures the electrical activity of your heart.')

        st.subheader("Resting ECG Results")
        st.write("This shows the results of your ECG taken while at rest.")
        restecg = st.selectbox('Resting ECG Results', [0, 1, 2], format_func=lambda x: {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}[x], key='form_restecg')
            
        st.subheader("Maximum Heart Rate Achieved")
        st.write("This is the highest heart rate achieved during exercise.")
        st.write("Normal maximum heart rate: 220 - your age")
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=250, key='form_thalach')

        st.header('Exercicse Test Results')
        st.subheader("Exercise Induced Angina")
        st.write("This indicates if you experienced chest pain during exercise.")
        exang = st.selectbox('Exercise Induced Angina',['No', 'Yes'], format_func = lambda x: 'Yes' if x == 'Yes' else 'No', key='form_exang')
            
        st.subheader("ST Depression Induced by Exercise Relative to Rest")
        oldpeak = st.number_input('ST Depression Induced by Exercise Relative to - Oldpeak', min_value = 0.0, max_value = 10.0, step = 0.1, key='form_oldpeak')
        slope = st.selectbox('Slope of Peak Exercise ST Segment - Slope', [0, 1, 2], format_func = lambda x: {0: 'Unsloping', 1:'Flat', 2: 'Downsloping'}[x], key='form_slope')

        st.header('🩺 Additional Tests')
        ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', [0, 1, 2, 3, 4], key='form_ca')

        # include_extra_tests = st.checkbox("Include Fluoroscopy & Thalassemia Data?")
        # if include_extra_tests:
        #     thal = st.selectbox('Thalassemia', [0, 1, 2, 3], 
        #                         format_func=lambda x: {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect", 3: "Unknown"}[x], key='form_thal') # Removed the feature as XGB model only accept 15 features
        # else:
        #     ca, thal = None, None

        st.header('🛠 Lifestye & Other Factors')
        bmi = st.number_input('Body Mass Index (BMI)', min_value=10.0, max_value=50.0, step=0.1, key='form_bmi')
        smoking = st.radio('Smoking Status', ['Non-smoker', 'Smoker'], key='form_smokingl')
        # alcohol = st.radio('Alcohol Consumption', ['No', 'Yes'], key='form_alcohol')
        # physical_activity = st.radio('Regular Physical Activity', ['No', 'Yes'], key='form_physical_activity')
        diabetes = st.radio('Diabetes', ['No', 'Yes'], key='form_diabetes')
        # family_history = st.radio('Family History of Heart Disease', ['No', 'Yes'], key='form_family_history')
        #stress_level = st.slider('Stress Level (1-10)', min_value=1, max_value=10, key='form_stress_level', help= '1 being the lowest and 10 being the highest')

        submitted = st.form_submit_button('Predict Heart Failure Risk')
        clear_button = st.form_submit_button('Clear Form', on_click=clear_form)

        if clear_button:
            clear_form()
            st.rerun() # Rerun the app to clear the inputs

        if submitted:  
            # Converting the inputs to numerical values
            sex_num = 0 if sex == 'Male' else 1
            fbs_num = 1 if fbs == 'Yes' else 0
            exang_num = 1 if exang == 'Yes' else 0
            smoking_num = 1 if smoking == 'Smoker' else 0
            #alcohol_num = 1 if alcohol == 'alcohol' else 0
            #physical_activity_num = 1 if physical_activity == 'physical_activity' else 0
            #family_history_num = 1 if family_history == 'family_history' else 0
            diabetes_num = 1 if diabetes == 'Yes' else 0
            #stress_level_num = 1 if stress_level == 'Yes' else 0
        
            # Create a numpy array for prediction
            input_data = np.array([[age, sex_num, cp, trestbps, chol, fbs_num, restecg, thalach, exang_num, oldpeak, slope, ca,
                                        bmi, smoking_num, diabetes_num]])
        
        
            # Make prediction
            scaler = StandardScaler()
            input_data_scaled  = scaler.fit_transform(input_data)
            prediction = model.predict(input_data_scaled)
            probability = model.predict_proba(input_data_scaled)[0][1] # Probability of positive class

            st.subheader(' 📢 Prediction Results:')
            # Interpret the prediction (e.g., 0 for no heart failure, 1 for heart failure)
            if prediction[0] == 0:
                st.success('🟢 Low risk of heart failure. Continue maintaining a healthy lifestyle!')
            else:
                st.error('🔴 High risk of heart failure. Please consult a heallthcare professional for further assessment.')

            st.info(f'Probability of heart faiilure: {probability:.2%}')

            # Prediction Probability Distribution
            st.write("""
            ## Note:
            This prediction is based on the provided information and should not be considered as a medical diagnosis. 
            Please consult with a healthcare professional for proper medical advice.
            """)
main()
