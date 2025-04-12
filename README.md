
# ❤️ Heart Failure Prediction Using Machine Learning ❤️

This poject involves a developing a machine learning model to predict the heart risk based on various clinical and lifestyle factors.

Disclaimer:

This app is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

Project Structure:

Data: 

Contains the dataset used for training and testing the model (Dataset used in the project was obtained from Kaggle)

Models:

It includes the trained machine learning models.

Notebook & Scripts:

 - heart_failure.ipynb: Contains the code for training the machine learning model
 - index.py: Scripts for making prediction using the training model

 Streamlit App:

 A web application build using Streamlit to interactively input the patient data and display predictions.

Usage

1. Data Preprocessing:
- Run heart_failure.ipynb notebook to clean and train the model

2. Prediction:
- Use index.py to make prediction with the trained model

3. Streamlit App:
- Run streamlit app runn index.py to launch the interactive web application

Model Details
- Features: The model use the following features to predict the heart failure risk:

  - Age
  - Sex
  - Chest Pain Type
  - Resting Blood Pressure
  - Serum Cholesterol
  - Fasting Blood Sugar
  - ECG Results
  - Maximum Heart Rate Achieved
  - Exercise Induced Angina
  - Old Peak (ST Depression)
  - Slope of Peak Exercise ST Segment
  - Fluoroscopy (Major Vessels Colored)
  - Body mass index (BMI)
  - Smoking Status
  - Diabetes Status

Model Type: 

For this project, XGB Classifier is used for its ability to handle multiple features and provide interpretable Results




## Libraries used in the project

- numpy
- pandas
- scikit-learning
- Streamlit
- matplotlib.pyplot
- seaborn
- pickle


```bash
  pip install numpy pandas scikit-learn Streamlit matplotlib.pyplot seaborn pickle
```
    
## Appendix

Dataset link:

https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data



