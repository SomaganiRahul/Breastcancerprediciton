import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv(r'C:\Users\nanis\OneDrive\Desktop\bcfinaaaal\breast-cancer-wisconsin-data_data.csv')

# Preprocessing
data = data.drop(columns=['id', 'Unnamed: 32'])
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Select 10 main input values
features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'
]

X = data[features]
y = data['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit app
st.title('Breast Cancer Prediction')

# State variable to track whether to show the form or results
if 'show_form' not in st.session_state:
    st.session_state.show_form = True

def user_input_features():
    st.markdown("<h2 style='text-align: center;'>User Input Parameters</h2>", unsafe_allow_html=True)
    radius_mean = st.number_input('Mean Radius', float(X['radius_mean'].min()), float(X['radius_mean'].max()))
    texture_mean = st.number_input('Mean Texture', float(X['texture_mean'].min()), float(X['texture_mean'].max()))
    perimeter_mean = st.number_input('Mean Perimeter', float(X['perimeter_mean'].min()), float(X['perimeter_mean'].max()))
    area_mean = st.number_input('Mean Area', float(X['area_mean'].min()), float(X['area_mean'].max()))
    smoothness_mean = st.number_input('Mean Smoothness', float(X['smoothness_mean'].min()), float(X['smoothness_mean'].max()))
    compactness_mean = st.number_input('Mean Compactness', float(X['compactness_mean'].min()), float(X['compactness_mean'].max()))
    concavity_mean = st.number_input('Mean Concavity', float(X['concavity_mean'].min()), float(X['concavity_mean'].max()))
    concave_points_mean = st.number_input('Mean Concave Points', float(X['concave points_mean'].min()), float(X['concave points_mean'].max()))
    symmetry_mean = st.number_input('Mean Symmetry', float(X['symmetry_mean'].min()), float(X['symmetry_mean'].max()))
    fractal_dimension_mean = st.number_input('Mean Fractal Dimension', float(X['fractal_dimension_mean'].min()), float(X['fractal_dimension_mean'].max()))

    data = {
        'radius_mean': radius_mean,
        'texture_mean': texture_mean,
        'perimeter_mean': perimeter_mean,
        'area_mean': area_mean,
        'smoothness_mean': smoothness_mean,
        'compactness_mean': compactness_mean,
        'concavity_mean': concavity_mean,
        'concave points_mean': concave_points_mean,
        'symmetry_mean': symmetry_mean,
        'fractal_dimension_mean': fractal_dimension_mean
    }
    features = pd.DataFrame(data, index=[0])
    return features

if st.session_state.show_form:
    input_df = user_input_features()
    if st.button('Predict'):
        # Scale user input
        input_scaled = scaler.transform(input_df)

        # Prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]

        # Store the prediction results in session state
        st.session_state.show_form = False
        st.session_state.prediction = prediction
        st.session_state.prediction_proba = prediction_proba
        st.rerun()

# Display prediction results in the center of the page
if not st.session_state.show_form:
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.markdown("<h2>Prediction Results</h2>", unsafe_allow_html=True)
    st.subheader('Prediction')
    if st.session_state.prediction == 0:
        st.write('The patient is negative and healthy')
    else:
        st.write('The patient has cancer')
        st.subheader('Precautions')
        st.write("""
        - Consult with a healthcare provider immediately.
        - Follow a healthy diet.
        - Maintain a healthy weight.
        - Avoid alcohol and tobacco.
        - Exercise regularly.
        - Get regular medical checkups.
        """)

    st.subheader('Prediction Probability')
    st.write(f'Negative Probability: {st.session_state.prediction_proba[0]:.2f}')
    st.write(f'Positive Probability: {st.session_state.prediction_proba[1]:.2f}')
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button('Back to Input Page'):
        st.session_state.show_form = True
        st.rerun()

        