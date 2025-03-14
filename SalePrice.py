import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
sklearn.set_config(transform_output="pandas")
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import pickle
import sklearn
sklearn.set_config(transform_output="pandas")

# Загрузите необходимые библиотеки и объект preprocessor
# Предполагается, что preprocessor уже обучен и сохранён
import joblib


import streamlit as st

# Задайте цвет фона через CSS
def set_background_color(color):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {color};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Установите желаемый цвет фона
set_background_color("white")  # Пример: светло-серый фон

# Загрузка обученного preprocessor
preprocessor = joblib.load('preprocessor.joblib')

# Загрузка тестовых данных
data_test = pd.read_csv('test.csv')



st.image('./ValueError.jpg',width=1200)
st.write("### мы №473 в мире, а ты нет :sunglasses:")

st.image('./media/leaderboard.jpg',width=1200)

# Заголовок приложения
st.title("Predictions with Preprocessor")

st.markdown('''
    ## House Prices - Advanced Regression Techniques
    ##### :gray[Predict sales prices and practice feature engineering, RFs, and gradient boosting]
    ''')

st.link_button("View competition", "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview")
# # Отображение объекта preprocessor
# st.subheader("Preprocessor Pipeline")
# st.write(preprocessor)

# Кнопка для выполнения предсказаний
if st.button("Run Test Predictions"):
    # Выполнение предсказаний
    test_predictions = preprocessor.predict(data_test)
    
    # Преобразование предсказаний из логарифмической шкалы
    test_predictions_exp = np.exp(test_predictions)
    
    # Создание DataFrame с результатами
    results = pd.DataFrame({
        'Id': data_test['Id'],
        'Predicted_SalePrice': test_predictions_exp
    })
    
    # Отображение результатов
    st.subheader("Prediction Results")
    st.dataframe(results)
    
    # Сохранение результатов в CSV
    results.to_csv('predictions.csv', index=False)
    st.success("Predictions saved to 'predictions.csv'")


st.markdown('''
#### <- Load Data
''')
uploaded_file = st.sidebar.file_uploader(
    "Choose a data", type='csv'
)

if 'show_text' not in st.session_state:
    st.session_state.show_text = False

if st.sidebar.button("Our team"):
    st.session_state.show_text = not st.session_state.show_text

if st.session_state.show_text:
    st.sidebar.markdown('''
    * Makson
    * Vovchik
    * Ziyarat
    ''')


if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if st.button("Run Predictions for Uploaded file"):
        # Выполнение предсказаний
        test_predictions = preprocessor.predict(df)
        
        # Преобразование предсказаний из логарифмической шкалы
        test_predictions_exp = np.exp(test_predictions)
        
        # Создание DataFrame с результатами
        results = pd.DataFrame({
            'Id': df['Id'],
            'Predicted_SalePrice': test_predictions_exp
        })
        
        # Отображение результатов
        st.subheader("Prediction Results")
        st.dataframe(results)
        
        # Сохранение результатов в CSV
        results.to_csv('predictions.csv', index=False)
        st.success("Predictions saved to 'predictions.csv'")


st.write("### Наши попытки")
st.image('./media/attempts.jpg',width=1200)
    