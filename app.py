import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Загрузка модели и стандарта нормализации
model = joblib.load('/content/linear_regression_model.pkl')
scaler = joblib.load('/content/standard_scaler.pkl')

# Функция для прогнозирования цены
def predict_price(total_square, rooms, floor):
    input_data = [[total_square, rooms, floor]]
    scaled_data = scaler.transform(input_data)
    predicted_price = model.predict(scaled_data)[0]
    return round(predicted_price, 2)

# Интерфейс Streamlit
st.title('Прогноз стоимости недвижимости')

# Поля ввода характеристик объекта
total_square = st.number_input('Общая площадь (м²)', min_value=1.0, step=1.0, format='%f')
rooms = st.number_input('Количество комнат', min_value=1, step=1)
floor = st.number_input('Этаж', min_value=1, step=1)

# Кнопка для расчета стоимости
if st.button('Рассчитать стоимость'):
    price_prediction = predict_price(total_square, rooms, floor)
    st.write(f'Прогнозируемая стоимость: ₽{price_prediction:,.2f}')