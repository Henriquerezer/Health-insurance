import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

# Carregar o modelo treinado
model = joblib.load("modelo_final.pkl")

# Função para processar os dados de entrada
def process_input(user_data):
    user_data['Gender'] = user_data['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
    user_data['Vehicle_Damage_Yes'] = user_data['Vehicle_Damage'].map({'No': 0, 'Yes': 1}).astype(int)
    user_data['Vehicle_Age_lt_1_Year'] = (user_data['Vehicle_Age'] == '< 1 Year').astype(int)
    user_data['Vehicle_Age_gt_2_Years'] = (user_data['Vehicle_Age'] == '> 2 Years').astype(int)
    user_data["premium_age_ratio"] = user_data["Annual_Premium"] / user_data["Age"]
    user_data["premium_vintage_ratio"] = user_data["Annual_Premium"] / user_data["Vintage"]
    ss = StandardScaler()
    mm = MinMaxScaler()
    num_features = ['Age', 'Vintage', 'premium_age_ratio', 'premium_vintage_ratio', 'Annual_Premium']
    user_data[num_features] = ss.fit_transform(user_data[num_features].fillna(0))
    user_data[['Annual_Premium']] = mm.fit_transform(user_data[['Annual_Premium']])
    model_features = ['Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured',
                      'Annual_Premium', 'Policy_Sales_Channel', 'Vintage', 'Vehicle_Age_lt_1_Year',
                      'Vehicle_Age_gt_2_Years', 'Vehicle_Damage_Yes', 'premium_age_ratio', 'premium_vintage_ratio']
    return user_data[model_features]

# Configuração da página
st.set_page_config(page_title='Classificador de Seguro Automóvel', layout='wide')

# Interface do Streamlit
st.title('Classificador de Interesse em Seguro Automóvel')
st.markdown("""
Este aplicativo prevê se um usuário estaria interessado em um plano de seguro automóvel com base em suas informações pessoais e de seguros anteriores.
Por favor, preencha as informações abaixo para obter uma previsão.
""")

# Colunas para entrada de dados
col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox('Gênero', ['Male', 'Female'])
    age = st.number_input('Idade', min_value=18, max_value=100)
    driving_license = st.selectbox('Possui Carteira de Motorista?', [1, 0])
with col2:
    region_code = st.number_input('Código da Região', min_value=0.0, max_value=50.0, step=0.1)
    previously_insured = st.selectbox('Já Possui Seguro Automóvel?', [0, 1])
    vehicle_age = st.selectbox('Idade do Veículo', ['< 1 Year', '1-2 Year', '> 2 Years'])
with col3:
    vehicle_damage = st.selectbox('Veículo já sofreu danos?', ['Yes', 'No'])
    annual_premium = st.number_input('Prêmio Anual (R$)', min_value=0.0, step=0.1)
    policy_sales_channel = st.number_input('Canal de Venda da Apólice', min_value=0.0, max_value=200.0, step=0.1)
    vintage = st.number_input('Antiguidade da Apólice (dias)', min_value=0, max_value=300)

submit = st.button('Classificar')

if submit:
    user_data = pd.DataFrame({
        'Gender': [gender], 'Age': [age], 'Driving_License': [driving_license], 
        'Region_Code': [region_code], 'Previously_Insured': [previously_insured], 
        'Vehicle_Age': [vehicle_age], 'Vehicle_Damage': [vehicle_damage],
        'Annual_Premium': [annual_premium], 'Policy_Sales_Channel': [policy_sales_channel], 'Vintage': [vintage]
    })
    processed_data = process_input(user_data)
    prediction = model.predict(processed_data)
    result = 'INTERESSADO' if prediction[0] == 1 else 'NÃO INTERESSADO'
    st.success(f'Resultado do Modelo de Classificação: **{result}**')

