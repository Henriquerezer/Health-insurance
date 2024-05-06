import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

# Carregando o modelo treinado
modelo = pickle.load(open('modelo.pkl', 'rb'))

# Função para tratar os dados
def preprocessar_dados(data):
    # Mapeamento de gênero
    data['Gender'] = data['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
    
    # Conversão de categorias em dummies
    data = pd.get_dummies(data, drop_first=True)
    data = data.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", 
                                "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
    
    # Conversão de tipos
    data['Vehicle_Age_lt_1_Year'] = data['Vehicle_Age_lt_1_Year'].astype(int)
    data['Vehicle_Age_gt_2_Years'] = data['Vehicle_Age_gt_2_Years'].astype(int)
    data['Vehicle_Damage_Yes'] = data['Vehicle_Damage_Yes'].astype(int)
    
    # Cálculo de novas variáveis
    data["premium_age_ratio"] = data["Annual_Premium"] / data["Age"]
    data["premium_vintage_ratio"] = data["Annual_Premium"] / data["Vintage"]
    
    # Normalização
    scaler = StandardScaler()
    data[['Age', 'Annual_Premium', 'Vintage']] = scaler.fit_transform(data[['Age', 'Annual_Premium', 'Vintage']])
    
    # Removendo id
    data = data.drop('id', axis=1)
    return data

# Interface do Streamlit
st.title('Classificador de Seguros')

# Formulário para entrada de dados
with st.form('form'):
    id = st.number_input('ID', min_value=1, step=1)
    gender = st.selectbox('Gender', options=['Male', 'Female'])
    age = st.slider('Age', 18, 100, 30)
    driving_license = st.selectbox('Driving License', options=[0, 1])
    region_code = st.number_input('Region Code', min_value=1.0, max_value=50.0, step=1.0)
    previously_insured = st.selectbox('Previously Insured', [0, 1])
    vehicle_age = st.selectbox('Vehicle Age', ['< 1 Year', '1-2 Year', '> 2 Years'])
    vehicle_damage = st.selectbox('Vehicle Damage', ['Yes', 'No'])
    annual_premium = st.number_input('Annual Premium', min_value=1000, step=1000)
    policy_sales_channel = st.number_input('Policy Sales Channel', min_value=1, step=1)
    vintage = st.slider('Vintage', 10, 365, 50)
    submit = st.form_submit_button('Classificar')

# Processamento após o envio
if submit:
    # Criando um dataframe com os dados
    data = pd.DataFrame([[id, gender, age, driving_license, region_code, previously_insured, 
                          vehicle_age, vehicle_damage, annual_premium, policy_sales_channel, vintage]],
                        columns=['id', 'Gender', 'Age', 'Driving_License', 'Region_Code', 
                                 'Previously_insured', 'Vehicle_Age', 'Vehicle_Damage', 
                                 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage'])
    
    # Processando os dados
    data = preprocessar_dados(data)
    
    # Fazendo a previsão
    resultado = modelo.predict(data)
    
    # Mostrando o resultado
    st.write('A previsão de resposta é: {}'.format(resultado[0]))

# Para rodar o Streamlit, salve este script como `app.py` e execute `streamlit run app.py` no terminal.
