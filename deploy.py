import streamlit as st
import numpy as np
import pandas as pd
import pickle
from replace_dicts import *


# Cargar el modelo entrenado
le,variables,min_max_scaler,knn = pickle.load('modelo_final.pickle')  

st.set_page_config(page_title="🧠 Predicción de Desercion - KNN", layout="wide")

# Título principal
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🔍 Predicción Inteligente de desercion de Talento con KNN</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Evalúa rápidamente el perfil de un colaborador para anticipar comportamientos clave</h4>", unsafe_allow_html=True)

# Layout en columnas para evitar scroll
col1, col2, col3 = st.columns(3)

with col1:
    JobInvolvement = st.selectbox("Nivel de involucramiento en el trabajo", ("Low","Medium","High","Very High"))
    PerformanceRating = st.selectbox("Calificación de desempeño del año anterior", ("Low","Good","Excelent","Outstanding"))
    Age = st.number_input("Edad", min_value=18, max_value=65, value=30)
    Education = st.selectbox("Nivel educativo", ('Below College','College','Bachelor','Master', 'Doctor'))
    MonthlyIncome = st.number_input("Ingreso mensual", min_value=1000.0, max_value=300000.0, value=5000.0)

with col2:
    NumCompaniesWorked = st.number_input("N° de empresas previas", value=None, placeholder="Escribe un numero...")
    PercentSalaryHike = st.slider("% incremento salarial",0.0, 25.0,5.0,step=0.5)
    TotalWorkingYears = st.number_input("Años totales de experiencia", 0.0, 40.0, 10.0,step=1.0)
    TrainingTimesLastYear = st.slider("Capacitaciones dictadas en el último año", 0.0, 10.0, 2.0, step=1.0)
    YearsAtCompany = st.number_input("Años en la empresa", 0.0, 40.0, 5.0,step=1.0)

with col3:
    YearsSinceLastPromotion = st.number_input("Años desde la última promoción", 0.0, 15.0, 1.0,step=1.0)
    YearsWithCurrManager = st.number_input("Años con el gerente actual", 0.0, 15.0, 3.0,step=1.0)
    EnvironmentSatisfaction = st.selectbox("Satisfacción con el ambiente laboral", ("Low","Medium","High","Very High"))
    JobSatisfaction = st.selectbox("Satisfacción con el atrabajo", ("Low","Medium","High","Very High"))
    WorkLifeBalance = st.selectbox("Balance Trabajo-Vida", ("Bad","Good","Better","Best"))

st.markdown("---")

# Categóricas (dummies)
col4, col5, col6 = st.columns(3)

with col4:
    BusinessTravel = st.selectbox("Frecuencia de viaje", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    Department = st.selectbox("Departamento", ["Human Resources", "Research & Development", "Sales"])
    EducationField = st.selectbox("Área de educación", ["Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"])

with col5:
    JobRole = st.selectbox("Rol del trabajo", [
        "Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager",
        "Manufacturing Director", "Research Director", "Research Scientist", "Sales Executive", "Sales Representative"
    ])
    MaritalStatus = st.selectbox("Estado civil", ["Divorced", "Married", "Single"])

# Procesamiento de entradas
def procesar_input():
    datos = {
        "JobInvolvement": job_involvement[JobInvolvement],
        "PerformanceRating": performance_rating[PerformanceRating],
        "Age": [Age],
        "Education": education[Education],
        "MonthlyIncome": [MonthlyIncome],
        "NumCompaniesWorked": [NumCompaniesWorked],
        "PercentSalaryHike": [PercentSalaryHike],
        "TotalWorkingYears": [TotalWorkingYears],
        "TrainingTimesLastYear": [TrainingTimesLastYear],
        "YearsAtCompany": [YearsAtCompany],
        "YearsSinceLastPromotion": [YearsSinceLastPromotion],
        "YearsWithCurrManager": [YearsWithCurrManager],
        "EnvironmentSatisfaction": enviroment_satisfaction[EnvironmentSatisfaction],
        "JobSatisfaction": job_satisfaction[JobSatisfaction],
        "WorkLifeBalance": worklife_balance[WorkLifeBalance],
        "BusinessTravel": [BusinessTravel],
        "Department": [Department],
        "EducationField": [EducationField],
        "JobRole": [JobRole],
        "MaritalStatus": [MaritalStatus]
    }
    return pd.DataFrame(datos)

# Botón de predicción
if st.button("🚀 Realizar predicción"):
    datos = procesar_input()
    datos=pd.get_dummies(datos,dtype='float64')
    data_preparada=datos.reindex(columns=variables,fill_value=0)
    resultado = knn.predict(data_preparada)[0]
    resultado_labeled=le.inverse_transform([resultado])[0]
    st.markdown("---")
    st.markdown(f"<h2 style='text-align: center; color: #2196F3;'>🔮 Predicción del Modelo: <span style='color:#FF5722'>{resultado_labeled}</span></h2>", unsafe_allow_html=True)
    st.success("✅ Predicción generada exitosamente. Analiza y toma decisiones con confianza.")
