# 🧠 Predicción de Deserción Laboral

[![Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

Este proyecto utiliza un modelo de aprendizaje automático (K-Nearest Neighbors - KNN) para predecir si un empleado está en riesgo de **renunciar** a su trabajo, con base en sus características laborales, demográficas y de satisfacción.

---

## 🚀 Características

- Clasificación binaria: `Retired` / `Not Retired`
- Interfaz de usuario desarrollada con **Streamlit**
- Entrada de variables numéricas y categóricas
- Modelo preentrenado listo para usar (`modelo_knn.pkl`)
- Código limpio y adaptable

---

## 🧠 Tecnologías Usadas

- Python 3.12.1
- Streamlit
- scikit-learn
- pandas / numpy
- joblib (para carga del modelo)

---

## 📂 Estructura del Proyecto

```
📁 final_mineria/
├── 📁 data
    ├── predictive_data_joined_prepared.csv      # Conjunto de entrenamiento
    ├── centroides_pf.csv   #Caracteristicas de los centroides   
├── deploy.py               # Interfaz principal de Streamlit
├── modelo_knn.pkl          # Modelo KNN entrenado
├── requirements.txt        # Librerías necesarias (ver abajo)
├── training.ipynb          # Notebook de entrenamiento del modelo
└── README.md               # Este archivo
```

---

## 📦 Instalación

1. Clona este repositorio:

```bash
git clone https://github.com/DanielGongora28/final_mineria.git
cd final_mineria
```

2. Instala los requerimientos:

```bash
pip install -r requirements.txt
```

3. Ejecuta la aplicación:

```bash
streamlit run deploy.py
```

---

## 🎯 Uso de la aplicación

1. Abre la interfaz en tu navegador (normalmente http://localhost:8501)
2. Ingresa los datos de un empleado (edad, ingreso, satisfacción, etc.)
3. Haz clic en **🚀 Realizar predicción**
4. El modelo mostrará si el empleado está en riesgo de renuncia

---

## 🧪 Modelo

- Algoritmo: **K-Nearest Neighbors**
- Entrenado sobre un dataset de 5841 registros y 39 variables
- Variables dummies para roles, viajes, educación, etc.
- Etiqueta objetivo: `¿Renunciará?`

---

## ☁️ Versión en línea

Si estás usando Streamlit Cloud, edita el enlace del badge arriba para apuntar a tu URL pública:

[![Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://predicciondesercionlaboral.streamlit.app/)


---

## 🤝 Contribuciones

Las contribuciones, ideas o mejoras son bienvenidas. Puedes:

- Abrir un issue
- Crear un pull request
- Proponer mejoras en el modelo o la interfaz

---