import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

# Configuración de la página
st.set_page_config(page_title="MNIST Digit Classifier", layout="wide")

st.title("🔢 Clasificador de Dígitos MNIST")
st.markdown("""
Esta aplicación permite entrenar diferentes modelos de Machine Learning para reconocer números escritos a mano
y validar sus resultados en tiempo real.
""")

# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    # Cargamos una versión reducida para velocidad, o completa (70k)
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist["data"], mnist["target"]
    # Normalización
    X = X / 255.0
    return X, y

with st.spinner("Cargando base de datos MNIST..."):
    X, y = load_data()

# --- SIDEBAR - CONFIGURACIÓN ---
st.sidebar.header("Configuración del Modelo")
model_type = st.sidebar.selectbox(
    "Selecciona el Algoritmo",
    ("Random Forest", "Neural Network (MLP)", "SVM (Soporte Vectorial)")
)

train_size = st.sidebar.slider("Tamaño de entrenamiento (muestras)", 1000, 10000, 5000)

# --- ENTRENAMIENTO ---
@st.cache_resource
def train_model(model_choice, X_data, y_data, samples):
    X_train, X_test, y_train, y_test = train_test_split(
        X_data[:20000], y_data[:20000], train_size=samples, random_state=42
    )
    
    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    elif model_choice == "Neural Network (MLP)":
        model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=20)
    else:
        model = SVC(kernel='rbf', probability=True)
        
    model.fit(X_train, y_train)
    
    # Métricas
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return model, acc, X_test, y_test, y_pred

model, accuracy, X_test, y_test, y_pred = train_model(model_type, X, y, train_size)

# --- MOSTRAR MÉTRICAS ---
st.header(f"📊 Desempeño: {model_type}")
col1, col2, col3 = st.columns(3)
col1.metric("Precisión (Accuracy)", f"{accuracy:.2%}")
col2.metric("Muestras Entrenadas", train_size)
col3.metric("Muestras de Test", len(X_test))

with st.expander("Ver Reporte Detallado"):
    st.dataframe(pd.DataFrame(report).transpose())

# --- SECCIÓN DE PRUEBAS ---
st.divider()
st.header("🔍 Prueba el modelo")

col_img, col_pred = st.columns([1, 1])

with col_img:
    index = st.number_input("Selecciona un índice del set de prueba (0 - {})".format(len(X_test)-1), 
                             min_value=0, max_value=len(X_test)-1, value=42)
    
    digit_image = X_test[index].reshape(28, 28)
    
    fig, ax = plt.subplots()
    ax.imshow(digit_image, cmap='gray')
    ax.axis('off')
    st.pyplot(fig)

with col_pred:
    prediction = model.predict(X_test[index].reshape(1, -1))
    
    st.subheader("Resultado de la Predicción:")
    st.markdown(f"<h1 style='text-align: center; color: #FF4B4B;'>{prediction[0]}</h1>", unsafe_allow_html=True)
    
    if prediction[0] == y_test[index]:
        st.success(f"¡Correcto! El valor real es {y_test[index]}")
    else:
        st.error(f"Error. El valor real era {y_test[index]}")

# --- MATRIZ DE CONFUSIÓN ---
st.divider()
if st.checkbox("Mostrar Matriz de Confusión"):
    st.subheader("Matriz de Confusión")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    im = ax_cm.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax_cm.figure.colorbar(im, ax=ax_cm)
    ax_cm.set(xticks=np.arange(cm.shape[1]),
              yticks=np.arange(cm.shape[0]),
              xlabel='Predicción', ylabel='Real')
    st.pyplot(fig_cm)
