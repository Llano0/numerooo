# =============================
# requirements.txt
# =============================
streamlit
scikit-learn
pandas
numpy
matplotlib
seaborn
pillow

# =============================
# main_app.py
# =============================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageOps

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# -----------------------------
# Configuración inicial
# -----------------------------
st.set_page_config(page_title="Clasificador MNIST Interactivo", layout="wide")
st.title("🧠 Clasificador de Dígitos Escritos a Mano (MNIST)")

st.markdown("""
Este sistema permite:
- Entrenar múltiples modelos
- Visualizar métricas de desempeño
- Probar imágenes propias
- Ver el dígito antes de clasificar
""")

# -----------------------------
# Cargar MNIST
# -----------------------------
@st.cache_data
def load_mnist():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype("float32")
    y = mnist.target.astype(int)
    return X, y

X, y = load_mnist()

# Para hacer la app más rápida
subset_size = st.sidebar.slider("Tamaño del dataset usado", 2000, 20000, 5000, step=1000)
X = X[:subset_size]
y = y[:subset_size]

# -----------------------------
# Sidebar modelos
# -----------------------------
st.sidebar.header("⚙️ Configuración del modelo")

model_option = st.sidebar.selectbox(
    "Modelo",
    ["KNN", "Decision Tree", "Random Forest", "SVM"]
)

test_size = st.sidebar.slider("Porcentaje test", 0.1, 0.4, 0.2)
random_state = st.sidebar.slider("Random state", 0, 100, 42)

# Hiperparámetros dinámicos
if model_option == "KNN":
    k = st.sidebar.slider("Vecinos (k)", 1, 7, 3)

elif model_option == "Decision Tree":
    max_depth = st.sidebar.slider("Max depth", 2, 20, 8)

elif model_option == "Random Forest":
    n_estimators = st.sidebar.slider("Árboles", 10, 150, 50)

elif model_option == "SVM":
    C = st.sidebar.slider("C", 0.1, 5.0, 1.0)

# -----------------------------
# División de datos
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=random_state
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Crear modelo
# -----------------------------
if model_option == "KNN":
    model = KNeighborsClassifier(n_neighbors=k)

elif model_option == "Decision Tree":
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

elif model_option == "Random Forest":
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

elif model_option == "SVM":
    model = SVC(C=C, probability=True)

# Entrenar
model.fit(X_train_scaled, y_train)
preds = model.predict(X_test_scaled)

# -----------------------------
# Métricas de desempeño
# -----------------------------
st.header("📊 Métricas del modelo")

accuracy = accuracy_score(y_test, preds)
st.metric("Accuracy", f"{accuracy:.3f}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Reporte de clasificación")
    report = classification_report(y_test, preds, output_dict=True)
    st.dataframe(report)

with col2:
    st.subheader("Matriz de confusión")
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, cmap="Blues", ax=ax)
    st.pyplot(fig)

# -----------------------------
# Visualización ejemplo del dataset
# -----------------------------
st.header("🔎 Ejemplo del dataset")
idx = st.slider("Selecciona un índice", 0, len(X_test)-1, 10)

img = X_test[idx].reshape(28, 28)
st.image(img, caption=f"Etiqueta real: {y_test[idx]}", width=150)

# -----------------------------
# Subir imagen propia
# -----------------------------
st.header("🧪 Probar con tu propia imagen")

uploaded_file = st.file_uploader("Sube una imagen de un dígito (28x28 o similar)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28, 28))

    st.subheader("Imagen procesada")
    st.image(image, width=150)

    img_array = np.array(image).reshape(1, -1).astype("float32")
    img_scaled = scaler.transform(img_array)

    pred = model.predict(img_scaled)[0]
    proba = model.predict_proba(img_scaled)[0]

    st.success(f"Predicción del modelo: {pred}")

    st.subheader("Probabilidades por clase")
    prob_dict = {str(i): float(p) for i, p in enumerate(proba)}
    st.bar_chart(prob_dict)
