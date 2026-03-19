import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Configuración de la CNN ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            # Entrada: (1, 8, 8)
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # Salida: (16, 4, 4)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

# --- Carga y Preprocesamiento ---
@st.cache_data
def prepare_data():
    digits = load_digits()
    X = digits.images.astype(np.float32)
    y = digits.target
    
    # Normalización simple (0-1)
    X /= 16.0 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Reshape para CNN: (Batch, Channels, H, W)
    X_train = X_train.reshape(-1, 1, 8, 8)
    X_test = X_test.reshape(-1, 1, 8, 8)
    
    return X_train, X_test, y_train, y_test

# --- Entrenamiento ---
def train_model(X_train, y_train):
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    inputs = torch.from_numpy(X_train)
    labels = torch.from_numpy(y_train).long()
    
    epochs = 50
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    return model

# --- Interfaz de Streamlit ---
st.title("🔢 Clasificador de Dígitos (CNN)")
st.write("Usando el dataset Digits de Scikit-Learn (8x8 píxeles).")

X_train, X_test, y_train, y_test = prepare_data()

if 'model' not in st.session_state:
    with st.spinner('Entrenando la Red Neuronal...'):
        st.session_state.model = train_model(X_train, y_train)
    st.success("Modelo entrenado con éxito.")

# Selección de imagen de prueba
idx = st.slider("Selecciona un índice del set de prueba:", 0, len(X_test)-1, 10)
sample_img = X_test[idx]
true_label = y_test[idx]

# Predicción
st.session_state.model.eval()
with torch.no_grad():
    input_tensor = torch.from_numpy(sample_img).unsqueeze(0)
    output = st.session_state.model(input_tensor)
    prediction = torch.argmax(output, dim=1).item()

# Mostrar resultados
col1, col2 = st.columns(2)
with col1:
    st.image(sample_img.reshape(8, 8), caption=f"Imagen Real", width=150, clamp=True)

with col2:
    st.metric("Predicción del Modelo", prediction)
    st.metric("Etiqueta Real", true_label)

if prediction == true_label:
    st.balloons()
