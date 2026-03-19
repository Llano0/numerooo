import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# 1. Cargar y preparar los datos
digits = load_digits()
X = digits.images  # Ya vienen en formato 8x8
y = digits.target

# Normalizar los píxeles (de 0-16 a 0-1) para que la red aprenda mejor
X = X.astype("float32") / 16.0

# Las CNN esperan una forma (batch, alto, ancho, canales)
# Como son imágenes en escala de grises, el canal es 1
X = np.expand_dims(X, -1)

# Convertir etiquetas a formato categórico (One-hot encoding)
y = keras.utils.to_categorical(y, 10)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Construir la arquitectura de la CNN
model = keras.Sequential([
    keras.Input(shape=(8, 8, 1)),
    
    # Capa Convolucional: Extrae características espaciales
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Segunda capa convolucional
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    
    # Aplanar para pasar a las capas densas (clasificación)
    layers.Flatten(),
    layers.Dropout(0.5), # Para evitar el sobreajuste
    layers.Dense(10, activation="softmax") # 10 neuronas para los dígitos 0-9
])

# 3. Compilar el modelo
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 4. Entrenar
print("Entrenando la red...")
model.fit(X_train, y_train, batch_size=32, epochs=20, validation_split=0.1, verbose=1)

# 5. Evaluar
score = model.evaluate(X_test, y_test, verbose=0)
print(f"\nPrecisión en el conjunto de prueba: {score[1]*100:.2f}%")
