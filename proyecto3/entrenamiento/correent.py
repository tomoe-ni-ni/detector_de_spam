import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
import pickle

# Cargar el dataset
df = pd.read_csv('dataset_correos_500.csv')

# Separar las características (X) y las etiquetas (y)
X = df['correo']
y = df['es_spam']

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorización de texto
from sklearn.feature_extraction.text import CountVectorizer

# Usar CountVectorizer para convertir los correos a una representación numérica
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

# Crear el modelo
model = Sequential()
model.add(Dense(128, input_shape=(X_train_vec.shape[1],), activation='relu'))  # Capa de entrada
model.add(Dropout(0.3))  # Regularización
model.add(Dense(64, activation='relu'))  # Capa oculta
model.add(Dropout(0.3))  # Regularización
model.add(Dense(1, activation='sigmoid'))  # Capa de salida

# Compilar el modelo
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train_vec, y_train, epochs=10, batch_size=32, validation_data=(X_test_vec, y_test))

# Evaluar el rendimiento en el conjunto de prueba
loss, accuracy = model.evaluate(X_test_vec, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Guardar el modelo en varios formatos

# 1. Guardar como .h5
SAVE_DIR = 'models/'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

model.save(os.path.join(SAVE_DIR, 'model.h5'))  # Guarda el modelo completo en formato HDF5

# 2. Guardar como .keras
model.save(os.path.join(SAVE_DIR, 'model.keras'), save_format='keras')

# 3. Guardar en formato JSON (solo la estructura)
model_json = model.to_json()
with open(os.path.join(SAVE_DIR, 'model.json'), "w") as json_file:
    json_file.write(model_json)

# Guardar los pesos por separado si usas JSON
model.save_weights(os.path.join(SAVE_DIR, 'model_weights.h5'))

# Opcional: Guardar el vectorizador en formato .pkl
with open(os.path.join(SAVE_DIR, 'vectorizer.pkl'), 'wb') as file:
    pickle.dump(vectorizer, file)

print(f'Modelos guardados en: {SAVE_DIR}')
