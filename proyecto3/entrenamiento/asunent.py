import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
import pickle

# Configurar directorios para guardar los modelos
SAVE_DIR = os.path.expanduser('/home/danijari1/models/')
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Cargar el CSV
df = pd.read_csv('correos.csv')

# Crear una nueva columna combinando 'asunto' y 'cuerpo'
df['texto'] = df['asunto'] + ' ' + df['cuerpo']

# Separar las características (X) y las etiquetas (y)
X = df['texto']
y = df['es_spam']

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir el texto a valores numéricos usando TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)  # Ajustar el número de características a 3000
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

# Guardar el vectorizador en formato .pkl
with open(os.path.join(SAVE_DIR, 'vectorizer.pkl'), 'wb') as file:
    pickle.dump(vectorizer, file)

# Crear el modelo con una cantidad mínima de neuronas para obtener buenos resultados
model = Sequential()

# Capa de entrada con 128 neuronas
model.add(Dense(128, input_shape=(X_train_tfidf.shape[1],), activation='relu'))
model.add(Dropout(0.2))

# Capa oculta con 64 neuronas
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

# Capa oculta con 32 neuronas
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

# Capa de salida
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train_tfidf, y_train, epochs=15, batch_size=32, validation_split=0.2)

# Evaluar el rendimiento en el conjunto de prueba
loss, accuracy = model.evaluate(X_test_tfidf, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Guardar el modelo en varios formatos

# 1. Guardar como .h5
model.save(os.path.join(SAVE_DIR, 'model.h5'))  # Guarda el modelo completo en formato HDF5

# 2. Guardar como .keras
model.save(os.path.join(SAVE_DIR, 'model.keras'), save_format='keras')

# 3. Guardar en formato JSON (solo la estructura)
model_json = model.to_json()
with open(os.path.join(SAVE_DIR, 'model.json'), "w") as json_file:
    json_file.write(model_json)

# Guardar los pesos por separado si usas JSON
model.save_weights(os.path.join(SAVE_DIR, 'model_weights.h5'))

print(f'Modelos guardados en: {SAVE_DIR}')
