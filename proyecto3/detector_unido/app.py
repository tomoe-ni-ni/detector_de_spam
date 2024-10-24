from flask import Flask, render_template, request
import os
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Definir el directorio de los archivos guardados
SAVE_DIR = '/home/johnrobin123/proyecto3/modelo/'  # Asegúrate de que este sea el path correcto

# Cargar los vectorizadores
vectorizers = {}
for item in ['nombre', 'asunto', 'link']:
    vectorizer_path = os.path.join(SAVE_DIR, f'{item}.pkl')
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Archivo {item}.pkl no encontrado en: {vectorizer_path}")
    with open(vectorizer_path, 'rb') as file:
        vectorizers[item] = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Obtener el tipo de análisis seleccionado por el usuario
        tipo = request.form.get('tipo', 'nombre')  # 'nombre', 'asunto' o 'link'
        input_text = request.form.get('input_text', '')  # Texto ingresado por el usuario

        # Verificar si el texto está vacío
        if not input_text:
            resultado = "Por favor, ingrese un texto para analizar."
            return render_template('index.html', prediction=resultado)

        # Verificar si el tipo de análisis es válido
        if tipo not in vectorizers:
            resultado = "Tipo de análisis no válido."
            return render_template('index.html', prediction=resultado)

        # Convertir el texto a la representación numérica usando el vectorizador correspondiente
        try:
            input_vector = vectorizers[tipo].transform([input_text]).toarray()

            # Cargar el modelo correspondiente solo cuando se realiza la predicción
            model_path = os.path.join(SAVE_DIR, f'{tipo}.h5')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"El archivo {tipo}.h5 no fue encontrado en: {model_path}")

            model = load_model(model_path)

            # Realizar predicción con el modelo cargado
            prediccion = model.predict(input_vector)

            # Interpretar el resultado según el tipo de análisis
            if tipo == 'nombre':
                resultado = "El nombre es SPAM" if prediccion[0][0] >= 0.5 else "El nombre NO es SPAM"
            elif tipo == 'asunto':
                resultado = "El asunto es SPAM" if prediccion[0][0] >= 0.5 else "El asunto NO es SPAM"
            elif tipo == 'link':
                resultado = "El enlace es MALICIOSO" if prediccion[0][0] >= 0.5 else "El enlace NO es malicioso"
        except Exception as e:
            resultado = f"Ocurrió un error durante la predicción: {str(e)}"

        # Renderizar la página de resultados con la predicción
        return render_template('index.html', prediction=resultado)

if __name__ == '__main__':
    app.run(debug=True)  # Cambia a False en producción
