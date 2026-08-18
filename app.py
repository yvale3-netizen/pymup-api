from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Cargar el modelo que descargaste de Colab
modelo = joblib.load('modelo_pymup.pkl')

@app.route('/')
def home():
    return "API Pymup ML - Funcionando!"

@app.route('/diagnosticar', methods=['POST'])
def diagnosticar():
    data = request.get_json()
    texto = data.get('texto', '')
    if not texto:
        return jsonify({'error': 'Texto vacio'}), 400
    
    prediccion = modelo.predict([texto])[0]
    probas = modelo.predict_proba([texto])[0]
    confianza = float(max(probas))
    
    return jsonify({
        'categoria': prediccion,
        'confianza': round(confianza * 100, 2)
    })

if __name__ == '__main__':
    app.run()