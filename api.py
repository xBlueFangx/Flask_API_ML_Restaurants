#Instalar librerias
#   pip install -U scikit-learn
#   pip install -U Flask
#   pip install -U flask-cors
#  python api.py 



# Importamos lo 3 métodos que utilizaremos
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from sklearn.tree import DecisionTreeClassifier
import joblib

# Creamos la instancia de Flask
app = Flask(__name__)
cors = CORS(app)

# Cargar el modelo entrenado
MODEL = joblib.load('modelo_RF_classificador.joblib') ##Algo pasa con el modelo

# Cargar el LabelEncoder
le_localidad = joblib.load('label_encoder_localidad.joblib')
le_tipo = joblib.load('label_encoder_tipo_comida.joblib')

#http://127.0.0.1:5000/predict?tipo=Lebanese&localidad=West%20Ahmedabad&costo=8.0
@app.route('/predict')
def predict():
    #Se recuperan los datos de la peticion
    tipo        = request.args.get('tipo')
    localidad   = request.args.get('localidad')
    cost        = request.args.get('costo')

    # Codificar el tipo y la localidad usando los LabelEncoders
    tipo_codificado = le_tipo.transform([tipo])[0]
    localidad_codificada = le_localidad.transform([localidad])[0]

    # La lista de caracteristicas que se utilizaran para la prediccion
    features = [[tipo_codificado, localidad_codificada, cost]]
    
    # Utilizamos el modelo para la predicción de los datos
    prediccion = MODEL.predict(features)
    
    # Creamos y enviamos la respuesta al cliente
    return jsonify(status='Predicción Completada', prediccion=prediccion)


#http://127.0.0.1:5000/catalogos/localidad
@app.route('/catalogos/localidad')
def catalogo_localidad():

    result = le_localidad.classes_

    return jsonify(status="OK", localidad=result.tolist())

#http://127.0.0.1:5000/catalogos/tipo
@app.route('/catalogos/tipo')
def catalogo_tipo():

    result = le_tipo.classes_

    return jsonify(status="OK", tipo=result.tolist())


if __name__ == '__main__':
    # Iniciamos el servidor
    app.run(debug=True)