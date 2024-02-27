from flask import Flask, jsonify, redirect, request
from flask_restful import Api, Resource, MethodNotAllowed, NotFound
from flask_cors import CORS
from flask import Flask, jsonify, redirect, request
from flask_restful import Api, Resource, MethodNotAllowed, NotFound
from util.common import domain, port, prefix, build_swagger_config_json
from resources.swaggerConfig import SwaggerConfig
from resources.bookResource import BooksGETResource, BookGETResource, BookPOSTResource, BookPUTResource, BookDELETEResource
# Removed the flask_swagger_ui import as we will use flasgger
import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from flasgger import Swagger

# Main
application = Flask(__name__)
app = application
app.config['PROPAGATE_EXCEPTIONS'] = True
CORS(app)
api = Api(app, prefix=prefix, catch_all_404s=True)

# Configure Flasgger
swagger_config = Swagger.DEFAULT_CONFIG
swagger_config['headers'] = []
swagger_config['specs'] = [
    {
        "endpoint": 'apispec_1',
        "route": '/apispec_1.json',
        "rule_filter": lambda rule: True,  # all in
        "model_filter": lambda tag: True,  # all in
    }
]
swagger_config['static_url_path'] = "/flasgger_static"
swagger_config['swagger_ui'] = True
swagger_config['specs_route'] = "/swagger/"

# Initialize Swagger only once with the config
swagger = Swagger(app, config=swagger_config)


# Error Handler
@app.errorhandler(NotFound)
def handle_method_not_found(e):
    response = jsonify({"message": str(e)})
    response.status_code = 404
    return response

@app.errorhandler(MethodNotAllowed)
def handle_method_not_allowed_error(e):
    response = jsonify({"message": str(e)})
    response.status_code = 405
    return response

@app.route('/')
def redirect_to_prefix():
    if prefix != '':
        return redirect(prefix)
    else:
        return jsonify({'message': 'This is the root of the API, perhaps you want to go to /<prefix>'}), 200


# Add Resource
api.add_resource(SwaggerConfig, '/swagger-config')
api.add_resource(BooksGETResource, '/books')
api.add_resource(BookGETResource, '/books/<int:id>')
api.add_resource(BookPOSTResource, '/books')
api.add_resource(BookPUTResource, '/books/<int:id>')
api.add_resource(BookDELETEResource, '/books/<int:id>')

# Model Setup
model_path = 'models/final_best_lr_processed_cash_out_data.pkl'
absolute_path = os.path.abspath(model_path)

cash_out_model_path = absolute_path
transfer_model_path = absolute_path

cash_out_model = joblib.load(cash_out_model_path)
transfer_model = joblib.load(transfer_model_path)


# Prediction Resources
def preprocess_input(input_data):
    df = pd.DataFrame([input_data])
    df = df[['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud']]
    
    # Assuming 'type' is categorical and needs to be encoded
    le_type = LabelEncoder()
    df['type'] = le_type.fit_transform(df['type'])
    
    # Return preprocessed DataFrame
    return df

def predict_fraud(model, preprocessed_input):
    prediction = model.predict(preprocessed_input)
    return int(prediction[0])

class PredictCashOut(Resource):
    def post(self): 
        """
        Predict if a cash-out transaction is fraudulent
        ---
        tags:
          - Prediction
        consumes:
          - application/json
        produces:
          - application/json
        parameters:
          - in: body
            name: body
            description: Transaction data
            required: true
            schema:
              type: object
              properties:
                step:
                  type: integer
                  description: The step in the simulation
                  example: 1
                type:
                  type: string
                  description: Type of transaction
                  example: "TRANSFER"
                amount:
                  type: number
                  description: Amount of the transaction
                  example: 1000.0
                nameOrig:
                  type: string
                  description: Customer who started the transaction
                  example: "C123456789"
                oldbalanceOrg:
                  type: number
                  description: Initial balance before the transaction
                  example: 5000.0
                newbalanceOrig:
                  type: number
                  description: New balance after the transaction
                  example: 4000.0
                nameDest:
                  type: string
                  description: Recipient of the transaction
                  example: "C987654321"
                oldbalanceDest:
                  type: number
                  description: Initial recipient balance before the transaction
                  example: 1000.0
                newbalanceDest:
                  type: number
                  description: New recipient balance after the transaction
                  example: 2000.0
        responses:
          200:
            description: The predicted result
            schema:
              type: object
              properties:
                prediction:
                  type: integer
                  description: 0 indicates 'not fraudulent', 1 indicates 'fraudulent'
        """
        try:
            input_data = request.get_json()
            preprocessed_input = preprocess_input(input_data)
            prediction = predict_fraud(cash_out_model, preprocessed_input)
            return jsonify({'prediction': prediction})
        except Exception as e:
            return jsonify({'error': str(e)})

class PredictTransfer(Resource):
    def post(self):
        """
        Predict if a cash-out transaction is fraudulent
        ---
        tags:
          - Prediction
        consumes:
          - application/json
        produces:
          - application/json
        parameters:
          - in: body
            name: body
            description: Transaction data
            required: true
            schema:
              type: object
              properties:
                step:
                  type: integer
                  description: The step in the simulation
                  example: 1
                type:
                  type: string
                  description: Type of transaction
                  example: "TRANSFER"
                amount:
                  type: number
                  description: Amount of the transaction
                  example: 1000.0
                nameOrig:
                  type: string
                  description: Customer who started the transaction
                  example: "C123456789"
                oldbalanceOrg:
                  type: number
                  description: Initial balance before the transaction
                  example: 5000.0
                newbalanceOrig:
                  type: number
                  description: New balance after the transaction
                  example: 4000.0
                nameDest:
                  type: string
                  description: Recipient of the transaction
                  example: "C987654321"
                oldbalanceDest:
                  type: number
                  description: Initial recipient balance before the transaction
                  example: 1000.0
                newbalanceDest:
                  type: number
                  description: New recipient balance after the transaction
                  example: 2000.0
        responses:
          200:
            description: The predicted result
            schema:
              type: object
              properties:
                prediction:
                  type: integer
                  description: 0 indicates 'not fraudulent', 1 indicates 'fraudulent'
        """
        try:
            input_data = request.get_json()
            preprocessed_input = preprocess_input(input_data)
            prediction = predict_fraud(transfer_model, preprocessed_input)
            return jsonify({'prediction': prediction})
        except Exception as e:
            return jsonify({'error': str(e)})

api.add_resource(PredictCashOut, '/predict_cash_out')
api.add_resource(PredictTransfer, '/predict_transfer')

if __name__ == '__main__':
    app.run(debug=True)
