from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

try:
    model = joblib.load('profitability_model.pkl')
except FileNotFoundError:
    model = None
    print("Model file not found. Please ensure 'profitability_model.pkl' exists.")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check the server logs.'}), 500
    
    try:
        data = request.get_json()
        
        if not all(key in data for key in ('worker_salary', 'raw_material_cost', 'targeted_materials_produced')):
            return jsonify({'error': 'Invalid input data. Please provide worker_salary, raw_material_cost, and targeted_materials_produced.'}), 400
        
        worker_salary = float(data['worker_salary'])
        raw_material_cost = float(data['raw_material_cost'])
        targeted_materials_produced = float(data['targeted_materials_produced'])
        
        input_features = [[worker_salary, raw_material_cost, targeted_materials_produced]]
        
        prediction = model.predict(input_features)
        
        return jsonify({'predicted_profit': prediction[0]})
    
    except ValueError:
        return jsonify({'error': 'Invalid data type. Please ensure all inputs are numbers.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to the Profit Prediction API!'}), 200

if __name__ == '__main__':
    app.run(debug=True)
