from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load your trained model
with open("banana_ripeness_softness.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Try to get JSON data
        data = request.get_json()
        if data and 'days' in data:
            days_str = data['days']
        
        # 2. If JSON failed, try to get form data (common in simple POST requests)
        else:
            days_str = request.form.get('days') or request.values.get('days')
            
        # Check if we got any value at all
        if days_str is None:
             raise ValueError("Input 'days' is missing.")

        # 3. Convert to float - this is where the error likely happens if input is non-numeric
        days = float(days_str)
        
        # 4. Prediction logic
        prediction = model.predict([[days]])
        return jsonify({'softness': round(prediction[0], 2)})
        
    except ValueError as ve:
        print(f"ValueError: {ve}") # Debugging for conversion issues
        return jsonify({'error': 'Invalid input! Please enter a number.'})
    except Exception as e:
        print(f"General Exception: {e}") # Debugging for other issues (like model loading)
        return jsonify({'error': 'An unexpected error occurred.'})
if __name__ == "__main__":
    app.run(debug=True)
