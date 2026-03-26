from flask import Flask, request, jsonify
from spam_detector import SpamDetector, TextPreprocessor  # Add TextPreprocessor here
import pickle

app = Flask(__name__)
detector = SpamDetector.load_model('spam_detector_model.pkl')

@app.route('/')
def home():
    return "Spam Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    email = data.get('email', '')
    
    prediction = detector.predict([email])[0]
    probability = detector.predict_proba([email])[0]
    
    return jsonify({
        'email': email[:100] + '...' if len(email) > 100 else email,
        'is_spam': bool(prediction),
        'spam_probability': float(probability[1]),
        'ham_probability': float(probability[0]),
        'result': 'SPAM' if prediction == 1 else 'HAM'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
