# Spam-Email-Recognizer
# Installation
pip install -r requirements.txt
Download required NLTK data:
Python
Copy
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
Usage
1. Train the Model
bash
Copy
python train.py
2. Run Predictions
bash
Copy
python predict.py
Type emails to check. Type quit to exit.
3. Run Web API (Optional)
bash
Copy
python app.py
Open browser to http://localhost:5000
Models Used
Logistic Regression (ensemble)
Linear SVM
TF-IDF with n-grams
Custom feature engineering
Project Structure
Copy
spam-email-detector/
├── spam_detector.py      # Core ML module
├── train.py              # Training script
├── predict.py            # CLI tool
├── app.py                # Flask web API
├── requirements.txt      # Dependencies
├── README.md             # Documentation
├── LICENSE               # MIT License
└── .gitignore            # Git ignore rules
Dataset
Uses SMS Spam Collection Dataset (5,574 messages)
Source: UCI Machine Learning Repository
4,827 ham (legitimate) messages
747 spam messages
