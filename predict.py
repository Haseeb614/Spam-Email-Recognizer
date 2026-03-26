import pickle
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
import re

def extract_advanced_features(texts):
    """Extract more powerful spam indicators"""
    features = []
    for text in texts:
        f = {}
        text_lower = text.lower()
        
        f['length'] = len(text)
        f['word_count'] = len(text.split())
        f['exclamation_count'] = text.count('!')
        f['question_count'] = text.count('?')
        f['dollar_count'] = text.count('$')
        f['caps_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        f['digit_count'] = sum(c.isdigit() for c in text)
        
        spam_words = ['free', 'winner', 'cash', 'prize', 'urgent', 'act now', 
                     'limited', 'click', 'buy', 'order', 'call now', 'credit',
                     'loan', 'debt', 'million', 'dollars', 'offer', 'winner',
                     'selected', 'claim', 'free entry', 'text win', 'urgent',
                     'account', 'verify', 'suspended', 'password', 'bank']
        
        f['spam_word_count'] = sum(1 for word in spam_words if word in text_lower)
        f['spam_word_ratio'] = f['spam_word_count'] / max(f['word_count'], 1)
        f['has_urgent'] = 1 if any(w in text_lower for w in ['urgent', 'immediate', 'now', 'today', 'expires']) else 0
        f['has_money'] = 1 if any(w in text_lower for w in ['$', 'dollar', 'cash', 'money', 'price', 'cost', 'free']) else 0
        f['has_phone'] = 1 if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text) else 0
        f['has_url'] = 1 if 'http' in text_lower or 'www' in text_lower else 0
        
        words = text.split()
        f['caps_words'] = sum(1 for w in words if w.isupper() and len(w) > 1)
        
        features.append(f)
    
    return pd.DataFrame(features)

# Load model
print("Loading improved model...")
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)

tfidf = model_data['tfidf']
preprocessor = model_data['preprocessor']
lr = model_data['lr']
svm = model_data['svm']
lr2 = model_data['lr2']

print("Spam Email Detector (IMPROVED)")
print("=" * 50)

while True:
    email = input("\nEnter email text (or 'quit' to exit): ")
    
    if email.lower() == 'quit':
        break
    
    # Preprocess
    email_clean = [preprocessor.clean_text(email)]
    email_tfidf = tfidf.transform(email_clean)
    email_features = extract_advanced_features([email])
    
    # Combine
    from scipy.sparse import hstack, csr_matrix
    email_combined = hstack([email_tfidf, csr_matrix(email_features.values)])
    
    # Predict with ensemble
    lr_proba = lr.predict_proba(email_combined)[0][1]
    svm_decision = svm.decision_function(email_combined)[0]
    svm_proba = 1 / (1 + np.exp(-svm_decision))
    lr2_proba = lr2.predict_proba(email_combined)[0][1]
    
    # Weighted ensemble
    ensemble_proba = (0.5 * lr_proba + 0.3 * svm_proba + 0.2 * lr2_proba)
    prediction = 1 if ensemble_proba > 0.5 else 0
    
    if prediction == 1:
        print(f"🚫 SPAM (confidence: {ensemble_proba:.2%})")
        print(f"   Why: Suspicious words or patterns detected")
    else:
        print(f"✅ HAM (confidence: {1-ensemble_proba:.2%})")
