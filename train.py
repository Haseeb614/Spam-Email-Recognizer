import pandas as pd
import numpy as np
from spam_detector import SpamDetector, TextPreprocessor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import re

# Load dataset
print("Loading dataset...")
df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'text'])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print(f"Total: {len(df)} | Ham: {sum(df['label']==0)} | Spam: {sum(df['label']==1)}")

# Enhanced feature engineering
def extract_advanced_features(texts):
    """Extract more powerful spam indicators"""
    features = []
    for text in texts:
        f = {}
        text_lower = text.lower()
        
        # Length features
        f['length'] = len(text)
        f['word_count'] = len(text.split())
        
        # Character features
        f['exclamation_count'] = text.count('!')
        f['question_count'] = text.count('?')
        f['dollar_count'] = text.count('$')
        f['caps_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        f['digit_count'] = sum(c.isdigit() for c in text)
        
        # Spam keywords (weighted)
        spam_words = ['free', 'winner', 'cash', 'prize', 'urgent', 'act now', 
                     'limited', 'click', 'buy', 'order', 'call now', 'credit',
                     'loan', 'debt', 'million', 'dollars', 'offer', 'winner',
                     'selected', 'claim', 'free entry', 'text win', 'urgent',
                     'account', 'verify', 'suspended', 'password', 'bank']
        
        f['spam_word_count'] = sum(1 for word in spam_words if word in text_lower)
        f['spam_word_ratio'] = f['spam_word_count'] / max(f['word_count'], 1)
        
        # Urgency indicators
        f['has_urgent'] = 1 if any(w in text_lower for w in ['urgent', 'immediate', 'now', 'today', 'expires']) else 0
        
        # Money indicators
        f['has_money'] = 1 if any(w in text_lower for w in ['$', 'dollar', 'cash', 'money', 'price', 'cost', 'free']) else 0
        
        # Contact indicators
        f['has_phone'] = 1 if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text) else 0
        f['has_url'] = 1 if 'http' in text_lower or 'www' in text_lower else 0
        
        # ALL CAPS words
        words = text.split()
        f['caps_words'] = sum(1 for w in words if w.isupper() and len(w) > 1)
        
        features.append(f)
    
    return pd.DataFrame(features)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'].tolist(), df['label'].tolist(), 
    test_size=0.2, random_state=42, stratify=df['label']
)

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# Get advanced features
print("Extracting features...")
X_train_features = extract_advanced_features(X_train)
X_test_features = extract_advanced_features(X_test)

# Text preprocessing
preprocessor = TextPreprocessor()
X_train_clean = [preprocessor.clean_text(t) for t in X_train]
X_test_clean = [preprocessor.clean_text(t) for t in X_test]

# TF-IDF with better parameters
print("Creating TF-IDF...")
tfidf = TfidfVectorizer(
    max_features=15000,      # More features
    ngram_range=(1, 3),      # Include trigrams
    min_df=2,
    max_df=0.9,
    sublinear_tf=True,
    use_idf=True
)

X_train_tfidf = tfidf.fit_transform(X_train_clean)
X_test_tfidf = tfidf.transform(X_test_clean)

# Combine TF-IDF with engineered features
from scipy.sparse import hstack, csr_matrix
X_train_combined = hstack([X_train_tfidf, csr_matrix(X_train_features.values)])
X_test_combined = hstack([X_test_tfidf, csr_matrix(X_test_features.values)])

print(f"Feature matrix shape: {X_train_combined.shape}")

# Train multiple models and ensemble
print("\nTraining models...")

# Model 1: Logistic Regression with tuning
print("Training Logistic Regression...")
lr = LogisticRegression(C=10, max_iter=1000, class_weight='balanced')
lr.fit(X_train_combined, y_train)

# Model 2: Linear SVM
print("Training SVM...")
svm = LinearSVC(C=1, max_iter=3000, class_weight='balanced')
svm.fit(X_train_combined, y_train)

# Model 3: Another LR with different params
lr2 = LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced')
lr2.fit(X_train_combined, y_train)

# Ensemble predictions
print("\nEvaluating...")

def get_proba(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    else:
        return model.decision_function(X)

# Average ensemble
lr_proba = get_proba(lr, X_test_combined)
svm_proba = get_proba(svm, X_test_combined)
lr2_proba = get_proba(lr2, X_test_combined)

# Weighted average (LR gets more weight)
ensemble_proba = (0.5 * lr_proba + 0.3 * svm_proba + 0.2 * lr2_proba)
ensemble_pred = (ensemble_proba > 0.5).astype(int)

print(f"\nAccuracy:  {accuracy_score(y_test, ensemble_pred):.4f}")
print(f"Precision: {precision_score(y_test, ensemble_pred):.4f}")
print(f"Recall:    {recall_score(y_test, ensemble_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, ensemble_pred):.4f}")

# Save the improved model
print("\nSaving improved model...")
with open('model.pkl', 'wb') as f:
    pickle.dump({
        'tfidf': tfidf,
        'preprocessor': preprocessor,
        'lr': lr,
        'svm': svm,
        'lr2': lr2,
        'feature_names': list(X_train_features.columns)
    }, f)

print("Done! Model saved.")
