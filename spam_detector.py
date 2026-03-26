"""
Spam Email Detection System
A complete ML pipeline with text preprocessing, feature engineering,
model training, and evaluation.
"""

import pandas as pd
import numpy as np
import re
import string
from collections import Counter
from typing import Tuple, List, Dict
import pickle
import json

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline, FeatureUnion

# Text processing
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Optional: Download NLTK data (run once)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')


class TextPreprocessor:
    """
    Comprehensive text preprocessing for email content.
    """
    
    def __init__(self, lowercase=True, remove_punctuation=True,
                 remove_stopwords=True, stemming=False, lemmatization=True):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        self.lemmatization = lemmatization
        
        self.stemmer = PorterStemmer() if stemming else None
        self.lemmatizer = WordNetLemmatizer() if lemmatization else None
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else None
        
        # Email-specific patterns
        self.url_pattern = re.compile(r'http\S+|www\S+|https\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.number_pattern = re.compile(r'\d+')
        self.special_chars = re.compile(r'[^a-zA-Z\s]')
        
    def clean_text(self, text: str) -> str:
        """Apply all cleaning steps."""
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Replace URLs with token
        text = self.url_pattern.sub(' URL ', text)
        
        # Replace emails with token
        text = self.email_pattern.sub(' EMAIL ', text)
        
        # Replace numbers with token
        text = self.number_pattern.sub(' NUM ', text)
        
        # Remove punctuation/special chars (keep tokens)
        if self.remove_punctuation:
            text = self.special_chars.sub(' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        
        # Stemming or Lemmatization
        if self.stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]
        elif self.lemmatization:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return ' '.join(tokens)
    
    def extract_email_features(self, text: str) -> Dict[str, float]:
        """
        Extract email-specific metadata features.
        """
        if not isinstance(text, str):
            return {}
        
        features = {}
        
        # Structural features
        features['length'] = len(text)
        features['num_exclamation'] = text.count('!')
        features['num_dollar'] = text.count('$')
        features['num_caps'] = sum(1 for c in text if c.isupper())
        features['caps_ratio'] = features['num_caps'] / len(text) if text else 0
        
        # Count suspicious words
        suspicious_words = ['free', 'winner', 'cash', 'prize', 'urgent', 'act now', 
                           'limited', 'offer', 'click', 'buy', 'order', 'call now',
                           'credit', 'loan', 'debt', 'million', 'dollars', 'nigerian']
        text_lower = text.lower()
        features['suspicious_words'] = sum(1 for word in suspicious_words if word in text_lower)
        
        # HTML/URL indicators
        features['has_html'] = 1 if '<html>' in text_lower or '<body>' in text_lower else 0
        features['num_urls'] = len(self.url_pattern.findall(text))
        
        return features


class SpamDetector:
    """
    Complete spam detection pipeline with model selection and evaluation.
    """
    
    def __init__(self, model_type='nb'):
        self.model_type = model_type
        self.preprocessor = TextPreprocessor()
        self.vectorizer = None
        self.model = None
        self.feature_names = None
        self.is_fitted = False
        
    def _get_model(self):
        """Initialize model based on type."""
        models = {
            'nb': MultinomialNB(alpha=0.1),
            'complement_nb': ComplementNB(),
            'logreg': LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced'),
            'svm': LinearSVC(C=1.0, class_weight='balanced', max_iter=2000),
            'rf': RandomForestClassifier(n_estimators=100, class_weight='balanced'),
            'gb': GradientBoostingClassifier(n_estimators=100)
        }
        return models.get(self.model_type, MultinomialNB())
    
    def create_pipeline(self, max_features=10000, ngram_range=(1, 2)):
        """
        Create sklearn pipeline with TF-IDF and classifier.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        
        self.model = self._get_model()
        
        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('classifier', self.model)
        ])
        
        return self.pipeline
    
    def prepare_data(self, texts: List[str], labels: List[int] = None) -> Tuple:
        """
        Preprocess texts and extract features.
        """
        # Clean texts
        cleaned_texts = [self.preprocessor.clean_text(text) for text in texts]
        
        # Extract additional features (optional - can be added to pipeline)
        email_features = pd.DataFrame([
            self.preprocessor.extract_email_features(text) for text in texts
        ])
        
        return cleaned_texts, email_features, labels
    
    def fit(self, texts: List[str], labels: List[int]):
        """
        Train the spam detection model.
        """
        cleaned_texts, _, _ = self.prepare_data(texts, labels)
        
        if not hasattr(self, 'pipeline'):
            self.create_pipeline()
        
        self.pipeline.fit(cleaned_texts, labels)
        self.is_fitted = True
        
        # Store feature names
        if hasattr(self.vectorizer, 'get_feature_names_out'):
            self.feature_names = self.vectorizer.get_feature_names_out()
        
        return self
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict spam (1) or ham (0) for new texts.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        cleaned_texts, _, _ = self.prepare_data(texts)
        return self.pipeline.predict(cleaned_texts)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Get probability scores for predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        cleaned_texts, _, _ = self.prepare_data(texts)
        
        # Some models don't have predict_proba
        if hasattr(self.pipeline, 'predict_proba'):
            return self.pipeline.predict_proba(cleaned_texts)
        elif hasattr(self.pipeline, 'decision_function'):
            # Convert decision function to probabilities using sigmoid
            decisions = self.pipeline.decision_function(cleaned_texts)
            return 1 / (1 + np.exp(-decisions))
        else:
            raise AttributeError("Model doesn't support probability prediction")
    
    def evaluate(self, texts: List[str], labels: List[int]) -> Dict:
        """
        Comprehensive model evaluation.
        """
        predictions = self.predict(texts)
        
        # Handle models without predict_proba for ROC-AUC
        try:
            probabilities = self.predict_proba(texts)
            if len(probabilities.shape) > 1:
                probabilities = probabilities[:, 1]
            roc_auc = roc_auc_score(labels, probabilities)
        except:
            roc_auc = None
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions),
            'recall': recall_score(labels, predictions),
            'f1_score': f1_score(labels, predictions),
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(labels, predictions).tolist()
        }
        
        print(f"\n{'='*50}")
        print(f"Model: {self.model_type.upper()}")
        print(f"{'='*50}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        if roc_auc:
            print(f"ROC-AUC:   {roc_auc:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(labels, predictions, target_names=['Ham', 'Spam']))
        
        return metrics
    
    def get_top_features(self, n=20) -> Tuple[List[str], List[str]]:
        """
        Get most important features for spam/ham classification.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Get feature log probabilities (for Naive Bayes)
        if hasattr(self.model, 'feature_log_prob_'):
            feature_log_prob = self.model.feature_log_prob_
            # Calculate ratio of spam to ham feature importance
            spam_importance = feature_log_prob[1] - feature_log_prob[0]
            
            # Top spam indicators
            top_spam_idx = np.argsort(spam_importance)[-n:][::-1]
            top_spam = [self.feature_names[i] for i in top_spam_idx]
            
            # Top ham indicators
            top_ham_idx = np.argsort(spam_importance)[:n]
            top_ham = [self.feature_names[i] for i in top_ham_idx]
            
            return top_spam, top_ham
        
        # For linear models (LogReg, SVM)
        elif hasattr(self.model, 'coef_'):
            coef = self.model.coef_[0]
            top_spam_idx = np.argsort(coef)[-n:][::-1]
            top_ham_idx = np.argsort(coef)[:n]
            
            top_spam = [self.feature_names[i] for i in top_spam_idx]
            top_ham = [self.feature_names[i] for i in top_ham_idx]
            
            return top_spam, top_ham
        
        else:
            return [], []
    
    def save_model(self, filepath: str):
        """Save trained model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'pipeline': self.pipeline,
                'preprocessor': self.preprocessor,
                'model_type': self.model_type,
                'feature_names': self.feature_names
            }, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load trained model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        detector = cls(model_type=data['model_type'])
        detector.pipeline = data['pipeline']
        detector.preprocessor = data['preprocessor']
        detector.feature_names = data['feature_names']
        detector.vectorizer = data['pipeline'].named_steps['tfidf']
        detector.model = data['pipeline'].named_steps['classifier']
        detector.is_fitted = True
        
        return detector


def compare_models(X_train, X_test, y_train, y_test):
    """
    Compare different algorithms for spam detection.
    """
    results = {}
    
    models = ['nb', 'complement_nb', 'logreg', 'svm']
    
    for model_type in models:
        detector = SpamDetector(model_type=model_type)
        detector.create_pipeline()
        detector.fit(X_train, y_train)
        metrics = detector.evaluate(X_test, y_test)
        results[model_type] = metrics
        
        top_spam, top_ham = detector.get_top_features(10)
        print(f"\nTop Spam indicators: {top_spam[:5]}")
        print(f"Top Ham indicators: {top_ham[:5]}")
    
    return results


# ============== DEMONSTRATION ==============

def create_sample_data():
    """
    Create sample spam/ham data for demonstration.
    In production, use real datasets like Enron, SpamAssassin, or Ling-Spam.
    """
    # Ham (legitimate) emails
    ham_samples = [
        "Hey, are we still meeting for lunch tomorrow at 12?",
        "Please find the attached report for Q3 sales figures.",
        "Thanks for your help with the project yesterday.",
        "Meeting rescheduled to 3pm in conference room B.",
        "Happy birthday! Hope you have a great day.",
        "Can you review the code changes and provide feedback?",
        "The package has been shipped and will arrive Thursday.",
        "Don't forget to submit your expense reports by Friday.",
        "Great job on the presentation today!",
        "Let me know when you're free to discuss the proposal.",
        "Your Amazon order #12345 has been confirmed.",
        "Weekly team sync notes attached.",
        "Reminder: Dentist appointment tomorrow at 10am.",
        "The meeting minutes from yesterday are attached.",
        "Can you pick up milk on your way home?",
        "Your password will expire in 7 days. Please update it.",
        "Thanks for the quick turnaround on this.",
        "See you at the gym later?",
        "Invoice #9876 has been paid. Thank you.",
        "Project deadline moved to next Monday."
    ]
    
    # Spam emails
    spam_samples = [
        "Congratulations! You've won $1,000,000! Call now to claim your prize!",
        "URGENT: You have won a free iPhone! Click here immediately!!!",
        "Buy cheap viagra pills now!!! 80% discount!!!",
        "Dear friend, I am a Nigerian prince and need your help transferring money.",
        "Act now! Limited time offer! Free cash waiting for you!",
        "You are a winner! Call 1-800-SCAM-NOW to claim your reward!",
        "Free entry to win a brand new car! No purchase necessary!",
        "URGENT: Your account will be suspended. Click here immediately!!!",
        "Make money fast!!! Work from home and earn $5000 weekly!!!",
        "Congratulations winner! You have been selected for a cash prize!",
        "Buy now!!! Cheap watches, pills, and loans!!!",
        "Dear beneficiary, your inheritance of $10M is waiting.",
        "Hot singles in your area want to meet you! Click here!",
        "Act immediately! Your credit card has been compromised!",
        "Free gift! You've been chosen! Call now!!!",
        "Lose weight fast with this miracle pill! Order now!",
        "You won the lottery! Send bank details to claim prize!",
        "Urgent business proposal - $50 million profit guaranteed!",
        "Click here for free money! No catch! Guaranteed!",
        "Final notice: Claim your free vacation package now!!!"
    ]
    
    texts = ham_samples + spam_samples
    labels = [0] * len(ham_samples) + [1] * len(spam_samples)
    
    return texts, labels


def main():
    """
    Main execution demonstrating the complete pipeline.
    """
    print("🚀 Spam Email Detection System")
    print("=" * 50)
    
    # 1. Load/Create Data
    print("\n1. Loading data...")
    texts, labels = create_sample_data()
    print(f"Total samples: {len(texts)} (Ham: {labels.count(0)}, Spam: {labels.count(1)})")
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Training: {len(X_train)}, Testing: {len(X_test)}")
    
    # 3. Compare models
    print("\n2. Training and comparing models...")
    results = compare_models(X_train, X_test, y_train, y_test)
    
    # 4. Train best model (Naive Bayes usually performs well for text)
    print("\n3. Training production model...")
    best_model = SpamDetector(model_type='nb')
    best_model.create_pipeline(max_features=5000, ngram_range=(1, 2))
    best_model.fit(X_train, y_train)
    
    # 5. Show feature importance
    print("\n4. Top predictive features:")
    top_spam, top_ham = best_model.get_top_features(15)
    print(f"Spam indicators:  {', '.join(top_spam[:10])}")
    print(f"Ham indicators:   {', '.join(top_ham[:10])}")
    
    # 6. Test on new examples
    print("\n5. Testing on new emails:")
    test_emails = [
        "Hey, want to grab coffee tomorrow afternoon?",
        "URGENT!!! You won FREE money!!! Click here now!!!",
        "Meeting notes from today's standup attached.",
        "Congratulations! Claim your prize of $1,000,000 now!"
    ]
    
    predictions = best_model.predict(test_emails)
    probabilities = best_model.predict_proba(test_emails)
    
    for email, pred, prob in zip(test_emails, predictions, probabilities):
        label = "SPAM" if pred == 1 else "HAM"
        confidence = prob[1] if pred == 1 else prob[0]
        print(f"\nEmail: {email[:50]}...")
        print(f"Prediction: {label} (confidence: {confidence:.2%})")
    
    # 7. Save model
    print("\n6. Saving model...")
    best_model.save_model('spam_detector_model.pkl')
    
    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()
