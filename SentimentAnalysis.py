import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

class SentimentAnalyzer:
    def __init__(self, hf_token=None):
        self.hf_token = hf_token
        self.models = {}
        self.setup_models()
    
    def setup_models(self):
        # Setup RoBERTa model for Twitter sentiment
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.labels = ['negative', 'neutral', 'positive']
        
        # Setup Hugging Face API if token provided
        if self.hf_token:
            self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
            self.headers = {"Authorization": f"Bearer {self.hf_token}"}
    
    def analyze_with_roberta(self, text):
        # Preprocess text for RoBERTa
        processed_text = self.preprocess_for_roberta(text)
        
        # Tokenize
        encoded_input = self.tokenizer(processed_text, return_tensors='pt', 
                                     truncation=True, max_length=512)
        
        # Get prediction
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        
        # Get the highest scoring sentiment
        max_score_idx = np.argmax(scores)
        sentiment = self.labels[max_score_idx]
        confidence = scores[max_score_idx]
        
        return {
            'sentiment': sentiment,
            'confidence': float(confidence),
            'scores': {label: float(score) for label, score in zip(self.labels, scores)}
        }
    
    def analyze_with_textblob(self, text):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def analyze_with_huggingface_api(self, text):
        if not self.hf_token:
            return None
        
        payload = {"inputs": text, "options": {"wait_for_model": True}}
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                top_result = max(result, key=lambda x: x['score'])
                return {
                    'sentiment': top_result['label'].lower(),
                    'confidence': top_result['score'],
                    'all_scores': result
                }
        return None
    
    def preprocess_for_roberta(self, text):
        # Specific preprocessing for RoBERTa model
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)
    
    def batch_analyze(self, texts, method='roberta'):
        results = []
        
        for text in texts:
            try:
                if method == 'roberta':
                    result = self.analyze_with_roberta(text)
                elif method == 'textblob':
                    result = self.analyze_with_textblob(text)
                elif method == 'huggingface_api':
                    result = self.analyze_with_huggingface_api(text)
                else:
                    result = self.analyze_with_roberta(text)  # default
                
                results.append(result)
            except Exception as e:
                print(f"Error analyzing text: {e}")
                results.append({'sentiment': 'neutral', 'confidence': 0.0})
        
        return results
