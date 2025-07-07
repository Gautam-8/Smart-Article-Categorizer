import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

# Embedding imports
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
import torch
import openai
from gensim.models import Word2Vec
from gensim.downloader import load

class EmbeddingModels:
    def __init__(self):
        self.models = {}
        self.classifiers = {}
        self.categories = ["Tech", "Finance", "Healthcare", "Sports", "Politics", "Entertainment"]
        self.model_names = ["word2vec", "bert", "sentence_bert", "openai"]
        
        # Initialize models
        self._load_models()
        
    def _load_models(self):
        """Load all embedding models"""
        print("Loading embedding models...")
        
        # 1. Word2Vec/GloVe (using pre-trained word2vec)
        try:
            self.models['word2vec'] = load('word2vec-google-news-300')
            print("✓ Word2Vec loaded")
        except:
            print("! Word2Vec not available, using random embeddings")
            self.models['word2vec'] = None
            
        # 2. BERT
        try:
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
            self.bert_model.eval()
            print("✓ BERT loaded")
        except:
            print("! BERT not available")
            self.bert_tokenizer = None
            self.bert_model = None
            
        # 3. Sentence-BERT
        try:
            self.models['sentence_bert'] = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Sentence-BERT loaded")
        except:
            print("! Sentence-BERT not available")
            self.models['sentence_bert'] = None
            
        # 4. OpenAI (will be loaded when needed)
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if openai.api_key:
            print("✓ OpenAI API key found")
        else:
            print("! OpenAI API key not found")
            
    def get_word2vec_embedding(self, text: str) -> np.ndarray:
        """Get Word2Vec embedding by averaging word vectors"""
        if self.models['word2vec'] is None:
            return np.random.rand(300)  # Fallback to random
            
        words = text.lower().split()
        embeddings = []
        
        for word in words:
            if word in self.models['word2vec']:
                embeddings.append(self.models['word2vec'][word])
                
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return np.random.rand(300)  # Fallback for unknown words
            
    def get_bert_embedding(self, text: str) -> np.ndarray:
        """Get BERT [CLS] token embedding"""
        if self.bert_tokenizer is None or self.bert_model is None:
            return np.random.rand(768)  # Fallback to random
            
        inputs = self.bert_tokenizer(text, return_tensors='pt', 
                                   truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # Use [CLS] token embedding
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
            
        return cls_embedding.numpy()
        
    def get_sentence_bert_embedding(self, text: str) -> np.ndarray:
        """Get Sentence-BERT embedding"""
        if self.models['sentence_bert'] is None:
            return np.random.rand(384)  # Fallback to random
            
        embedding = self.models['sentence_bert'].encode(text)
        return embedding
        
    def get_openai_embedding(self, text: str) -> np.ndarray:
        """Get OpenAI embedding"""
        if not openai.api_key:
            return np.random.rand(1536)  # Fallback to random
            
        try:
            response = openai.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return np.array(response.data[0].embedding)
        except:
            return np.random.rand(1536)  # Fallback on error
            
    def get_embedding(self, text: str, model_name: str) -> np.ndarray:
        """Get embedding for given text using specified model"""
        if model_name == "word2vec":
            return self.get_word2vec_embedding(text)
        elif model_name == "bert":
            return self.get_bert_embedding(text)
        elif model_name == "sentence_bert":
            return self.get_sentence_bert_embedding(text)
        elif model_name == "openai":
            return self.get_openai_embedding(text)
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
    def train_classifiers(self, articles: List[Tuple[str, str]]):
        """Train logistic regression classifiers for each embedding model"""
        print("Training classifiers...")
        
        for model_name in self.model_names:
            print(f"Training {model_name} classifier...")
            
            # Get embeddings for all articles
            X = []
            y = []
            
            for text, category in articles:
                embedding = self.get_embedding(text, model_name)
                X.append(embedding)
                y.append(category)
                
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train classifier
            clf = LogisticRegression(random_state=42, max_iter=1000)
            clf.fit(X_train, y_train)
            
            # Evaluate
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted'
            )
            
            self.classifiers[model_name] = {
                'model': clf,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            print(f"  {model_name}: Accuracy={accuracy:.3f}, F1={f1:.3f}")
            
    def predict(self, text: str, model_name: str) -> Dict:
        """Predict category for given text using specified model"""
        if model_name not in self.classifiers:
            return {"error": f"Model {model_name} not trained"}
            
        embedding = self.get_embedding(text, model_name)
        embedding = embedding.reshape(1, -1)
        
        clf = self.classifiers[model_name]['model']
        prediction = clf.predict(embedding)[0]
        probabilities = clf.predict_proba(embedding)[0]
        
        # Get confidence scores for all categories
        confidence_scores = {}
        for i, category in enumerate(clf.classes_):
            confidence_scores[category] = float(probabilities[i])
            
        return {
            "prediction": prediction,
            "confidence": float(max(probabilities)),
            "all_scores": confidence_scores,
            "model_metrics": {
                "accuracy": self.classifiers[model_name]['accuracy'],
                "precision": self.classifiers[model_name]['precision'],
                "recall": self.classifiers[model_name]['recall'],
                "f1": self.classifiers[model_name]['f1']
            }
        }
        
    def predict_all(self, text: str) -> Dict:
        """Get predictions from all models"""
        results = {}
        
        for model_name in self.model_names:
            results[model_name] = self.predict(text, model_name)
            
        return results
        
    def save_models(self, filepath: str):
        """Save trained classifiers"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.classifiers, f)
            
    def load_models(self, filepath: str):
        """Load trained classifiers"""
        with open(filepath, 'rb') as f:
            self.classifiers = pickle.load(f) 