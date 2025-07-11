
import numpy as np
import requests
import re
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
import torch

# def download_glove():
#     if not os.path.exists("glove.6B.300d.txt"):
#         print("📦 Downloading GloVe embeddings...")
#         url = "http://nlp.stanford.edu/data/glove.6B.zip"
#         r = requests.get(url)
#         with open("glove.zip", "wb") as f:
#             f.write(r.content)
#         with zipfile.ZipFile("glove.zip", "r") as zip_ref:
#             zip_ref.extract("glove.6B.300d.txt", path=".")
#         print("✅ GloVe ready!")

# download_glove()

class GloveEmbedder:
    def __init__(self, glove_path="glove.6B.300d.txt"):
        self.embedding_dim = 300
        self.glove = self.load_glove(glove_path)

    def load_glove(self, filepath):
        print("🔄 Loading GloVe embeddings...")
        glove = {}
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                vector = list(map(float, parts[1:]))
                glove[word] = vector
        print("✅ GloVe loaded.")
        return glove

    def preprocess(self, text):
        return re.sub(r"[^a-zA-Z\s]", "", text.lower()).split()

    def embed(self, text):
        words = self.preprocess(text)
        vectors = [self.glove[word] for word in words if word in self.glove]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.embedding_dim)

class NomicEmbedder:
    def __init__(self, endpoint="http://localhost:1234/v1/embeddings"):
        self.endpoint = endpoint

    def embed(self, text: str) -> np.ndarray:
        payload = {
            "input": [text],
            "model": "text-embedding-nomic-embed-text-v1.5"
        }
        response = requests.post(self.endpoint, json=payload)
        response.raise_for_status()
        return np.array(response.json()["data"][0]["embedding"])



class BERTEmbedder:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode

    def embed(self, text: str) -> np.ndarray:
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
            return cls_embedding.squeeze().numpy()



class SBERTEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        return np.array(self.model.encode(text, convert_to_numpy=True))
