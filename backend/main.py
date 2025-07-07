from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
import os
import sys

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.embeddings import EmbeddingModels
from data.sample_articles import sample_articles, categories

app = FastAPI(title="Smart Article Categorizer API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
embedding_models = None
is_trained = False

class ArticleRequest(BaseModel):
    text: str
    model_name: str = "all"  # "all", "word2vec", "bert", "sentence_bert", "openai"

class TrainRequest(BaseModel):
    retrain: bool = False

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global embedding_models, is_trained
    
    print("Starting Smart Article Categorizer API...")
    
    # Initialize embedding models
    embedding_models = EmbeddingModels()
    
    # Check if models are already trained
    model_path = "trained_models.pkl"
    if os.path.exists(model_path):
        try:
            embedding_models.load_models(model_path)
            is_trained = True
            print("✓ Pre-trained models loaded successfully")
        except:
            print("! Failed to load pre-trained models, will train on first request")
    else:
        print("! No pre-trained models found, will train on first request")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Smart Article Categorizer API",
        "version": "1.0.0",
        "status": "running",
        "trained": is_trained
    }

@app.get("/categories")
async def get_categories():
    """Get available categories"""
    return {"categories": categories}

@app.get("/models")
async def get_models():
    """Get available embedding models"""
    return {
        "models": ["word2vec", "bert", "sentence_bert", "openai"],
        "descriptions": {
            "word2vec": "Word2Vec/GloVe - Average word vectors",
            "bert": "BERT - [CLS] token embeddings",
            "sentence_bert": "Sentence-BERT - Direct sentence embeddings",
            "openai": "OpenAI - text-embedding-ada-002"
        }
    }

@app.post("/train")
async def train_models(request: TrainRequest = TrainRequest(retrain=False)):
    """Train or retrain all embedding models"""
    global embedding_models, is_trained
    
    if is_trained and not request.retrain:
        return {
            "message": "Models already trained. Use retrain=true to force retraining.",
            "trained": True
        }
    
    if embedding_models is None:
        raise HTTPException(status_code=500, detail="Embedding models not initialized")
    
    try:
        print("Training models with sample data...")
        embedding_models.train_classifiers(sample_articles)
        
        # Save trained models
        embedding_models.save_models("trained_models.pkl")
        is_trained = True
        
        return {
            "message": "Models trained successfully",
            "trained": True,
            "sample_data_size": len(sample_articles)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict")
async def predict_article(request: ArticleRequest):
    """Predict article category using specified model(s)"""
    global embedding_models, is_trained
    
    if embedding_models is None:
        raise HTTPException(status_code=500, detail="Embedding models not initialized")
    
    if not is_trained:
        # Auto-train if not trained yet
        try:
            print("Auto-training models...")
            embedding_models.train_classifiers(sample_articles)
            embedding_models.save_models("trained_models.pkl")
            is_trained = True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Auto-training failed: {str(e)}")
    
    try:
        if request.model_name == "all":
            # Get predictions from all models
            results = embedding_models.predict_all(request.text)
            return {
                "text": request.text,
                "predictions": results,
                "model_count": len(results)
            }
        else:
            # Get prediction from specific model
            if request.model_name not in ["word2vec", "bert", "sentence_bert", "openai"]:
                raise HTTPException(status_code=400, detail="Invalid model name")
            
            result = embedding_models.predict(request.text, request.model_name)
            return {
                "text": request.text,
                "model": request.model_name,
                "prediction": result
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model-performance")
async def get_model_performance():
    """Get performance metrics for all trained models"""
    global embedding_models, is_trained
    
    if embedding_models is None:
        raise HTTPException(status_code=500, detail="Embedding models not initialized")
    
    if not is_trained:
        raise HTTPException(status_code=400, detail="Models not trained yet")
    
    try:
        performance = {}
        for model_name in embedding_models.model_names:
            if model_name in embedding_models.classifiers:
                metrics = embedding_models.classifiers[model_name]
                performance[model_name] = {
                    "accuracy": metrics['accuracy'],
                    "precision": metrics['precision'],
                    "recall": metrics['recall'],
                    "f1": metrics['f1']
                }
        
        return {"performance": performance}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": embedding_models is not None,
        "trained": is_trained
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 