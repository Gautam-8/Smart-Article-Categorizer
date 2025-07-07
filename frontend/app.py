import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Smart Article Categorizer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .prediction-result {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2e7d32;
    }
    .confidence-score {
        font-size: 1rem;
        color: #666;
    }
    .metric-card {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_categories():
    """Get available categories from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/categories")
        if response.status_code == 200:
            return response.json()["categories"]
        return []
    except:
        return []

def get_models():
    """Get available models from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        if response.status_code == 200:
            return response.json()
        return {}
    except:
        return {}

def predict_article(text: str, model_name: str = "all"):
    """Get prediction from API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"text": text, "model_name": model_name},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}

def get_model_performance():
    """Get model performance metrics"""
    try:
        response = requests.get(f"{API_BASE_URL}/model-performance")
        if response.status_code == 200:
            return response.json()["performance"]
        return {}
    except:
        return {}

def train_models():
    """Train models via API"""
    try:
        response = requests.post(f"{API_BASE_URL}/train", json={"retrain": True})
        if response.status_code == 200:
            return response.json()
        return {"error": f"Training failed: {response.status_code}"}
    except Exception as e:
        return {"error": f"Training error: {str(e)}"}

def display_prediction_results(results: Dict[str, Any]):
    """Display prediction results in a nice format"""
    if "predictions" in results:
        # Multiple model predictions
        st.markdown("### 🎯 Predictions from All Models")
        
        # Create columns for each model
        cols = st.columns(2)
        
        for i, (model_name, prediction) in enumerate(results["predictions"].items()):
            with cols[i % 2]:
                if "error" in prediction:
                    st.error(f"**{model_name.upper()}**: {prediction['error']}")
                else:
                    st.markdown(f"""
                    <div class="model-card">
                        <h4>{model_name.upper()}</h4>
                        <div class="prediction-result">📂 {prediction['prediction']}</div>
                        <div class="confidence-score">Confidence: {prediction['confidence']:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show top 3 confidence scores
                    if "all_scores" in prediction:
                        scores_df = pd.DataFrame(
                            list(prediction["all_scores"].items()),
                            columns=["Category", "Score"]
                        ).sort_values("Score", ascending=False)
                        
                        fig = px.bar(
                            scores_df.head(3),
                            x="Score",
                            y="Category",
                            orientation="h",
                            title=f"{model_name.upper()} - Top 3 Categories",
                            color="Score",
                            color_continuous_scale="Blues"
                        )
                        fig.update_layout(height=300, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
        
        # Model comparison
        st.markdown("### 📊 Model Comparison")
        comparison_data = []
        
        for model_name, prediction in results["predictions"].items():
            if "error" not in prediction:
                comparison_data.append({
                    "Model": model_name.upper(),
                    "Prediction": prediction["prediction"],
                    "Confidence": prediction["confidence"],
                    "Accuracy": prediction.get("model_metrics", {}).get("accuracy", 0),
                    "F1 Score": prediction.get("model_metrics", {}).get("f1", 0)
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)
            
            # Confidence comparison chart
            fig = px.bar(
                df,
                x="Model",
                y="Confidence",
                title="Model Confidence Comparison",
                color="Confidence",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig, use_container_width=True)

def main():
    # Header
    st.markdown('<div class="main-header">📰 Smart Article Categorizer</div>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("🚨 Backend API is not running. Please start the FastAPI server first.")
        st.code("uvicorn backend.main:app --reload --port 8000")
        return
    
    # Sidebar
    st.sidebar.title("🔧 Settings")
    
    # Get available data
    categories = get_categories()
    models_info = get_models()
    
    if categories:
        st.sidebar.success(f"✅ API Connected - {len(categories)} categories available")
    else:
        st.sidebar.error("❌ Failed to connect to API")
        return
    
    # Display available categories
    st.sidebar.markdown("### 📂 Available Categories")
    for category in categories:
        st.sidebar.markdown(f"• {category}")
    
    # Display available models
    st.sidebar.markdown("### 🤖 Available Models")
    if "descriptions" in models_info:
        for model, desc in models_info["descriptions"].items():
            st.sidebar.markdown(f"**{model.upper()}**: {desc}")
    
    # Training section
    st.sidebar.markdown("### 🎯 Model Training")
    if st.sidebar.button("🔄 Retrain Models"):
        with st.spinner("Training models..."):
            result = train_models()
            if "error" in result:
                st.sidebar.error(f"Training failed: {result['error']}")
            else:
                st.sidebar.success("Models trained successfully!")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### ✍️ Enter Article Text")
        
        # Sample articles for testing
        sample_articles = {
            "Tech": "Apple announces new iPhone with advanced AI capabilities and improved battery life lasting up to 20 hours.",
            "Finance": "Stock market reaches record highs as investors show confidence in economic recovery prospects.",
            "Healthcare": "New cancer treatment shows promising results in clinical trials with 85% success rate.",
            "Sports": "Championship game draws record viewership as two undefeated teams battle for the title.",
            "Politics": "Presidential election campaign intensifies as candidates debate key policy issues facing nation.",
            "Entertainment": "Blockbuster movie breaks box office records with opening weekend earnings exceeding expectations."
        }
        
        # Sample article selector
        selected_sample = st.selectbox(
            "📝 Or choose a sample article:",
            [""] + list(sample_articles.keys())
        )
        
        default_text = sample_articles.get(selected_sample, "")
        
        # Text input
        article_text = st.text_area(
            "Article Text:",
            value=default_text,
            height=150,
            placeholder="Enter your article text here..."
        )
        
        # Prediction button
        if st.button("🔮 Classify Article", type="primary"):
            if article_text.strip():
                with st.spinner("Analyzing article..."):
                    results = predict_article(article_text)
                    
                    if "error" in results:
                        st.error(f"Error: {results['error']}")
                    else:
                        display_prediction_results(results)
            else:
                st.warning("Please enter some article text first!")
    
    with col2:
        st.markdown("### 📈 Model Performance")
        
        # Get and display model performance
        performance = get_model_performance()
        
        if performance:
            for model_name, metrics in performance.items():
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{model_name.upper()}</h4>
                    <p><strong>Accuracy:</strong> {metrics['accuracy']:.2%}</p>
                    <p><strong>F1 Score:</strong> {metrics['f1']:.2%}</p>
                    <p><strong>Precision:</strong> {metrics['precision']:.2%}</p>
                    <p><strong>Recall:</strong> {metrics['recall']:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Performance comparison chart
            perf_df = pd.DataFrame(performance).T
            perf_df.index.name = "Model"
            perf_df = perf_df.reset_index()
            
            fig = px.bar(
                perf_df,
                x="Model",
                y=["accuracy", "f1", "precision", "recall"],
                title="Model Performance Comparison",
                barmode="group"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🔄 Train models to see performance metrics")
    
    # Footer
    st.markdown("---")
    st.markdown("### 🔍 How it works")
    st.markdown("""
    1. **Word2Vec/GloVe**: Averages word vectors for document representation
    2. **BERT**: Uses [CLS] token embeddings from pre-trained BERT model
    3. **Sentence-BERT**: Direct sentence embeddings using all-MiniLM-L6-v2
    4. **OpenAI**: Uses text-embedding-ada-002 API (requires API key)
    
    Each model is trained with Logistic Regression for classification.
    """)

if __name__ == "__main__":
    main() 