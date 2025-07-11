from embedding_models import BERTEmbedder, GloveEmbedder, SBERTEmbedder, NomicEmbedder
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from joblib import dump

# Dataset
data = {
    "text": [
        # Tech
        "Apple launches new AI chip for iPhones",
        "Microsoft announces cloud partnership with OpenAI",
        "Google’s quantum computer achieves new milestone",
        "Meta unveils next-gen AR headset for developers",
        "Tesla introduces self-driving software upgrade",
        "Amazon uses robots to automate delivery system",
        "NVIDIA announces breakthrough in GPU performance",
        "Intel reveals its new AI-powered processor",

        # Finance
        "Stocks fall as inflation fears rise",
        "Investors see Bitcoin as safe haven",
        "Federal Reserve raises interest rates again",
        "Gold prices surge amid market uncertainty",
        "Nasdaq hits record high led by tech stocks",
        "Oil prices drop after OPEC announcement",
        "Global markets react to economic data",
        "IMF forecasts slowdown in global growth",

        # Healthcare
        "AI predicts cancer risk with high accuracy",
        "New breakthrough in heart disease treatment",
        "Hospitals adopt robotic surgery systems",
        "COVID-19 booster shots approved for children",
        "Scientists develop wearable diabetes monitor",
        "Cancer vaccine shows promising trial results",
        "New drug reduces Alzheimer’s symptoms",
        "Mental health programs expanded in schools",

        # Sports
        "India wins T20 World Cup final",
        "Ronaldo scores a hat trick in UCL",
        "Olympics 2024 preparations begin in Paris",
        "Messi signs with Inter Miami",
        "FIFA introduces new World Cup format",
        "Serena Williams announces retirement from tennis",
        "NBA Finals set between Lakers and Celtics",
        "Usain Bolt's record broken in 100m sprint",

        # Politics
        "PM addresses the nation on economic reforms",
        "Elections results spark political debates",
        "New policies target environmental sustainability",
        "Senate passes climate change legislation",
        "Leaders meet to discuss international trade",
        "New immigration law triggers protests",
        "UN votes on humanitarian aid package",
        "Government unveils budget with tax cuts",

        # Entertainment
        "Netflix launches sci-fi thriller series",
        "Marvel announces Phase 5 movie lineup",
        "Taylor Swift tops Billboard charts again",
        "Oscars 2025: full list of nominations revealed",
        "Christopher Nolan’s film breaks box office records",
        "BTS announces global comeback tour",
        "Streaming wars heat up with new entrants",
        "Grammy Awards ceremony dazzles audiences"
    ],
    "label": [
        "Tech"] * 8 +
        ["Finance"] * 8 +
        ["Healthcare"] * 8 +
        ["Sports"] * 8 +
        ["Politics"] * 8 +
        ["Entertainment"] * 8
}

df = pd.DataFrame(data)
print(df.head())

# ⏱ Train + Collect Scores
def train_embeddings(df, embedder, model_name):
    print(f"\n🔷 Training on {model_name} Embeddings...")

    # Embed
    X = np.array([embedder.embed(text) for text in df["text"]])
    y = df["label"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Train
    clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Evaluation
    print(f"📊 {model_name} Classifier Performance:\n")
    print(classification_report(y_test, y_pred))

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")

    # Save model
    os.makedirs("models", exist_ok=True)
    dump(clf, f"models/{model_name}_classifier.joblib")
    print(f"✅ {model_name} classifier saved to: models/{model_name}_classifier.joblib")

    # Return metrics
    return {
        "Model": model_name.upper(),
        "Accuracy": round(acc, 3),
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "F1": round(f1, 3)
    }

# 🔁 Train All Models
scores = []
scores.append(train_embeddings(df, GloveEmbedder(), "glove"))
scores.append(train_embeddings(df, BERTEmbedder(), "bert"))
scores.append(train_embeddings(df, SBERTEmbedder(), "sbert"))
scores.append(train_embeddings(df, NomicEmbedder(), "nomic"))

# 💾 Save evaluation summary
os.makedirs("metrics", exist_ok=True)
eval_df = pd.DataFrame(scores)
eval_df.to_csv("metrics/eval_scores.csv", index=False)
print("\n✅ All model metrics saved to: metrics/eval_scores.csv")
