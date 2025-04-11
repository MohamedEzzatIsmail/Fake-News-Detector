import pandas as pd
import string
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
import nltk

# Only once
nltk.download('stopwords')

# ----------- Load and Preprocess Dataset ------------
def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df = df[['title', 'text', 'label']].dropna()
    df['content'] = df['title'] + " " + df['text']
    df['content'] = df['content'].apply(clean_text)
    return df[['content', 'label']]

# ----------- Train the Model ------------
def train_model(df):
    X = df['content']
    y = df['label']

    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("🔍 Evaluation:\n")
    print(classification_report(y_test, y_pred))
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    return model, vectorizer

# ----------- Save and Load Model ------------
def save_model(model, vectorizer, path="model.joblib"):
    joblib.dump({'model': model, 'vectorizer': vectorizer}, path)

def load_model(path="model.joblib"):
    return joblib.load(path)

# ----------- Predict ------------
def predict_news(news_text, loaded_model):
    clean = clean_text(news_text)
    vec = loaded_model['vectorizer'].transform([clean])
    prediction = loaded_model['model'].predict(vec)[0]
    return "📰 REAL" if prediction == 0 else "🚨 FAKE"

# ----------- Main App Flow ------------
def main():
    model_path = "model.joblib"

    if not os.path.exists(model_path):
        print("📦 Training model...")
        data = load_and_prepare_data("train.csv")
        model, vectorizer = train_model(data)
        save_model(model, vectorizer, model_path)
        print("✅ Model saved.")
    else:
        print("📂 Loading existing model...")

    model_data = load_model(model_path)

    while True:
        news = input("\n🧾 Enter news article text (or type 'exit'): ")
        if news.lower() == "exit":
            break
        result = predict_news(news, model_data)
        print("Result:", result)

if __name__ == "__main__":
    main()
