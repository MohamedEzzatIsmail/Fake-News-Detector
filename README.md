# 📰 Fake News Detector

A Python-based machine learning project to classify news articles as **REAL** or **FAKE** using Natural Language Processing (NLP) techniques.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![ML](https://img.shields.io/badge/ML-TF--IDF%20%2B%20LogReg-orange)

---

## 🚀 Features

- Detects fake vs. real news articles based on text content
- Uses TF-IDF vectorization and Logistic Regression
- Preprocessing with NLTK (stopwords, cleaning)
- Model persistence with Joblib
- Interactive CLI for manual testing

---

## 📁 Project Structure

fake-news-detector/ │ ├── fake_news_detector.py # Main code ├── news.csv # Dataset (Kaggle) ├── model.joblib # Saved model + vectorizer └── README.md # Project documentation


## 📊 Dataset

- 📂 **Source**: [Fake and Real News Dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- 🔍 **Columns used**:  
  - `title`: Headline  
  - `text`: Full article body  
  - `label`: `0 = REAL`, `1 = FAKE`

---

## 🧠 How It Works

### 1. Preprocessing
- Lowercase conversion
- Punctuation removal
- Stopword filtering (`nltk`)
- Merges `title` and `text`

### 2. Feature Extraction
- TF-IDF Vectorizer (`max_features=5000`)

### 3. Model Training
- Logistic Regression (`sklearn`)
- Trained on 80% of the dataset

### 4. Prediction
- User inputs a news article via CLI
- Returns "REAL" or "FAKE" label

---

## ⚙️ Setup

### ✅ Requirements

Install dependencies:

```
pip install pandas scikit-learn joblib nltk
Download stopwords:


import nltk
nltk.download('stopwords')
▶️ Run the Project

python fake_news_detector.py
🧪 Sample Predictions

🧾 Enter news article text (or type 'exit'): Bill Gates Creates Microchip Vaccine to Track People
Result: 🚨 FAKE

🧾 Enter news article text (or type 'exit'): NASA Confirms New Moon Mission
Result: 📰 REAL
📈 Model Evaluation
Metric	Score
Accuracy	~93–95%
Classifier	Logistic Regression
Vectorizer	TF-IDF (5,000 features)
🛠️ Future Enhancements
Add deep learning model (LSTM or BERT)

Build a web app (Flask/Streamlit)

Use live news APIs for real-time predictions

Browser extension for one-click analysis

📄 License
This project is licensed under the MIT License.

🙋‍♂️ Author
Project Type: Machine Learning, NLP, Text Classification
Languages: Python, NLTK, Scikit-learn
Keywords: Fake News, Logistic Regression, TF-IDF, Python, ML, NLP

