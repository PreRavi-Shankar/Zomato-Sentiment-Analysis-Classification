# Zomato-Sentiment-Analysis-Classification

# 🍽️ Zomato Restaurant Review Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green?style=flat)
![Accuracy](https://img.shields.io/badge/Accuracy-90%25+-brightgreen?style=flat)
![Project Type](https://img.shields.io/badge/Project-Classification-purple?style=flat)
![Contribution](https://img.shields.io/badge/Contribution-Individual-red?style=flat)

> **Automated sentiment classification system for Zomato restaurant reviews using NLP and Machine Learning — classifying customer feedback as Positive or Negative with 90%+ accuracy.**

---

## 📌 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Models Used](#models-used)
- [Results](#results)
- [Key Insights](#key-insights)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Business Impact](#business-impact)
- [Author](#author)

---

## 📖 Overview

In the restaurant industry, online reviews heavily influence customer decisions. Zomato receives thousands of reviews daily — making it impossible to manually read and analyse each one. This project builds an **automated ML pipeline** that reads a restaurant review and instantly classifies it as **Positive** or **Negative**, enabling restaurants to monitor satisfaction trends and respond to unhappy customers in real time.

---

## ❓ Problem Statement

> *Build an automated machine learning system that accurately classifies Zomato restaurant reviews as positive or negative based on review text — handling natural language nuances and providing reliable predictions that restaurants can act upon.*

**Task Type:** Binary Text Classification  
**Input:** Raw customer review text  
**Output:** Sentiment label → `Positive` or `Negative`

---

## 📊 Dataset

| File | Description |
|------|-------------|
| Restaurant Metadata | Info on **105 restaurants** — name, cuisine, cost, hours |
| Reviews Dataset | **10,000 customer reviews** — text, ratings (1–5 stars), reviewer info, timestamps |

**Label Engineering:**
- Rating ≥ 4 → **Positive** (1)
- Rating < 4 → **Negative** (0)
- Class distribution: ~**63% Positive**, ~37% Negative

---

## 🔄 Project Pipeline

```
Raw Data
   ↓
Data Exploration & EDA
   ↓
Label Engineering (ratings → binary sentiment)
   ↓
Text Preprocessing
  - Lowercase conversion
  - Special character & URL removal
  - Extra whitespace elimination
   ↓
Feature Extraction (TF-IDF Vectorization)
  - 5,000 features
  - Unigrams + Bigrams
   ↓
Model Training & Comparison
  - Logistic Regression
  - Naive Bayes
  - Random Forest
  - Linear SVM
   ↓
Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
   ↓
Feature Importance Analysis
   ↓
Predictions on New Reviews
   ↓
Model Saved with Pickle
```

---

## 🤖 Models Used

| Model | Type |
|-------|------|
| Logistic Regression | Linear Classifier |
| Multinomial Naive Bayes | Probabilistic Classifier |
| Random Forest | Ensemble (Bagging) |
| Linear SVM (LinearSVC) | Support Vector Machine |

---

## 📈 Results

| Model | Accuracy | Notes |
|-------|----------|-------|
| **Logistic Regression** | **~90%+** | ✅ Best performer |
| Linear SVM | High | Strong baseline |
| Random Forest | Good | Slightly lower recall |
| Naive Bayes | Good | Fast, decent precision |

- **Best Model:** Logistic Regression
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
- Correctly classifies **1800+ positive** and **1100+ negative** reviews with only ~150–200 misclassifications

---

## 💡 Key Insights

**Top Positive Sentiment Words:**
`amazing` · `excellent` · `delicious` · `outstanding` · `highly recommended`

**Top Negative Sentiment Words:**
`terrible` · `worst` · `disappointing` · `never going back` · `pathetic`

**Other Findings:**
- 63% of Zomato reviews are positive — reflecting generally good customer experiences
- Positive reviews tend to be **longer** on average than negative ones
- Bigrams like `not good`, `very bad` significantly improve classification accuracy over unigrams alone

---

## 🛠️ Tech Stack

```
Language        : Python 3.8+
Data Handling   : Pandas, NumPy
Visualization   : Matplotlib, Seaborn
NLP             : Scikit-learn (TfidfVectorizer)
ML Models       : Scikit-learn (LogisticRegression, MultinomialNB, RandomForestClassifier, LinearSVC)
Evaluation      : Scikit-learn metrics (accuracy, precision, recall, f1, roc_auc, confusion_matrix)
Model Saving    : Pickle
Environment     : Google Colab / Jupyter Notebook
```

---

## 📁 Project Structure

```
Zomato-Sentiment-Analysis-Classification/
│
├── Zomato_Sentiment_Analysis.ipynb   # Main notebook (all code + analysis)
├── sentiment_model.pkl               # Saved best model (Logistic Regression)
├── tfidf_vectorizer.pkl              # Saved TF-IDF vectorizer
├── README.md                         # Project documentation
│
└── data/
    ├── zomato_restaurants.csv        # Restaurant metadata (105 restaurants)
    └── zomato_reviews.csv            # Reviews dataset (10,000 reviews)
```

---

## ▶️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/PreRavi-Shankar/Zomato-Sentiment-Analysis-Classification.git
cd Zomato-Sentiment-Analysis-Classification
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

**3. Run the notebook**
```bash
jupyter notebook Zomato_Sentiment_Analysis.ipynb
```
Or open directly in **Google Colab**.

**4. Predict on a new review**
```python
import pickle

# Load saved model and vectorizer
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Predict
review = "The food was absolutely amazing and service was great!"
vectorized = vectorizer.transform([review.lower()])
prediction = model.predict(vectorized)[0]
print("Sentiment:", "Positive 😊" if prediction == 1 else "Negative 😞")
```

---

## 💼 Business Impact

| Use Case | Benefit |
|----------|---------|
| Real-time review monitoring | Instantly flag negative reviews for action |
| Customer satisfaction tracking | Monitor sentiment trends over time |
| Competitor analysis | Compare sentiment across restaurants |
| Service improvement | Identify specific pain points from negative keywords |
| Scalability | Handles thousands of reviews automatically — no manual reading |

---

## 👨‍💻 Author

**Ravi Shankar Kumar**  
B.Tech CSE (AI & ML) — Amity University Jharkhand (2023–27)

[![GitHub](https://img.shields.io/badge/GitHub-PreRavi--Shankar-black?style=flat&logo=github)](https://github.com/PreRavi-Shankar)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ravi%20Shankar%20Kumar-blue?style=flat&logo=linkedin)](https://linkedin.com/in/ravi-shankar-kumar)
[![LeetCode](https://img.shields.io/badge/LeetCode-PreRavi__Shankar-orange?style=flat&logo=leetcode)](https://leetcode.com/u/PreRavi_Shankar)

---

⭐ **If you found this project useful, please consider giving it a star!**
