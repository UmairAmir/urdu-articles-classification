# 📰 Article Categorization Using Machine Learning Models

This project implements and compares multiple **machine learning algorithms** to classify Urdu news articles into predefined categories such as **sports, entertainment, politics, business, and technology**.  
It combines **traditional ML models** (Naive Bayes, Logistic Regression) with a **Neural Network**, offering a complete pipeline from **data scraping to evaluation**.

---

## 📚 Project Overview

With the rapid growth of online content, automatic classification of news articles is crucial for efficient content management and user experience.  
This project focuses on developing and evaluating models that can accurately classify articles based on their textual content.

**Key Highlights:**
- Dataset of **2,500 Urdu news articles** scraped from major sources (Express, Dunya, Jang, Geo, Samaa).  
- **Data preprocessing** includes cleaning, tokenization, and Bag-of-Words feature extraction.  
- **Three classifiers implemented:**
  - Multinomial Naive Bayes  
  - Logistic Regression  
  - Feedforward Neural Network  
- **Highest accuracy:** 96.15% (Neural Network)  

---

## 🧠 Models Implemented

| Model | Description | Accuracy |
|--------|--------------|-----------|
| **Neural Network** | Deep learning approach with ReLU activations, BatchNorm, and Dropout regularization | **96.15%** |
| **Logistic Regression** | Simple, interpretable model for linearly separable data | 95.07% |
| **Multinomial Naive Bayes** | Probabilistic model based on word frequency and Bayes’ theorem | 94.00% |

---

## 🧩 Repository Structure

| File | Description |
|------|--------------|
| `Group20_WebScraping.ipynb` | Web scraping script for collecting Urdu news articles |
| `Group20_LogisticRegression.ipynb` / `.py` | Logistic Regression model implementation |
| `Group20_MultinomialNaiveBayes.ipynb` / `.py` | Multinomial Naive Bayes classifier |
| `Group20_NeuralNetwork.ipynb` / `.py` | Feedforward Neural Network model |
| `Group20_Report.pdf` | Complete project report with methodology, results, and discussion |

---

## ⚙️ Methodology

1. **Data Scraping**
   - Scraped Urdu-language news articles using `BeautifulSoup`.
   - Preserved text encoding to avoid data loss.
   - Sources: Express, Dunya, Jang, Geo, Samaa.

2. **Data Processing & EDA**
   - Cleaned text (removed punctuation, links, numbers).
   - Performed exploratory analysis:
     - Class distribution
     - Word frequency
     - Word clouds for each category.

3. **Tokenization & Feature Extraction**
   - Used **spaCy** for tokenization.
   - Built **Bag of Words (BoW)** model for vectorization.

4. **Model Training & Evaluation**
   - Compared **Neural Network, Logistic Regression, Naive Bayes** models.
   - Evaluated on held-out test data using **accuracy, precision, recall, F1-score**.

---

## 📈 Results Summary

| Metric | Neural Network | Logistic Regression | Naive Bayes |
|---------|----------------|--------------------|--------------|
| Accuracy | 96.15% | 95.07% | 94.00% |
| Best Trade-off | High accuracy, robust generalization | Fast, interpretable | Lightweight, simple |

> The Neural Network achieved the best performance but required longer training time.

---

## 🧰 Tools & Libraries

- **Python 3.x**
- **NumPy**, **Pandas**
- **scikit-learn**
- **PyTorch**
- **BeautifulSoup4**
- **spaCy**
- **Matplotlib / Seaborn**

---

## 🚀 How to Run

1. **Clone this repository:**
   ```bash
   git clone https://github.com/<your-username>/ArticleCategorization.git
   cd ArticleCategorization
