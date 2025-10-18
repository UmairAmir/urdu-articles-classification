# %% [markdown]
# Imports
#

# %%
import spacy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# %% [markdown]
# EDA
#

# %%
df = pd.read_csv('Group20_RawData.csv')
df.head()

# %%
print(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

# %%
df.info()

# %%
df.describe()

# %%
print(df['gold_label'].value_counts())

# %%
print(df.isnull().sum())

# %%
plt.figure(figsize=(10, 6))
sns.countplot(x='gold_label', data=df,
              order=df['gold_label'].value_counts().index)
plt.title('Distribution of Gold Labels')
plt.xlabel('Gold Labels')
plt.ylabel('Count')
plt.xticks(rotation=30)
plt.show()

# %%
df['gold_label'].unique()

# %%
df['gold_label'] = df['gold_label'].str.strip()

# %%
df['title_length'] = df['title'].apply(len)
df['title_length'].describe()

# %%
sns.histplot(df['title_length'], bins=20, kde=True)
plt.title('Title Length Distribution')
plt.show()

# %%
vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=20)
bow_matrix = vectorizer.fit_transform(df['title'].fillna(''))
print(vectorizer.get_feature_names_out())

# %%
df['content'] = df['content'].fillna('')
df['content_length'] = df['content'].apply(len)
df['content_length'].describe()

# %%
df['word_count'] = df['content'].apply(lambda x: len(str(x).split()))
sns.histplot(df['word_count'], bins=20, kde=True)
plt.title('Word Count Distribution')
plt.show()

# %%
df[df['content_length'] < 50]

# %% [markdown]
# DATA CLEANING
#

# %%
df.drop(columns=['link'], inplace=True)

# %%
df = df[df['title'].notna() | df['content'].notna()]

# %%


def clean_text(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


df['title'] = df['title'].apply(clean_text)
df['content'] = df['content'].apply(clean_text)

# %% [markdown]
# Tokenization and Bag of Words
#

# %%
nlp = spacy.blank('ur')


def tokenize_text(text):
    if pd.isna(text):
        return []
    doc = nlp(text)
    return [token.text for token in doc]


df['title_tokens'] = df['title'].apply(tokenize_text)
df['content_tokens'] = df['content'].apply(tokenize_text)
df[['title', 'title_tokens', 'content', 'content_tokens']].head()

# %%


class BagOfWords:
    def __init__(self):
        self.vocab = {}

    def fit(self, documents):
        unique_words = set(word for doc in documents for word in doc)
        self.vocab = {word: idx for idx, word in enumerate(unique_words)}

    def transform(self, documents):
        rows = []
        for doc in documents:
            word_counts = Counter(doc)
            row = [word_counts.get(word, 0) for word in self.vocab]
            rows.append(row)
        return np.array(rows)


bow = BagOfWords()
bow.fit(df['title_tokens'] + df['content_tokens'])
X = bow.transform(df['title_tokens'] + df['content_tokens'])

# %% [markdown]
# Sklearn Logistic Regression to verify
#

# %%
le = LabelEncoder()
y = le.fit_transform(df['gold_label'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# %% [markdown]
# Train Test Split and Label Encoding
#

# %%
le = LabelEncoder()
y = le.fit_transform(df['gold_label'])

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=69)
print("Training Data")
print(X_train.shape)
print(y_train.shape)
print("Testing Data")
print(X_test.shape)
print(y_test.shape)

# %% [markdown]
# Logistic Regression from Scratch
#

# %%


class LogisticRegressionScratch:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return [0 if i > 0.5 else 1 for i in y_predicted]


class LogisticRegressionOvR:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.models = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.models = []
        for cls in self.classes:
            y_binary = (y == cls).astype(int)
            model = LogisticRegressionScratch(self.lr, self.epochs)
            model.fit(X, y_binary)
            self.models.append(model)

    def predict(self, X):
        predictions = np.array(
            [model.sigmoid(np.dot(X, model.weights) + model.bias) for model in self.models])
        return np.argmax(predictions, axis=0)

# %%


model = LogisticRegressionOvR(lr=0.1, epochs=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# %%
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[
      str(label) for label in np.unique(y_test)]))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
