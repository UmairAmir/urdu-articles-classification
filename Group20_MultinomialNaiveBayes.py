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
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
import re
from sklearn.model_selection import train_test_split

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
# Multinomial Naive Bayes from Scratch
#

# %%


class NaiveBayes:
    def __init__(self):
        self.class_probs = {}
        self.word_probs = {}
        self.vocab_size = 0
        self.classes = []

    def fit(self, X, y):
        n_docs, total_words = X.shape
        self.vocab_size = total_words
        class_word_counts = {}
        class_counts = {}
        for class_index in np.unique(y):
            temp_class = X[y == class_index]
            class_counts[class_index] = len(temp_class)
            class_word_counts[class_index] = np.sum(temp_class, axis=0)
        self.classes = np.unique(y)
        for class_index in self.classes:
            self.class_probs[class_index] = np.log(
                class_counts[class_index] / n_docs)
        for class_index in self.classes:
            total_words_in_class = np.sum(class_word_counts[class_index])
            self.word_probs[class_index] = np.log(
                (class_word_counts[class_index] + 1) / (total_words_in_class + self.vocab_size))

    def predict(self, X):
        predictions = []
        for doc in X:
            class_scores = {}
            for class_index in self.classes:
                class_scores[class_index] = self.class_probs[class_index]
                class_scores[class_index] += np.sum(
                    doc * self.word_probs[class_index])
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
        return np.array(predictions)


# %%
nb = NaiveBayes()
nb.fit(X_train, y_train)
y_test_pred = nb.predict(X_test)
accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {accuracy}")
precision = precision_score(y_test, y_test_pred, average='macro')
recall = recall_score(y_test, y_test_pred, average='macro')
f1 = f1_score(y_test, y_test_pred, average='macro')
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=[
      str(label) for label in np.unique(y_test)]))
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# %% [markdown]
# Sklearn Naive Bayes to verify
#

# %%
model = BernoulliNB()
model.fit(X_train, y_train)
y_pred_new = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_new)
print("Bernoulli Implemenation\n")
print("Accuracy:", accuracy)
print("\n")
model2 = MultinomialNB()
model2.fit(X_train, y_train)
y_pred_new = model2.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_new)
print("Multinomial Implemenation\n")
print("Accuracy:", accuracy)
