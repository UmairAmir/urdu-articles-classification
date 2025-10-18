# %% [markdown]
# Imports

# %%
import spacy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

# %% [markdown]
# EDA

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
# Train Test Split and Lable Encoding

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
# Neural Network

# %%
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# %%


class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x


# %%
input_size = X_train.shape[1]
num_classes = len(le.classes_)
model = NeuralNet(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %%
num_epochs = 50
train_losses = []
train_accuracies = []
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_train_tensor).sum().item() / \
        y_train_tensor.size(0)
    train_losses.append(loss.item())
    train_accuracies.append(accuracy)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy * 100:.2f}%")

# %%
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_classes = torch.argmax(y_pred, axis=1)
final_accuracy = (y_pred_classes == y_test_tensor).sum(
).item() / y_test_tensor.size(0)
print(f"Final Test Accuracy: {final_accuracy * 100:.2f}%")

precision = precision_score(y_test_tensor, y_pred_classes, average='macro')
recall = recall_score(y_test_tensor, y_pred_classes, average='macro')
f1 = f1_score(y_test_tensor, y_pred_classes, average='macro')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_tensor,
      y_pred_classes, target_names=le.classes_))
conf_matrix = confusion_matrix(y_test_tensor, y_pred_classes)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
epochs = np.arange(1, num_epochs + 1)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Training Loss", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs. Epoch")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label="Training Accuracy", color="green")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Epoch")
plt.legend()
plt.tight_layout()
plt.show()
