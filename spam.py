# scripts/spam.py
# Email Spam Detection Project
# Ready-to-run with real dataset

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')  # Ignore minor warnings

# --- Step 1: Load real dataset ---
# Make sure you downloaded the SMS Spam Collection Dataset from Kaggle
# and placed it in data/spam.csv
df = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only relevant columns
df = df[['v1', 'v2']]  # 'v1' = label, 'v2' = text
df.columns = ['label', 'text']  # Rename columns

# Convert label to numeric: 'spam' = 1, 'ham' = 0
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print("Dataset Loaded. Sample data:")
print(df.head())

# --- Step 2: Prepare Features and Target ---
X = df['text']
y = df['label']

# Convert text to numeric features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# --- Step 3: Split Data into Training and Testing ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# --- Step 4: Train Naive Bayes Model ---
model = MultinomialNB()
model.fit(X_train, y_train)

# --- Step 5: Evaluate Model ---
y_pred = model.predict(X_test)
print("\nModel Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Step 6: Test Prediction ---
test_emails = [
    'Free money now',
    'Hello friend, how are you?',
    'Congratulations! You won a prize'
]

for email in test_emails:
    pred = model.predict(vectorizer.transform([email]))
    print(f"Prediction for '{email}':", "Spam" if pred[0]==1 else "Not Spam")