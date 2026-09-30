import pandas as pd
import re
import nltk
import joblib
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


print("================================")
print(" NEWS ARTICLE CLASSIFIER")
print("================================")


# Download NLTK data
print("\nChecking NLTK data...")

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

print("\nLoading dataset...")

dataset_path = "dataset/news.csv"

if not os.path.exists(dataset_path):
    print("\nERROR: dataset/news.csv was not found!")
    print("Please create the dataset first.")
    exit()


data = pd.read_csv(dataset_path)


print("Dataset loaded successfully!")
print("Number of articles:", len(data))


# Check columns

if "text" not in data.columns or "category" not in data.columns:

    print("\nERROR!")
    print("CSV must contain these columns:")
    print("text, category")

    exit()


# Remove empty rows

data = data.dropna(
    subset=["text", "category"]
)


print("Categories:")

print(
    data["category"].value_counts()
)


# --------------------------------------------------
# 2. TEXT PREPROCESSING
# --------------------------------------------------

stop_words = set(
    stopwords.words("english")
)

lemmatizer = WordNetLemmatizer()


def preprocess_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove special characters
    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    # Tokenization
    words = text.split()

    # Stopword removal + lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


print("\nPreprocessing articles...")

data["clean_text"] = data["text"].apply(
    preprocess_text
)


# --------------------------------------------------
# 3. TRAIN / TEST SPLIT
# --------------------------------------------------

X = data["clean_text"]
y = data["category"]


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)


print("\nTraining articles:", len(X_train))
print("Testing articles:", len(X_test))


# --------------------------------------------------
# 4. TF-IDF
# --------------------------------------------------

print("\nCreating TF-IDF features...")


vectorizer = TfidfVectorizer(

    max_features=5000,

    ngram_range=(1, 2),

    min_df=1

)


X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)


print(
    "TF-IDF features:",
    X_train_tfidf.shape
)


# --------------------------------------------------
# 5. TRAIN MODEL
# --------------------------------------------------

print("\nTraining Logistic Regression model...")


model = LogisticRegression(

    max_iter=1000

)


model.fit(

    X_train_tfidf,

    y_train

)


print("Model training completed!")


# --------------------------------------------------
# 6. TEST MODEL
# --------------------------------------------------

predictions = model.predict(
    X_test_tfidf
)


accuracy = accuracy_score(

    y_test,

    predictions

)


print("\n================================")
print(" MODEL RESULTS")
print("================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions
    )
)


# --------------------------------------------------
# 7. SAVE MODEL
# --------------------------------------------------

print("\nSaving model...")


os.makedirs(
    "model",
    exist_ok=True
)


joblib.dump(

    model,

    "model/news_classifier.pkl"

)


joblib.dump(

    vectorizer,

    "model/tfidf_vectorizer.pkl"

)


print("\n================================")
print(" SUCCESS!")
print("================================")

print(
    "Model saved to:"
)

print(
    "model/news_classifier.pkl"
)

print(
    "TF-IDF saved to:"
)

print(
    "model/tfidf_vectorizer.pkl"
)

print("\nYou can now run:")
print("python app.py")