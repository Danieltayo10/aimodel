import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import shap

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob  # For sentiment analysis

MODEL_PATH = "model.pkl"

# =======================
# EXISTING ML FUNCTIONS
# =======================

def prepare_data(df, target_col):
    y = df[target_col]
    X = df.drop(columns=[target_col])
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    X[numeric_cols] = X[numeric_cols].fillna(0)
    X[categorical_cols] = X[categorical_cols].fillna("UNKNOWN")
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    return X_encoded, y

def train_model(df, target_col, problem_type="Auto Detect"):
    X, y = prepare_data(df, target_col)
    if problem_type == "Auto Detect":
        if y.dtype.kind in "if":
            problem_type = "Regression"
        else:
            problem_type = "Classification"

    if problem_type == "Regression":
        model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
    else:
        model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    if problem_type == "Regression":
        preds = model.predict(X_test)
        metric = mean_absolute_error(y_test, preds)
    else:
        preds = model.predict(X_test)
        metric = accuracy_score(y_test, preds)

    joblib.dump((model, X.columns.tolist(), problem_type), MODEL_PATH)

    return {"metric": round(float(metric), 4), "problem_type": problem_type, "features_used": len(X.columns)}

def load_model():
    return joblib.load(MODEL_PATH)

def predict(model, feature_columns, input_data):
    df = pd.DataFrame([input_data])
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]
    pred = model.predict(df)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0].max()
        return pred, round(float(proba), 3)
    return pred, None

def explain(model, feature_columns, input_data):
    df = pd.DataFrame([input_data])[feature_columns]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(df)
    return dict(zip(feature_columns, shap_values.values[0]))

# =======================
# NEW TEXT / AI FUNCTIONS
# =======================

def analyze_sentiment(text):
    """Return sentiment polarity and label"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.05:
        label = "Positive"
    elif polarity < -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return {"polarity": round(polarity, 3), "label": label}

def extract_topics(texts, n_topics=5, n_top_words=5):
    """Return LDA topic modeling results"""
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    topics = []
    feature_names = vectorizer.get_feature_names_out()
    for idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-n_top_words:][::-1]]
        topics.append({"topic_id": idx+1, "keywords": top_words})
    return topics

def extract_keywords(texts, n_keywords=10):
    """Return top TF-IDF keywords from a list of texts"""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(texts)
    sums = X.sum(axis=0)
    terms = vectorizer.get_feature_names_out()
    data = []
    for col, term in enumerate(terms):
        data.append((term, sums[0, col]))
    data = sorted(data, key=lambda x: x[1], reverse=True)[:n_keywords]
    return [term for term, score in data]
