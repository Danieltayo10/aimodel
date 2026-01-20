import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import shap

MODEL_PATH = "model.pkl"

def prepare_data(df, target_col):
    y = df[target_col]
    X = df.drop(columns=[target_col])

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Handle missing values
    X[numeric_cols] = X[numeric_cols].fillna(0)
    X[categorical_cols] = X[categorical_cols].fillna("UNKNOWN")

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    return X_encoded, y

def train_model(df, target_col, problem_type="Auto Detect"):
    X, y = prepare_data(df, target_col)

    # Auto-detect if problem_type is Auto
    if problem_type == "Auto Detect":
        if y.dtype.kind in "if":  # numeric -> regression
            problem_type = "Regression"
        else:
            problem_type = "Classification"

    if problem_type == "Regression":
        model = GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42
        )
    else:  # Classification
        model = GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

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

    # Fill missing columns
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]
    pred = model.predict(df)[0]

    if hasattr(model, "predict_proba"):  # classification probability
        proba = model.predict_proba(df)[0].max()
        return pred, round(float(proba), 3)
    return pred, None

def explain(model, feature_columns, input_data):
    df = pd.DataFrame([input_data])[feature_columns]
    explainer = shap.Explainer(model)
    shap_values = explainer(df)
    return dict(zip(feature_columns, shap_values.values[0]))
