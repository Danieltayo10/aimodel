import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import shap

MODEL_PATH = "model.pkl"

def prepare_data(df, target_col):
    # Separate features and target
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Identify column types
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Handle missing values
    X[numeric_cols] = X[numeric_cols].fillna(0)
    X[categorical_cols] = X[categorical_cols].fillna("UNKNOWN")

    # Encode categorical features
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    return X_encoded, y


def train_model(df, target_col):
    X, y = prepare_data(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    joblib.dump((model, X.columns.tolist()), MODEL_PATH)

    return {
        "mae": round(mae, 4),
        "features_used": len(X.columns)
    }


def load_model():
    return joblib.load(MODEL_PATH)


def predict(model, feature_columns, input_data):
    df = pd.DataFrame([input_data])

    # Fill missing columns
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    prediction = model.predict(df)[0]
    return round(float(prediction), 4)


def explain(model, feature_columns, input_data):
    df = pd.DataFrame([input_data])[feature_columns]
    explainer = shap.Explainer(model)
    shap_values = explainer(df)

    return dict(zip(feature_columns, shap_values.values[0]))
