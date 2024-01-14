import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump


def load_data(file_path):
    return pd.read_csv(file_path)


def save_model(model, filename):
    model_path = Path.home() / "Desktop" / "56870" / "code" / "models" / filename
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)


def main():
    data_path = Path.home() / "Desktop" / "56870" / "code" / "data" / \
        "parsed" / "merged_wine_processed.csv"
    data = load_data(data_path)

    X = data.drop('type', axis=1)
    y = data['type']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
    save_model(log_reg, 'task2_logistic_regression_type.pk1')

    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    y_pred_rfc = rfc.predict(X_test)
    print("Random Forest Classifier Accuracy:",
          accuracy_score(y_test, y_pred_rfc))
    save_model(rfc, 'task2_random_forest_type.pk1')


if __name__ == "__main__":
    main()
