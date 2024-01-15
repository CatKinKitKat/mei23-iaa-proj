from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


def load_data(file_path):
    return pd.read_csv(file_path)


def save_model(model, filename):
    model_path = Path.home() / "Desktop" / "56870" / "code" / "models" / filename
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)


def confusion_matrix_heatmap(y_test, y_pred, name):
    cm_gbc = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm_gbc, annot=True, fmt="d", cmap="Blues")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("Classifier Confusion Matrix")

    fig_save_path = Path.home() / "Desktop" / "56870" / "code" / "data" / "out" / "graphs" / f"task2_{name}_matrix.png"
    fig_save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_save_path)


def main():
    data_path = Path.home() / "Desktop" / "56870" / "code" / "data" / "parsed" / "merged_wine_processed.csv"
    data = load_data(data_path)

    x = data.drop('type', axis=1)
    y = data['type']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)
    y_pred = log_reg.predict(x_test)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
    save_model(log_reg, 'task2_logistic_regression_type.pk1')

    confusion_matrix_heatmap(y_test, y_pred, 'log_reg')

    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    y_pred_rfc = rfc.predict(x_test)
    print("Random Forest Classifier Accuracy:", accuracy_score(y_test, y_pred_rfc))
    save_model(rfc, 'task2_random_forest_type.pk1')

    confusion_matrix_heatmap(y_test, y_pred_rfc, 'rfc')


if __name__ == "__main__":
    main()
