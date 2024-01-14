import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
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

    data = pd.get_dummies(data, columns=['type'])

    X = data.drop('quality', axis=1)
    y = data['quality']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    y_pred = gbc.predict(X_test)
    print("Gradient Boosting Classifier Accuracy:",
          accuracy_score(y_test, y_pred))
    save_model(gbc, 'task3_gradient_boosting_quality.pk1')

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    print("K-Nearest Neighbors Classifier Accuracy:",
          accuracy_score(y_test, y_pred_knn))
    save_model(knn, 'task3_knn_quality.pk1')


if __name__ == "__main__":
    main()
