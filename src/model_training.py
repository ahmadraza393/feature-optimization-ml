from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def train_models(X_train, y_train, regularization='l2'):
    """Train multiple ML models with optional regularization"""
    models = {}

    # Logistic Regression
    models['LogisticRegression'] = LogisticRegression(penalty=regularization, max_iter=1000)
    models['LogisticRegression'].fit(X_train, y_train)

    # Random Forest
    models['RandomForest'] = RandomForestClassifier()
    models['RandomForest'].fit(X_train, y_train)

    # Support Vector Machine
    models['SVM'] = SVC(kernel='rbf', C=1.0)
    models['SVM'].fit(X_train, y_train)

    return models
