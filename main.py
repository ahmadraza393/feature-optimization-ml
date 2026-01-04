from src.data_preprocessing import load_data, save_processed_data
from src.feature_engineering import scale_features
from src.model_training import train_models
from src.evaluation import evaluate_models
from sklearn.model_selection import train_test_split

# Load dataset
data = load_data('data/raw/heart_disease.csv')

# Split features and target
X = data.drop('target', axis=1)
y = data['target']

# Feature scaling
X = scale_features(X, X.columns, method='standard')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = train_models(X_train, y_train, regularization='l2')

# Evaluate
results = evaluate_models(models, X_test, y_test)

# Print results
for model_name, metrics in results.items():
    print(f"\n{model_name} Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
