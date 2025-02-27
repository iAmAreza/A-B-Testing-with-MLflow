import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load the Iris dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize MLflow
mlflow.set_experiment("AB_Testing_Experiment")

# Model A: Random Forest
with mlflow.start_run(run_name="Model_A_RandomForest"):
    model_a = RandomForestClassifier(n_estimators=50, random_state=42)
    model_a.fit(X_train, y_train)
    predictions_a = model_a.predict(X_test)
    
    # Log metrics
    accuracy_a = accuracy_score(y_test, predictions_a)
    f1_score_a = f1_score(y_test, predictions_a, average='weighted')
    mlflow.log_metric("accuracy", accuracy_a)
    mlflow.log_metric("f1_score", f1_score_a)
    
    # Log model
    mlflow.sklearn.log_model(model_a, "model_a")
    
    # Print classification report
    print("Model A - Random Forest Classification Report:\n", classification_report(y_test, predictions_a))

# Model B: Gradient Boosting
with mlflow.start_run(run_name="Model_B_GradientBoosting"):
    model_b = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model_b.fit(X_train, y_train)
    predictions_b = model_b.predict(X_test)
    
    # Log metrics
    accuracy_b = accuracy_score(y_test, predictions_b)
    f1_score_b = f1_score(y_test, predictions_b, average='weighted')
    mlflow.log_metric("accuracy", accuracy_b)
    mlflow.log_metric("f1_score", f1_score_b)
    
    # Log model
    mlflow.sklearn.log_model(model_b, "model_b")
    
    # Print classification report
    print("Model B - Gradient Boosting Classification Report:\n", classification_report(y_test, predictions_b))

# Instructions to view the MLflow UI
print("\nRun the following command to view results in the MLflow UI:")
print("mlflow ui")
