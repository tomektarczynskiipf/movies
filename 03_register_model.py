# Databricks notebook source
from pyspark.sql import SparkSession
import mlflow
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss
from mlflow.tracking import MlflowClient

# COMMAND ----------

# Create a Spark session
spark = SparkSession.builder.appName("Load Movies Data").getOrCreate()

# Load the table into a Spark DataFrame
spark_df = spark.sql("SELECT * FROM movies.tmp.dataset_01")

# COMMAND ----------

# Convert to pandas DataFrame
df = spark_df.toPandas()

# COMMAND ----------

# Initialize MLflow
mlflow.set_experiment('/Shared/MOVIES/EXPERIMENTS/Movies experiments')

# COMMAND ----------

# Load dataset
X, y = df[['year', 'eu_sales', 'jp_sales', 'other_sales']], df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# COMMAND ----------

import mlflow
import mlflow.xgboost
from sklearn.metrics import log_loss, accuracy_score
import xgboost as xgb
from mlflow.tracking import MlflowClient

# Assume dtrain, dtest, X, y_test are already defined and loaded.

def check_alias_exists(client, model_name, alias_name):
    try:
        model_versions = client.search_model_versions(f"name='{model_name}'")
        for version in model_versions:
            aliases = client.get_model_version(model_name, version.version).aliases
            if any(alias == alias_name for alias in aliases):
                return True
    except mlflow.exceptions.RestException:
        # Model or versions might not exist, so no aliases exist either
        pass
    return False

client = MlflowClient()
model_name = "movies.dat.prod_model"
alias_name = "model_2024_05_20"

# Check if the alias already exists
if check_alias_exists(client, model_name, alias_name):
    raise ValueError(f"Alias '{alias_name}' already exists for model '{model_name}'")

with mlflow.start_run(run_name="model depth 3"):
    # Train model
    params = {
        "objective": "reg:logistic",
        "learning_rate": 0.1,
        "colsample_bytree": 1,
        "subsample": 1,
        "max_depth": 3,
        "seed": 42,
    }
    model = xgb.train(params, dtrain, evals=[(dtrain, "train")])

    # Evaluate model
    y_proba = model.predict(dtest)
    y_pred = (y_proba > 0.5) * 1
    loss = log_loss(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metrics({"log_loss_test": loss, "accuracy_test": acc})

    # Take the first row of the training dataset as the model input example.
    input_example = X.iloc[[0]]
    # Log the model
    signature = mlflow.models.infer_signature(input_example, y_proba)
    mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
    )

    # Register the model
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    registered_model_info = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

# Create an alias for the registered model version
model_version = registered_model_info.version
client.set_registered_model_alias(name=model_name, alias=alias_name, version=model_version)

