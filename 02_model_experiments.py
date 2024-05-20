# Databricks notebook source
from pyspark.sql import SparkSession
import mlflow
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss

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

mlflow.xgboost.autolog()

# COMMAND ----------

df.columns

# COMMAND ----------

# Load dataset
X, y = df[['year', 'eu_sales', 'jp_sales', 'other_sales']], df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# COMMAND ----------

 with mlflow.start_run(run_name = "Depth 1"):
        # train model
        params = {
            "objective": "reg:logistic",
            "learning_rate": 0.1,
            "colsample_bytree": 1,
            "subsample": 1,
            "max_depth": 1,
            "seed": 42,
        }
        model = xgb.train(params, dtrain, evals=[(dtrain, "train")])

        # evaluate model
        y_proba = model.predict(dtest)
        y_pred = (y_proba > 0.5) * 1 
        loss = log_loss(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)

        # log metrics
        mlflow.log_metrics({"log_loss_test": loss, "accuracy_test": acc})

# COMMAND ----------

 with mlflow.start_run(run_name = "Depth 2"):
        # train model
        params = {
            "objective": "reg:logistic",
            "learning_rate": 0.1,
            "colsample_bytree": 1,
            "subsample": 1,
            "max_depth": 2,
            "seed": 42,
        }
        model = xgb.train(params, dtrain, evals=[(dtrain, "train")])

        # evaluate model
        y_proba = model.predict(dtest)
        y_pred = (y_proba > 0.5) * 1 
        loss = log_loss(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)

        # log metrics
        mlflow.log_metrics({"log_loss_test": loss, "accuracy_test": acc})

# COMMAND ----------

 with mlflow.start_run(run_name = "Depth 3"):
        # train model
        params = {
            "objective": "reg:logistic",
            "learning_rate": 0.1,
            "colsample_bytree": 1,
            "subsample": 1,
            "max_depth": 3,
            "seed": 42,
        }
        model = xgb.train(params, dtrain, evals=[(dtrain, "train")])

        # evaluate model
        y_proba = model.predict(dtest)
        y_pred = (y_proba > 0.5) * 1 
        loss = log_loss(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)

        # log metrics
        mlflow.log_metrics({"log_loss_test": loss, "accuracy_test": acc})
