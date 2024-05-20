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

# Load dataset
X, y = df[['year', 'eu_sales', 'jp_sales', 'other_sales']], df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# COMMAND ----------

import mlflow
import pandas as pd

# Define the model name and alias
model_name = "movies.dat.prod_model"
alias_name = "model_2024_05_20"

# Load the model using the alias
model_uri = f"models:/{model_name}@{alias_name}"
model = mlflow.pyfunc.load_model(model_uri)


# Perform inference
predictions = model.predict(X_test)

loss = log_loss(y_test, predictions)

threshold = 1 / 10**5
if abs(loss - 0.30573104446862814) > threshold:
    raise ValueError(f"Loss {loss} differs from 0.30573104446862814 by more than {threshold}")
