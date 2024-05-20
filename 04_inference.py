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
X = df[['year', 'eu_sales', 'jp_sales', 'other_sales']]

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
predictions = model.predict(X)

# COMMAND ----------

df['predictions'] = predictions

# COMMAND ----------

# Create or get the Spark session
spark = SparkSession.builder.appName("Upload DataFrame").getOrCreate()

# Convert pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(df)

# Define the table name and database
database_name = 'movies.dat'
table_name = 'predictions'

# Write Spark DataFrame to a new table in the Databricks database
spark_df.write.saveAsTable(f"{database_name}.{table_name}", mode='overwrite')
