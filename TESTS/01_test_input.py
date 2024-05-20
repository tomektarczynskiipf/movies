# Databricks notebook source
from pyspark.sql import SparkSession
import pandas as pd

# COMMAND ----------

# Create a Spark session
spark = SparkSession.builder.appName("Load Movies Data").getOrCreate()

# Load the table into a Spark DataFrame
spark_df = spark.sql("SELECT * FROM movies.dat.input")

# Convert to pandas DataFrame
pandas_df = spark_df.toPandas()

# COMMAND ----------

reference_benchmarks = {'num_columns': 12, 'column_names': ['Rank', 'Name', 'Platform', 'Year', 'Genre', 'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales', '_rescued_data'], 'column_sums': {'Rank': 135350581, 'Year': 32752543, 'NA_Sales': 4333.43, 'EU_Sales': 2409.12, 'JP_Sales': 1284.25, 'Other_Sales': 789.01, 'Global_Sales': 8820.31}}

# COMMAND ----------

def assert_dataset(new_df, reference_benchmarks):
    # Assert number of columns
    assert len(new_df.columns) == reference_benchmarks['num_columns'], "Column count mismatch."
    
    # Assert names of columns
    assert set(new_df.columns) == set(reference_benchmarks['column_names']), "Column names do not match."
    
    # Assert sum of numerical columns
    for col in reference_benchmarks['column_sums']:
        new_sum = new_df[col].sum()
        ref_sum = reference_benchmarks['column_sums'][col]
        assert new_sum == ref_sum, f"Sum mismatch for column {col}: expected {ref_sum}, got {new_sum}"


# Assume 'reference_benchmarks' is already defined either by copying the output or loading from a file
try:
    assert_dataset(pandas_df, reference_benchmarks)
    print("All tests passed, the new dataset matches the reference.")
except AssertionError as e:
    print(f"Test failed: {e}")
