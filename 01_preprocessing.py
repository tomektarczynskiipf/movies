# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

# Create a Spark session
spark = SparkSession.builder.appName("Load Movies Data").getOrCreate()

# Load the table into a Spark DataFrame
spark_df = spark.sql("SELECT * FROM movies.dat.input")

# COMMAND ----------

# Convert to pandas DataFrame
pandas_df = spark_df.toPandas()

pandas_df.head()

# COMMAND ----------

pandas_df.drop(columns=['_rescued_data'], inplace=True)

# COMMAND ----------

# Convert all column names to lowercase
pandas_df.columns = [col.lower() for col in pandas_df.columns]

# Display the modified DataFrame to verify the change
print(pandas_df.head())

# COMMAND ----------

pandas_df['target'] = (pandas_df['na_sales'] > 0) * 1

# COMMAND ----------

# Create or get the Spark session
spark = SparkSession.builder.appName("Upload DataFrame").getOrCreate()

# Convert pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(pandas_df)

# Define the table name and database
database_name = 'movies.tmp'
table_name = 'dataset_01'

# Write Spark DataFrame to a new table in the Databricks database
spark_df.write.saveAsTable(f"{database_name}.{table_name}", mode='overwrite')

