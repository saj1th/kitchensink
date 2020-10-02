# Databricks notebook source
# MAGIC %md
# MAGIC ## Moving to Production

# COMMAND ----------

import mlflow
import mlflow.pyfunc
from pyspark.sql.functions import col
from pyspark.sql.types import StringType

# Get latest model
latest_model_detail = mlflow.tracking.MlflowClient().get_latest_versions("westca_life_expectancy", stages=['Production'])[0]
model_udf = mlflow.pyfunc.spark_udf(spark, latest_model_detail.source)

future_input_df = spark.read.table("life_expectancy.featurized").drop("WHOSIS_000001").filter("Year > 2016")

quoted_cols = list(map(lambda c: f"`{c}`", future_input_df.columns))
with_prediction_df = future_input_df.withColumn("WHOSIS_000001", model_udf(*quoted_cols))

# COMMAND ----------

# Unencode country for display
country_cols = [c for c in with_prediction_df.columns if c.startswith("Country_")]
def unencode_country(*is_country):
  for i in range(len(country_cols)):
    if is_country[i]:
      return country_cols[i][-3:]
    
unencode_country_udf = udf(unencode_country, StringType())

country_unencoded_df = with_prediction_df.withColumn("Country", unencode_country_udf(*country_cols)).drop(*country_cols)

display(country_unencoded_df.select(col("Year"), col("Country"), col("WHOSIS_000001").alias("LifeExpectancy")).orderBy("Year", "Country"))

# COMMAND ----------

input_df = spark.read.table("life_expectancy.input")
display(input_df.filter("Year <= 2016").select(col("Year"), col("Country"), col("WHOSIS_000001").alias("LifeExpectancy")).union(
  country_unencoded_df.select(col("Year"), col("Country"), col("WHOSIS_000001").alias("LifeExpectancy"))))

# COMMAND ----------


