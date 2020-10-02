# Databricks notebook source
# MAGIC %md 
# MAGIC 
# MAGIC ## 3. Model Interpretation & Deployment
# MAGIC 
# MAGIC <img src="https://sa-iot.s3.ca-central-1.amazonaws.com/collateral/mlops.png" width="900">
# MAGIC 
# MAGIC 
# MAGIC From there the question is, what features seem to predict life expectancy? Given the relatively limited span of data, and the limitations of what models can tell us about causality, interpretation requires some care. We applied the package `shap` using Databricks to attempt to explain what the model learned in a principled way, and can now view the logged plot:

# COMMAND ----------

import mlflow
import shap

client = mlflow.tracking.MlflowClient()
latest_model_detail = client.get_latest_versions("westca_life_expectancy", stages=['Staging'])[0]
client.download_artifacts(latest_model_detail.run_id, "summary_plot.png", "/dbfs/FileStore/tmp/sajith/")

# COMMAND ----------

# MAGIC %md
# MAGIC ![](/files/tmp/sajith/summary_plot.png)

# COMMAND ----------

# MAGIC %md
# MAGIC The two most important features that stand out by far are:
# MAGIC - Mortality from CVD, cancer, diabetes or CRD between exact ages 30 and 70 (%)
# MAGIC - Year
# MAGIC 
# MAGIC The horizontal axis units are years of life expectancy. It is not the rate of change of life expectancy with respect to the feature value, but the average effect (positive or negative) that the feature's particular value explains per country and year over the data set. Each country and year is a dot, and red dots indicate high values of the feature.
# MAGIC 
# MAGIC Year is self-explanatory; clearly there is generally an upward trend in life expectancy per time, with an average absolute of effect of 0.3 years. But mortality from cardiac diseases, cancer, and diabetes explains even more. Higher %s obviously explain lower life expectancy, as seen at the left.
# MAGIC 
# MAGIC None of these necessarily cause life expectancy directly, but as a first pass, these are suggestive of factors that at least correlate over the last 20 years.
# MAGIC 
# MAGIC SHAP can produce an interaction plot to further study the effect of the most-significant feature. Its built-in `matplotlib`-based plots render directly.

# COMMAND ----------

import databricks.koalas as ks

input_ks = spark.read.table("westca_life_expectancy.featurized").to_koalas()
X = input_ks.drop('WHOSIS_000001', axis=1).to_pandas()
y = input_ks['WHOSIS_000001'].to_pandas()

# COMMAND ----------

import numpy as np
import mlflow
import mlflow.xgboost
import shap

model = mlflow.xgboost.load_model(latest_model_detail.source)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X, y=y)

feature = "SH.DYN.NCOM.ZS"
feature_loc = X.columns.get_loc(feature)
interactions = explainer.shap_interaction_values(X).mean(axis=0)
interactions[feature_loc] = 0 # don't consider feature itself as interaction
max_interaction = np.argmax(np.abs(interactions[feature_loc]))

code_lookup_df = spark.read.table("westca_life_expectancy.descriptions")
code_lookup = dict([(r['Code'], r['Description']) for r in code_lookup_df.collect()])
display_cols = [code_lookup[c] if c in code_lookup else c for c in X.columns]

def abbrev(c):
  return c if len(c) < 32 else c[0:32]+"..."
abbrev_display_cols = [abbrev(c) for c in display_cols]
display(shap.dependence_plot(abbrev(code_lookup["SH.DYN.NCOM.ZS"]), shap_values, X, x_jitter=0.5, alpha=0.5, interaction_index=max_interaction, feature_names=abbrev_display_cols))

# COMMAND ----------

# MAGIC %md
# MAGIC Each point is a country and year. This plots the mortality rate mentioned above (`SH.DYN.NCOM.ZS`) versus SHAP value -- the effect on predicted life expectancy that this particular mortality rate has in that time and place. Of course, higher mortality rates are associated with lower predicted life expectancy.
# MAGIC 
# MAGIC Colors correspond to years, which was selected as a feature that most strongly interacts with mortality rate. It's also not surprising that in later years (red), mortality rate is lower and thus life expectancy higher. There is a mild secondary trend here, seen if comparing the curve of blue points (longer go) to red point (more recent). Predicted life expectancy, it might be said, varies less with this mortality rate recently than in the past.

# COMMAND ----------

# MAGIC %md
# MAGIC The United States stood out as an outlier in the life expectancy plot above. We might instead ask, how is the USA different relative to other countries. SHAP can help explain how features explain predicted life expectancy differently.

# COMMAND ----------

us_delta = shap_values[X['Country_USA']].mean(axis=0) - shap_values[~X['Country_USA']].mean(axis=0)
importances = list(zip([float(f) for f in us_delta], display_cols))
top_importances = sorted(importances, key=lambda p: abs(p[0]), reverse=True)[:10]
display(spark.createDataFrame(top_importances, ["Mean SHAP delta", "Feature"]))

# COMMAND ----------

# MAGIC %md
# MAGIC Mortality rate due to cardiac disease, diabetes and cancer stands out in the USA. On average, it explains almost a year less life expectancy than in other countris.
# MAGIC 
# MAGIC This model can now be moved to Production, for consumption and deployment for inference:

# COMMAND ----------

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

future_input_df = spark.read.table("westca_life_expectancy.featurized").drop("WHOSIS_000001").filter("Year > 2016")

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

input_df = spark.read.table("westca_life_expectancy.input")
display(input_df.filter("Year <= 2016").select(col("Year"), col("Country"), col("WHOSIS_000001").alias("LifeExpectancy")).union(
  country_unencoded_df.select(col("Year"), col("Country"), col("WHOSIS_000001").alias("LifeExpectancy"))))
