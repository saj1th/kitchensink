# Databricks notebook source
# MAGIC %md 
# MAGIC 
# MAGIC ## 2. Modeling
# MAGIC 
# MAGIC <img src="https://sa-iot.s3.ca-central-1.amazonaws.com/collateral/mlops.png" width="900">
# MAGIC 
# MAGIC 
# MAGIC The cleaned and joined data might be handed over to a data scientist for analysis. It can be re-read from the Delta table. 
# MAGIC 
# MAGIC At this point, it may be data scientists taking over, and they can continue in Python using the same data set.

# COMMAND ----------

input_df = spark.read.table("westca_life_expectancy.input")
display(input_df.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Engineering
# MAGIC Many columns' values are mostly missing, or entirely empty. 
# MAGIC 
# MAGIC However, because the data is organized by time, it's reasonable to forward/back fill data by country and year to impute missing values. 
# MAGIC 
# MAGIC This isn't is as good as having actual values, but as the dimensions here are generally slow-changing over years, it is likely to help the analysis.

# COMMAND ----------

input_pd = input_df.toPandas()
input_pd = input_pd.sort_values('Year').groupby('Country').ffill()
input_pd = input_pd.sort_values('Year').groupby('Country').bfill()
input_df = spark.createDataFrame(input_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC First, take a look at some suggested features and how they correlate, especially with the target (life expectancy, `WHOSIS_000001`, the left/top column/row). 
# MAGIC 
# MAGIC In Databricks it's easy to use standard libraries like `seaborn` for this.

# COMMAND ----------

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

selected_indicators = [
  "WHOSIS_000001",        # Target - life expectancy
  "WHS9_85",             # Literacy rate among adults aged >= 15 years (%)
  "WHS9_96",             # Population living in urban areas (%)
  "WHS7_156",            # Per capita total expenditure on health at average exchange rate (US$)
  #"WHS7_104",            # Per capita government expenditure on health at average exchange rate (US$)
  #"WHS7_113",            # General government expenditure on health as a percentage of total government expenditure
  #"WHS7_143",            # Total expenditure on health as a percentage of gross domestic product
  #"WHS6_140",            # Number of community health workers
  #"WHS6_150",            # Community health workers density (per 10 000 population)
  #"WHS9_CS",             # Cellular subscribers (per 100 population)
  #"SI.DST.FRST.20",       # Income share held by lowest 20%
  "SI.POV.DDAY",          # Poverty headcount ratio at $1.90 a day (2011 PPP) (% of population)
  #"SM.POP.NETM",         # Net Migration
  "NY.GDP.PCAP.CD",       # GDP per capita (current US$)
  #"NY.GDP.PCAP.CN",      # GDP per capita (current LCU)
  #"NY.GNP.MKTP.CD",       # GNI (current US$)
  "BAR.NOED.15UP.ZS",    # Barro-Lee: Percentage of population age 15+ with no education
  #"BAR.NOED.15UP.FE.ZS", # Barro-Lee: Percentage of female population age 15+ with no education
  #"BAR.POP.15UP",        # Barro-Lee: Population in thousands, age 15+, total
  #"BAR.SCHL.1519",       # Barro-Lee: Average years of total schooling, age 15-19, total
  "OpioidsDeaths"
]

def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)

pairplot_pd = input_df.select(['Country'] + list(map(lambda c: f"`{c}`", selected_indicators))).toPandas().fillna(0)
g = sns.pairplot(pairplot_pd, hue='Country', palette='Paired')
g.map_upper(hide_current_axis)
display()

# COMMAND ----------

# MAGIC %md
# MAGIC In the scatterplot, points are years and countries. Data from different countries are colored differently. Note that the USA (dark blue) stands out easily on many dimensions, particularly in the last row: opioid-related deaths.
# MAGIC 
# MAGIC Many key features aren't particularly correlated. A few are, like "GDP per capita (current US$)" vs "Per capita total expenditure on health at average exchange rate (US$)"; naturally nations with more economic production per capita spend more on health care. Again, the US stands out for spending relatively _more_ per capita than would be expected from GDP.
# MAGIC 
# MAGIC To make further sense of this, it's necessary to prepare the data for a machine learning model that can attempt to relate these many input features to the desired outcome, life expectancy.

# COMMAND ----------

# MAGIC %fs rm --recurse=true /Users/sajith.appukuttan@databricks.com/life_expectancy/featurized

# COMMAND ----------

# DBTITLE 1,Simple One-Hot Encode Country Column
from pyspark.sql.functions import col

# Simple one-hot encoding
countries = sorted(map(lambda r: r['Country'], input_df.select("Country").distinct().collect()))

with_countries_df = input_df
for country in countries:
  with_countries_df = with_countries_df.withColumn(f"Country_{country}", col("Country") == country)
  
with_countries_df = with_countries_df.drop("Country")
with_countries_df.write.format("delta").save("/Users/sajith.appukuttan@databricks.com/life_expectancy/featurized")

# COMMAND ----------

# MAGIC %sql
# MAGIC USE westca_life_expectancy;
# MAGIC CREATE TABLE IF NOT EXISTS featurized USING DELTA LOCATION '/Users/sajith.appukuttan@databricks.com/life_expectancy/featurized';

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Now, a 'gold' table of featurized data is available for data scientists and modelers to use for the modeling.
# MAGIC 
# MAGIC ### Modeling
# MAGIC 
# MAGIC From here, data scientists may work in other frameworks like `pandas` to handle this small data set. In fact, those familiar with `pandas` can even manipulate and query data using Spark using the same API, via `koalas`, and then return to `pandas` DataFrames using the same API.

# COMMAND ----------

import databricks.koalas as ks

input_ks = spark.read.table("westca_life_expectancy.featurized").to_koalas()
input_ks = input_ks[input_ks['Year'] <= 2016]

# Train/test split on <= 2014 vs 2015-2016
input_ks_train = input_ks[input_ks['Year'] <= 2014]
input_ks_test = input_ks[input_ks['Year'] > 2014]

X_ks_train = input_ks_train.drop('WHOSIS_000001', axis=1)
y_ks_train = input_ks_train['WHOSIS_000001']
X_ks_test = input_ks_test.drop('WHOSIS_000001', axis=1)
y_ks_test = input_ks_test['WHOSIS_000001']

X = input_ks.drop('WHOSIS_000001', axis=1).to_pandas()
y = input_ks['WHOSIS_000001'].to_pandas()
X_train = X_ks_train.to_pandas()
X_test =  X_ks_test.to_pandas()
y_train = y_ks_train.to_pandas()
y_test =  y_ks_test.to_pandas()

# COMMAND ----------

# MAGIC %md
# MAGIC The data set is actually quite small -- 255 rows by about 1000 columns -- and consumes barely 2MB of memory. It's trivial to fit a model to this data with standard packages like `scikit-learn` or `xgboost`. However each of these models requires tuning, and needs building of 100 or more models to find the best combination.
# MAGIC 
# MAGIC In Databricks, the tool `hyperopt` can be use to build these models on a Spark cluster in parallel. The results are logged automatically to `mlflow`.

# COMMAND ----------

from math import exp
import xgboost as xgb
from hyperopt import fmin, hp, tpe, SparkTrials, STATUS_OK
import mlflow

def params_to_xgb(params):
  return {
    'objective':        'reg:squarederror',
    'eval_metric':      'rmse',
    'max_depth':        int(params['max_depth']),
    'learning_rate':    exp(params['log_learning_rate']), # exp() here because hyperparams are in log space
    'reg_alpha':        exp(params['log_reg_alpha']),
    'reg_lambda':       exp(params['log_reg_lambda']),
    'gamma':            exp(params['log_gamma']),
    'min_child_weight': exp(params['log_min_child_weight']),
    'importance_type':  'total_gain',
    'seed':             0
  }

def train_model(params):
  train = xgb.DMatrix(data=X_train, label=y_train)
  test = xgb.DMatrix(data=X_test, label=y_test)
  booster = xgb.train(params=params_to_xgb(params), dtrain=train, num_boost_round=1000,\
                      evals=[(test, "test")], early_stopping_rounds=50)
  mlflow.log_param('best_iteration', booster.attr('best_iteration'))
  return {'status': STATUS_OK, 'loss': booster.best_score, 'booster': booster.attributes()}

search_space = {
  'max_depth':            hp.quniform('max_depth', 20, 60, 1),
  # use uniform over loguniform here simply to make metrics show up better in mlflow comparison, in logspace
  'log_learning_rate':    hp.uniform('log_learning_rate', -3, 0),
  'log_reg_alpha':        hp.uniform('log_reg_alpha', -5, -1),
  'log_reg_lambda':       hp.uniform('log_reg_lambda', 1, 8),
  'log_gamma':            hp.uniform('log_gamma', -6, -1),
  'log_min_child_weight': hp.uniform('log_min_child_weight', -1, 3)
}

spark_trials = SparkTrials(parallelism=6)
best_params = fmin(fn=train_model, space=search_space, algo=tpe.suggest, max_evals=96, trials=spark_trials)

# COMMAND ----------

# MAGIC %md
# MAGIC The resulting runs and their hyperparameters can be visualized in the mlflow tracking server, via Databricks:
# MAGIC 
# MAGIC <img width="800" src="https://databricks-knowledge-repo-images.s3.us-east-2.amazonaws.com/ML/gartner_2020/hyperopt.png"/>
# MAGIC 
# MAGIC Root mean squared error was about 0.3 years, compared to life expectancies ranging from about 77 to 83 years. With a best set of hyperparameters chosen, the final model is re-fit and logged with `mlflow`, along with an analysis of feature importance from `shap`:

# COMMAND ----------

code_lookup_df = spark.read.table("westca_life_expectancy.descriptions")
code_lookup = dict([(r['Code'], r['Description']) for r in code_lookup_df.collect()])
display_cols = [code_lookup[c] if c in code_lookup else c for c in X.columns]

# COMMAND ----------

# DBTITLE 1,Refit Model with best parameters
import mlflow
import mlflow.xgboost
import shap
import matplotlib.pyplot as plt

plt.close()

with mlflow.start_run() as run:
  best_iteration = int(spark_trials.best_trial['result']['booster']['best_iteration'])
  booster = xgb.train(params=params_to_xgb(best_params), dtrain=xgb.DMatrix(data=X, label=y), num_boost_round=best_iteration)
  mlflow.log_params(best_params)
  mlflow.log_param('best_iteration', best_iteration)
  mlflow.xgboost.log_model(booster, "xgboost")

  shap_values = shap.TreeExplainer(booster).shap_values(X, y=y)
  shap.summary_plot(shap_values, X, feature_names=display_cols, plot_size=(14,6), max_display=10, show=False)
  plt.savefig("summary_plot.png", bbox_inches="tight")
  plt.close()
  mlflow.log_artifact("summary_plot.png")
  
  best_run = run.info

# COMMAND ----------

import mlflow
import mlflow.xgboost
import shap
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC This model can then be registered as the current candidate model for further evaluation in the Model Registry:
# MAGIC 
# MAGIC <img width="800" src="https://databricks-knowledge-repo-images.s3.us-east-2.amazonaws.com/ML/gartner_2020/registry.png"/>

# COMMAND ----------

import time

model_name = "westca_life_expectancy"
client = mlflow.tracking.MlflowClient()
try:
  client.create_registered_model(model_name)
except Exception as e:
  pass

model_version = client.create_model_version(model_name, f"{best_run.artifact_uri}/xgboost", best_run.run_id)

time.sleep(3) # Just to make sure it's had a second to register
client.update_model_version(model_name, model_version.version, stage="Staging", description="Current candidate")

# COMMAND ----------


