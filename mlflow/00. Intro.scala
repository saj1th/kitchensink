// Databricks notebook source
// DBTITLE 1,End-to-end Demo: Predicting Life Expectancy
// MAGIC %md
// MAGIC <img src="https://sa-iot.s3.ca-central-1.amazonaws.com/collateral/mlops.png" width="900">

// COMMAND ----------

// MAGIC %md 
// MAGIC ## 1. Data Preparation
// MAGIC 
// MAGIC 
// MAGIC Data comes from several sources.
// MAGIC 
// MAGIC - The WHO Health Indicators (primary data) for:
// MAGIC   - the USA: https://data.humdata.org/dataset/who-data-for-united-states-of-america
// MAGIC   - similarly for other developed nations: Australia, Denmark, Finland, France, Germany, Iceland, Italy, New Zealand, Norway, Portugal, Spain, Sweden, the UK
// MAGIC - The World Bank Health Indicators (supplementary data) for:
// MAGIC   - the USA: https://data.humdata.org/dataset/world-bank-combined-indicators-for-united-states
// MAGIC   - similarly for other developed nations
// MAGIC - Our World In Data (Drug Use)
// MAGIC   - https://ourworldindata.org/drug-use
// MAGIC   
// MAGIC ### Health Indicators primary data
// MAGIC 
// MAGIC The "health indicators" datasets are the primary data sets. They are CSV files, and are easily read by Spark. However, they don't have a consistent schema. Some contains extra "DATASOURCE" columns, which can be ignored.

// COMMAND ----------

// MAGIC %scala
// MAGIC import org.apache.spark.sql.DataFrame
// MAGIC 
// MAGIC val rawDataPath = "/mnt/databricks-datasets-private/ML/gartner_2020/"
// MAGIC 
// MAGIC def transformCSV(path: String): DataFrame = {
// MAGIC   // Second row contains odd "comment" lines that intefere with schema validation. Filter, then parse as CSV
// MAGIC   val withoutComment = spark.read.text(path).filter(!$"value".startsWith("#")).as[String]
// MAGIC   spark.read.option("inferSchema", true).option("header", true).csv(withoutComment)
// MAGIC }
// MAGIC 
// MAGIC // Some "Health Indicators" files have three extra "DATASOURCE" columns; ignore them
// MAGIC var rawHealthIndicators =
// MAGIC   transformCSV(rawDataPath + "health_indicators/format2").union(
// MAGIC   transformCSV(rawDataPath + "health_indicators/format1").drop("DATASOURCE (CODE)", "DATASOURCE (DISPLAY)", "DATASOURCE (URL)"))
// MAGIC 
// MAGIC display(rawHealthIndicators)

// COMMAND ----------

// MAGIC %fs rm --recurse=true /tmp/sajith/healthindicators

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC rawHealthIndicators
// MAGIC     .select("GHO (CODE)", "GHO (DISPLAY)")
// MAGIC     .distinct()
// MAGIC     .toDF("Code", "Description")
// MAGIC     .write.format("delta")
// MAGIC     .save("/tmp/sajith/healthindicators")
// MAGIC 
// MAGIC display(spark.read.format("delta").load("/tmp/sajith/healthindicators").orderBy("Code"))

// COMMAND ----------

// MAGIC %md
// MAGIC The data needs some basic normalization and filtering:
// MAGIC - Remove any variables that are effectively variations on life expectancy, as this is the variable to be explained
// MAGIC - Use published data only
// MAGIC - For now, use data for both sexes, not male/female individually
// MAGIC - Correctly parse the Value / Display Value, which are inconsistently available
// MAGIC - Flatten year ranges like "2012-2017" to individual years
// MAGIC - Keep only data from 2000 onwards
// MAGIC 
// MAGIC Finally, the data needs to be 'pivoted' to contain indicator values as columns.

// COMMAND ----------

// MAGIC %scala
// MAGIC import org.apache.spark.sql.functions.{explode, when, udf}
// MAGIC import org.apache.spark.sql.types.FloatType
// MAGIC 
// MAGIC // Can't use life expectancy at 60
// MAGIC rawHealthIndicators = rawHealthIndicators.filter($"GHO (CODE)" =!= "WHOSIS_000015")
// MAGIC 
// MAGIC // Keep just PUBLISHED data, not VOID
// MAGIC rawHealthIndicators = rawHealthIndicators.filter($"PUBLISHSTATE (CODE)" === "PUBLISHED").drop("PUBLISHSTATE (CODE)")
// MAGIC 
// MAGIC // Use stats for both sexes now, not male/female separately. It's either NULL or BTSX
// MAGIC rawHealthIndicators = rawHealthIndicators.filter(($"SEX (CODE)".isNull) || ($"SEX (CODE)" === "BTSX")).drop("SEX (CODE)")
// MAGIC 
// MAGIC // Use Numeric where available, otherwise Display Value, as value. Low/High/StdErr/StdDev are unevenly available, so drop
// MAGIC rawHealthIndicators = rawHealthIndicators.
// MAGIC   withColumn("Value", when($"Numeric".isNull, $"Display Value").otherwise($"Numeric"))
// MAGIC 
// MAGIC // Some "year" values are like 2012-2017. Explode to a value for each year in the range
// MAGIC val yearsToRangeUDF = udf { (s: String) =>
// MAGIC     if (s.contains("-")) {
// MAGIC       val Array(start, end) = s.split("-")
// MAGIC       (start.toInt to end.toInt).toArray
// MAGIC     } else {
// MAGIC       Array(s.toInt)
// MAGIC     }
// MAGIC   }
// MAGIC rawHealthIndicators = rawHealthIndicators.withColumn("Year", explode(yearsToRangeUDF($"YEAR (CODE)")))
// MAGIC 
// MAGIC // Rename columns, while dropping everything but Year, Country, GHO CODE, and Value
// MAGIC rawHealthIndicators = rawHealthIndicators.select(
// MAGIC   $"GHO (CODE)".alias("GHO"), $"Year", $"COUNTRY (CODE)".alias("Country"), $"Value".cast(FloatType))
// MAGIC 
// MAGIC // Keep only 2000-2018 at most
// MAGIC rawHealthIndicators = rawHealthIndicators.filter("Year >= 2000 AND Year <= 2018")
// MAGIC 
// MAGIC // avg() because some values will exist twice because of WORLDBANKINCOMEGROUP; value is virtually always the same
// MAGIC val healthIndicatorsDF = rawHealthIndicators.groupBy("Country", "Year").pivot("GHO").avg("Value")
// MAGIC healthIndicatorsDF.createOrReplaceTempView("healthIndicators")

// COMMAND ----------

// MAGIC %sql
// MAGIC SELECT * FROM healthIndicators ORDER BY Country, Year

// COMMAND ----------

// MAGIC %md
// MAGIC This data set contains life expectancy (`WHOSIS_000001`), so we can already compare life expectancy from 2000-2016 across countries. The USA is an outlier, it seems.

// COMMAND ----------

// MAGIC %sql
// MAGIC SELECT Year, Country, WHOSIS_000001 AS LifeExpectancy FROM healthIndicators WHERE Year <= 2016

// COMMAND ----------

// MAGIC %md
// MAGIC ### Indicators supplementary data
// MAGIC The data from the World Bank can likewise be normalized, filtered and analyzed.

// COMMAND ----------

// MAGIC %scala
// MAGIC var rawIndicators = transformCSV(rawDataPath + "indicators")
// MAGIC display(rawIndicators)

// COMMAND ----------

// MAGIC %md
// MAGIC This data set has 2,283 (!) features for these countries, by year. As above, many are highly correlated, and even redundant for comparative purposes. For example, figures in local currency are less useful for comparison than the ones expressed in US$. Likewise to limit the scale of the feature set, male/female figures reported separately are removed.

// COMMAND ----------

// MAGIC %scala
// MAGIC rawIndicators.select("Indicator Code", "Indicator Name").distinct().toDF("Code", "Description").
// MAGIC   write.format("delta").mode("append").save("/tmp/sajith/healthindicators")
// MAGIC display(spark.read.format("delta").load("/tmp/sajith/healthindicators").orderBy("Code"))

// COMMAND ----------

// MAGIC %scala
// MAGIC // Keep only 2000-2018 at most
// MAGIC rawIndicators = rawIndicators.filter("Year >= 2000 AND Year <= 2018")
// MAGIC 
// MAGIC // Can't use life expectancy from World Bank, or mortality rates or survival rates -- too closely related to life expectancy
// MAGIC rawIndicators = rawIndicators.
// MAGIC   filter(!$"Indicator Code".startsWith("SP.DYN.LE")).
// MAGIC   filter(!$"Indicator Code".startsWith("SP.DYN.AMRT")).filter(!$"Indicator Code".startsWith("SP.DYN.TO"))
// MAGIC 
// MAGIC // Don't use gender columns separately for now
// MAGIC rawIndicators = rawIndicators.
// MAGIC   filter(!$"Indicator Code".endsWith(".FE") && !$"Indicator Code".endsWith(".MA")).
// MAGIC   filter(!$"Indicator Code".contains(".FE.") && !$"Indicator Code".contains(".MA."))
// MAGIC 
// MAGIC // Don't use local currency variants
// MAGIC rawIndicators = rawIndicators.
// MAGIC   filter(!$"Indicator Code".endsWith(".CN") && !$"Indicator Code".endsWith(".KN")).
// MAGIC   filter(!$"Indicator Code".startsWith("PA.") && !$"Indicator Code".startsWith("PX."))
// MAGIC 
// MAGIC rawIndicators = rawIndicators.select($"Country ISO3".alias("Country"), $"Year", $"Indicator Code".alias("Indicator"), $"Value")
// MAGIC 
// MAGIC val indicatorsDF = rawIndicators.groupBy("Country", "Year").pivot("Indicator").avg("Value")
// MAGIC indicatorsDF.createOrReplaceTempView("indicators")

// COMMAND ----------

// MAGIC %sql
// MAGIC SELECT * FROM indicators

// COMMAND ----------

// MAGIC %md
// MAGIC ## Overdoses
// MAGIC 
// MAGIC One issue that comes to mind when thinking about life expectancy, given the unusual downward trend in life expectancy in the USA, is drug-related deaths. These have been a newsworthy issue for the USA for several years. Our World In Data provides drug overdose data by country, year, and type:

// COMMAND ----------

// MAGIC %scala
// MAGIC var rawOverdoses = spark.read.option("inferSchema", true).option("header", true).csv(rawDataPath + "overdoses_world")
// MAGIC display(rawOverdoses)

// COMMAND ----------

// MAGIC %scala
// MAGIC // Rename some columns for compatibility
// MAGIC rawOverdoses = rawOverdoses.drop("Entity").
// MAGIC   toDF("Country", "Year", "CocaineDeaths", "IllicitDrugDeaths", "OpioidsDeaths", "AlcoholDeaths", "OtherIllicitDeaths", "AmphetamineDeaths")
// MAGIC rawOverdoses = rawOverdoses.filter("Year >= 2000 AND Year <= 2018")
// MAGIC rawOverdoses.createOrReplaceTempView("rawOverdoses")

// COMMAND ----------

// MAGIC %md
// MAGIC These three data sets, having been filtered and normalized, can now be joined by country and year, to produce the raw input for further analysis. Join and write to a Delta table as a 'silver' table of cleaned data.

// COMMAND ----------

// MAGIC %fs rm --recurse=true /tmp/sajith/lifeinput

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC spark
// MAGIC     .sql("SELECT * FROM healthIndicators LEFT OUTER JOIN indicators USING (Country, Year) LEFT OUTER JOIN rawOverdoses USING (Country, Year)")
// MAGIC     .write.format("delta")
// MAGIC     .save("/tmp/sajith/lifeinput")

// COMMAND ----------

// MAGIC %sql
// MAGIC CREATE DATABASE IF NOT EXISTS westca_life_expectancy;
// MAGIC USE westca_life_expectancy;
// MAGIC 
// MAGIC CREATE TABLE IF NOT EXISTS input USING DELTA LOCATION '/tmp/sajith/lifeinput';
// MAGIC CREATE TABLE IF NOT EXISTS descriptions USING DELTA LOCATION '/tmp/sajith/healthindicators';
