# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC Use kaggle dataset: 
# MAGIC * https://www.kaggle.com/datasets/vikashrajluhaniwal/fashion-images/data
# MAGIC
# MAGIC put the data set in volumes
# MAGIC

# COMMAND ----------

# %sh
# cd ~
# rm -rf data
# cp /Volumes/srituc/fashion/images-volume/data.zip ./
# unzip data.zip
# mkdir /Volumes/srituc/fashion/images-volume/data/
# cp -R ~/data/* /Volumes/srituc/fashion/images-volume/data/

# COMMAND ----------

# %sh
# cd ~
# ls /Volumes/srituc/fashion/images-volume/data/

# COMMAND ----------

from pyspark.sql.functions import concat, lit
data = spark.read.format("csv").option("header", "true").load("/Volumes/srituc/fashion/images-volume/data/fashion.csv")
data_with_files = data.withColumn("file_path", concat(lit("/Volumes/srituc/fashion/images-volume/data/"), 
                              "Category", lit("/"), 
                              "Gender", lit("/"), 
                              lit("Images/images_with_product_ids/"), "Image"))
data_with_files.write.format("delta").mode("overwrite").saveAsTable("srituc.default.demodata")

# COMMAND ----------


