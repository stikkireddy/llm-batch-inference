# Databricks notebook source
# post1 hot fix is needed if you want to use dbrx in distributed if you can get multiple 4xA100 gpus :-)
%pip install -U 'vllm==0.4.0.post1' 'openai'
dbutils.library.restartPython()

# COMMAND ----------

## magic arguments
from utils.uc_helpers import stage_registered_model

# this is used to shard the model across the gpus per node
number_of_gpus_per_node = 1
# total number of workers with gpus
number_of_worker_nodes = 4
# records per batch (make this from 1024-4096) changing wont oom but larger batches makes longer tasks
records_per_batch = 1024

total_num_gpus = number_of_gpus_per_node * number_of_worker_nodes

# copy the model from uc to dbfs or volumes to load into each node
model_staged_path = stage_registered_model(
  catalog="databricks_llama_2_models",
  schema="models",
  model_name="llama_2_7b_chat_hf",
  # model_name="llama_2_13b_chat_hf",
  # model_name="llama_2_70b_chat_hf",
  version=3,
  # change this local base path to some dbfs or volume location for distributed
  local_base_path="/dbfs/tmp/sri/registry",
  overwrite=False # if this is false it will use what ever is there existing
)

# COMMAND ----------

import sys

# the python virtual env executable that notebooks create
port = "8000"
spark.conf.set("vllm.python.executable", str(sys.executable))
# number of gpus per worker 
spark.conf.set("vllm.num_gpus", str(number_of_gpus_per_node))
# vllm port
spark.conf.set("vllm.port", port)

model_path = str(model_staged_path / "model")
tokenizer_path = str(model_staged_path / "components/tokenizer")

# path to model
spark.conf.set("vllm.model", model_path)
# path to tokenizer config
spark.conf.set("vllm.tokenizer", tokenizer_path)

# COMMAND ----------

# DBTITLE 1,key variables
print(f"Python exec: {spark.conf.get('vllm.python.executable')}")
print(f"vLLM Port: {spark.conf.get('vllm.port')}")
print(f"vLLM Num GPUs: {spark.conf.get('vllm.num_gpus')}")
print(f"vLLM Model Path: {spark.conf.get('vllm.model')}")
print(f"vLLM Tokenizer Path: {spark.conf.get('vllm.tokenizer')}")

# COMMAND ----------

# MAGIC %scala
# MAGIC  
# MAGIC import scala.concurrent.duration._
# MAGIC import sys.process._
# MAGIC import java.net.Socket
# MAGIC import scala.concurrent.duration._
# MAGIC import scala.util.Success
# MAGIC
# MAGIC val pythonExecutable: String = spark.conf.get("vllm.python.executable")
# MAGIC val numGpus: String = spark.conf.get("vllm.num_gpus")
# MAGIC val port: String = spark.conf.get("vllm.port")
# MAGIC val modelPath: String = spark.conf.get("vllm.model")
# MAGIC val tokenizerPath: String = spark.conf.get("vllm.tokenizer")
# MAGIC
# MAGIC def isPortInUse(port: Int): Boolean = {
# MAGIC     try {
# MAGIC       new Socket("0.0.0.0", port).close()
# MAGIC       true
# MAGIC     } catch {
# MAGIC       case _: Exception => false
# MAGIC     }
# MAGIC   }
# MAGIC
# MAGIC def retryWithBackoff(port: Int, maxAttempts: Int = 5, initialDelay: FiniteDuration = 1.second): Boolean = {
# MAGIC   var attempt = 1
# MAGIC   var delay = initialDelay
# MAGIC   var portInUse = isPortInUse(port)
# MAGIC
# MAGIC   while (!portInUse && attempt <= maxAttempts) {
# MAGIC     println(s"Port $port is not in use. Retrying in $delay... (Attempt $attempt/$maxAttempts)")
# MAGIC     Thread.sleep(delay.toMillis)
# MAGIC     portInUse = isPortInUse(port)
# MAGIC     delay *= 2
# MAGIC     attempt += 1
# MAGIC   }
# MAGIC
# MAGIC   if (portInUse) {
# MAGIC     println(s"Port $port is in use.")
# MAGIC     true
# MAGIC   } else {
# MAGIC     println("Reached max retry attempts. Giving up.")
# MAGIC     false
# MAGIC   }
# MAGIC }
# MAGIC
# MAGIC def retryWithFixedDelay(port: Int, maxAttempts: Int = 5, delay: FiniteDuration = 10.seconds): Boolean = {
# MAGIC   var attempt = 1
# MAGIC   var portInUse = isPortInUse(port)
# MAGIC
# MAGIC   while (!portInUse && attempt <= maxAttempts) {
# MAGIC     println(s"Port $port is not in use. Retrying in $delay... (Attempt $attempt/$maxAttempts)")
# MAGIC     Thread.sleep(delay.toMillis)
# MAGIC     portInUse = isPortInUse(port)
# MAGIC     attempt += 1
# MAGIC   }
# MAGIC
# MAGIC   if (portInUse) {
# MAGIC     println(s"Port $port is in use.")
# MAGIC     true
# MAGIC   } else {
# MAGIC     println("Reached max retry attempts. Giving up.")
# MAGIC     false
# MAGIC   }
# MAGIC }
# MAGIC
# MAGIC
# MAGIC def runBash(cmd: String): Unit = {
# MAGIC   val res = sc.runOnEachExecutor(() => {
# MAGIC     val cmdResult = Seq("bash", "-c", cmd).!!
# MAGIC     cmdResult
# MAGIC   }, 500.seconds)
# MAGIC
# MAGIC   res.foreach { case (index, output) =>
# MAGIC     println(s"Node: $index")
# MAGIC     output match {
# MAGIC       case Success(outputString) => println(outputString)
# MAGIC       case _ => println("Command execution failed")
# MAGIC     }
# MAGIC   }
# MAGIC }
# MAGIC
# MAGIC def runCommandAndPrintResults(): Unit = {
# MAGIC   runBash("tail output.log || echo 'File does not exist'")
# MAGIC }
# MAGIC
# MAGIC // convenience for listing python processes for debugging
# MAGIC def findAllPythonProcs(): Unit = {
# MAGIC   runBash("ps aux | grep python")
# MAGIC }
# MAGIC
# MAGIC // convenience for listing python processes for debugging
# MAGIC def findAllVEnvPythonProcs(): Unit = {
# MAGIC   runBash(s"ps aux | grep '$pythonExecutable'")
# MAGIC }
# MAGIC
# MAGIC def verifyvLLM(): Unit = {
# MAGIC   val res = sc.runOnEachExecutor(() => {
# MAGIC     retryWithFixedDelay(port.toInt, maxAttempts=30)
# MAGIC     // larger models like llama 70b mixtral and dbrx may take 900s to load
# MAGIC   }, 900.seconds)
# MAGIC   res.foreach { case (index, output) =>
# MAGIC     println(s"Node: $index")
# MAGIC     output match {
# MAGIC       case Success(outputString) => println("vLLM ready to connect")
# MAGIC       case _ => println("Failed to wait for vLLM")
# MAGIC     }
# MAGIC   }
# MAGIC }
# MAGIC
# MAGIC def runvLLM(): Unit = {
# MAGIC   runBash("rm -f output.log || echo 'file doesnt exist'")
# MAGIC   runBash(s"nohup $pythonExecutable -m vllm.entrypoints.openai.api_server --model '$modelPath' --tokenizer '$tokenizerPath' --tensor-parallel-size $numGpus --port $port > output.log 2>&1 &")
# MAGIC   println("Started vLLM now waiting for server to be in a connectable state!")
# MAGIC   verifyvLLM
# MAGIC }
# MAGIC
# MAGIC def killvLLM(): Unit = {
# MAGIC   runBash(s"kill -9 $$(lsof -ti :$port) || echo 'Nothing running on that port: $port'")
# MAGIC   runBash(s"$pythonExecutable -c 'import ray; ray.shutdown()' && echo 'Started attempt to shutdown vllm'")
# MAGIC }

# COMMAND ----------

# MAGIC %scala
# MAGIC runvLLM

# COMMAND ----------

import pandas as pd
import random

word_list = [
    "apple", "banana", "carrot", "dog", "elephant", 
    "frog", "guitar", "house", "ice cream", "jacket", 
    "kite", "lion", "moon", "nest", "orange", 
    "piano", "queen", "rabbit", "snake", "tree"
]

def generate_random_questions(word_list, n):
    questions = []
    for _ in range(n):
        question_type = random.choice(["Spell", "Define"])  # Choose between "Spell" or "Define"
        if question_type == "Spell":
            word = random.choice(word_list)
            questions.append(f"[INST] <<SYS>>Answer to my quesiton in one sentence! Here is an example: \n Question: What day is today? Answer: Today is monday! <</SYS>> How do you spell '{word}'?[/INST]")
        elif question_type == "Define":
            word = random.choice(word_list)
            questions.append(f"[INST] <<SYS>> Answer to my quesiton in one sentence! Here is an example: \nQuestion: What day is today? Answer: Today is monday!<</SYS>> What is the definition of '{word}'?[/INST]")
    return questions
  
questions = generate_random_questions(word_list=word_list, n=100000)

df = spark.createDataFrame(pd.DataFrame({"text": questions}))
display(df)

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType
import pandas as pd
import math

from openai import OpenAI

@pandas_udf(StringType())
def generate_text(s: pd.Series) -> pd.Series:
    client = OpenAI(
        base_url=f"http://0.0.0.0:{port}/v1",
        api_key="something"
    )
    completions = client.completions.create(
      model=model_path, 
      prompt=s.tolist(),
      max_tokens=500 
    )
    choices = pd.Series([choice.text for choice in completions.choices])
    # Your custom logic here, using s as the input Pandas Series
    return choices

# repartition if you want your batches to be larger, ideally you want batches of size 128 or higher so that most of the work is done on the gpu in terms of packing
# records_per_batch is just a magic number that seems to correspond well to vllm
# partitions is max(total_num_gpus, total_records/(records_per_batch))

total_records = df.count()
num_partitions = max(total_num_gpus, math.ceil(total_records / records_per_batch))
print(f"Sharding into {num_partitions} partitions")
df_with_generations = df.repartition(num_partitions).withColumn("generated", generate_text(df["text"]))
df_with_generations.cache()
# modify this to change to a target table
df_with_generations.write.format("delta").mode("overwrite").saveAsTable("main.default.test_generations")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * FROM main.default.test_generations

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   sum(length(text))/4 as input_tokens, 
# MAGIC   sum(length(generated))/4 as output_tokens, 
# MAGIC   count(1) as total_ct 
# MAGIC FROM main.default.test_generations

# COMMAND ----------

# MAGIC %scala
# MAGIC // you may need to wait a little bit, this notebook is tailored for batch so may take a while for gpu memory to be released
# MAGIC killvLLM

# COMMAND ----------


