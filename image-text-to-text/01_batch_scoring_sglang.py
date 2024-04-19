# Databricks notebook source
# side effect of some issues with them using gemma model with vllm feature that is deprecated in vllm 0.4.0.post1
%pip install 'git+https://github.com/sgl-project/sglang.git@c93293c#egg=sglang[all]&subdirectory=python'
%pip install -U filelock
dbutils.library.restartPython()

# COMMAND ----------

model_path = "liuhaotian/llava-v1.6-vicuna-7b"
tokenizer_path = "llava-hf/llava-1.5-7b-hf"
num_images_per_spark_task = 32

# COMMAND ----------

import sys

# the python virtual env executable that notebooks create
port = "30000"
number_of_gpus_per_node = int(spark.conf.get("spark.executor.resource.gpu.amount"))
num_workers = int(spark.conf.get("spark.databricks.clusterUsageTags.clusterWorkers"))

spark.conf.set("sglang.python.executable", str(sys.executable))
# number of gpus per worker 
spark.conf.set("sglang.num_gpus", str(number_of_gpus_per_node))
# sglang port
spark.conf.set("sglang.port", port)

# path to model
spark.conf.set("sglang.model", model_path)
# path to tokenizer config
spark.conf.set("sglang.tokenizer", tokenizer_path)

# COMMAND ----------

print(f"Python exec: {spark.conf.get('sglang.python.executable')}")
print(f"num workers: {num_workers}")
print(f"sglang Port: {spark.conf.get('sglang.port')}")
print(f"sglang Num GPUs: {spark.conf.get('sglang.num_gpus')}")
print(f"sglang Model Path: {spark.conf.get('sglang.model')}")
print(f"sglang Tokenizer Path: {spark.conf.get('sglang.tokenizer')}")

# COMMAND ----------

# MAGIC %scala
# MAGIC  
# MAGIC import scala.concurrent.duration._
# MAGIC import sys.process._
# MAGIC import java.net.Socket
# MAGIC import scala.concurrent.duration._
# MAGIC import scala.util.Success
# MAGIC
# MAGIC val pythonExecutable: String = spark.conf.get("sglang.python.executable")
# MAGIC val numGpus: String = spark.conf.get("sglang.num_gpus")
# MAGIC val port: String = spark.conf.get("sglang.port")
# MAGIC val modelPath: String = spark.conf.get("sglang.model")
# MAGIC val tokenizerPath: String = spark.conf.get("sglang.tokenizer")
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
# MAGIC   println(s"Attempting to run command: $cmd")
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
# MAGIC def verifySglang(): Unit = {
# MAGIC   val res = sc.runOnEachExecutor(() => {
# MAGIC     retryWithFixedDelay(port.toInt, maxAttempts=30)
# MAGIC     // larger models like llama 70b mixtral and dbrx may take 900s to load
# MAGIC   }, 900.seconds)
# MAGIC   res.foreach { case (index, output) =>
# MAGIC     println(s"Node: $index")
# MAGIC     output match {
# MAGIC       case Success(outputString) => println("sglang ready to connect")
# MAGIC       case _ => println("Failed to wait for sglang")
# MAGIC     }
# MAGIC   }
# MAGIC }
# MAGIC
# MAGIC def runSglang(): Unit = {
# MAGIC   runBash("rm -f output.log || echo 'file doesnt exist'")
# MAGIC   runBash(s"nohup $pythonExecutable -m sglang.launch_server --model-path '$modelPath' --tokenizer-path '$tokenizerPath' --tp-size $numGpus --port $port --host 0.0.0.0 > output.log 2>&1 &")
# MAGIC   println("Started sglang server now waiting for server to be in a connectable state!")
# MAGIC   verifySglang
# MAGIC }
# MAGIC
# MAGIC def killSglang(): Unit = {
# MAGIC   runBash(s"kill -9 $$(lsof -ti :$port) || echo 'Nothing running on that port: $port'")
# MAGIC   runBash(s"ps aux | grep $port | grep python | awk '{print $$2}' | xargs sudo kill -9")
# MAGIC   runBash(s"$pythonExecutable -c 'import ray; ray.shutdown()' && echo 'Started attempt to shutdown sglang'")
# MAGIC }

# COMMAND ----------

# MAGIC %scala
# MAGIC runSglang

# COMMAND ----------

path = "/Volumes/srituc/fashion/images-volume/data/Apparel/Girls/Images/images_with_product_ids/42419.jpg"
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image = mpimg.imread(path)
plt.imshow(image)
plt.axis('off')  # Turn off axis
plt.show()


# COMMAND ----------

from typing import Iterator
import pandas as pd

from pyspark.sql.functions import pandas_udf

spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "512")

@pandas_udf("string")
def describe(path_iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
    import sglang as sgl
    from sglang import RuntimeEndpoint
    
    sgl.set_default_backend(RuntimeEndpoint(f"http://0.0.0.0:{port}"))

    @sgl.function
    def image_qa(s, image_path, question):
        s += sgl.user(sgl.image(image_path) + question)
        s += sgl.assistant(sgl.gen("answer"))

    def batch_vision(paths):
        states = image_qa.run_batch(
            [{"image_path": p, "question": "What product is this an image of? Describe in detail!"} for p in paths],
            max_new_tokens=128,
            progress_bar=True,
        )
        return [s["answer"] for s in states]

    for path_chunk in path_iterator:
        paths = path_chunk.tolist()
        resp = batch_vision(paths)
        yield pd.Series(resp)


# COMMAND ----------

import math
data = spark.sql("SELECT * FROM srituc.default.demodata")
total_ct = data.count()
print(f"Total number of images: {total_ct}")
num_partitions = max(num_workers, math.ceil(total_ct/num_images_per_spark_task))
print(f"Total number of partitions: {num_partitions}")
data.repartition(num_partitions)\
  .withColumn("caption", describe("file_path"))\
  .write\
  .format("delta")\
  .mode("overwrite")\
  .saveAsTable("srituc.default.demodata_described")

# COMMAND ----------

# MAGIC %scala
# MAGIC killSglang

# COMMAND ----------


