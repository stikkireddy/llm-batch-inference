# Databricks notebook source
import sys

# the python virtual env executable that notebooks create
spark.conf.set("vllm.python.executable", str(sys.executable))
# the python virtual env executable that notebooks create
port = "8000"
# vllm port
spark.conf.set("vllm.port", port)

# COMMAND ----------

# MAGIC %scala
# MAGIC  
# MAGIC import scala.concurrent.duration._
# MAGIC import sys.process._
# MAGIC import java.net.Socket
# MAGIC import scala.util.Success
# MAGIC import scala.io.Source
# MAGIC import scala.util.{Using}
# MAGIC
# MAGIC val pythonExecutable: String = spark.conf.get("vllm.python.executable")
# MAGIC
# MAGIC def fetchMetrics(): Unit = {
# MAGIC   val res = sc.runOnEachExecutor(() => {
# MAGIC     Using(Source.fromURL("http://localhost:8000/metrics"))(_.mkString)
# MAGIC   }, 500.seconds)
# MAGIC
# MAGIC   res.foreach { case (index, output) =>
# MAGIC     println(s"Node: $index")
# MAGIC     output match {
# MAGIC       case scala.util.Success(metrics) => println(metrics)  // Raw JSON string
# MAGIC       case scala.util.Failure(exception) => println(s"An error occurred: ${exception.getMessage}")
# MAGIC     }
# MAGIC   }
# MAGIC }
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
# MAGIC def showvLLMLogs(): Unit = {
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
# MAGIC def nvidiaSMI(): Unit = {
# MAGIC   runBash("nvidia-smi")
# MAGIC }
# MAGIC

# COMMAND ----------

# MAGIC %scala
# MAGIC nvidiaSMI

# COMMAND ----------

# MAGIC %scala
# MAGIC fetchMetrics

# COMMAND ----------

# MAGIC %scala 
# MAGIC showvLLMLogs

# COMMAND ----------

# MAGIC %scala
# MAGIC findAllPythonProcs

# COMMAND ----------


