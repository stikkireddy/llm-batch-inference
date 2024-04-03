# Databricks notebook source
# MAGIC %pip install 'vllm==0.3.3' 'ray[default]>=2.3.0'
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,get_default_ray_configs
from utils.ray_helpers import get_default_ray_cluster_config, get_proxy_url
ray_config = get_default_ray_cluster_config(spark)
ray_config.to_dict()

# COMMAND ----------

from utils.uc_helpers import stage_registered_model

model_staged_path = stage_registered_model(
  catalog="databricks_llama_2_models",
  schema="models",
  # model_name="llama_2_7b_chat_hf",
  model_name="llama_2_70b_chat_hf",
  version=3,
  local_base_path="/local_disk0/models",
  overwrite=False # if this is false it will use what ever is there existing
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Make fake data and table to simulate source table

# COMMAND ----------

import random
import pandas as pd
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
  
questions = generate_random_questions(word_list=word_list, n=100)

df = spark.createDataFrame(pd.DataFrame({"text": questions}))
display(df)

# COMMAND ----------

model_path = str(model_staged_path / "model")
tokenizer_path = str(model_staged_path / "components/tokenizer")
model_path, tokenizer_path

# COMMAND ----------

from vllm import LLM, SamplingParams
from typing import Iterator
from pyspark.sql.functions import pandas_udf

model = LLM(model=model_path,
            tokenizer=tokenizer_path,
            tensor_parallel_size=ray_config.num_gpus_head_node)

# COMMAND ----------

params = SamplingParams( temperature = 0.1 , top_p = 0.6 , max_tokens=150)

def generate_in_batch(batch: pd.Series) -> pd.Series:
    responses = []
    outputs = model.generate(batch.tolist(), params)
    for output in outputs:
      responses.append(' '.join([o.text for o in output.outputs]))
    return pd.Series(responses)

def chunk_dataframe(df, chunk_size=4096):
    return [df[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]

# Process each chunk
def process_chunk(chunk, column_name: str, new_column_name: str):
    chunk['column_to_process'] = generate_in_batch(chunk[column_name])
    return chunk

pdf = df.toPandas()
df_chunks = chunk_dataframe(pdf)
processed_chunks = [process_chunk(chunk, column_name="text", new_column_name="generated_text") for chunk in df_chunks]
processed_pdf = pd.concat(processed_chunks, ignore_index=True)
processed_df = spark.createDataFrame(processed_pdf)

# COMMAND ----------

display(processed_df)

# COMMAND ----------

import ray
ray.shutdown()

# COMMAND ----------


