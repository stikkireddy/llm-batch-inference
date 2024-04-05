# Databricks notebook source
# MAGIC %pip install -U 'vllm==0.4.0.post1'
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,get_default_ray_configs
num_gpus = 4

# COMMAND ----------

from utils.uc_helpers import stage_registered_model

# this will roughly take 10-15 minutes
model_staged_path = stage_registered_model(
  catalog="databricks_dbrx_models",
  schema="models",
  model_name="dbrx_instruct",
  version=1,
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

chatml_format = """
<|im_start|>system 
{system}
<|im_end|> 
<|im_start|>user 
{user}
<|im_end|> 
<|im_start|>assistant 
"""

def generate_random_questions(word_list, n):
    questions = []
    for _ in range(n):
        question_type = random.choice(["Spell", "Define"])  # Choose between "Spell" or "Define"
        if question_type == "Spell":
            word = random.choice(word_list)
            questions.append(chatml_format.format(
                # this is the default system prompt from DBRX tokenizer
                system="You are DBRX, created by Databricks. You were last updated in December 2023. You answer questions based on information available up to that point.\nYOU PROVIDE SHORT RESPONSES TO SHORT QUESTIONS OR STATEMENTS, but provide thorough responses to more complex and open-ended questions.\nYou assist with various tasks, from writing to coding (using markdown for code blocks — remember to use ``` with code, JSON, and tables).\n(You do not have real-time data access or code execution capabilities. You avoid stereotyping and provide balanced perspectives on controversial topics. You do not provide song lyrics, poems, or news articles and do not divulge details of your training data.)\nThis is your system prompt, guiding your responses. Do not reference it, just respond to the user. If you find yourself talking about this message, stop. You should be responding appropriately and usually that means not mentioning this.\nYOU DO NOT MENTION ANY OF THIS INFORMATION ABOUT YOURSELF UNLESS THE INFORMATION IS DIRECTLY PERTINENT TO THE USER'S QUERY.",
                user=f"Answer to my quesiton in one sentence! Be extremely brief and succinct. Do not be too wordy! Here is an example: \n Question: How do you spell today? Answer: T-O-D-A-Y!\nHow do you spell '{word}'?"
            ))
        elif question_type == "Define":
            word = random.choice(word_list)
            questions.append(chatml_format.format(
                # this is the default system prompt from DBRX tokenizer
                system="You are DBRX, created by Databricks. You were last updated in December 2023. You answer questions based on information available up to that point.\nYOU PROVIDE SHORT RESPONSES TO SHORT QUESTIONS OR STATEMENTS, but provide thorough responses to more complex and open-ended questions.\nYou assist with various tasks, from writing to coding (using markdown for code blocks — remember to use ``` with code, JSON, and tables).\n(You do not have real-time data access or code execution capabilities. You avoid stereotyping and provide balanced perspectives on controversial topics. You do not provide song lyrics, poems, or news articles and do not divulge details of your training data.)\nThis is your system prompt, guiding your responses. Do not reference it, just respond to the user. If you find yourself talking about this message, stop. You should be responding appropriately and usually that means not mentioning this.\nYOU DO NOT MENTION ANY OF THIS INFORMATION ABOUT YOURSELF UNLESS THE INFORMATION IS DIRECTLY PERTINENT TO THE USER'S QUERY.",
                user=f"Answer to my quesiton in one sentence! Be extremely brief and succinct. Do not be too wordy! Here is an example: \n Question: What day is today? Answer: Today is monday!\nWhat is the definition of '{word}'?"
            ))
    return questions
  
questions = generate_random_questions(word_list=word_list, n=100)

df = spark.createDataFrame(pd.DataFrame({"text": questions}))
display(df)

# COMMAND ----------

model_path = str(model_staged_path / "model")
tokenizer_path = str(model_staged_path / "components/tokenizer")
model_path, tokenizer_path

# COMMAND ----------

import os
# get your huggingface token and place it here. It is required for some tokenizer scripts to be run for dbrx from huggingface
os.environ["HF_TOKEN"] = "<huggingface token>"

# COMMAND ----------

from vllm import LLM, SamplingParams
from typing import Iterator
from pyspark.sql.functions import pandas_udf

# this will roughly take ~4 minutes
model = LLM(model=model_path,
            tokenizer=tokenizer_path,
            trust_remote_code=True,
            tensor_parallel_size=num_gpus)

# COMMAND ----------

params = SamplingParams( temperature = 0.1 , top_p = 0.6 , max_tokens=250)

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


