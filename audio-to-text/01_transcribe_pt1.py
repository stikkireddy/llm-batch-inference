# Databricks notebook source
# ensure torch is 2.0.1 so flash_attn is used and make sure all of torch libs have same version of cuda
%pip install optimum accelerate
%pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
dbutils.library.restartPython()

# COMMAND ----------

# constants
num_workers = 2

# you should only have one gpu per worker
# we want to process at most 4096 files per model load into memory
# make this smaller if you want less tasks
max_batch_size = 4096

audio_metadata_table = "main.default.test_audio_files"

audio_transcription_table = "main.default.test_audio_files_transcription"

# COMMAND ----------

import math
audio_table = spark.table(audio_metadata_table)
total_audio_files = audio_table.count()

num_batches = math.ceil(total_audio_files/max_batch_size)
# this will gaurantee all nodes being utilized
num_partitions = max(num_workers, num_batches)
audio_table = audio_table.repartition(num_partitions).cache()
audio_table.display()

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
import pandas as pd
from typing import Iterator

@pandas_udf("string")
def transcribe(path_iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
    import torch
    from transformers import pipeline
    from transformers.utils import is_flash_attn_2_available
    import json
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3", # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
        # model="distil-whisper/large-v2",
        torch_dtype=torch.float16,
        device="cuda:0", # or mps for Mac devices
        # Comment this out if you plan on using T4 gpus, flash attention is only supported on ampere (A10/A100) architecture
        model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
    )

    def add_suffix(path: str):
        outputs = pipe(
            path,
            chunk_length_s=30,
            batch_size=24,
            return_timestamps=True,
        )
        return json.dumps(outputs["chunks"])

    for path_chunk in path_iterator:
        yield path_chunk.apply(add_suffix)

audio_table.withColumn("transcription", transcribe("file_path")).write.mode("overwrite").saveAsTable(audio_transcription_table)

# COMMAND ----------


