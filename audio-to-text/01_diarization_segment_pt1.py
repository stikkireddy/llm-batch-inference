# Databricks notebook source
# ensure torch is 2.0.1 so flash_attn is used and make sure all of torch libs have same version of cuda
%pip install optimum accelerate 'pyannote-audio==3.0.1'
%pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
%pip install onnxruntime-gpu
dbutils.library.restartPython()

# COMMAND ----------

# constants
num_workers = 2

# you should only have one gpu per worker
# we want to process at most 4096 files per model load into memory
max_batch_size = 4096

audio_metadata_table = "main.default.test_audio_files"

audio_diarization_table = "main.default.test_audio_files_diarization"

hf_token = "<hf token for diarization with terms agreed>"

# COMMAND ----------

import math
audio_table = spark.table(audio_metadata_table)
total_audio_files = audio_table.count()
print(f"Processing {total_audio_files} audio files")

num_batches = math.ceil(total_audio_files/max_batch_size)
# this will gaurantee all nodes being utilized
num_partitions = max(num_workers, num_batches)
audio_table = audio_table.repartition(num_partitions).cache()
audio_table.display()

# COMMAND ----------


def preprocess_inputs(inputs):
    import requests
    import torch
    from transformers.pipelines.audio_utils import ffmpeg_read
    from torchaudio import functional as F
    import numpy as np

    if isinstance(inputs, str):
        if inputs.startswith("http://") or inputs.startswith("https://"):
            # We need to actually check for a real protocol, otherwise it's impossible to use a local file
            # like http_huggingface_co.png
            inputs = requests.get(inputs).content
        else:
            with open(inputs, "rb") as f:
                inputs = f.read()

    if isinstance(inputs, bytes):
        inputs = ffmpeg_read(inputs, 16000)

    if isinstance(inputs, dict):
        # Accepting `"array"` which is the key defined in `datasets` for better integration
        if not ("sampling_rate" in inputs and ("raw" in inputs or "array" in inputs)):
            raise ValueError(
                "When passing a dictionary to ASRDiarizePipeline, the dict needs to contain a "
                '"raw" key containing the numpy array representing the audio and a "sampling_rate" key, '
                "containing the sampling_rate associated with that array"
            )

        _inputs = inputs.pop("raw", None)
        if _inputs is None:
            # Remove path which will not be used from `datasets`.
            inputs.pop("path", None)
            _inputs = inputs.pop("array", None)
        in_sampling_rate = inputs.pop("sampling_rate")
        inputs = _inputs
        if in_sampling_rate != 16000:
            inputs = F.resample(
                torch.from_numpy(inputs), in_sampling_rate, 16000
            ).numpy()

    if not isinstance(inputs, np.ndarray):
        raise ValueError(f"We expect a numpy ndarray as input, got `{type(inputs)}`")
    if len(inputs.shape) != 1:
        raise ValueError(
            "We expect a single channel audio input for ASRDiarizePipeline"
        )

    # diarization model expects float32 torch tensor of shape `(channels, seq_len)`
    diarizer_inputs = torch.from_numpy(inputs).float()
    diarizer_inputs = diarizer_inputs.unsqueeze(0)

    return inputs, diarizer_inputs


def diarize_audio(diarizer_inputs, diarization_pipeline, num_speakers, min_speakers, max_speakers):
    diarization = diarization_pipeline(
        {"waveform": diarizer_inputs, "sample_rate": 16000},
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    print("finished diarization")
    segments = []
    for segment, track, label in diarization.itertracks(yield_label=True):
        print(f"finished track: {track}, label: {label}")
        segments.append(
            {
                "segment": {"start": segment.start, "end": segment.end},
                "track": track,
                "label": label,
            }
        )

    # diarizer output may contain consecutive segments from the same speaker (e.g. {(0 -> 1, speaker_1), (1 -> 1.5, speaker_1), ...})
    # we combine these segments to give overall timestamps for each speaker's turn (e.g. {(0 -> 1.5, speaker_1), ...})
    new_segments = []
    prev_segment = cur_segment = segments[0]

    for i in range(1, len(segments)):
        cur_segment = segments[i]

        # check if we have changed speaker ("label")
        if cur_segment["label"] != prev_segment["label"] and i < len(segments):
            # add the start/end times for the super-segment to the new list
            new_segments.append(
                {
                    "segment": {
                        "start": prev_segment["segment"]["start"],
                        "end": cur_segment["segment"]["start"],
                    },
                    "speaker": prev_segment["label"],
                }
            )
            prev_segment = segments[i]

    # add the last segment(s) if there was no speaker change
    new_segments.append(
        {
            "segment": {
                "start": prev_segment["segment"]["start"],
                "end": cur_segment["segment"]["end"],
            },
            "speaker": prev_segment["label"],
        }
    )

    return new_segments

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
import pandas as pd
from typing import Iterator

@pandas_udf("string")
def diarize(path_iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
    import torch
    import json
    from pyannote.audio import Pipeline

    diarization_pipeline = Pipeline.from_pretrained(
        checkpoint_path="pyannote/speaker-diarization-3.0",
        use_auth_token=hf_token
    )
    diarization_pipeline.to(
        torch.device(f"cuda:0")
    )
    
    def add_suffix(path: str):
        inputs, diarizer_inputs = preprocess_inputs(inputs=path)
        segments = diarize_audio(diarizer_inputs, diarization_pipeline, None, 1, 6)
        return json.dumps(segments)

    for path_chunk in path_iterator:
        yield path_chunk.apply(add_suffix)

audio_table.withColumn("diarization", diarize("file_path")).write.mode("overwrite").saveAsTable(audio_diarization_table)

# COMMAND ----------


