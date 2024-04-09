# Databricks notebook source

audio_metadata_table = "main.default.test_audio_files"

audio_diarization_table = "main.default.test_audio_files_diarization"

audio_transcription_table = "main.default.test_audio_files_transcription"

audio_final_silver_table = "main.default.test_audio_final_table"

# COMMAND ----------

def post_process_segments_and_transcripts(new_segments, transcript, group_by_speaker) -> list:
    import numpy as np
    import sys

    # get the end timestamps for each chunk from the ASR output
    end_timestamps = np.array(
        [chunk["timestamp"][-1] if chunk["timestamp"][-1] is not None else sys.float_info.max for chunk in transcript])
    segmented_preds = []

    # align the diarizer timestamps and the ASR timestamps
    for segment in new_segments:
        # get the diarizer end timestamp
        end_time = segment["segment"]["end"]
        # find the ASR end timestamp that is closest to the diarizer's end timestamp and cut the transcript to here
        upto_idx = np.argmin(np.abs(end_timestamps - end_time))

        if group_by_speaker:
            segmented_preds.append(
                {
                    "speaker": segment["speaker"],
                    "text": "".join(
                        [chunk["text"] for chunk in transcript[: upto_idx + 1]]
                    ),
                    "timestamp": (
                        transcript[0]["timestamp"][0],
                        transcript[upto_idx]["timestamp"][1],
                    ),
                }
            )
        else:
            for i in range(upto_idx + 1):
                segmented_preds.append({"speaker": segment["speaker"], **transcript[i]})

        # crop the transcripts and timestamp lists according to the latest timestamp (for faster argmin)
        transcript = transcript[upto_idx + 1:]
        end_timestamps = end_timestamps[upto_idx + 1:]

        if len(end_timestamps) == 0:
            break 

    return segmented_preds

# COMMAND ----------

def post_process_transcript_diarization(transcript: str, diarization: str):
    import json
    chunks = json.loads(transcript)
    segments = json.loads(diarization)
    return post_process_segments_and_transcripts(segments, chunks, group_by_speaker=False)

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
merge_transcript_diarization = udf(post_process_transcript_diarization, StringType())

# COMMAND ----------

spark.sql(f"""
          SELECT src.file_ids, src.file_name, src.file_path, t.transcription, d.diarization
          FROM {audio_metadata_table} src
          LEFT JOIN {audio_transcription_table} t
          ON t.file_ids = src.file_ids
          LEFT JOIN {audio_diarization_table} d
          ON d.file_ids = src.file_ids
          """)\
            .withColumn("mergedTranscriptionDiarization", 
                        merge_transcript_diarization('transcription', 'diarization'))\
            .write\
            .format("delta"
            .mode("overwrite")\
            .saveAsTable(audio_final_silver_table)
