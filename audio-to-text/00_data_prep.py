# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Make sure that the data is in a table

# COMMAND ----------

# if you want to use youtube audio
# %pip install 'moviepy' 'pytube'
# dbutils.library.restartPython()

# COMMAND ----------

# import logging
# import os
# from pathlib import Path
# from typing import Optional

# from moviepy.editor import AudioFileClip
# from pytube import YouTube

# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# def download_and_convert_to_mp3(url: str,
#                                 output_path: str = "output",
#                                 filename: str = "test") -> Optional[str]:
#     try:
#         yt = YouTube(url)
#         audio_stream = yt.streams.filter(only_audio=True).first()

#         if audio_stream is None:
#             logging.warning("No audio streams found")
#             return None

#         Path(output_path).mkdir(parents=True, exist_ok=True)

#         mp3_file_path = os.path.join(output_path, filename + ".mp3")
#         logging.info(f"Downloading started... {mp3_file_path}")

#         downloaded_file_path = audio_stream.download(output_path)

#         audio_clip = AudioFileClip(downloaded_file_path)
#         audio_clip.write_audiofile(mp3_file_path, codec="libmp3lame", verbose=False, logger=None)
#         audio_clip.close()

#         if Path(downloaded_file_path).suffix != ".mp3":
#             os.remove(downloaded_file_path)

#         logging.info(f"Download and conversion successful. File saved at: {mp3_file_path}")
#         return str(mp3_file_path)

#     except Exception as e:
#         logging.error(f"An error occurred: {e}")
#         return None

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /dbfs/sri/tmp/output/audio

# COMMAND ----------

# Define the data
# do not save raw audio in tables, it will blow up your table if you dont partition properly
# make sure these are in volumes
data = [
    ("/dbfs/sri/tmp/output/audio/assistant.mp3", "databricks_assistant.mp3", 0),
    ("/dbfs/sri/tmp/output/audio/test.mp3", "dais_summit_keynote.mp3", 1),
]

# Define the schema
schema = ["file_path", "file_name", "file_ids"]

# Create a DataFrame
df = spark.createDataFrame(data, schema)

# COMMAND ----------

df.write.format("delta").mode("overwrite").saveAsTable("main.default.test_audio_transcriptions")

# COMMAND ----------


