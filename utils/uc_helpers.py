from pathlib import Path
import mlflow
import shutil
import os

def copy_folder_recursively(source: str, destination: str):
    # Create the destination directory if it does not exist
    os.makedirs(destination, exist_ok=True)
    
    for item in os.listdir(source):
        source_item = os.path.join(source, item)
        destination_item = os.path.join(destination, item)
        
        if os.path.isdir(source_item):
            # If the item is a directory, call the function recursively
            copy_folder_recursively(source_item, destination_item)
        else:
            # If the item is a file, copy it to the destination
            shutil.copy2(source_item, destination_item)

def stage_registered_model(
  *,
  catalog: str, 
  schema: str, 
  model_name: str,
  version: int,
  local_base_path: str,
  overwrite: bool = False
) -> Path:
  local_base_path = Path(local_base_path)
  local_model_path = local_base_path / catalog / schema / model_name / str(version)   
  if overwrite is False and local_model_path.exists() and any(local_model_path.iterdir()):
    print(f"Model path {local_model_path} already exists and contains content.")
    return local_model_path
  if overwrite is True and local_model_path.exists():
    shutil.rmtree(str(local_model_path))
    print(f"Existing model path {local_model_path} removed.")

  mlflow.set_registry_uri("databricks-uc")
  model_uri = f"models:/{catalog}.{schema}.{model_name}/{version}"
  if str(local_base_path).startswith("/dbfs") or str(local_base_path).startswith("/Volumes"):
    local_disk_path = "/local_disk0" + str(local_model_path)
    if overwrite is True and Path(local_disk_path).exists():
      shutil.rmtree(local_disk_path)
      print(f"Existing model path in local disk {local_disk_path} removed.")
    mlflow.artifacts.download_artifacts(model_uri, dst_path=local_disk_path)
    print(f"Model artifacts downloaded to local disk {local_disk_path}")
    copy_folder_recursively(source=str(local_disk_path), destination=str(local_model_path))
    print(f"Model artifacts copied to dbfs or volume {local_model_path}")
  else:
    mlflow.artifacts.download_artifacts(model_uri, dst_path=local_model_path)
  print(f"Model from uc registry saved in: {local_model_path}")
  return local_model_path
  