from pathlib import Path
import mlflow

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
  mlflow.set_registry_uri("databricks-uc")
  model_uri = f"models:/{catalog}.{schema}.{model_name}/{version}"
  mlflow.artifacts.download_artifacts(model_uri, dst_path=local_model_path)
  print(f"Model from uc registry saved in: {model_root_path}")
  return local_model_path
  