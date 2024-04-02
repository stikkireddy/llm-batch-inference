from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, StructType, StructField
import torch
import os
import socket
import subprocess
import re
from dataclasses import dataclass, asdict
from typing import List, Optional
import math

node_details_schema = StructType([
    StructField("num_gpus", IntegerType(), False),
    StructField("num_cpus", IntegerType(), False),
])

@dataclass
class DatabricksNode:
  num_cpus: int
  num_gpus: int
  is_worker: bool # false is driver node

@dataclass
class RayClusterConfig:
  min_worker_nodes: int
  max_worker_nodes: int
  num_cpus_per_node: int
  num_gpus_per_node: int
  num_cpus_head_node: int
  num_gpus_head_node: int
  single_node_ray: bool = False

  def to_dict(self):
    return asdict(self)

def get_node_details(_):
    num_gpus = torch.cuda.device_count()
    num_cpus = os.cpu_count()
    return [(num_gpus, num_cpus)]

def is_not_autoscaling_cluster(spark) -> bool:
  return spark.conf.get("spark.databricks.clusterUsageTags.clusterScalingType") == "fixed_size"

def get_worker_count(spark) -> int:
  return int(spark.conf.get("spark.databricks.clusterUsageTags.clusterWorkers", "0"))

def _get_worker_node_configs(
  spark, 
  num_workers: int,
  ignore_auto_scaling = False
) -> List[DatabricksNode]:
  num_partitions = num_workers  # This should be number of workers
  df = spark.range(num_partitions).repartition(num_partitions)

  # Apply the function to each partition and get distinct results
  resources_rdd = df.rdd.mapPartitions(get_node_details)
  resources_df = spark.createDataFrame(resources_rdd, node_details_schema) \
    .distinct()

  # Show the distinct IP addresses and their resources
  worker_pdf = resources_df.toPandas()
  assert worker_pdf.shape[0] == 1, "The resources for workers should not be heterogeneous"
  nodes = []
  for n in worker_pdf.to_dict("records"):
    nodes.append(DatabricksNode(
      num_cpus=n["num_cpus"],
      num_gpus=n["num_gpus"],
      is_worker=True
    ))
  return nodes

def get_node_configs(
  spark, 
  num_workers: int,
  ignore_auto_scaling = False
) -> List[DatabricksNode]:
  if ignore_auto_scaling is False:
    assert is_not_autoscaling_cluster(spark) == True, "Cluster should be not autoscaling"

  nodes = []
  
  if num_workers > 0:
    nodes += _get_worker_node_configs(spark, num_workers, ignore_autoscaling)
  
  driver_details = get_node_details(None)
  nodes.append(
    DatabricksNode(
      num_cpus=driver_details[0][1],
      num_gpus=driver_details[0][0],
      is_worker=False
    )
  )
  return nodes

def get_ray_gpu_cluster_config(
  nodes: List[DatabricksNode], 
  num_workers: int,
  percent_cpu_head_node=0.8,
) -> RayClusterConfig:
  data = {
    "min_worker_nodes": None,
    "max_worker_nodes": None,
    "num_cpus_per_node": None,
    "num_gpus_per_node": None,
    "num_cpus_head_node": None,
    "num_gpus_head_node": None,
  }

  if percent_cpu_head_node >= 1:
    raise Exception("percent_cpu_head_node must be less than 1")

  if num_workers == 0:
    data["single_node_ray"] = True
  
  if num_workers > 0:
    num_gpus_per_node = set()
    num_cpus_per_node = set()
    for n in nodes:
      if n.is_worker is True:
        num_gpus_per_node.add(n.num_gpus)
        num_cpus_per_node.add(n.num_cpus)
    if len(num_gpus_per_node) != 1:
      raise Exception("Invalid worker config you need all your workers to have the same gpus")
    # TODO: maybe in future we can support heterogenious
    if len(num_cpus_per_node) != 1:
      raise Exception("Invalid worker config you need all your workers to have the same number of cpus")

    num_gpus_per_node = list(num_gpus_per_node)
    num_cpus_per_node = list(num_cpus_per_node)
    data["min_worker_nodes"] = num_workers
    data["max_worker_nodes"] = num_workers
    data["num_cpus_per_node"] = num_cpus_per_node[0]
    if num_gpus_per_node[0] != 0:
      data["num_gpus_per_node"] = num_gpus_per_node[0]

  for n in nodes:
    if n.is_worker is False:
      if n.num_gpus != 0:
        data["num_gpus_head_node"] = n.num_gpus
      data["num_cpus_head_node"] = math.floor(n.num_cpus * percent_cpu_head_node)

  return RayClusterConfig(**data)


def get_default_ray_cluster_config(
  spark, 
  ignore_auto_scaling = False,
  percent_cpu_head_node=0.8
) -> Optional[RayClusterConfig]:
  num_workers = get_worker_count(spark)
  # if num_workers == 0:
  #   return None
  nodes = get_node_configs(spark, num_workers, ignore_auto_scaling)
  return get_ray_gpu_cluster_config(
    nodes,
    num_workers, 
    percent_cpu_head_node
  )

@dataclass
class ProxySettings:
    proxy_url: str
    port: str
    url_base_path: str
    url_base_path_no_port: Optional[str] = None

    def get_proxy_url(self, ensure_ends_with_slash=False):
        """
        For certain apps that use relative paths like "assets/index-*.js" we need to ensure that the url ends
        with a slash.
        """
        if ensure_ends_with_slash is True:
            return self.proxy_url.rstrip("/") + "/"
        return self.proxy_url
      
def _get_cloud(spark) -> str:
    workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
    # TODO: support gcp
    if workspace_url.endswith("azuredatabricks.net"):
        return "azure"
    if workspace_url.endswith("gcp.databricks.com"):
        return "gcp"
    return "aws"

def _get_cluster_id(spark) -> str:
    return spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
  
def _get_org_id(spark) -> str:
    return spark.conf.get("spark.databricks.clusterUsageTags.clusterOwnerOrgId")

def _get_cloud_proxy_settings(cloud: str, org_id: str, cluster_id: str, port: int) -> ProxySettings:
    cloud_norm = cloud.lower()
    if cloud_norm not in ["aws", "azure"]:
        raise Exception("only supported in aws or azure")
    prefix_url_settings = {
        "aws": "https://dbc-dp-",
        "azure": "https://adb-dp-",
    }
    suffix_url_settings = {
        "azure": "azuredatabricks.net",
    }
    if cloud_norm == "aws":
        suffix = remove_lowest_subdomain_from_host(ctx.host)
        suffix_url_settings["aws"] = suffix

    org_shard = ""
    # org_shard doesnt need a suffix of "." for dnsname its handled in building the url
    # only azure right now does dns sharding
    # gcp will need this
    if cloud_norm == "azure":
        org_shard_id = int(org_id) % 20
        org_shard = f".{org_shard_id}"

    url_base_path_no_port = f"/driver-proxy/o/{org_id}/{cluster_id}"
    url_base_path = f"{url_base_path_no_port}/{port}/"
    return ProxySettings(
        proxy_url=f"{prefix_url_settings[cloud_norm]}{org_id}{org_shard}.{suffix_url_settings[cloud_norm]}{url_base_path}",
        port=str(port),
        url_base_path=url_base_path,
        url_base_path_no_port=url_base_path_no_port
    )

def get_proxy_url(spark, port: int) -> ProxySettings:
    cloud = _get_cloud(spark)
    cluster_id = _get_cluster_id(spark)
    org_id = _get_org_id(spark)
    return _get_cloud_proxy_settings(cloud, org_id, cluster_id, port)
