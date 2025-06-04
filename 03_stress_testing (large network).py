# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/).

# COMMAND ----------

catalog = "mlops_pj"
schema = "supply_chain_stress_test"


# COMMAND ----------

# MAGIC %pip install -r ./requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import mlflow
import random
import numpy as np
import pandas as pd
import cloudpickle
import pyomo.environ as pyo
import scripts.utils as utils
import ray
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster, MAX_NUM_WORKER_NODES

# Generate a synthetic 3-tier network dataset for optimization 
dataset = utils.generate_data(N1=200, N2=500, N3=1000)

# Assign a random ttr to each disrupted node
random.seed(777)
disrupted_nodes = {node: random.randint(1, 10) for node in dataset['tier2'] + dataset['tier3']}

# COMMAND ----------

import requests
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
host_name = ctx.tags().get("browserHostName").get()
host_token = ctx.apiToken().get()
cluster_id = ctx.tags().get("clusterId").get()

response = requests.get(
    f'https://{host_name}/api/2.1/clusters/get?cluster_id={cluster_id}',
    headers={'Authorization': f'Bearer {host_token}'}
  ).json()

if "autoscale" in response:
  min_node = response['autoscale']["min_workers"]
  max_node = response['autoscale']["max_workers"]

# COMMAND ----------

Start Ray Cluster

import os
import ray
import numpy as np 
from mlflow.utils.databricks_utils import get_databricks_env_vars
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster, MAX_NUM_WORKER_NODES


# Cluster cleanup
restart = True
if restart is True:
  try:
    shutdown_ray_cluster()
  except:
    pass

  try:
    ray.shutdown()
  except:
    pass

# Set configs based on your cluster size
num_cpu_cores_per_worker = 4 # total cpu to use in each worker node 
num_cpus_head_node = 4 # Cores to use in driver node (total_cores - 2)

# Set databricks credentials as env vars
mlflow_dbrx_creds = get_databricks_env_vars("databricks")
os.environ["DATABRICKS_HOST"] = mlflow_dbrx_creds['DATABRICKS_HOST']
os.environ["DATABRICKS_TOKEN"] = mlflow_dbrx_creds['DATABRICKS_TOKEN']

ray_conf = setup_ray_cluster(
  min_worker_nodes=min_node,
  max_worker_nodes=max_node,
  num_cpus_head_node= num_cpus_head_node,
  num_cpus_per_node=num_cpu_cores_per_worker,
  num_gpus_head_node=0,
  num_gpus_worker_node=0
)
os.environ['RAY_ADDRESS'] = ray_conf[0]

# COMMAND ----------

from pyomo.common.timing import TicTocTimer
for name in ["highs"]:
    print(name, pyo.SolverFactory(name).available())

# COMMAND ----------

df = pd.DataFrame.from_dict(disrupted_nodes, orient='index', columns=['ttr']).reset_index(names='node')
df = ray.data.from_pandas(df)

# COMMAND ----------

class Solver:
    """
    Class which whill read audio files and convert them to numpy arrays
    """
    
    def __init__(self,data = dataset):
        self.data = dataset


    def __call__(self, row):
        """Run the Pyomo model for a single disrupted scenario."""
        disrupted = [row['node']]
        solver = utils.build_and_solve_multi_tier_ttr(self.data, disrupted, row['ttr'])
        row['termination_condition'] = str(solver.iloc[0]['termination_condition'])
        row['profit_loss'] = solver.iloc[0]['profit_loss']
        return row




# COMMAND ----------

Solver()(df.take(1)[0])

# COMMAND ----------

df_new =df.repartition(300).map(Solver,
       num_cpus = 1,
       concurrency=(3,20))
pandas_df = df_new.to_pandas()

# COMMAND ----------

pandas_df

# COMMAND ----------

spark.createDataFrame(pandas_df).write.mode("overwrite").saveAsTable(f"ryuta.supply_chain_stress_test.result")

# COMMAND ----------

highest_risk_nodes = pandas_df.sort_values(by="profit_loss", ascending=False)[0:10]
highest_risk_nodes

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | pyomo | An object-oriented algebraic modeling language in Python for structured optimization problems | BSD | https://pypi.org/project/pyomo/
# MAGIC | coinor-cbc | COIN-OR Branch-and-Cut solver | Eclipse Public License - v 2.0 | https://github.com/coin-or/Cbc
