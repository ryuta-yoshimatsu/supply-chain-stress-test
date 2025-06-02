# Databricks notebook source
catalog = "ryuta"
schema = "supply_chain_stress_test"
volume = "init_script"

# COMMAND ----------

# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/).

# COMMAND ----------

#%sh apt-get update && apt-get install -y coinor-cbc

# COMMAND ----------

# MAGIC %md
# MAGIC install_cbc.sh script must be specified as an init script for the cluster.

# COMMAND ----------

# overwrite if the file already exists
dbutils.fs.put(f"/Volumes/{catalog}/{schema}/{volume}/install_cbc.sh", open("./scripts/install_cbc.sh").read(), True)

# COMMAND ----------

# MAGIC %md
# MAGIC Restart cluster with the init script

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
max_worker_nodes = 8        # total cpu to use in each worker node (total_cores - 1 to leave one core for spark)
num_cpus_worker_node = 8     # Cores to use in driver node (total_cores - 1)

# If the cluster has four workers with 8 CPUs each as an example
setup_ray_cluster(
  max_worker_nodes=8, 
  num_cpus_worker_node=8, 
  num_gpus_worker_node=0,
  )

# Pass any custom configuration to ray.init
ray.init(ignore_reinit_error=True)

# COMMAND ----------

@ray.remote
def solve_for_scenario(nodes, ttr, data):
    """Run the Pyomo model for a single disrupted scenario."""
    disrupted = [nodes]
    return utils.build_and_solve_multi_tier_ttr(data, disrupted, ttr)

# Launch a Ray task per scenario
futures = [
    solve_for_scenario.remote(nodes, ttr, dataset)
    for nodes, ttr in disrupted_nodes.items()
]

# Gather results back to the driver
objectives = ray.get(futures)

# Optionally combine into one Pandas / Spark DataFrame
combined_df = pd.concat(objectives, ignore_index=True)

spark.createDataFrame(combined_df).write.mode("overwrite").saveAsTable(f"ryuta.supply_chain_stress_test.result")

# COMMAND ----------

highest_risk_nodes = combined_df.sort_values(by="objective_value", ascending=False)[0:10]
highest_risk_nodes

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | pyomo | An object-oriented algebraic modeling language in Python for structured optimization problems | BSD | https://pypi.org/project/pyomo/
# MAGIC | coinor-cbc | COIN-OR Branch-and-Cut solver | Eclipse Public License - v 2.0 | https://github.com/coin-or/Cbc
