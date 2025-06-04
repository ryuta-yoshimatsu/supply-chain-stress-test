# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/).
# MAGIC 
# MAGIC ## Overview
# MAGIC This notebook demonstrates how to perform large-scale supply chain stress testing using distributed computation with Ray on Databricks. It covers data generation, Ray cluster setup, distributed optimization, and result analysis.

# COMMAND ----------

catalog = "mlops_pj"
schema = "supply_chain_stress_test"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster Configuration and Performance Note
# MAGIC This notebook was tested on the following Databricks cluster configuration:
# MAGIC - **Databricks Runtime Version:** 16.4 LTS ML (includes Apache Spark 3.5.2, Scala 2.12)
# MAGIC - **Photon Acceleration:** Enabled (Photon boosts Apache Spark workloads; not all ML workloads will see an improvement)
# MAGIC - **Worker Type:** Standard_D4ds_v5 (16 GB Memory, 4 Cores)
# MAGIC - **Number of Workers:** 4
# MAGIC - **Driver Type:** Standard_DS4_v2 (28 GB Memory, 8 Cores)
# MAGIC > **Note:** Performance may vary depending on the cluster size, node types, and workload characteristics. For large-scale distributed computation, ensure sufficient resources are allocated to avoid bottlenecks.

# COMMAND ----------


 # Install Required Packages
# The following cell installs all required Python packages as specified in the requirements.txt file and restarts the Python environment to ensure all dependencies are loaded.

# MAGIC %pip install -r ./requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries and Generate Synthetic Data
# MAGIC Here, we import all necessary libraries and generate a synthetic 3-tier supply chain network dataset for optimization. We also assign random time-to-recovery (ttr) values to each disrupted node.

import os
import random
import pandas as pd
import pyomo.environ as pyo
from pyomo.common.timing import TicTocTimer
import scripts.utils as utils

# Generate a synthetic 3-tier network dataset for optimization 
dataset = utils.generate_data(N1=200, N2=500, N3=1000)

# Assign a random ttr (time-to-recovery) to each disrupted node
random.seed(777)
disrupted_nodes = {node: random.randint(1, 10) for node in dataset['tier2'] + dataset['tier3']}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieve Databricks Cluster Information
# MAGIC This function retrieves the minimum and maximum number of worker nodes from the Databricks cluster context using the REST API. This information is used to configure the Ray cluster for distributed computation.

# Databricks-only: get cluster context and min/max nodes
def get_min_max_nodes():
    try:
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
            return response['autoscale']["min_workers"], response['autoscale']["max_workers"]
    except Exception as e:
        print(f"Warning: Could not fetch min/max nodes from Databricks context: {e}")
    return 1, 1  # fallback for local/testing

min_node, max_node = get_min_max_nodes()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ray Cluster Initialization
# MAGIC Ray is a distributed execution framework that enables scalable parallel computation. Here, we initialize a Ray cluster on Databricks using the `setup_ray_cluster` utility. The number of worker nodes and CPU cores per node are set based on the Databricks cluster configuration. Environment variables for Databricks authentication are also set for Ray workers.

import ray
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster

# Cluster cleanup: shut down any existing Ray cluster and Ray context to ensure a clean start
restart = True
if restart is True:
    try:
        shutdown_ray_cluster()
    except Exception:
        pass
    try:
        ray.shutdown()
    except Exception:
        pass

# Set configs based on your cluster size
num_cpu_cores_per_worker = 4 # total cpu to use in each worker node 
num_cpus_head_node = 4 # Cores to use in driver node (total_cores - 2)

# Set databricks credentials as env vars for Ray workers
try:
    from mlflow.utils.databricks_utils import get_databricks_env_vars
    mlflow_dbrx_creds = get_databricks_env_vars("databricks")
    os.environ["DATABRICKS_HOST"] = mlflow_dbrx_creds['DATABRICKS_HOST']
    os.environ["DATABRICKS_TOKEN"] = mlflow_dbrx_creds['DATABRICKS_TOKEN']
except Exception as e:
    print(f"Warning: Could not set Databricks env vars: {e}")

# Start the Ray cluster with the specified configuration
ray_conf = setup_ray_cluster(
    min_worker_nodes=min_node,
    max_worker_nodes=max_node,
    num_cpus_head_node=num_cpus_head_node,
    num_cpus_per_node=num_cpu_cores_per_worker,
    num_gpus_head_node=0,
    num_gpus_worker_node=0
)
os.environ['RAY_ADDRESS'] = ray_conf[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check Pyomo Solver Availability
# MAGIC This cell checks the availability of the 'highs' solver in Pyomo, which is used for optimization. If unavailable, ensure the solver is installed in your environment.

for name in ["highs"]:
    print(name, pyo.SolverFactory(name).available())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Data for Distributed Computation
# MAGIC Convert the disrupted nodes dictionary to a pandas DataFrame, then to a Ray Dataset for distributed processing.

df = pd.DataFrame.from_dict(disrupted_nodes, orient='index', columns=['ttr']).reset_index(names='node')
df = ray.data.from_pandas(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the Solver Class
# MAGIC The `Solver` class encapsulates the logic for running the Pyomo model for each disrupted scenario. It is designed to be used with Ray's distributed map operation.

class Solver:
    """
    Callable class to run the Pyomo model for a single disrupted scenario.
    """
    
    def __init__(self, data=dataset):
        self.data = dataset

    def __call__(self, row):
        """Run the Pyomo model for a single disrupted scenario."""
        disrupted = [row['node']]
        # Call the utility function to build and solve the optimization model
        solver = utils.build_and_solve_multi_tier_ttr(self.data, disrupted, row['ttr'])
        row['termination_condition'] = str(solver.iloc[0]['termination_condition'])
        row['profit_loss'] = solver.iloc[0]['profit_loss']
        return row

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the Solver on a Single Row
# MAGIC This cell tests the `Solver` class on a single row to ensure correctness before distributed execution.

Solver()(df.take(1)[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Distributed Computation with Ray Data API
# MAGIC The following cell demonstrates distributed computation using Ray's Data API:
# MAGIC - The Ray Dataset is repartitioned into 300 partitions to increase parallelism and optimize resource utilization across the cluster.
# MAGIC - The `map` function applies the `Solver` class to each partition in parallel, with each task using 1 CPU and a concurrency window of (3, 20).
# MAGIC - The results are collected as a pandas DataFrame for further analysis.

df_new = df.repartition(300).map(Solver,
       num_cpus=1,
       concurrency=(3,20))
pandas_df = df_new.to_pandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Results to Delta Table
# MAGIC The results are saved to a Delta table for persistent storage and further analysis using Spark SQL. This step is Databricks-specific.

# Databricks-only: save to Delta table
try:
    spark.createDataFrame(pandas_df).write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.result")
except Exception as e:
    print(f"Warning: Could not save to Delta table: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyze Highest Risk Nodes
# MAGIC The top 10 nodes with the highest profit loss are identified for further investigation.

highest_risk_nodes = pandas_df.sort_values(by="profit_loss", ascending=False)[0:10]
highest_risk_nodes

# COMMAND ----------

# MAGIC %md
# MAGIC ## License and Third-Party Libraries
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | pyomo | An object-oriented algebraic modeling language in Python for structured optimization problems | BSD | https://pypi.org/project/pyomo/
# MAGIC | highspy | Linear optimization solver (HiGHS) | MIT | https://github.com/ERGO-Code/HiGHS
