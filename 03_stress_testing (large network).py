# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/).

# COMMAND ----------

# MAGIC %md
# MAGIC # Stress-Test Large Networks and Analyze the Results
# MAGIC
# MAGIC This notebook demonstrates how to perform stress testing on a large supply chain network. While the previous notebooks focused on a small network (35 nodes), modern supply chains often consist of tens of thousands of suppliers and sub-suppliers. To run comprehensive stress tests on such large-scale networks, a scalable setup is essential. We leverage distributed computation using Ray on Databricks to achieve this. This notebook covers network generation (1,700 nodes), Ray cluster setup, distributed optimization, and result analysis.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster Configuration
# MAGIC This notebook was tested on the following Databricks cluster configuration:
# MAGIC - **Databricks Runtime Version:** 16.4 LTS ML (includes Apache Spark 3.5.2, Scala 2.12)
# MAGIC - **Photon Acceleration:** Disabled (Photon boosts Apache Spark workloads; not all ML workloads will see an improvement)
# MAGIC - **Driver Type:** Standard_DS4_v2 (28 GB Memory, 8 Cores)
# MAGIC - **Worker Type:** Standard_E4d_v4 (32 GB Memory, 4 Cores, **memory optimized**)
# MAGIC - **Number of Workers:** 4
# MAGIC > **Note:** Performance may vary depending on the cluster size, node types, and workload characteristics. For large-scale distributed computation, ensure sufficient resources are allocated to avoid bottlenecks.

# COMMAND ----------

# DBTITLE 1,Install requirements
# MAGIC %pip install -r ./requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import modules
import os
import random
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.common.timing import TicTocTimer
import scripts.utils as utils

# COMMAND ----------

# MAGIC %md
# MAGIC We will write the results of our optimization to Delta tables. Update the `catalog` and `schema` names below to specify where you want the results to be saved.

# COMMAND ----------

catalog = "supply_chain_stress_test"
schema = "results"

# Make sure that the catalog and the schema exist
_ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}") 
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}") 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Data
# MAGIC We generate a synthetic 3-tier supply chain network dataset for optimization. We also assign random time-to-recovery (ttr) values to each disrupted node.

# COMMAND ----------

# Generate a synthetic 3-tier network dataset for optimization 
dataset = utils.generate_data(N1=200, N2=500, N3=1000) # DO NOT CHANGE!

# Assign a random ttr (time-to-recovery) to each disrupted node
random.seed(777) # DO NOT CHANGE!
disrupted_nodes = {node: random.randint(1, 10) for node in dataset['tier2'] + dataset['tier3']}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieve Databricks Cluster Information
# MAGIC This function retrieves the minimum and maximum number of worker nodes from the Databricks cluster context using the REST API. This information is used to configure the Ray cluster for distributed computation.

# COMMAND ----------

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
    return 1, response['num_workers']  # fallback for local/testing

min_node, max_node = get_min_max_nodes()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ray Cluster Initialization
# MAGIC Ray is a distributed execution framework that enables scalable parallel computation. Here, we initialize a Ray cluster on Databricks using the `setup_ray_cluster` utility. The number of worker nodes and CPU cores per node are set based on the Databricks cluster configuration. Environment variables for Databricks authentication are also set for Ray workers.

# COMMAND ----------

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
num_cpus_head_node = 4 # Cores to use in driver node (total_cores - 4)

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
# MAGIC ## Prepare Data for Distributed Computation
# MAGIC Here, we convert the disrupted nodes dictionary we defined above to a pandas DataFrame, then to a Ray Dataset for distributed processing.

# COMMAND ----------

df = pd.DataFrame.from_dict(disrupted_nodes, orient='index', columns=['ttr']).reset_index(names='node')
df = ray.data.from_pandas(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Multi-Tier TTR Model
# MAGIC
# MAGIC ### Define the Solver Class
# MAGIC The `TTRSolver` class encapsulates the logic for running the `utils.build_and_solve_multi_tier_ttr` function for each disrupted scenario. It is designed to be used with Ray's [distributed map](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map.html) operation.

# COMMAND ----------

class TTRSolver:
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
        row['lost_profit'] = solver.iloc[0]['lost_profit']
        return row

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the Solver on a Single Row
# MAGIC This cell tests the `TTRSolver` class on a single row to ensure correctness before distributed execution.

# COMMAND ----------

TTRSolver()(df.take(1)[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distributed Computation with Ray Data API
# MAGIC The following cell demonstrates distributed computation using [Ray's Data API](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map.html):
# MAGIC - The Ray Dataset is repartitioned into 300 partitions to increase parallelism and optimize resource utilization across the cluster.
# MAGIC - The `map` function applies the `TTRSolver` class to each partition in parallel, with each task using 1 CPU and a concurrency window of (4, 20) you can adjust the concurreny based on your cluster setup.
# MAGIC - The results are collected as a pandas DataFrame for further analysis.
# MAGIC
# MAGIC **The following cell will run in about a minute. On a single-node cluster without distributed computation, the same calculation would take approximately an hour to complete.**

# COMMAND ----------

df_ttr = df.repartition(300).map(TTRSolver,
       num_cpus=1,
       concurrency=(4,20))
pandas_df_ttr = df_ttr.to_pandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Highest Risk Nodes
# MAGIC Let's examine the top 10 nodes with the highest lost profit.

# COMMAND ----------

highest_risk_nodes = pandas_df_ttr.sort_values(by="lost_profit", ascending=False)[0:10]
display(highest_risk_nodes)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Total Spend vs. Lost Profit
# MAGIC
# MAGIC Let's imagine we have a global budget for risk mitigation in our supply chain, and each node receives some portion of that budget. The purpose of this analysis is to identify which nodes are over- or under-invested based on the risk exposure we previously computed. For simplicity, we randomly assign the total spend on risk mitigation measures for each node.
# MAGIC

# COMMAND ----------

np.random.seed(42) # DO NOT CHANGE!
pandas_df_ttr["total_spend"] = np.abs(np.random.normal(loc=0, scale=50, size=len(pandas_df_ttr))).astype(int)

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.scatter(pandas_df_ttr["lost_profit"], pandas_df_ttr["total_spend"])
plt.xlabel("Lost Profit")
plt.ylabel("Total Spend")
plt.title("Total Spend vs Lost Profit")

rect_1 = patches.Rectangle((1900, -5), 3100, 110, linewidth=2, edgecolor='red', facecolor='none')
rect_2 = patches.Rectangle((-50, 100), 1000, 100, linewidth=2, edgecolor='red', facecolor='none')
plt.gca().add_patch(rect_1)
plt.gca().add_patch(rect_2)

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The scatter plot above shows total spend on supplier sites for risk mitigation on the vertical axis and lost profit on the horizontal axis. This visualization helps quickly identify areas where risk mitigation investment is undersized relative to the potential impact of a node failure (right box), as well as areas where investment may be oversized relative to the risk (left box and potentially all nodes with zero lost profit). Both regions highlight opportunities to reassess and optimize the investment strategy—either to strengthen network resiliency or reduce unnecessary costs.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Multi-Tier TTS Model
# MAGIC
# MAGIC ### Define the Solver Class
# MAGIC The `TTSSolver` class encapsulates the logic for running the `utils.build_and_solve_multi_tier_tts` function for each disruption scenario. It will reuse the same Ray cluster defined above.
# MAGIC

# COMMAND ----------

class TTSSolver:
    """
    Callable class to run the Pyomo model for a single disrupted scenario.
    """
    
    def __init__(self, data=dataset):
        self.data = dataset

    def __call__(self, row):
        """Run the Pyomo model for a single disrupted scenario."""
        disrupted = [row['node']]
        # Call the utility function to build and solve the optimization model
        solver = utils.build_and_solve_multi_tier_tts(self.data, disrupted)
        row['termination_condition'] = str(solver.iloc[0]['termination_condition'])
        row['tts'] = solver.iloc[0]['tts']
        return row

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the Solver on a Single Row
# MAGIC This cell tests the `TTSSolver` class on a single row to ensure correctness before distributed execution.

# COMMAND ----------

TTSSolver()(df.take(1)[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run the Solver at Scale
# MAGIC
# MAGIC Let's solve the TTS model at scale. The following cell will run for approximately 30 minutes using the cluster configuration mentioned above.

# COMMAND ----------

df_tts = df.repartition(300).map(TTSSolver,
                                 num_cpus=1,
                                 concurrency=(4,20))
pandas_df_tts = df_tts.to_pandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analyze Results

# COMMAND ----------

import matplotlib.pyplot as plt

pandas_df_tts['delta'] = pandas_df_tts['ttr'] - pandas_df_tts['tts']
ax = pandas_df_tts.hist(column='delta', bins=20, grid=False, edgecolor='black', figsize=(10, 6))
plt.title('Histogram of TTR - TTS')
plt.xlabel('TTR - TTS')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
display(ax)

# COMMAND ----------

# MAGIC %md
# MAGIC Note that TTS represents the maximum amount of time the network can operate without performance loss when a specific node is disrupted. It becomes particularly important when a node’s TTR exceeds its TTS.
# MAGIC
# MAGIC Refer to the histogram above, which shows the distribution of differences between TTR and TTS for each node. Nodes with a negative TTR − TTS are generally not a concern—assuming the provided TTR values are accurate. However, nodes with a positive TTR − TTS may incur financial loss, especially those with a large gap.
# MAGIC
# MAGIC To enhance network resiliency, companies can engage in discussions with their suppliers to reduce TTR, increase TTS or explore alternative sourcing and diversification strategies.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Shutdown Ray Cluster

# COMMAND ----------

try:
    shutdown_ray_cluster()
except Exception:
    pass
try:
    ray.shutdown()
except Exception:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Delta Tables

# COMMAND ----------

# Databricks-only: save to Delta table
try:
    spark.createDataFrame(pandas_df_ttr).write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.stress_test_result_ttr")
except Exception as e:
    print(f"Warning: Could not save to Delta table: {e}")

try:
    spark.createDataFrame(pandas_df_tts).write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.stress_test_result_tts")
except Exception as e:
    print(f"Warning: Could not save to Delta table: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wrap Up
# MAGIC
# MAGIC In this notebook, we explored how to perform stress testing on a large supply chain network. We leveraged Ray on Databricks to distribute the simulation of thousands of disruption scenarios. We then analyzed the distribution of risk exposures across these scenarios and identified nodes that may require additional investment, as well as those that may have been previously over-invested.
# MAGIC
# MAGIC This concludes the main part of the solution accelerator. The next notebook, `04_appendix`, is optional. It dives into the mathematical formulation of the optimization problem, discusses key assumptions, and outlines ways to further extend the model.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | pyomo | An object-oriented algebraic modeling language in Python for structured optimization problems | BSD | https://pypi.org/project/pyomo/
# MAGIC | highspy | Linear optimization solver (HiGHS) | MIT | https://pypi.org/project/highspy/
