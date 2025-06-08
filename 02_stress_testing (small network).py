# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/).

# COMMAND ----------

# MAGIC %md
# MAGIC # Stress-Test Small Networks and Analyze the Results
# MAGIC
# MAGIC TODO: This notebook demonstrates

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster Configuration
# MAGIC This notebook was tested on the following Databricks cluster configuration:
# MAGIC - **Databricks Runtime Version:** 16.4 LTS ML (includes Apache Spark 3.5.2, Scala 2.12)
# MAGIC - **Single Node:** Standard_DS4_v2 (28 GB Memory, 8 Cores)
# MAGIC - **Photon Acceleration:** Disabled (Photon boosts Apache Spark workloads; not all ML workloads will see an improvement)

# COMMAND ----------

# MAGIC %pip install -r ./requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import mlflow
import random
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import scripts.utils as utils

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Dataset

# COMMAND ----------

# Generate a synthetic 3-tier network dataset for optimization 
dataset = utils.generate_data(N1=5, N2=10, N3=20)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Multi-Tier Time To Recover (TTR) Model

# COMMAND ----------

# Assign a random ttr to each disrupted node
random.seed(777)
disrupted_nodes = {node: random.randint(1, 10) for node in dataset['tier2'] + dataset['tier3']}

lost_profit = []
for disrupted_node in disrupted_nodes:
    disrupted = [disrupted_node]
    df = utils.build_and_solve_multi_tier_ttr(dataset, disrupted, disrupted_nodes[disrupted_node])
    lost_profit.append(df)

# COMMAND ----------

lost_profit = pd.concat(lost_profit, ignore_index=True)
display(lost_profit)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Highest Risk Nodes
# MAGIC The top 5 nodes with the highest lost profit are identified for further investigation.

# COMMAND ----------

# Visualizes the 3-tier network
utils.visualize_network(dataset)

highest_risk_nodes = lost_profit.sort_values(by="lost_profit", ascending=False)[0:5]
display(highest_risk_nodes)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inspect Specific Failures

# COMMAND ----------

scenario = "T2_8"

df = utils.build_and_solve_multi_tier_ttr(dataset, [scenario], disrupted_nodes[scenario], True)
model = df["model"].values[0]
records = []
for v in model.component_data_objects(ctype=pyo.Var, active=True):
    idx  = v.index()
    record  = {
        "var_name"  : v.parent_component().name,
        "index"     : idx,
        "value"     : pyo.value(v),
    }
    records.append(record)

# COMMAND ----------

demand = pd.DataFrame.from_dict(dataset["d"], orient='index').reset_index()
demand = demand.rename(columns={"index": "node", 0: "demand"})
demand["demand"] *= disrupted_nodes[scenario]

prod = pd.DataFrame([record for record in records if record["var_name"] == "u"])
prod = prod.rename(columns={"index": "node", "value": "production"})
prod = prod.drop("var_name", axis=1).sort_values(by="node").reset_index(drop=True)

stock = pd.DataFrame.from_dict(dataset["s"], orient='index').reset_index()
stock = stock.rename(columns={"index": "node", 0: "stock"})

lost = pd.DataFrame([record for record in records if record["var_name"] == "l"])
lost = lost.drop("var_name", axis=1).rename(columns={"index": "node", "value": "lost_demand"})

merged = demand.merge(prod, on="node", how="right")
merged = merged.merge(stock, on="node")
merged = merged.merge(lost, on="node", how="left")

display(merged)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Multi-Tier Time To Survive (TTS) Model

# COMMAND ----------

tts = []
for disrupted_node in disrupted_nodes:
    disrupted = [disrupted_node]
    df = utils.build_and_solve_multi_tier_tts(dataset, disrupted)
    df["ttr"] = disrupted_nodes[disrupted_node]
    tts.append(df)

# COMMAND ----------

tts = pd.concat(tts, ignore_index=True)

# COMMAND ----------

# Show where tts is shorter than ttr
display(tts[tts["tts"] < tts["ttr"]])

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | pyomo | An object-oriented algebraic modeling language in Python for structured optimization problems | BSD | https://pypi.org/project/pyomo/
# MAGIC | highspy | Linear optimization solver (HiGHS) | MIT | https://pypi.org/project/highspy/
