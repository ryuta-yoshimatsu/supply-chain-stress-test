# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/).

# COMMAND ----------

# MAGIC %md
# MAGIC # Stress-Test Small Networks and Analyze the Results
# MAGIC
# MAGIC This notebook demonstrates how to run stress tests on a small supply chain network and analyze the results. We will introduce two key concepts—time-to-recover (TTR) and time-to-survive (TTS)—which are essential for understanding the risk exposure of disruption scenarios.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster Configuration
# MAGIC This notebook was tested on the following Databricks cluster configuration:
# MAGIC - **Databricks Runtime Version:** 16.4 LTS ML (includes Apache Spark 3.5.2, Scala 2.12)
# MAGIC - **Single Node** 
# MAGIC     - Azure: Standard_DS4_v2 (28 GB Memory, 8 Cores)
# MAGIC     - AWS: m5d.2xlarge (32 GB Memory, 8 Cores)
# MAGIC - **Photon Acceleration:** Disabled (Photon boosts Apache Spark workloads; not all ML workloads will see an improvement)

# COMMAND ----------

# DBTITLE 1,Install requirements
# MAGIC %pip install -r ./requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import modules
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
# MAGIC
# MAGIC As in the previous notebook, we will generate a supply chain network along with the corresponding operational data.

# COMMAND ----------

# Generate a synthetic 3-tier supply chain network dataset for optimization
# N1: number of product nodes
# N2: number of direct supplier nodes
# N3: number of sub-supplier nodes
dataset = utils.generate_data(N1=5, N2=10, N3=20) # DO NOT CHANGE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Multi-Tier Time-to-Recover (TTR) Model
# MAGIC
# MAGIC TTR (Time-to-Recover) is a key input to the optimization problem. It represents the time required for a node—or a group of nodes—to return to normal operations after a disruption. TTR is typically obtained from suppliers or estimated through internal assessments.
# MAGIC
# MAGIC In stress testing, we simulate disruptive scenarios in which one or more nodes fail. A failure can be either partial or complete and lasts for the duration of the TTR. In this solution accelerator, we simulate complete failures for each node. This is modeled by temporarily removing the node and re-optimizing the network based on the new topology. The objective function can be defined flexibly based on your business goals. In this exercise, we define it as the total lost profit across all products, which we aim to minimize. For the detailed formulation and model specification, see the function `utils.build_and_solve_multi_tier_ttr` in the script `scripts/utils.py` or refer to the notebook `04_appendix`.
# MAGIC
# MAGIC After finding the optimized configuration—through material and inventory reallocation and production adjustments—we obtain the cumulative lost profit incurred during the TTR period. This quantifies the impact of each disruption scenario. We repeat this process for all nodes to compare and assess the relative risk exposure associated with each one. 
# MAGIC

# COMMAND ----------

# Assign a random TTR to each disrupted node; in practice, TTR is a predefined input variable
random.seed(777) # DO NOT CHANGE!
disrupted_nodes = {node: random.randint(1, 10) for node in dataset['tier2'] + dataset['tier3']}

# COMMAND ----------

# MAGIC %md
# MAGIC We iterate through all nodes in Tier 2 and Tier 3 and simulate their respective disruption events.

# COMMAND ----------

lost_profit = []
for disrupted_node in disrupted_nodes:
    disrupted = [disrupted_node]
    df = utils.build_and_solve_multi_tier_ttr(dataset, disrupted, disrupted_nodes[disrupted_node])
    lost_profit.append(df)

lost_profit = pd.concat(lost_profit, ignore_index=True)
display(lost_profit)

# COMMAND ----------

# MAGIC %md
# MAGIC The result above indicates that not every failure leads to a finite lost profit. The magnitude of lost profit is influenced by several factors, including inventory levels, production capacity, supplier diversification, TTR, and more.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Highest Risk Nodes
# MAGIC Let's examine the top 5 nodes with the highest lost profit. We'll visualize the network again to help interpret the results more intuitively.

# COMMAND ----------

# Visualizes the 3-tier network
utils.visualize_network(dataset)

highest_risk_nodes = lost_profit.sort_values(by="lost_profit", ascending=False)[0:5]
display(highest_risk_nodes)

# COMMAND ----------

# MAGIC %md
# MAGIC Having `T2_10` as the node with the highest risk exposure is not surprising, as it is the sole supplier of a specific material type. If this node fails, the production of `T1_1`, `T1_3`, and `T1_5` will halt once existing inventory is depleted.
# MAGIC
# MAGIC `T2_8` is a more subtle case. At first glance, its high risk is not obvious. However, upon examining the nodes it supplies—`T1_1`, `T1_3`, and `T1_5`—we see that for `T1_1` and `T1_3`, `T2_8` is the only source of a specific material. This means its failure directly halts production at those nodes.
# MAGIC
# MAGIC For the same reason, `T3_15` is also a high-risk node, with disruption effects that propagate across multiple tiers. `T3_14` is in a similar situation; however, its TTR is 5—significantly shorter than `T3_15`'s TTR of 10. As a result, `T3_14` does not appear on the riskiest node list.
# MAGIC
# MAGIC Let's take a closer look at a specific failure scenario by inspecting the values of the decision variables identified by the linear solver. In the `utils.build_and_solve_multi_tier_ttr` function, you can set the parameter `return_model=True` to retrieve the model and access the decision variable values. Let's look into the `T2_8` failure.
# MAGIC

# COMMAND ----------

scenario = "T2_8"

# Solve the multi-tier TTR model for the given scenario
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
# MAGIC The first thing to notice in the output above is that the production volume of `T2_8` is zero due to the simulated failure (`production`). Next, we observe losses at `T1_1` and `T1_3` (`lost_demand`). In the case of `T1_3`, there is no production, so once the existing stock is depleted, the remaining demand directly becomes lost demand. For `T1_1`, it appears that all of `T2_8`'s inventory (1,569 units) was routed to this node, allowing some production to continue. The solver prioritized `T1_1` because it has a higher profit margin than `T1_3` (as shown in the `01_operational_data` notebook).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Multi-Tier Time-to-Survive (TTS) Model
# MAGIC
# MAGIC Time-to-Survive (TTS) provides a complementary perspective on the risk associated with node failures. Unlike TTR, TTS is not an input but an output—a decision variable. When a disruption affects a node or group of nodes, TTS represents the amount of time the reconfigured network can continue to meet demand without incurring any loss. The risk becomes more critical when TTR exceeds TTS by a significant margin.
# MAGIC
# MAGIC The TTS model closely resembles the TTR model, with minor modifications to the objective function and constraints. For more details, see the function `utils.build_and_solve_multi_tier_tts` in the script `scripts/utils.py` or refer to the `04_appendix` notebook.
# MAGIC
# MAGIC Let's again iterate through all nodes in Tier 2 and Tier 3.

# COMMAND ----------

tts = []
for disrupted_node in disrupted_nodes:
    disrupted = [disrupted_node]
    df = utils.build_and_solve_multi_tier_tts(dataset, disrupted)
    df["ttr"] = disrupted_nodes[disrupted_node]
    tts.append(df)

# Concatenate all dataframes in the list into a single dataframe
tts = pd.concat(tts, ignore_index=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Now, let's show where TTS is shorter than TTR.

# COMMAND ----------

# Show where tts is shorter than ttr
display(tts[tts["tts"] < tts["ttr"]])

# COMMAND ----------

# MAGIC %md
# MAGIC The result above highlights all (sub)supplier sites where the time-to-recover (TTR) exceeds the time-to-survive (TTS). To enhance network resiliency, these nodes can be reassessed. Potential actions include reducing TTR through supplier renegotiations, increasing TTS by building inventory buffers, or diversifying the sourcing strategy.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wrap Up
# MAGIC
# MAGIC In this notebook, we demonstrated how to run stress tests on a small supply chain network and analyze the results. We introduced two key concepts—time-to-recover (TTR) and time-to-survive (TTS)—and applied them in our analysis.
# MAGIC
# MAGIC In the next notebook, `03_stress_testing (large network)`, we will run similar stress tests on a much larger network consisting of thousands of nodes.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | pyomo | An object-oriented algebraic modeling language in Python for structured optimization problems | BSD | https://pypi.org/project/pyomo/
# MAGIC | highspy | Linear optimization solver (HiGHS) | MIT | https://pypi.org/project/highspy/
