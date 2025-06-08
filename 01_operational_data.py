# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/).

# COMMAND ----------

# MAGIC %md
# MAGIC # Prepare the Dataset
# MAGIC
# MAGIC This notebook generates a synthetic three-tier supply chain network along with operational data, which will be used in later notebooks for stress testing. We will review the properties of the supply chain network and the requirements for the operational data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster Configuration
# MAGIC This notebook was tested on the following Databricks cluster configuration:
# MAGIC - **Databricks Runtime Version:** 16.4 LTS ML (includes Apache Spark 3.5.2, Scala 2.12)
# MAGIC - **Single Node:** Standard_DS4_v2 (28 GB Memory, 8 Cores)
# MAGIC - **Photon Acceleration:** Disabled (Photon boosts Apache Spark workloads; not all ML workloads will see an improvement)

# COMMAND ----------

# DBTITLE 1,Install requirements
# MAGIC %pip install -r ./requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import modules
import random
import numpy as np
import pandas as pd
import scripts.utils as utils

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Supply Chain Network and Data
# MAGIC
# MAGIC Let's generate a three-tier supply chain network consisting of raw-material suppliers (Tier 3), direct suppliers (Tier 2), and finished-goods plants (Tier 1). The utility function `generate_data` produces the network topology (directed edges) and key operational parameters, including inventory, capacity, demand, and profit margins. This data structure is designed to support optimization and stress-testing models that analyze how disruptions propagate through multi-tier supply chains.

# COMMAND ----------

# Generate a synthetic 3-tier supply chain network dataset for optimization
# N1: number of product nodes
# N2: number of direct supplier nodes
# N3: number of sub-supplier nodes
dataset = utils.generate_data(N1=5, N2=10, N3=20)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's visualize the network.
# MAGIC
# MAGIC - Product (finished-goods plants: ●) at the top
# MAGIC - Tier 2 (direct suppliers: ■) in the middle
# MAGIC - Tier 3 (sub-suppliers: ▲) at the bottom
# MAGIC - Nodes with the same color produce the same `supplier_material_type`.
# MAGIC - Grey is used for Product nodes that have no material-type code.
# MAGIC - Edges run from a parent node to an immediate child node, illustrating material flow (Tier 3 ➜ Tier 2 ➜ Product).

# COMMAND ----------

# Visualizes the 3-tier network
utils.visualize_network(dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Operational Data

# COMMAND ----------

# MAGIC %md
# MAGIC  Variable                  | What it represents                                                                                 |
# MAGIC  ------------------------- | -------------------------------------------------------------------------------------------------- |
# MAGIC  **tier1 / tier2 / tier3** | Lists of node IDs in each tier.                                                                    |
# MAGIC  **edges**                 | Directed links `(source, target)` showing which node supplies which.                               |
# MAGIC  **supplier\_material\_type**  | Material type each supplier produces.                                    |
# MAGIC  **f**                     | Profit margin for each Tier 1 node’s finished product.                                             |
# MAGIC  **s**                     | On-hand inventory units at every node.                                                             |
# MAGIC  **d**                     | Demand per one time unit for Tier 1 products.                                       |
# MAGIC  **c**                     | Production capacity per one time unit at each node.                                                          |
# MAGIC  **r**                     | Number of material types (k) required to make one unit of node j.              |
# MAGIC  **N\_minus**              | For each node j (Tier 1 or 2), the set of material types it requires.                              |
# MAGIC  **N\_plus**               | For each supplier i (Tier 2 or 3), the set of downstream nodes j it feeds.                     |
# MAGIC  **P**                     | For each `(j, material)` pair, list of upstream suppliers i that can provide it (multi-sourcing view). |

# COMMAND ----------

for key in dataset.keys():
  print(f"{key}: {dataset[key]}", "\n")

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | pyomo | An object-oriented algebraic modeling language in Python for structured optimization problems | BSD | https://pypi.org/project/pyomo/
# MAGIC | highspy | Linear optimization solver (HiGHS) | MIT | https://pypi.org/project/highspy/
