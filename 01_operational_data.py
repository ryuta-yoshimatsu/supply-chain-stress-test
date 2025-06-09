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
# MAGIC In the real world, the supply chain network already exists, so the primary task is to collect data and map the network. However, in this solution accelerator, we generate a synthetic dataset for the entire supply chain. This allows us to control the setup and better understand the methodology in depth.
# MAGIC
# MAGIC Let’s generate a three-tier supply chain network consisting of raw-material suppliers (Tier 3), direct suppliers (Tier 2), and finished-goods plants (Tier 1). The utility function `generate_data` outputs the network topology (directed edges) and key operational parameters, including inventory, capacity, demand, and profit margins. This data structure is designed to support optimization and stress-testing models that simulate how disruptions propagate through multi-tier supply chains.

# COMMAND ----------

# Generate a synthetic 3-tier supply chain network dataset for optimization
# N1: number of product nodes
# N2: number of direct supplier nodes
# N3: number of sub-supplier nodes
dataset = utils.generate_data(N1=5, N2=10, N3=20)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's visualize the network.

# COMMAND ----------

# Visualizes the 3-tier network
utils.visualize_network(dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC - Product (finished-goods plants: ●) at the top
# MAGIC - Tier 2 (direct suppliers: ■) in the middle
# MAGIC - Tier 3 (sub-suppliers: ▲) at the bottom
# MAGIC - Nodes with the same color produce and supply the same `supplier_material_type`.
# MAGIC - Grey is used for Product nodes that have no material-type code.
# MAGIC - Edges run from a parent node to a child node, illustrating material flow (Tier 3 ➜ Tier 2 ➜ Product).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Operational Data
# MAGIC
# MAGIC The operational data required to run stress tests on your supply chain network depends on how the optimization problem is formulated—that is, the objective function and its constraints. In this solution accelerator, we follow the formulation presented in the [paper](https://dspace.mit.edu/handle/1721.1/101782). The table below outlines the data elements used in this approach.
# MAGIC
# MAGIC For more details on the problem formulation and guidance on how to extend the model further, refer to the paper or the appendix notebook: `04_appendix`.

# COMMAND ----------

# MAGIC %md
# MAGIC  Variable                  | What it represents                                                                                 |
# MAGIC  ------------------------- | -------------------------------------------------------------------------------------------------- |
# MAGIC  **tier1 / tier2 / tier3** | Lists of node IDs in each tier.                                                                    |
# MAGIC  **edges**                 | Directed links `(source, target)` showing which node supplies which.                               |
# MAGIC  **material\_type**  | List of all material types. 
# MAGIC  **supplier\_material\_type**  | Material type each supplier produces and supplies.                                    |
# MAGIC  **f**                     | Profit margin for each Tier 1 node’s finished product.                                             |
# MAGIC  **s**                     | On-hand inventory units at every node.                                                             |
# MAGIC  **d**                     | Demand per time unit for Tier 1 products.                                       |
# MAGIC  **c**                     | Production capacity per time unit at each node.                                                          |
# MAGIC  **r**                     | Number of material types (k) required to make one unit of node j.              |
# MAGIC  **N\_minus**              | For each node j (Tier 1 or 2), the set of material types it requires.                              |
# MAGIC  **N\_plus**               | For each supplier i (Tier 2 or 3), the set of downstream nodes j it feeds.                     |
# MAGIC  **P**                     | For each `(j, material_part)` pair, a list of upstream suppliers i that provides it (multi-sourcing view). |

# COMMAND ----------

# MAGIC %md  
# MAGIC Let's take a peek at the `dataset` dictionary that contains all the data we need.

# COMMAND ----------

for key in dataset.keys():
  print(f"{key}: {dataset[key]}", "\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wrap Up
# MAGIC
# MAGIC In this notebook, we generated a synthetic three-tier supply chain network along with the corresponding operational data. We also reviewed the structure of the network and the key requirements for the operational data.
# MAGIC
# MAGIC In the next notebook, `02_stress_testing (small network)`, we will run multiple stress tests on the small network constructed here.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | pyomo | An object-oriented algebraic modeling language in Python for structured optimization problems | BSD | https://pypi.org/project/pyomo/
# MAGIC | highspy | Linear optimization solver (HiGHS) | MIT | https://pypi.org/project/highspy/
