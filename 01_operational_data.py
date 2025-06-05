# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/).

# COMMAND ----------

# MAGIC %md
# MAGIC # Prepare the Dataset
# MAGIC
# MAGIC TODO: This notebook demonstrates...

# COMMAND ----------

# MAGIC %pip install -r ./requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import random
import numpy as np
import pandas as pd
import scripts.utils as utils

# COMMAND ----------

# MAGIC %md
# MAGIC Build the three-tier network exactly with “close” Tier 3 → Tier 2 links (each Tier-3 node picks the nearest Tier-2 node, with a random tie-break when it sits between two). Guarantee no isolation – every Tier-1 node receives ≥ 1 edge from Tier-2, and every Tier-2 node receives ≥ 1 edge from Tier-3 (the assertion would fail otherwise).

# COMMAND ----------

# Generate a synthetic 3-tier network dataset for optimization 
dataset = utils.generate_data(N1=5, N2=10, N3=20)

# COMMAND ----------

# MAGIC %md
# MAGIC Visualise the tiers cleanly with distinctive shapes (●, ■, ▲) and a layered layout.

# COMMAND ----------

# Visualizes the 3-tier network
utils.visualize_network(dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC TODO: Walk through each variable

# COMMAND ----------

print("Tier sizes  :", len(dataset['tier1']), len(dataset['tier2']), len(dataset['tier3']))
print("Edges       :", len(dataset['edges']))
print("f:", {j: dataset['f'][j] for j in dataset['tier1']})
print("d:", {j: dataset['d'][j] for j in dataset['tier1']})
print("s:", {j: dataset['s'][j] for j in dataset['tier1']})
print("c:", {j: dataset['c'][j] for j in dataset['tier2']})

# COMMAND ----------

dataset

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | pyomo | An object-oriented algebraic modeling language in Python for structured optimization problems | BSD | https://pypi.org/project/pyomo/
# MAGIC | highspy | Linear optimization solver (HiGHS) | MIT | https://pypi.org/project/highspy/
