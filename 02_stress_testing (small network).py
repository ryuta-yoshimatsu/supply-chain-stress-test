# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/).

# COMMAND ----------

# MAGIC %sh apt-get update && apt-get install -y coinor-cbc

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

# COMMAND ----------

# Generate a synthetic 3-tier network dataset for optimization 
dataset = utils.generate_data(N1=5, N2=10, N3=20)

# COMMAND ----------

# Assign a random ttr to each disrupted node
random.seed(777)
disrupted_nodes = {node: random.randint(1, 10) for node in dataset['tier2'] + dataset['tier3']}

objectives = []
for disrupted_node in disrupted_nodes:
    disrupted = [disrupted_node]
    df = utils.build_and_solve_multi_tier_ttr(dataset, disrupted, disrupted_nodes[disrupted_node])
    objectives.append(df)

# COMMAND ----------

objectives = pd.concat(objectives, ignore_index=True)
display(objectives)

# COMMAND ----------

highest_risk_nodes = objectives.sort_values(by="objective_value", ascending=False)[0:5]
highest_risk_nodes

# COMMAND ----------

disrupted = ["T2_10"]
df = utils.build_and_solve_multi_tier_ttr(dataset, disrupted, disrupted_nodes[disrupted_node], True)

model = df["model"].values[0]
records = []
for v in model.component_data_objects(ctype=pyo.Var, active=True):
    idx  = v.index()
    record  = {
        "var_name"  : v.parent_component().name,
        "index"     : idx,
        "value"     : pyo.value(v),
        "lb"        : v.lb,
        "ub"        : v.ub,
        "fixed"     : v.fixed,
    }
    records.append(record)

for record in records:
        record["index"] = (record["index"],) if isinstance(record["index"], str) else record["index"]

display(pd.DataFrame.from_records(records))

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | pyomo | An object-oriented algebraic modeling language in Python for structured optimization problems | BSD | https://pypi.org/project/pyomo/
# MAGIC | coinor-cbc | COIN-OR Branch-and-Cut solver | Eclipse Public License - v 2.0 | https://github.com/coin-or/Cbc
