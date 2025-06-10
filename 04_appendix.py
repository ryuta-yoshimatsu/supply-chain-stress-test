# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/).

# COMMAND ----------

# MAGIC %md
# MAGIC # Appendix
# MAGIC
# MAGIC In this notebook, we explore the mathematical formulation of the optimization problem, define the variables and discuss key assumptions. This solution accelerator is based closely on the models presented in the [paper](https://dspace.mit.edu/handle/1721.1/101782) (with slight modifications), and we recommend referring to it for further details.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Multi-Tier TTR Model
# MAGIC
# MAGIC The multi-tier time-to-recover (TTR) model represents a supply chain as a directed graph of materials and production sites. For a disruption lasting t(n), it chooses production quantities (u), inter-tier flows (y) and lost sales (l) that minimise total weighted loss across all finished products. 
# MAGIC
# MAGIC **Constraints**: (1) A bill-of-materials constraint limits each node’s output to the scarcest upstream material; (2) a flow-balance constraint caps shipments by on-hand inventory plus new production. (3) Disrupted nodes produce nothing. (4) Further constraints match cumulative demand, capturing unmet demand as loss, (5) and bound plant throughput by installed capacity. 
# MAGIC
# MAGIC The resulting linear program remains tractable for thousands of nodes.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="images/multi-tier-ttr.png" alt="Multi-Tier TTR" width="650">

# COMMAND ----------

# MAGIC %md
# MAGIC ## Multi-Tier TTS Model
# MAGIC
# MAGIC The multi-tier time-to-survive (TTS) model asks: given a disruption at a specific node, how long can the network continue meeting demand with **no** lost sales? It employs the same directed-graph representation as the TTR model, but its linear program maximises the survival horizon, t, rather than minimising lost profit.
# MAGIC
# MAGIC **Constraints**: Identical to the TTR model except for the fourth, where no loss is allowed; demand must be fully satisfied.

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC <img src="images/multi-tier-tts.png" alt="Multi-Tier TTS" width="650">

# COMMAND ----------

# MAGIC %md
# MAGIC ## Assumptions
# MAGIC
# MAGIC In both the TTR and TTS models, we simplify by assuming that processing lead times are negligible compared to the disruption's impact. We further assume that the costs associated with rerouting materials and manufacturing changeovers are also negligible relative to the disruption's effect. These assumptions are often reasonable in the context of high-impact disruptions, where their effects far outweigh those of these secondary factors. See the [paper](https://dspace.mit.edu/handle/1721.1/101782) for more details.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | pyomo | An object-oriented algebraic modeling language in Python for structured optimization problems | BSD | https://pypi.org/project/pyomo/
# MAGIC | highspy | Linear optimization solver (HiGHS) | MIT | https://pypi.org/project/highspy/
