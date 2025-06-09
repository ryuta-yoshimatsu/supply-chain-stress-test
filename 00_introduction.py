# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/).

# COMMAND ----------

# MAGIC %md
# MAGIC <img src='https://github.com/databricks-industry-solutions/.github/raw/main/profile/solacc_logo_wide.png' width="1000" ></img>
# MAGIC
# MAGIC # Stress Test Supply Chain Networks at Scale
# MAGIC
# MAGIC Numerous past disruptions have exposed the fragility of global supply chains, challenging the efficiency and reliability of traditional operating models. In response, companies have adopted strategies such as reshoring production and stockpiling critical materials. While these measures aim to improve continuity, they often result in higher costs and increased financial risk. To better anticipate and manage such challenges, a well-cited [paper](https://dspace.mit.edu/handle/1721.1/101782) proposes stress testing supply chains using digital twins—virtual models built from real operational data. By simulating a range of disruption scenarios, businesses can identify vulnerabilities, assess potential impacts, and make informed, proactive decisions. This approach is gaining traction as a means to strengthen supply chain resilience.

# COMMAND ----------

# MAGIC %md <img src="images/cartoon.png" alt="Simplified Supply Chain Network" width="800">

# COMMAND ----------

# MAGIC %md
# MAGIC TODO: Write about the method described in the paper here.

# COMMAND ----------

# MAGIC %md
# MAGIC The remainder notebooks provide a detailed implementation of the solution and perform a comprehensive analysis on Databricks.
# MAGIC
# MAGIC TODO: Write about the sturcture of the solution acclerator here.
# MAGIC
# MAGIC TODO: Go thorugh the tools: pyomo, highs and ray here.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | pyomo | An object-oriented algebraic modeling language in Python for structured optimization problems | BSD | https://pypi.org/project/pyomo/
# MAGIC | highspy | Linear optimization solver (HiGHS) | MIT | https://pypi.org/project/highspy/
