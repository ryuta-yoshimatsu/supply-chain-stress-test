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

# MAGIC %md <img src="images/cartoon.png" alt="Simplified Supply Chain Network" width="1000">

# COMMAND ----------

# MAGIC %md
# MAGIC ## How does this method work?
# MAGIC
# MAGIC The methodology outlined in the paper—and implemented in this solution accelerator—models the supply chain as a directed graph of materials, plants, and flows. For each node, we capture inventory levels, production capacity, estimated time-to-recover (TTR) after a disruption, and, for product nodes, profit contribution. To simulate the failure of a specific node or a set of nodes, we “knock out” the corresponding node(s) for the duration of their TTR, then solve a linear optimisation problem that reallocates inventory, reroutes materials, and idles plants to minimise lost profit (or another selected metric such as volume or sales). We run this simulation across a comprehensive set of disruption scenarios at scale. Resulting impacts are ranked from negligible to catastrophic, while a complementary time-to-survive (TTS) metric indicates how long operations can continue if a node fails, highlighting sensitivity to TTR assumptions.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Why do we use this method for stress-testing?
# MAGIC
# MAGIC First, this method avoids the need for probability estimation, allowing business stakeholders to focus on visible vulnerabilities rather than guessing the likelihood of rare events. Second, because it is optimisation-based and relies only on MRP and purchasing / demand data, it scales to thousands of suppliers and can be refreshed weekly—or even daily. Third, the risk exposure metric uncovers hidden, low-spend materials whose failure could severely disrupt production, enabling smarter prioritisation of inventory buffers or dual sourcing. Fourth, the TTS metric highlights where TTR estimates may be unreliable, helping guide supplier discussions. Collectively, these advantages reduce analysis time from years to days and ensure mitigation efforts target the most critical risks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Which tools do we use?
# MAGIC
# MAGIC This method is flexible and can be implemented using variety of tools. In this solution accelerator, we will use [Pyomo](https://pyomo.readthedocs.io/en/stable/index.html) as a solver wrapper and [HiGHS](https://github.com/ERGO-Code/HiGHS) as a solver. Both are open source tools and are permitted for commercial use.
# MAGIC
# MAGIC Using Pyomo with the HiGHS solver lets analysts model and solve linear optimisation problems quickly and transparently. Pyomo provides a Python-native, algebraic modelling language, so analysts describe decisions, constraints and objectives in readable code that sits beside their data pipelines and dashboards. Because models are ordinary Python objects, you can version-control them, parameterise them from Pandas DataFrames, and embed them in notebooks, APIs or scheduled jobs without proprietary software. HiGHS supplies a state-of-the-art, open-source LP engine that rivals commercial solvers for speed and robustness, handling millions of variables while offering duals, reduced costs and warm-start capabilities. Together, Pyomo and HiGHS give you a zero-licence-fee stack that scales from a laptop to the cloud, fits inside Docker containers and integrates with ML workflows. That combination shortens prototype-to-production cycles, demystifies optimisation for stakeholders and frees budget for mitigation actions instead of solver licences, while ensuring auditability and regulatory compliance.

# COMMAND ----------

# MAGIC %md
# MAGIC The remainder notebooks provide a detailed implementation of the solution and perform a comprehensive analysis on Databricks.
# MAGIC
# MAGIC TODO: Write about the sturcture of the solution acclerator here.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | pyomo | An object-oriented algebraic modeling language in Python for structured optimization problems | BSD | https://pypi.org/project/pyomo/
# MAGIC | highspy | Linear optimization solver (HiGHS) | MIT | https://pypi.org/project/highspy/
